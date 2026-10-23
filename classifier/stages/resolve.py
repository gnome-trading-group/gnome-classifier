import logging
from datetime import datetime, timezone

from classifier.db import ClassifierDB

logger = logging.getLogger(__name__)


def deactivate_closed_listings(
    closed_by_exchange: dict[int, set[str]],
    registry,
    db: ClassifierDB,
    debug: bool = False,
) -> dict:
    """Deactivates the active listings of contracts their venue no longer trades, and securities left with none.

    Never resolves an event: a venue's word that a market closed can be wrong (Polymarket US once listed every open
    market as closed), and a listing it turns off comes back on when a fetch sees the market open again. Only a
    settlement value resolves an event (resolve_settled_events), or an event leaving its venue's feed for a day (stale).
    """
    listing_ids_to_deactivate: list[int] = []
    candidate_security_ids: set[int] = set()

    for exchange_id, closed_security_ids in closed_by_exchange.items():
        if not closed_security_ids:
            continue
        rows = db.get_active_listings_by_exchange_security(exchange_id, list(closed_security_ids))
        for listing_id, security_id, _ in rows:
            listing_ids_to_deactivate.append(listing_id)
            candidate_security_ids.add(security_id)

    if not listing_ids_to_deactivate:
        return {"securities_deactivated": 0, "listings_deactivated": 0}

    if debug:
        logger.info("[DEBUG] closed: deactivating %d listings", len(listing_ids_to_deactivate))
    registry.bulk_patch_listings([{"listing_id": lid, "active": False} for lid in listing_ids_to_deactivate])

    # Only deactivate securities whose remaining active listings are all being deactivated
    still_have_active = db.get_securities_with_active_listings(list(candidate_security_ids))
    security_ids_to_deactivate = list(candidate_security_ids - still_have_active)
    if security_ids_to_deactivate:
        registry.bulk_patch_securities([{"security_id": sid, "active": False} for sid in security_ids_to_deactivate])

    logger.info("Closed: %d securities deactivated, %d listings deactivated",
                len(security_ids_to_deactivate), len(listing_ids_to_deactivate))
    return {
        "securities_deactivated": len(security_ids_to_deactivate),
        "listings_deactivated": len(listing_ids_to_deactivate),
    }


def resolve_settled_events(
    event_ids: set[int],
    just_settled: set[int],
    registry,
    db: ClassifierDB,
) -> tuple[list[int], list[str]]:
    """Resolves those of these events whose every outcome has a settlement value, returning their ids and titles.

    `just_settled` are event contract ids recorded a moment ago, counted as settled even if the database read below
    doesn't show them yet (the dry run records into a stub).
    """
    settlements = db.get_unresolved_event_settlements(list(event_ids))
    resolved_event_ids = sorted(
        event_id for event_id, contracts in settlements.items()
        if all(is_settled or ec_id in just_settled for ec_id, is_settled in contracts.items())
    )
    if not resolved_event_ids:
        return [], []

    now = datetime.now(timezone.utc).isoformat()
    registry.bulk_patch_events([{"event_id": eid, "resolved": True, "resolved_at": now} for eid in resolved_event_ids])
    db.delete_embeddings(resolved_event_ids)
    event_info = db.get_events(resolved_event_ids)
    return resolved_event_ids, [event_info[eid]["title"] for eid in resolved_event_ids if eid in event_info]
