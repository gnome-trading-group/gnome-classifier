import logging
from datetime import datetime, timezone

from classifier.db import ClassifierDB

logger = logging.getLogger(__name__)


def update_stale_tracker(
    tracker: dict[str, dict],
    active_by_exchange: dict[int, set[str]],
    failed_exchange_ids: set[int],
    miss_threshold: int,
    max_messages: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Returns (stale_messages, new_tracker)."""
    new_tracker: dict[str, dict] = {}
    all_stale: list[dict] = []

    for tk, entry in tracker.items():
        exchange_id = entry["exchange_id"]
        native_event_id = entry["native_event_id"]
        miss_count = entry["miss_count"]

        if exchange_id in failed_exchange_ids:
            new_tracker[tk] = entry
            continue

        if native_event_id in active_by_exchange.get(exchange_id, set()):
            new_tracker[tk] = {"exchange_id": exchange_id, "native_event_id": native_event_id, "miss_count": 0}
        else:
            miss_count += 1
            if miss_count >= miss_threshold:
                all_stale.append({"type": "stale", "exchange_id": exchange_id, "native_event_id": native_event_id})
            else:
                new_tracker[tk] = {"exchange_id": exchange_id, "native_event_id": native_event_id, "miss_count": miss_count}

    for exchange_id, native_ids in active_by_exchange.items():
        for native_event_id in native_ids:
            tk = f"{exchange_id}:{native_event_id}"
            if tk not in new_tracker:
                new_tracker[tk] = {"exchange_id": exchange_id, "native_event_id": native_event_id, "miss_count": 0}

    stale_messages = all_stale[:max_messages]
    if len(all_stale) > max_messages:
        logger.info("Capping stale from %d to %d messages", len(all_stale), max_messages)

    for msg in all_stale[max_messages:]:
        tk = f"{msg['exchange_id']}:{msg['native_event_id']}"
        new_tracker[tk] = {"exchange_id": msg["exchange_id"], "native_event_id": msg["native_event_id"], "miss_count": miss_threshold}

    return stale_messages, new_tracker


def deactivate_stale_events(
    stale_native_keys: list[tuple[int, str]],
    registry,
    db: ClassifierDB,
    debug: bool = False,
) -> dict:
    listing_ids_to_deactivate: list[int] = []
    candidate_security_ids: set[int] = set()

    for exchange_id, native_event_id in stale_native_keys:
        event_id = db.get_exchange_event(exchange_id, native_event_id)
        if event_id is None:
            continue

        security_ids = db.get_security_ids_for_events([event_id])
        if not security_ids:
            continue

        listings = db.get_active_listings_for_securities(list(security_ids))
        for lid, sid, lex_id, _ in listings:
            if lex_id == exchange_id:
                listing_ids_to_deactivate.append(lid)
                candidate_security_ids.add(sid)

    if not candidate_security_ids:
        return {"events_resolved": 0, "securities_deactivated": 0, "listings_deactivated": 0, "resolved_event_ids": [], "resolved_event_names": []}

    if debug:
        logger.info("[DEBUG] stale: deactivating %d listings", len(listing_ids_to_deactivate))
    if listing_ids_to_deactivate:
        registry.bulk_patch_listings([
            {"listing_id": lid, "active": False} for lid in listing_ids_to_deactivate
        ])

    still_active = db.get_securities_with_active_listings(list(candidate_security_ids))
    security_ids_to_deactivate = list(candidate_security_ids - still_active)

    if debug and security_ids_to_deactivate:
        for sid in security_ids_to_deactivate:
            logger.info("[DEBUG] stale: security_id=%d fully deactivated (no active listings)", sid)

    if security_ids_to_deactivate:
        registry.bulk_patch_securities([
            {"security_id": sid, "active": False} for sid in security_ids_to_deactivate
        ])

    event_ids_to_check: set[int] = set()
    for sid in security_ids_to_deactivate:
        event_ids_to_check.update(db.get_event_ids_for_security(sid))

    resolved_event_ids: list[int] = []
    for eid in event_ids_to_check:
        if db.get_active_security_count_for_event(eid) == 0:
            resolved_event_ids.append(eid)

    event_info = db.get_events(resolved_event_ids) if resolved_event_ids else {}

    if resolved_event_ids:
        now = datetime.now(timezone.utc).isoformat()
        registry.bulk_patch_events([
            {"event_id": eid, "resolved": True, "resolved_at": now}
            for eid in resolved_event_ids
        ])
        db.delete_embeddings(resolved_event_ids)

    if debug:
        for eid in resolved_event_ids:
            title = event_info.get(eid, {}).get("title", "?")
            logger.info("[DEBUG] stale: event_id=%d %r fully resolved", eid, title[:80])

    logger.info(
        "Stale cleanup: %d events resolved, %d securities deactivated, %d listings deactivated",
        len(resolved_event_ids), len(security_ids_to_deactivate), len(listing_ids_to_deactivate),
    )
    return {
        "events_resolved": len(resolved_event_ids),
        "securities_deactivated": len(security_ids_to_deactivate),
        "listings_deactivated": len(listing_ids_to_deactivate),
        "resolved_event_ids": resolved_event_ids,
        "resolved_event_names": [event_info[eid]["title"] for eid in resolved_event_ids if eid in event_info],
    }
