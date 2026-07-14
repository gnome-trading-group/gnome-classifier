import logging
from datetime import datetime, timezone

from classifier.db import ClassifierDB

logger = logging.getLogger(__name__)


def detect_resolved_events(
    resolved_by_exchange: dict[int, set[str]],
    registry,
    db: ClassifierDB,
) -> dict:
    listing_ids_to_deactivate: list[int] = []
    candidate_security_ids: set[int] = set()

    for exchange_id, resolved_security_ids in resolved_by_exchange.items():
        if not resolved_security_ids:
            continue
        rows = db.get_active_listings_by_exchange_security(
            exchange_id, list(resolved_security_ids)
        )
        for listing_id, security_id, _ in rows:
            listing_ids_to_deactivate.append(listing_id)
            candidate_security_ids.add(security_id)

    if not candidate_security_ids:
        return {
            "events_resolved": 0,
            "securities_deactivated": 0,
            "listings_deactivated": 0,
        }

    if listing_ids_to_deactivate:
        registry.bulk_patch_listings([
            {"listing_id": lid, "active": False} for lid in listing_ids_to_deactivate
        ])

    # Only deactivate securities whose remaining active listings are all being deactivated
    still_have_active = db.get_securities_with_active_listings(list(candidate_security_ids))
    security_ids_to_deactivate = list(candidate_security_ids - still_have_active)

    if security_ids_to_deactivate:
        registry.bulk_patch_securities([
            {"security_id": sid, "active": False} for sid in security_ids_to_deactivate
        ])

    event_ids_to_check: set[int] = set()
    for security_id in security_ids_to_deactivate:  # only fully-resolved securities
        event_ids_to_check.update(db.get_event_ids_for_security(security_id))

    resolved_event_ids: list[int] = []
    for event_id in event_ids_to_check:
        if db.get_active_security_count_for_event(event_id) == 0:
            resolved_event_ids.append(event_id)

    if resolved_event_ids:
        now = datetime.now(timezone.utc).isoformat()
        registry.bulk_patch_events([
            {"event_id": eid, "resolved": True, "resolved_at": now}
            for eid in resolved_event_ids
        ])

    logger.info(
        "Resolution: %d events resolved, %d securities deactivated, %d listings deactivated",
        len(resolved_event_ids), len(security_ids_to_deactivate), len(listing_ids_to_deactivate),
    )
    return {
        "events_resolved": len(resolved_event_ids),
        "securities_deactivated": len(security_ids_to_deactivate),
        "listings_deactivated": len(listing_ids_to_deactivate),
    }
