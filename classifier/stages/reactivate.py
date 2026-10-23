import logging
from collections.abc import Container
from datetime import timedelta

from classifier.constants import REACTIVATION_WARNING_THRESHOLD
from classifier.db import ClassifierDB

logger = logging.getLogger(__name__)


def reactivate_listings(
    exchange_ids: set[int],
    fetched_keys: Container[str],
    registry,
    db: ClassifierDB,
    max_age: timedelta,
) -> dict:
    """Turns back on the listings deactivated within `max_age`, with no settlement value, that a fetch of their exchange
    lists as open again, along with their securities, and un-resolves their events.

    Every way a listing is turned off reads a venue that can be wrong (a bulk list of closed markets, a feed missing an
    event for a day), while an exchange's fetch lists only what trades now. Only exchanges whose fetch succeeded are
    passed in; `fetched_keys` holds that fetch's "exchange_id:exchange_security_id" keys.
    """
    listing_ids: list[int] = []
    security_ids: set[int] = set()
    for exchange_id in sorted(exchange_ids):
        reopened = [
            (listing_id, security_id)
            for listing_id, security_id, exchange_security_id in db.get_retired_unsettled_listings(exchange_id, max_age)
            if f"{exchange_id}:{exchange_security_id}" in fetched_keys
        ]
        if len(reopened) >= REACTIVATION_WARNING_THRESHOLD:
            # Counted by the ListingsReactivated metric filter (cdk/lib/stacks/classifier-stack.ts); keep the wording.
            logger.warning("Exchange %d lists %d retired listings as open again; reactivating them",
                           exchange_id, len(reopened))
        listing_ids.extend(listing_id for listing_id, _ in reopened)
        security_ids.update(security_id for _, security_id in reopened)

    if not listing_ids:
        return {"listings_reactivated": 0, "securities_reactivated": 0, "events_unresolved": 0}

    registry.bulk_patch_listings([{"listing_id": lid, "active": True} for lid in listing_ids])
    registry.bulk_patch_securities([{"security_id": sid, "active": True} for sid in sorted(security_ids)])
    event_ids = sorted(db.get_resolved_event_ids_for_securities(sorted(security_ids)))
    if event_ids:
        registry.bulk_patch_events([{"event_id": eid, "resolved": False, "resolved_at": None} for eid in event_ids])

    logger.info("Reactivation: %d listings, %d securities, %d events un-resolved",
                len(listing_ids), len(security_ids), len(event_ids))
    return {
        "listings_reactivated": len(listing_ids),
        "securities_reactivated": len(security_ids),
        "events_unresolved": len(event_ids),
    }
