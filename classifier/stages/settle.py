import logging

from classifier.db import ClassifierDB
from classifier.stages.resolve import detect_resolved_events

logger = logging.getLogger(__name__)


def record_settlements(
    adapters,
    exchange_by_code: dict,
    registry,
    db: ClassifierDB,
    lookback_days: int,
    max_candidates: int,
    cursors: dict[str, int],
) -> dict:
    """Records the final settlement value of every recently deactivated, unsettled outcome its venue has settled.

    Looks each candidate up on its venue rather than relying on the resolve pipeline, which only reports ids, never
    revisits a market it has reported, and can't see Hyperliquid outcomes once they settle (they leave outcomeMeta).
    Adapters return only values their venue has made final, so a market that merely closed is left for a later cycle.

    Each exchange looks at no more than `max_candidates` per cycle, continuing from where its last cycle stopped
    (`cursors`, by exchange code, updated in place) and wrapping around, so a backlog drains over several cycles
    without holding up the fetch loop, and outcomes that never settle can't starve the rest.
    """
    recorded = 0
    failed: list[str] = []
    settled_by_exchange: dict[int, set[str]] = {}
    for adapter in adapters:
        exchange = exchange_by_code.get(adapter.exchange_code)
        if not exchange:
            continue
        after_id = cursors.get(adapter.exchange_code, 0)
        page_size = min(max_candidates, getattr(adapter, "settle_page_size", max_candidates))
        candidates = db.get_unsettled_contracts(
            exchange.exchange_id, lookback_days, after_id, page_size,
            include_active=getattr(adapter, "settle_active_listings", False),
        )
        # A short page means the end was reached, so the next cycle starts again from the beginning.
        cursors[adapter.exchange_code] = candidates[-1][0] if len(candidates) == page_size else 0
        if not candidates:
            continue
        try:
            prices = adapter.fetch_settlements({security_id for _, security_id in candidates})
        except Exception as e:
            logger.error("Failed to fetch settlements from %s: %s", adapter.exchange_code, e)
            failed.append(adapter.exchange_code)
            continue
        items = [
            {"event_contract_id": event_contract_id, "settlement_price": prices[security_id]}
            for event_contract_id, security_id in candidates
            if security_id in prices
        ]
        if items:
            registry.bulk_patch_event_contracts(items)
            recorded += len(items)
            settled_by_exchange[exchange.exchange_id] = {security_id for security_id in prices}
        logger.info("Recorded %d of %d candidate settlements from %s", len(items), len(candidates), adapter.exchange_code)
    # A final value means the market is over: retire whatever is still active, as resolve would had it seen it.
    resolution = detect_resolved_events(settled_by_exchange, registry, db) if settled_by_exchange else {}
    return {
        "settlements_recorded": recorded,
        "failed_exchanges": failed,
        "listings_deactivated": resolution.get("listings_deactivated", 0),
        "events_resolved": resolution.get("events_resolved", 0),
    }
