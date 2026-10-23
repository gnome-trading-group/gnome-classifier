import logging
from datetime import timedelta

from classifier.adapters.settlement import MarketState
from classifier.db import ClassifierDB
from classifier.stages.resolve import deactivate_closed_listings, resolve_settled_events

logger = logging.getLogger(__name__)


def empty_counts() -> dict:
    return {
        "settlements_recorded": 0,
        "listings_deactivated": 0,
        "securities_deactivated": 0,
        "events_resolved": 0,
        "resolved_event_ids": [],
        "resolved_event_names": [],
        "failed_exchanges": [],
    }


def merge_counts(total: dict, part: dict) -> dict:
    for key, value in part.items():
        total[key] = total[key] + value
    return total


def apply_market_states(exchange_id: int, states: dict[str, MarketState], registry, db: ClassifierDB) -> dict:
    """Records the settlement values among `states`, deactivates what the venue no longer trades, and resolves the
    events whose every outcome now has a value."""
    counts = empty_counts()
    prices = {security_id: state.settlement for security_id, state in states.items() if state.settlement is not None}
    contracts = db.get_unsettled_event_contracts(exchange_id, sorted(prices))
    items = [
        {"event_contract_id": event_contract_id, "settlement_price": prices[security_id]}
        for security_id, (event_contract_id, _) in contracts.items()
    ]
    if items:
        registry.bulk_patch_event_contracts(items)
        counts["settlements_recorded"] = len(items)

    closed = {security_id for security_id, state in states.items() if not state.open}
    if closed:
        counts.update(deactivate_closed_listings({exchange_id: closed}, registry, db))

    if items:
        event_ids = {event_id for _, event_id in contracts.values()}
        resolved_ids, resolved_names = resolve_settled_events(
            event_ids, {item["event_contract_id"] for item in items}, registry, db)
        counts.update(events_resolved=len(resolved_ids), resolved_event_ids=resolved_ids,
                      resolved_event_names=resolved_names)
    return counts


def _look_up(adapter, exchange_id: int, security_ids: set[str], registry, db: ClassifierDB) -> dict | None:
    """The venue's states for these contracts, applied; None if the venue couldn't be asked."""
    try:
        states = adapter.fetch_market_states(security_ids)
    except Exception as e:
        # Counted by the FetchCycleFailures metric filter (cdk/lib/stacks/classifier-stack.ts); keep the wording.
        logger.error("Failed to fetch settlements from %s: %s", adapter.exchange_code, e)
        return None
    return apply_market_states(exchange_id, states, registry, db)


def check_closures(
    adapters,
    exchange_by_code: dict,
    left_feed_by_exchange: dict[int, set[str]],
    registry,
    db: ClassifierDB,
    max_lookups: int,
) -> dict:
    """Asks each venue about the contracts that just left its feed: those it says are over are deactivated, and settled
    if their value is final, a fetch after they close rather than up to a resolve interval later.

    A contract that is still open (the feed missed it) or that the venue doesn't know (stale cleanup's case) is left
    alone. So many leaving at once is a feed fault rather than markets closing, and is left to the resolve cycle.
    """
    counts = empty_counts()
    for adapter in adapters:
        exchange = exchange_by_code.get(adapter.exchange_code)
        left_feed = left_feed_by_exchange.get(exchange.exchange_id) if exchange else None
        if not left_feed:
            continue
        if len(left_feed) > max_lookups:
            # Counted by the ClosureChecksSkipped metric filter (cdk/lib/stacks/classifier-stack.ts); keep the wording.
            logger.warning("%d contracts left the %s feed at once; not checking them this cycle",
                           len(left_feed), adapter.exchange_code)
            continue
        result = _look_up(adapter, exchange.exchange_id, left_feed, registry, db)
        if result is None:
            counts["failed_exchanges"].append(adapter.exchange_code)
            continue
        logger.info("Closure check on %s: %d contracts left the feed; %d listings deactivated, %d settled",
                    adapter.exchange_code, len(left_feed), result["listings_deactivated"],
                    result["settlements_recorded"])
        merge_counts(counts, result)
    return counts


def record_settlements(
    adapters,
    exchange_by_code: dict,
    registry,
    db: ClassifierDB,
    max_age: timedelta,
    max_candidates: int,
    cursors: dict[str, int] | None = None,
) -> dict:
    """Records the final settlement value of every outcome deactivated within `max_age` that its venue has settled.

    Looks each candidate up on its venue rather than relying on the resolve pipeline, which only reports ids, never
    revisits a market it has reported, and can't see Hyperliquid outcomes once they settle (they leave outcomeMeta).
    Adapters report only values their venue has made final, so a market that merely closed is left for a later cycle.

    With `cursors` (by exchange code, updated in place) this is the sweep of the whole lookback: each exchange looks at
    no more than `max_candidates` per cycle, continuing from where its last cycle stopped and wrapping around, so a
    backlog drains over several cycles and outcomes that never settle can't starve the rest. It also looks at the
    active outcomes of venues that never retire a settled listing. Without, it is the pass over recently deactivated
    outcomes after each fetch, which are few enough to take from the start every time.
    """
    counts = empty_counts()
    for adapter in adapters:
        exchange = exchange_by_code.get(adapter.exchange_code)
        if not exchange:
            continue
        sweep = cursors is not None
        after_id = cursors.get(adapter.exchange_code, 0) if sweep else 0
        page_size = min(max_candidates, getattr(adapter, "settle_page_size", max_candidates))
        candidates = db.get_unsettled_contracts(
            exchange.exchange_id, max_age, after_id, page_size,
            include_active=sweep and getattr(adapter, "settle_active_listings", False),
        )
        if sweep:
            # A short page means the end was reached, so the next cycle starts again from the beginning.
            cursors[adapter.exchange_code] = candidates[-1][0] if len(candidates) == page_size else 0
        if not candidates:
            continue
        result = _look_up(adapter, exchange.exchange_id, {security_id for _, security_id in candidates}, registry, db)
        if result is None:
            counts["failed_exchanges"].append(adapter.exchange_code)
            continue
        logger.info("Recorded %d of %d %scandidate settlements from %s", result["settlements_recorded"],
                    len(candidates), "" if sweep else "recent ", adapter.exchange_code)
        merge_counts(counts, result)
    return counts
