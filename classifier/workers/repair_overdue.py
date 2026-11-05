"""Settles and resolves events still open long after they expired. The regular cycles never look at them again: settle
takes outcomes deactivated within its lookback, and resolve, closure checks and stale cleanup only notice markets as
they leave the feed, so an event missed then stays open for good.

Each outcome is looked up on its venue as the settle stage does, without the hourly CLOB budget that paces the regular
cycles (most of a Polymarket International backlog is only on the CLOB, closed with its winner): values are recorded, closed listings deactivated and
events whose every outcome has a value resolved, by the same steps. On a venue whose lookup leaves out only markets it
doesn't have (`omits_only_unknown_markets`), an event with nothing open and no value pending whose markets are partly or
wholly gone is resolved as stale cleanup would, without values for the gone ones.

Run it where the workers run (it reads the database), as a one-off task on the Fetch task definition with the command
`repair-overdue --exchange KALSHI` (dry run) or with `--apply` added.
"""
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from classifier.adapters import ADAPTERS
from classifier.adapters.polymarket_intl import PolymarketIntlAdapter
from classifier.adapters.settlement import MarketState
from classifier.db import ClassifierDB
from classifier.stages.fetch import fetch_exchanges
from classifier.stages.resolve import resolve_settled_events
from classifier.stages.settle import apply_market_states, empty_counts, merge_counts
from classifier.stages.stale import deactivate_stale_events
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)

EVENTS_PER_BATCH = 1000


def unpaced_adapters() -> list:
    return [PolymarketIntlAdapter(clob_lookups_per_hour=None) if isinstance(adapter, PolymarketIntlAdapter) else adapter
            for adapter in ADAPTERS]


@dataclass
class Plan:
    """What the venue's answer means for a batch of overdue events."""
    states: dict[str, MarketState] = field(default_factory=dict)
    # Events whose every outcome now has a value.
    settled: list[int] = field(default_factory=list)
    # Events with nothing open or pending whose markets the venue no longer has, keyed for stale cleanup.
    gone: list[tuple[int, str]] = field(default_factory=list)
    # Events left open, by why.
    left: Counter = field(default_factory=Counter)


def plan_batch(outcomes: list[tuple[int, str, str]], states: dict[str, MarketState], trust_missing: bool) -> Plan:
    plan = Plan(states=states)
    by_event: dict[tuple[int, str], list[str]] = defaultdict(list)
    for event_id, native_event_id, security_id in outcomes:
        by_event[(event_id, native_event_id)].append(security_id)
    for (event_id, native_event_id), security_ids in by_event.items():
        found = [states[sid] for sid in security_ids if sid in states]
        if any(state.open for state in found):
            plan.left["still open on the venue"] += 1
        elif any(state.settlement is None for state in found):
            plan.left["closed, value not final yet"] += 1
        elif len(found) == len(security_ids):
            plan.settled.append(event_id)
        elif trust_missing:
            plan.gone.append((event_id, native_event_id))
        else:
            plan.left["not found on the venue"] += 1
    return plan


def repair_overdue(registry: RegistryClient, db: ClassifierDB, exchange_code: str | None, expired_days: int,
                   apply: bool) -> None:
    exchange_by_code = fetch_exchanges(registry, exchange_code)
    for adapter in unpaced_adapters():
        exchange = exchange_by_code.get(adapter.exchange_code)
        if not exchange:
            continue
        trust_missing = getattr(adapter, "omits_only_unknown_markets", False)
        outcomes = db.get_overdue_outcomes(exchange.exchange_id, timedelta(days=expired_days))
        event_ids = sorted({event_id for event_id, _, _ in outcomes})
        logger.info("%s: %d events expired over %d days ago and unresolved, %d outcomes", adapter.exchange_code,
                    len(event_ids), expired_days, len(outcomes))

        counts, settled, gone, left = empty_counts(), 0, 0, Counter()
        for start in range(0, len(event_ids), EVENTS_PER_BATCH):
            last = event_ids[min(start + EVENTS_PER_BATCH, len(event_ids)) - 1]
            batch = [o for o in outcomes if event_ids[start] <= o[0] <= last]
            plan = plan_batch(batch, adapter.fetch_market_states({sid for _, _, sid in batch}), trust_missing)
            settled, gone, left = settled + len(plan.settled), gone + len(plan.gone), left + plan.left
            if apply:
                merge_counts(counts, apply_market_states(exchange.exchange_id, plan.states, registry, db))
                # Events whose values were all recorded before but that were never resolved get no new value above.
                counts["events_resolved"] += len(resolve_settled_events(set(plan.settled), set(), registry, db)[0])
                if plan.gone:
                    stale = deactivate_stale_events([(exchange.exchange_id, n) for _, n in plan.gone], registry, db)
                    counts["events_resolved"] += stale["events_resolved"]
                    counts["listings_deactivated"] += stale["listings_deactivated"]
            logger.info("%s: %d/%d events looked up", adapter.exchange_code, min(start + EVENTS_PER_BATCH,
                                                                                len(event_ids)), len(event_ids))

        logger.info("%s: %s %d events with every value final, %d whose markets the venue no longer has; left open: %s",
                    adapter.exchange_code, "resolved" if apply else "would resolve", settled, gone, dict(left))
        if apply:
            logger.info("%s: recorded %d settlements, deactivated %d listings, resolved %d events",
                        adapter.exchange_code, counts["settlements_recorded"], counts["listings_deactivated"],
                        counts["events_resolved"])
    if not apply:
        logger.info("Dry run: nothing written. Re-run with --apply to settle and resolve.")
