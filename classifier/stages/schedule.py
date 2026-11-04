"""Events' scheduled start times (`sm.event.start_time`) as the venues list them now: Kalshi's milestone, Polymarket's
game start time, Hyperliquid's scheduled start. A venue moves a rescheduled game here while its ticker and rules keep
the original time. The registry logs each change (`sm.event_schedule_change`).
"""
import logging
from collections.abc import Iterable
from datetime import datetime

from classifier.adapters.types import AdapterContract

logger = logging.getLogger(__name__)

PATCH_CHUNK = 500


def event_starts(contracts: Iterable[AdapterContract]) -> dict[tuple[int, str], tuple[str, str]]:
    """Each event's (start time, source) from its contracts, where the venue gives one."""
    starts: dict[tuple[int, str], tuple[str, str]] = {}
    for contract in contracts:
        if contract.event_start_time is not None:
            starts[(contract.exchange_id, contract.exchange_event_native_id)] = (
                contract.event_start_time, contract.event_start_source)
    return starts


def _same(stored: datetime | None, start: str) -> bool:
    return stored is not None and stored.replace(tzinfo=None) == datetime.fromisoformat(start).replace(tzinfo=None)


def write_start_times(db, registry, starts: dict[tuple[int, str], tuple[str, str]]) -> int:
    """Writes the start times that differ from what the registry has; returns how many. Only readings are written: a
    venue that gives no start time this cycle (Kalshi's milestone window passed the game) leaves the stored one."""
    stored = db.get_event_start_times(list(starts))
    items = []
    for key, (event_id, current) in stored.items():
        start, source = starts[key]
        if _same(current, start):
            continue
        if current is not None:
            # Log wording kept stable for searching the logs for reschedules.
            logger.info("event rescheduled: event=%d from=%s to=%s source=%s", event_id, current.isoformat(), start,
                        source)
        items.append({"event_id": event_id, "start_time": start, "start_time_source": source})
    for i in range(0, len(items), PATCH_CHUNK):
        registry.bulk_patch_events(items[i:i + PATCH_CHUNK])
    if items:
        logger.info("Wrote start times for %d of %d events", len(items), len(stored))
    return len(items)
