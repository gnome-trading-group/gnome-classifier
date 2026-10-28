import logging
from collections.abc import Iterable

from classifier.adapters.types import AdapterContract

logger = logging.getLogger(__name__)


def event_volumes(contracts: Iterable[AdapterContract]) -> dict[tuple[int, str], float]:
    """Each event's 24h volume from its contracts: adapters put the event's volume on every contract it has."""
    volumes: dict[tuple[int, str], float] = {}
    for contract in contracts:
        if contract.event_volume is not None:
            volumes[(contract.exchange_id, contract.exchange_event_native_id)] = contract.event_volume
    return volumes


class VolumeWriter:
    """Writes events' 24h volume to the registry (`sm.event.volume_24h`), which collection rules select events by.

    A reading is written when it moved at least ``min_change`` (a share) from the last one written, or that one is
    ``max_age_seconds`` old, so the registry follows volume without a write per event every fetch. Remembers what it
    last wrote in memory only, so a restarted worker writes every known event once.
    """

    def __init__(self):
        self._written: dict[int, tuple[float, float]] = {}

    def write(
        self,
        db,
        registry,
        volumes: dict[tuple[int, str], float],
        now: float,
        min_change: float,
        max_age_seconds: float,
    ) -> int:
        """Writes the readings that changed enough or went stale for events the registry has; returns how many."""
        event_ids = db.get_exchange_events(list(volumes))
        items = []
        for key, event_id in event_ids.items():
            volume = volumes[key]
            if self._due(event_id, volume, now, min_change, max_age_seconds):
                items.append({"event_id": event_id, "volume_24h": volume})
        if not items:
            return 0
        registry.bulk_patch_events(items)
        for item in items:
            self._written[item["event_id"]] = (item["volume_24h"], now)
        logger.info("Wrote 24h volume for %d of %d events", len(items), len(event_ids))
        return len(items)

    def _due(self, event_id: int, volume: float, now: float, min_change: float, max_age_seconds: float) -> bool:
        last = self._written.get(event_id)
        if last is None:
            return True
        last_volume, written_at = last
        if now - written_at >= max_age_seconds:
            return True
        if last_volume == 0:
            return volume != 0
        return abs(volume - last_volume) / abs(last_volume) >= min_change
