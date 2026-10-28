from classifier.stages.volume import VolumeWriter, event_volumes
from tests.test_fetch import _make_contract


class _Db:
    def __init__(self, event_ids):
        self._event_ids = event_ids

    def get_exchange_events(self, keys):
        return {key: self._event_ids[key] for key in keys if key in self._event_ids}


class _Registry:
    def __init__(self):
        self.patches = []

    def bulk_patch_events(self, items):
        self.patches.append(items)
        return items


def _write(writer, registry, volumes, now):
    db = _Db({(1, "A"): 10, (1, "B"): 11})
    return writer.write(db, registry, volumes, now, min_change=0.10, max_age_seconds=3600)


def test_an_events_volume_comes_from_its_contracts_and_contracts_without_one_are_skipped():
    contracts = [_make_contract("A", "a1", 500.0), _make_contract("A", "a2", 500.0), _make_contract("B", "b1")]
    assert event_volumes(contracts) == {(1, "A"): 500.0}


def test_every_known_event_is_written_once_and_unknown_events_never():
    registry = _Registry()
    assert _write(VolumeWriter(), registry, {(1, "A"): 500.0, (1, "B"): 0.0, (1, "Z"): 9.0}, now=0) == 2
    assert registry.patches == [[{"event_id": 10, "volume_24h": 500.0}, {"event_id": 11, "volume_24h": 0.0}]]


def test_a_reading_is_rewritten_only_when_it_moved_enough_or_the_last_one_is_old():
    writer, registry = VolumeWriter(), _Registry()
    _write(writer, registry, {(1, "A"): 1000.0, (1, "B"): 0.0}, now=0)

    assert _write(writer, registry, {(1, "A"): 1050.0, (1, "B"): 0.0}, now=900) == 0
    assert _write(writer, registry, {(1, "A"): 1150.0, (1, "B"): 5.0}, now=1800) == 2
    assert _write(writer, registry, {(1, "A"): 1150.0, (1, "B"): 5.0}, now=1800 + 3600) == 2
    assert [len(p) for p in registry.patches] == [2, 2, 2]


def test_nothing_is_sent_when_nothing_is_due():
    writer, registry = VolumeWriter(), _Registry()
    _write(writer, registry, {(1, "A"): 1000.0}, now=0)
    assert _write(writer, registry, {(1, "A"): 1000.0}, now=60) == 0
    assert len(registry.patches) == 1
