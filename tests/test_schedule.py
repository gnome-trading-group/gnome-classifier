import dataclasses
import logging
from datetime import datetime

from classifier.adapters.types import utc_iso
from classifier.stages.entities import create_entities
from classifier.stages.schedule import event_starts, write_start_times
from tests.test_fetch import _make_contract


def started(native_id, security_id, start, source="kalshi-milestone"):
    return dataclasses.replace(_make_contract(native_id, security_id), event_start_time=start, event_start_source=source)


class _Db:
    def __init__(self, stored):
        self.stored = stored

    def get_event_start_times(self, keys):
        return {key: self.stored[key] for key in keys if key in self.stored}


class _Registry:
    def __init__(self):
        self.patches = []

    def bulk_patch_events(self, items):
        self.patches.append(items)
        return items


def test_venue_times_become_utc():
    assert utc_iso("2026-11-04 17:45:00+00") == "2026-11-04T17:45:00+00:00"
    assert utc_iso("2026-11-07T17:00:00Z") == "2026-11-07T17:00:00+00:00"
    assert utc_iso("20261104-2000") == "2026-11-04T20:00:00+00:00"
    assert utc_iso("2026-11-04T17:45:00") is None and utc_iso("soon") is None and utc_iso(None) is None


def test_an_events_start_comes_from_its_contracts_and_events_without_one_are_left_out():
    contracts = [started("A", "a1", "2026-11-04T02:00:00+00:00"), _make_contract("B", "b1")]
    assert event_starts(contracts) == {(1, "A"): ("2026-11-04T02:00:00+00:00", "kalshi-milestone")}


def test_only_changed_start_times_are_written_and_a_move_is_logged(caplog):
    db = _Db({(1, "A"): (10, datetime(2026, 11, 5, 23)), (1, "B"): (11, datetime(2026, 11, 4, 2)), (1, "C"): (12, None)})
    registry = _Registry()
    starts = {(1, "A"): ("2026-11-08T23:00:00+00:00", "kalshi-milestone"),
              (1, "B"): ("2026-11-04T02:00:00+00:00", "kalshi-milestone"),
              (1, "C"): ("2026-11-04T17:45:00+00:00", "polymarket-game-start"),
              (1, "Z"): ("2026-11-04T17:45:00+00:00", "polymarket-game-start")}
    with caplog.at_level(logging.INFO):
        assert write_start_times(db, registry, starts) == 2
    assert registry.patches == [[
        {"event_id": 10, "start_time": "2026-11-08T23:00:00+00:00", "start_time_source": "kalshi-milestone"},
        {"event_id": 12, "start_time": "2026-11-04T17:45:00+00:00", "start_time_source": "polymarket-game-start"},
    ]]
    assert caplog.text.count("event rescheduled") == 1 and "event rescheduled: event=10" in caplog.text


def test_nothing_is_written_when_nothing_moved():
    registry = _Registry()
    db = _Db({(1, "A"): (10, datetime(2026, 11, 4, 2))})
    assert write_start_times(db, registry, {(1, "A"): ("2026-11-04T02:00:00+00:00", "kalshi-milestone")}) == 0
    assert registry.patches == []


def test_a_new_event_is_created_with_its_start(stub_registry, stub_db):
    create_entities(stub_registry, None, [started("A", "a1", "2026-11-04T02:00:00+00:00")], db=stub_db)
    (event_id,) = [e.event_id for e in stub_registry._events]
    assert stub_registry._start_by_event[event_id] == ("2026-11-04T02:00:00+00:00", "kalshi-milestone")
