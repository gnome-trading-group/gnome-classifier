from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from classifier.adapters.settlement import CLOSED, OPEN, PRICE_SCALE, settled
from classifier.workers import repair_overdue as repair
from classifier.workers.repair_overdue import plan_batch

OUTCOMES = [
    (1, "WON", "A:yes"), (1, "WON", "A:no"),
    (2, "LIVE", "B:yes"), (2, "LIVE", "B:no"),
    (3, "PENDING", "C:yes"),
    (4, "DELETED", "D:yes"), (4, "DELETED", "D:no"),
    (5, "HALF", "E-1:yes"), (5, "HALF", "E-2:yes"),
]
STATES = {
    "A:yes": settled(PRICE_SCALE), "A:no": settled(0),
    "B:yes": OPEN, "B:no": OPEN,
    "C:yes": CLOSED,
    "E-1:yes": settled(0),
}


def test_an_event_resolves_on_values_and_is_otherwise_left_open_by_why():
    plan = plan_batch(OUTCOMES, STATES, trust_missing=False)
    assert plan.settled == [1]
    assert plan.gone == []
    assert plan.left == {"still open on the venue": 1, "closed, value not final yet": 1, "not found on the venue": 2}


def test_where_the_venue_leaves_out_only_markets_it_lacks_an_event_with_gone_markets_is_stale():
    plan = plan_batch(OUTCOMES, STATES, trust_missing=True)
    assert plan.settled == [1]
    assert plan.gone == [(4, "DELETED"), (5, "HALF")]
    assert plan.left == {"still open on the venue": 1, "closed, value not final yet": 1}


@pytest.fixture
def venue(monkeypatch):
    adapter = MagicMock(exchange_code="KALSHI", omits_only_unknown_markets=True)
    adapter.fetch_market_states.return_value = STATES
    monkeypatch.setattr(repair, "unpaced_adapters", lambda: [adapter])
    monkeypatch.setattr(repair, "fetch_exchanges", lambda registry, code: {
        "KALSHI": SimpleNamespace(exchange_id=5)})
    db = MagicMock()
    db.get_overdue_outcomes.return_value = OUTCOMES
    return adapter, db


def test_a_dry_run_looks_every_outcome_up_and_writes_nothing(venue):
    adapter, db = venue
    registry = MagicMock()
    repair.repair_overdue(registry, db, "KALSHI", 7, apply=False)
    adapter.fetch_market_states.assert_called_once_with({sid for _, _, sid in OUTCOMES})
    assert not [c for c in registry.mock_calls if c[0].startswith("bulk_patch")]


def test_applying_records_values_resolves_settled_events_and_retires_gone_ones(venue, monkeypatch):
    _, db = venue
    applied, resolved, stale = [], [], []
    monkeypatch.setattr(repair, "apply_market_states",
                        lambda exchange_id, states, registry, db: applied.append(states) or repair.empty_counts())
    monkeypatch.setattr(repair, "resolve_settled_events",
                        lambda event_ids, just_settled, registry, db: resolved.append(event_ids) or ([], []))
    monkeypatch.setattr(repair, "deactivate_stale_events",
                        lambda keys, registry, db: stale.append(keys) or {"events_resolved": 2, "listings_deactivated": 3})
    repair.repair_overdue(MagicMock(), db, "KALSHI", 7, apply=True)
    assert applied == [STATES]
    assert resolved == [{1}]
    assert stale == [[(5, "DELETED"), (5, "HALF")]]


def test_the_repair_looks_polymarket_international_up_on_the_clob_without_the_hourly_budget():
    pm_intl = [a for a in repair.unpaced_adapters() if a.exchange_code == "POLYMARKET_INTL"]
    assert len(pm_intl) == 1
    assert all(pm_intl[0]._take_clob_lookup() for _ in range(5_000))
