import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from classifier.adapters.hyperliquid import MAX_SETTLEMENT_LOOKUPS, HyperliquidAdapter
from classifier.adapters.kalshi import KalshiAdapter
from classifier.adapters.polymarket_intl import PolymarketIntlAdapter
from classifier.adapters.polymarket_us import PolymarketUsAdapter
from classifier.adapters.settlement import PRICE_SCALE, to_price
from classifier.runtime_config import ClassifierConfig
from classifier.stages.settle import record_settlements
from classifier.workers.fetch import FetchRunner
from gnomepy.registry.types import Event, EventContract, Exchange, Listing, Security, SecurityType

FIXTURES = Path(__file__).parent / "fixtures"
KALSHI = json.loads((FIXTURES / "kalshi_settlement_markets.json").read_text())["markets"]
PM_INTL = json.loads((FIXTURES / "polymarket_intl_settlement_markets.json").read_text())["markets"]
PM_US = json.loads((FIXTURES / "polymarket_us_settlement_markets.json").read_text())["markets"]
HL_SETTLED = json.loads((FIXTURES / "hyperliquid_settled_outcomes.json").read_text())["outcomes"]
CENT = PRICE_SCALE // 100


def _get_session(body):
    session = MagicMock()
    session.get.return_value = MagicMock(json=lambda: body)
    return session


# ── Exact prices ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,price", [
    ("1.0000", PRICE_SCALE), ("0", 0), ("0.47", 47 * CENT), (0.5, 50 * CENT), ("1.0", PRICE_SCALE),
    ("0.123456789", 123_456_789),
])
def test_to_price_is_exact(raw, price):
    assert to_price(raw) == price


@pytest.mark.parametrize("raw", [None, "", "abc", "1.5", "-0.1", "0.0000000001", "NaN"])
def test_to_price_refuses_what_is_not_a_settlement_value(raw):
    assert to_price(raw) is None


# ── Kalshi ────────────────────────────────────────────────────────────────────

def test_kalshi_records_only_finalized_markets_and_complements_the_no_side():
    yes, no, scalar, closed, determined, disputed = (m["ticker"] for m in KALSHI)
    session = _get_session({"markets": KALSHI})
    ids = {f"{yes}:yes", f"{yes}:no", f"{no}:yes", f"{no}:no", scalar, f"{closed}:yes", f"{determined}:yes",
           f"{disputed}:no"}
    assert KalshiAdapter(session=session).fetch_settlements(ids) == {
        f"{yes}:yes": PRICE_SCALE, f"{yes}:no": 0,
        f"{no}:yes": 0, f"{no}:no": PRICE_SCALE,
        scalar: 47 * CENT,
    }
    params = session.get.call_args.kwargs["params"]
    assert set(params["tickers"].split(",")) == {yes, no, scalar, closed, determined, disputed}


# ── Polymarket International ──────────────────────────────────────────────────

def _pm_intl_ids(market: dict) -> list[str]:
    return [f"{market['conditionId']}:{token}" for token in json.loads(market["clobTokenIds"])]


def test_polymarket_intl_records_only_uma_resolved_markets_per_token():
    yes_wins, no_wins, closed_unresolved, placeholder, proposed, disputed, half = PM_INTL
    ids = {i for m in PM_INTL for i in _pm_intl_ids(m)}
    session = _get_session(PM_INTL)
    settlements = PolymarketIntlAdapter(session=session).fetch_settlements(ids)
    y, n = _pm_intl_ids(yes_wins), _pm_intl_ids(no_wins)
    h = _pm_intl_ids(half)
    assert settlements == {y[0]: PRICE_SCALE, y[1]: 0, n[0]: 0, n[1]: PRICE_SCALE, h[0]: 50 * CENT, h[1]: 50 * CENT}
    assert session.get.call_args.kwargs["params"]["closed"] == "true"


# ── Polymarket US ─────────────────────────────────────────────────────────────

def test_polymarket_us_records_each_sides_own_payout_once_resolved():
    long_loses, long_wins, fractional, resolving, closed = (m["slug"] for m in PM_US)
    ids = {f"{slug}:{leg}" for slug in (long_loses, long_wins, fractional, resolving, closed) for leg in ("long", "short")}
    settlements = PolymarketUsAdapter(session=_get_session({"markets": PM_US})).fetch_settlements(ids)
    assert settlements == {
        f"{long_loses}:long": 0, f"{long_loses}:short": PRICE_SCALE,
        f"{long_wins}:long": PRICE_SCALE, f"{long_wins}:short": 0,
        f"{fractional}:long": 58 * CENT, f"{fractional}:short": 42 * CENT,
    }


# ── Hyperliquid ───────────────────────────────────────────────────────────────

def _hl_session():
    session = MagicMock()
    session.post.side_effect = lambda url, json, timeout: MagicMock(
        json=lambda: HL_SETTLED.get(str(json["outcome"])))
    return session


def test_hyperliquid_settles_yes_at_the_fraction_and_no_at_its_complement():
    session = _hl_session()
    settlements = HyperliquidAdapter(session=session).fetch_settlements({"#970", "#971", "#10010", "#10011", "#14730"})
    assert settlements == {"#970": 0, "#971": PRICE_SCALE, "#10010": PRICE_SCALE, "#10011": 0}
    assert sorted(c.kwargs["json"]["outcome"] for c in session.post.call_args_list) == [97, 1001, 1473]
    assert {c.kwargs["json"]["type"] for c in session.post.call_args_list} == {"settledOutcome"}


def test_hyperliquid_caps_lookups_per_cycle_oldest_outcomes_first():
    session = MagicMock()
    session.post.return_value = MagicMock(json=lambda: None)
    HyperliquidAdapter(session=session).fetch_settlements({f"#{10 * o}" for o in range(1, MAX_SETTLEMENT_LOOKUPS + 10)})
    looked_up = [c.kwargs["json"]["outcome"] for c in session.post.call_args_list]
    assert looked_up == list(range(1, MAX_SETTLEMENT_LOOKUPS + 1))


# ── Settle stage ──────────────────────────────────────────────────────────────

def _seed(registry, listing_id: int, security_id: int, exchange_security_id: str, active: bool = False):
    registry._securities.append(Security(
        security_id=security_id, symbol=f"S{security_id}", type=SecurityType.EVENT_CONTRACT, description=None,
        contract_type=7, base_currency_id=None, quote_currency_id=None, settle_currency_id=None, inverse=False,
        is_quanto=False, expiry=None, strike_price=None, active=active, underlying_security_id=None, asset_class=5,
        date_modified="", date_created=""))
    registry._listings.append(Listing(
        listing_id=listing_id, security_id=security_id, exchange_id=2, exchange_security_id=exchange_security_id,
        exchange_security_symbol="", date_modified="", date_created="", active=active))
    registry._event_contracts.append(EventContract(
        event_contract_id=security_id, event_id=1, security_id=security_id, outcome_label="Yes", date_created=""))


def test_record_settlements_writes_final_values_and_skips_the_rest(stub_registry, stub_db):
    _seed(stub_registry, 10, 100, "T:yes")
    _seed(stub_registry, 11, 101, "T:no")
    _seed(stub_registry, 12, 102, "OPEN:yes", active=True)
    _seed(stub_registry, 13, 103, "PENDING:yes")
    adapter = SimpleNamespace(exchange_code="KALSHI", fetch_settlements=MagicMock(
        return_value={"T:yes": PRICE_SCALE, "T:no": 0}))
    exchanges = {"KALSHI": Exchange(exchange_id=2, exchange_code="KALSHI", exchange_name="", region="", schema_type="",
                                    date_modified="", date_created="")}

    assert record_settlements([adapter], exchanges, stub_registry, stub_db, 30, 100, {}) == {
        "settlements_recorded": 2, "failed_exchanges": [], "listings_deactivated": 0, "events_resolved": 0}
    assert adapter.fetch_settlements.call_args.args[0] == {"T:yes", "T:no", "PENDING:yes"}
    assert stub_registry._settlement_by_event_contract == {100: PRICE_SCALE, 101: 0}
    # Recorded ones aren't candidates any more; the pending one is asked about again next cycle.
    record_settlements([adapter], exchanges, stub_registry, stub_db, 30, 100, {})
    assert adapter.fetch_settlements.call_args.args[0] == {"PENDING:yes"}


def test_record_settlements_skips_a_failing_venue(stub_registry, stub_db):
    _seed(stub_registry, 10, 100, "T:yes")
    adapter = SimpleNamespace(exchange_code="KALSHI", fetch_settlements=MagicMock(side_effect=RuntimeError("down")))
    exchanges = {"KALSHI": Exchange(exchange_id=2, exchange_code="KALSHI", exchange_name="", region="", schema_type="",
                                    date_modified="", date_created="")}
    assert record_settlements([adapter], exchanges, stub_registry, stub_db, 30, 100, {}) == {
        "settlements_recorded": 0, "failed_exchanges": ["KALSHI"], "listings_deactivated": 0, "events_resolved": 0}


def test_settle_cycle_does_nothing_while_the_flag_is_off(monkeypatch):
    calls = []
    monkeypatch.setattr("classifier.workers.fetch.record_settlements", lambda *a: calls.append(a))
    monkeypatch.setattr("classifier.workers.fetch.init_db", lambda: calls.append("db"))
    rc = SimpleNamespace(config=ClassifierConfig())
    FetchRunner()._run_settle(rc, MagicMock())
    assert calls == []

    rc.config.feature_flags.settlement_enabled = True
    monkeypatch.setattr("classifier.workers.fetch.fetch_exchanges", lambda registry: {})
    monkeypatch.setattr("classifier.workers.fetch.init_db", lambda: "db")
    monkeypatch.setattr("classifier.workers.fetch.record_settlements", lambda *a: calls.append(a) or {})
    FetchRunner()._run_settle(rc, MagicMock())
    assert len(calls) == 1 and calls[0][3] == "db" and calls[0][4:6] == (30, 20_000)


def test_record_settlements_rotates_through_a_backlog_a_page_per_cycle(stub_registry, stub_db):
    for i in range(5):
        _seed(stub_registry, 10 + i, 100 + i, f"T{i}:yes")
    # Nothing settles, as for markets that closed but never got a final value: each cycle must still move on.
    adapter = SimpleNamespace(exchange_code="KALSHI", fetch_settlements=MagicMock(return_value={}))
    exchanges = {"KALSHI": Exchange(exchange_id=2, exchange_code="KALSHI", exchange_name="", region="", schema_type="",
                                    date_modified="", date_created="")}
    cursors: dict[str, int] = {}
    seen = []
    for _ in range(4):
        record_settlements([adapter], exchanges, stub_registry, stub_db, 30, 2, cursors)
        seen.append(sorted(adapter.fetch_settlements.call_args.args[0]))
    assert seen == [["T0:yes", "T1:yes"], ["T2:yes", "T3:yes"], ["T4:yes"], ["T0:yes", "T1:yes"]]


def test_hyperliquid_settles_active_outcomes_and_retires_them(stub_registry, stub_db):
    # Hyperliquid listings stay active after settling (the outcome just leaves outcomeMeta), so the settle cycle
    # looks at active ones too, and retires what it settles.
    stub_registry._events.append(Event(event_id=1, title="NFL", description=None, category=None, tags=None,
                                       resolved=False, resolved_at=None, expiry=None, date_modified="", date_created=""))
    _seed(stub_registry, 10, 100, "#62090", active=True)
    _seed(stub_registry, 11, 101, "#62091", active=True)
    _seed(stub_registry, 12, 102, "#14730", active=True)
    adapter = HyperliquidAdapter(session=_hl_session())
    HL_SETTLED["6209"] = {"settleFraction": "1.0"}
    exchanges = {"HYPERLIQUID": Exchange(exchange_id=2, exchange_code="HYPERLIQUID", exchange_name="", region="",
                                         schema_type="", date_modified="", date_created="")}
    try:
        result = record_settlements([adapter], exchanges, stub_registry, stub_db, 30, 20_000, {})
    finally:
        del HL_SETTLED["6209"]
    assert result["settlements_recorded"] == 2
    assert stub_registry._settlement_by_event_contract == {100: PRICE_SCALE, 101: 0}
    assert {l.exchange_security_id: l.active for l in stub_registry._listings} == {
        "#62090": False, "#62091": False, "#14730": True}


def test_active_listings_of_other_venues_are_not_candidates(stub_registry, stub_db):
    _seed(stub_registry, 10, 100, "OPEN:yes", active=True)
    assert stub_db.get_unsettled_contracts(2, 30, 0, 100) == []
    assert stub_db.get_unsettled_contracts(2, 30, 0, 100, include_active=True) == [(100, "OPEN:yes")]


def test_hyperliquid_candidates_are_paged_to_its_lookup_budget(stub_registry, stub_db):
    for i in range(MAX_SETTLEMENT_LOOKUPS + 5):
        _seed(stub_registry, 1000 + i, 2000 + i, f"#{(5000 + i) * 10}", active=True)
    adapter = SimpleNamespace(exchange_code="HYPERLIQUID", settle_active_listings=True,
                              settle_page_size=MAX_SETTLEMENT_LOOKUPS, fetch_settlements=MagicMock(return_value={}))
    exchanges = {"HYPERLIQUID": Exchange(exchange_id=2, exchange_code="HYPERLIQUID", exchange_name="", region="",
                                         schema_type="", date_modified="", date_created="")}
    cursors: dict[str, int] = {}
    record_settlements([adapter], exchanges, stub_registry, stub_db, 30, 20_000, cursors)
    assert len(adapter.fetch_settlements.call_args.args[0]) == MAX_SETTLEMENT_LOOKUPS
    record_settlements([adapter], exchanges, stub_registry, stub_db, 30, 20_000, cursors)
    assert len(adapter.fetch_settlements.call_args.args[0]) == 5
