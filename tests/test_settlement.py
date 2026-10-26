import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from classifier.adapters.hyperliquid import MAX_SETTLEMENT_LOOKUPS, HyperliquidAdapter
from classifier.adapters.kalshi import KalshiAdapter
from classifier.adapters.polymarket_intl import CLOB_API_URL, PolymarketIntlAdapter
from classifier.adapters.polymarket_us import PolymarketUsAdapter
from classifier.adapters.settlement import CLOSED, OPEN, PRICE_SCALE, settled, to_price
from classifier.runtime_config import ClassifierConfig
from classifier.stages.settle import check_closures, empty_counts, record_settlements
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


def _prices(states: dict) -> dict:
    return {security_id: state.settlement for security_id, state in states.items() if state.settlement is not None}


# ── Kalshi ────────────────────────────────────────────────────────────────────

def test_kalshi_reports_finalized_values_complementing_the_no_side_and_the_rest_closed():
    yes, no, scalar, closed, determined, disputed = (m["ticker"] for m in KALSHI)
    session = _get_session({"markets": KALSHI + [{"ticker": "KXOPEN-1", "status": "active"}]})
    ids = {f"{yes}:yes", f"{yes}:no", f"{no}:yes", f"{no}:no", scalar, f"{closed}:yes", f"{determined}:yes",
           f"{disputed}:no", "KXOPEN-1:yes", "KXUNKNOWN-1:yes"}
    states = KalshiAdapter(session=session).fetch_market_states(ids)
    assert _prices(states) == {
        f"{yes}:yes": PRICE_SCALE, f"{yes}:no": 0,
        f"{no}:yes": 0, f"{no}:no": PRICE_SCALE,
        scalar: 47 * CENT,
    }
    assert states[f"{closed}:yes"] == states[f"{determined}:yes"] == states[f"{disputed}:no"] == CLOSED
    assert states["KXOPEN-1:yes"] == OPEN
    assert "KXUNKNOWN-1:yes" not in states
    params = session.get.call_args.kwargs["params"]
    assert set(params["tickers"].split(",")) == {yes, no, scalar, closed, determined, disputed, "KXOPEN-1",
                                                  "KXUNKNOWN-1"}


# ── Polymarket International ──────────────────────────────────────────────────

def _pm_intl_ids(market: dict) -> list[str]:
    return [f"{market['conditionId']}:{token}" for token in json.loads(market["clobTokenIds"])]


def test_polymarket_intl_reports_only_uma_resolved_values_per_token():
    yes_wins, no_wins, closed_unresolved, placeholder, proposed, disputed, half = PM_INTL
    ids = {i for m in PM_INTL for i in _pm_intl_ids(m)}
    session = _get_session(PM_INTL)
    states = PolymarketIntlAdapter(session=session).fetch_market_states(ids)
    y, n = _pm_intl_ids(yes_wins), _pm_intl_ids(no_wins)
    h = _pm_intl_ids(half)
    assert _prices(states) == {y[0]: PRICE_SCALE, y[1]: 0, n[0]: 0, n[1]: PRICE_SCALE, h[0]: 50 * CENT, h[1]: 50 * CENT}
    assert all(states[i] == CLOSED for m in (closed_unresolved, proposed, disputed) for i in _pm_intl_ids(m))
    assert all(states[i] == OPEN for i in _pm_intl_ids(placeholder))
    # Every market turned up among the closed ones, so the open ones weren't asked for.
    assert [c.kwargs["params"]["closed"] for c in session.get.call_args_list] == ["true"]


def _routed_session(gamma_closed: list[dict], gamma_open: list[dict], clob: dict[str, dict]):
    session = MagicMock()

    def get(url, params=None, timeout=None):
        if url.startswith(CLOB_API_URL):
            market = clob.get(url.rsplit("/", 1)[1])
            return MagicMock(status_code=200 if market else 404, json=lambda: market)
        wanted = set(params["condition_ids"])
        markets = gamma_closed if params["closed"] == "true" else gamma_open
        return MagicMock(json=lambda: [m for m in markets if m["conditionId"] in wanted])

    session.get.side_effect = get
    return session


def test_polymarket_intl_asks_the_clob_about_markets_gamma_no_longer_returns():
    clob = {
        "0xold": {"condition_id": "0xold", "closed": True, "tokens": [
            {"token_id": "1", "price": 0, "winner": False}, {"token_id": "2", "price": 1, "winner": True}]},
        "0xpending": {"condition_id": "0xpending", "closed": True, "tokens": [
            {"token_id": "3", "price": 0.4, "winner": False}, {"token_id": "4", "price": 0.6, "winner": False}]},
        "0xplaceholder": {"condition_id": "0xplaceholder", "closed": False, "tokens": [
            {"token_id": "", "price": 0, "winner": False}]},
    }
    open_market = {"conditionId": "0xopen", "closed": False, "clobTokenIds": "[\"5\", \"6\"]"}
    session = _routed_session([], [open_market], clob)
    ids = {"0xold:1", "0xold:2", "0xpending:3", "0xplaceholder:", "0xopen:5", "0xgone:7"}
    states = PolymarketIntlAdapter(session=session).fetch_market_states(ids)
    assert states == {"0xold:1": settled(0), "0xold:2": settled(PRICE_SCALE), "0xpending:3": CLOSED, "0xopen:5": OPEN}
    clob_urls = [c.args[0] for c in session.get.call_args_list if c.args[0].startswith(CLOB_API_URL)]
    assert sorted(u.rsplit("/", 1)[1] for u in clob_urls) == ["0xgone", "0xold", "0xpending", "0xplaceholder"]


def test_polymarket_intl_clob_lookups_are_budgeted_per_hour(monkeypatch):
    monkeypatch.setattr("classifier.adapters.polymarket_intl.CLOB_LOOKUPS_PER_HOUR", 2)
    session = _routed_session([], [], {})
    adapter = PolymarketIntlAdapter(session=session)
    adapter.fetch_market_states({"0xa:1", "0xb:1", "0xc:1"})
    adapter.fetch_market_states({"0xd:1"})
    assert sum(c.args[0].startswith(CLOB_API_URL) for c in session.get.call_args_list) == 2
    adapter._clob_window_start -= 3600
    adapter.fetch_market_states({"0xd:1"})
    assert sum(c.args[0].startswith(CLOB_API_URL) for c in session.get.call_args_list) == 3


# ── Polymarket US ─────────────────────────────────────────────────────────────

def test_polymarket_us_reports_each_sides_own_payout_once_resolved():
    long_loses, long_wins, fractional, resolving, closed = (m["slug"] for m in PM_US)
    open_market = {"slug": "aec-open", "active": True, "closed": False, "status": "MARKET_STATUS_OPEN"}
    slugs = (long_loses, long_wins, fractional, resolving, closed, "aec-open")
    ids = {f"{slug}:{leg}" for slug in slugs for leg in ("long", "short")}
    states = PolymarketUsAdapter(session=_get_session({"markets": PM_US + [open_market]})).fetch_market_states(ids)
    assert _prices(states) == {
        f"{long_loses}:long": 0, f"{long_loses}:short": PRICE_SCALE,
        f"{long_wins}:long": PRICE_SCALE, f"{long_wins}:short": 0,
        f"{fractional}:long": 58 * CENT, f"{fractional}:short": 42 * CENT,
    }
    assert states[f"{resolving}:long"] == states[f"{closed}:short"] == CLOSED
    assert states["aec-open:long"] == states["aec-open:short"] == OPEN


def test_polymarket_us_ignores_markets_it_was_not_asked_about():
    # The gateway has ignored its filters before (2026-10-22, closed=true); only the asked-for slugs may be answered.
    states = PolymarketUsAdapter(session=_get_session({"markets": PM_US})).fetch_market_states({"other:long"})
    assert states == {}


# ── Hyperliquid ───────────────────────────────────────────────────────────────

def _hl_session():
    session = MagicMock()
    session.post.side_effect = lambda url, json, timeout: MagicMock(
        json=lambda: HL_SETTLED.get(str(json["outcome"])))
    return session


def test_hyperliquid_settles_yes_at_the_fraction_and_no_at_its_complement():
    session = _hl_session()
    states = HyperliquidAdapter(session=session).fetch_market_states({"#970", "#971", "#10010", "#10011", "#14730"})
    assert states == {"#970": settled(0), "#971": settled(PRICE_SCALE), "#10010": settled(PRICE_SCALE),
                      "#10011": settled(0)}
    assert sorted(c.kwargs["json"]["outcome"] for c in session.post.call_args_list) == [97, 1001, 1473]
    assert {c.kwargs["json"]["type"] for c in session.post.call_args_list} == {"settledOutcome"}


def test_hyperliquid_caps_lookups_per_cycle_oldest_outcomes_first():
    session = MagicMock()
    session.post.return_value = MagicMock(json=lambda: None)
    HyperliquidAdapter(session=session).fetch_market_states({f"#{10 * o}" for o in range(1, MAX_SETTLEMENT_LOOKUPS + 10)})
    looked_up = [c.kwargs["json"]["outcome"] for c in session.post.call_args_list]
    assert looked_up == list(range(1, MAX_SETTLEMENT_LOOKUPS + 1))


# ── Settle stage ──────────────────────────────────────────────────────────────

KALSHI_EXCHANGES = {"KALSHI": Exchange(exchange_id=2, exchange_code="KALSHI", exchange_name="", region="",
                                       date_modified="", date_created="")}
LOOKBACK = timedelta(days=30)


def _seed(registry, listing_id: int, security_id: int, exchange_security_id: str, active: bool = False,
          event_id: int = 1):
    registry._securities.append(Security(
        security_id=security_id, symbol=f"S{security_id}", type=SecurityType.EVENT_CONTRACT, description=None,
        contract_type=7, base_currency_id=None, quote_currency_id=None, settle_currency_id=None, inverse=False,
        is_quanto=False, expiry=None, strike_price=None, active=active, underlying_security_id=None, asset_class=5,
        date_modified="", date_created=""))
    registry._listings.append(Listing(
        listing_id=listing_id, security_id=security_id, exchange_id=2, exchange_security_id=exchange_security_id,
        exchange_security_symbol="", date_modified="", date_created="", active=active))
    registry._event_contracts.append(EventContract(
        event_contract_id=security_id, event_id=event_id, security_id=security_id, outcome_label="Yes", date_created=""))


def _seed_event(registry, event_id: int = 1):
    registry._events.append(Event(event_id=event_id, title=f"Event {event_id}", description=None, category=None,
                                  tags=None, resolved=False, resolved_at=None, expiry=None, date_modified="",
                                  date_created=""))


def _adapter(code: str = "KALSHI", **kwargs):
    return SimpleNamespace(exchange_code=code, **kwargs)


def test_record_settlements_writes_final_values_and_skips_the_rest(stub_registry, stub_db):
    _seed(stub_registry, 10, 100, "T:yes")
    _seed(stub_registry, 11, 101, "T:no")
    _seed(stub_registry, 12, 102, "OPEN:yes", active=True)
    _seed(stub_registry, 13, 103, "PENDING:yes")
    adapter = _adapter(fetch_market_states=MagicMock(return_value={
        "T:yes": settled(PRICE_SCALE), "T:no": settled(0), "PENDING:yes": CLOSED}))

    result = record_settlements([adapter], KALSHI_EXCHANGES, stub_registry, stub_db, LOOKBACK, 100, {})
    assert result["settlements_recorded"] == 2 and result["failed_exchanges"] == []
    assert adapter.fetch_market_states.call_args.args[0] == {"T:yes", "T:no", "PENDING:yes"}
    assert stub_registry._settlement_by_event_contract == {100: PRICE_SCALE, 101: 0}
    # Recorded ones aren't candidates any more; the pending one is asked about again next cycle.
    record_settlements([adapter], KALSHI_EXCHANGES, stub_registry, stub_db, LOOKBACK, 100, {})
    assert adapter.fetch_market_states.call_args.args[0] == {"PENDING:yes"}


def test_an_event_resolves_when_its_last_outcome_settles_and_is_announced(stub_registry, stub_db):
    _seed_event(stub_registry)
    _seed(stub_registry, 10, 100, "T:yes")
    _seed(stub_registry, 11, 101, "T:no")
    adapter = _adapter(fetch_market_states=MagicMock(return_value={"T:yes": settled(PRICE_SCALE)}))
    result = record_settlements([adapter], KALSHI_EXCHANGES, stub_registry, stub_db, LOOKBACK, 100, {})
    assert result["events_resolved"] == 0 and stub_registry._events[0].resolved is False

    adapter.fetch_market_states.return_value = {"T:no": settled(0)}
    result = record_settlements([adapter], KALSHI_EXCHANGES, stub_registry, stub_db, LOOKBACK, 100, {})
    assert (result["events_resolved"], result["resolved_event_ids"], result["resolved_event_names"]) == (
        1, [1], ["Event 1"])
    assert stub_registry._events[0].resolved is True


def test_record_settlements_skips_a_failing_venue(stub_registry, stub_db):
    _seed(stub_registry, 10, 100, "T:yes")
    adapter = _adapter(fetch_market_states=MagicMock(side_effect=RuntimeError("down")))
    result = record_settlements([adapter], KALSHI_EXCHANGES, stub_registry, stub_db, LOOKBACK, 100, {})
    assert result["settlements_recorded"] == 0 and result["failed_exchanges"] == ["KALSHI"]


def test_the_recent_pass_leaves_older_outcomes_to_the_sweep(stub_registry, stub_db):
    _seed(stub_registry, 10, 100, "RECENT:yes")
    _seed(stub_registry, 11, 101, "OLD:yes")
    stub_registry._listing_modified_at[11] = datetime.now(timezone.utc) - timedelta(days=2)
    adapter = _adapter(fetch_market_states=MagicMock(return_value={}))

    record_settlements([adapter], KALSHI_EXCHANGES, stub_registry, stub_db, timedelta(hours=6), 100)
    assert adapter.fetch_market_states.call_args.args[0] == {"RECENT:yes"}
    record_settlements([adapter], KALSHI_EXCHANGES, stub_registry, stub_db, LOOKBACK, 100, {})
    assert adapter.fetch_market_states.call_args.args[0] == {"RECENT:yes", "OLD:yes"}


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
    monkeypatch.setattr("classifier.workers.fetch.record_settlements", lambda *a: calls.append(a) or empty_counts())
    FetchRunner()._run_settle(rc, MagicMock())
    assert len(calls) == 1 and calls[0][3] == "db" and calls[0][4:6] == (timedelta(days=30), 20_000)


def test_record_settlements_rotates_through_a_backlog_a_page_per_cycle(stub_registry, stub_db):
    for i in range(5):
        _seed(stub_registry, 10 + i, 100 + i, f"T{i}:yes")
    # Nothing settles, as for markets that closed but never got a final value: each cycle must still move on.
    adapter = _adapter(fetch_market_states=MagicMock(return_value={}))
    cursors: dict[str, int] = {}
    seen = []
    for _ in range(4):
        record_settlements([adapter], KALSHI_EXCHANGES, stub_registry, stub_db, LOOKBACK, 2, cursors)
        seen.append(sorted(adapter.fetch_market_states.call_args.args[0]))
    assert seen == [["T0:yes", "T1:yes"], ["T2:yes", "T3:yes"], ["T4:yes"], ["T0:yes", "T1:yes"]]


def test_hyperliquid_settles_active_outcomes_and_retires_them(stub_registry, stub_db):
    # Hyperliquid listings stay active after settling (the outcome just leaves outcomeMeta), so the sweep looks at
    # active ones too, and retires what it settles.
    _seed_event(stub_registry)
    _seed(stub_registry, 10, 100, "#62090", active=True)
    _seed(stub_registry, 11, 101, "#62091", active=True)
    _seed(stub_registry, 12, 102, "#14730", active=True)
    adapter = HyperliquidAdapter(session=_hl_session())
    HL_SETTLED["6209"] = {"settleFraction": "1.0"}
    exchanges = {"HYPERLIQUID": Exchange(exchange_id=2, exchange_code="HYPERLIQUID", exchange_name="", region="",
                                         date_modified="", date_created="")}
    try:
        result = record_settlements([adapter], exchanges, stub_registry, stub_db, LOOKBACK, 20_000, {})
    finally:
        del HL_SETTLED["6209"]
    assert result["settlements_recorded"] == 2
    assert stub_registry._settlement_by_event_contract == {100: PRICE_SCALE, 101: 0}
    assert {l.exchange_security_id: l.active for l in stub_registry._listings} == {
        "#62090": False, "#62091": False, "#14730": True}


def test_active_listings_of_other_venues_are_not_candidates(stub_registry, stub_db):
    _seed(stub_registry, 10, 100, "OPEN:yes", active=True)
    assert stub_db.get_unsettled_contracts(2, LOOKBACK, 0, 100) == []
    assert stub_db.get_unsettled_contracts(2, LOOKBACK, 0, 100, include_active=True) == [(100, "OPEN:yes")]


def test_only_the_sweep_looks_at_active_outcomes(stub_registry, stub_db):
    _seed(stub_registry, 10, 100, "#62090", active=True)
    adapter = _adapter("HYPERLIQUID", settle_active_listings=True, fetch_market_states=MagicMock(return_value={}))
    exchanges = {"HYPERLIQUID": Exchange(exchange_id=2, exchange_code="HYPERLIQUID", exchange_name="", region="",
                                         date_modified="", date_created="")}
    record_settlements([adapter], exchanges, stub_registry, stub_db, timedelta(hours=6), 100)
    assert not adapter.fetch_market_states.called


def test_hyperliquid_candidates_are_paged_to_its_lookup_budget(stub_registry, stub_db):
    for i in range(MAX_SETTLEMENT_LOOKUPS + 5):
        _seed(stub_registry, 1000 + i, 2000 + i, f"#{(5000 + i) * 10}", active=True)
    adapter = _adapter("HYPERLIQUID", settle_active_listings=True, settle_page_size=MAX_SETTLEMENT_LOOKUPS,
                       fetch_market_states=MagicMock(return_value={}))
    exchanges = {"HYPERLIQUID": Exchange(exchange_id=2, exchange_code="HYPERLIQUID", exchange_name="", region="",
                                         date_modified="", date_created="")}
    cursors: dict[str, int] = {}
    record_settlements([adapter], exchanges, stub_registry, stub_db, LOOKBACK, 20_000, cursors)
    assert len(adapter.fetch_market_states.call_args.args[0]) == MAX_SETTLEMENT_LOOKUPS
    record_settlements([adapter], exchanges, stub_registry, stub_db, LOOKBACK, 20_000, cursors)
    assert len(adapter.fetch_market_states.call_args.args[0]) == 5


# ── Closure check ─────────────────────────────────────────────────────────────

def test_closure_check_retires_closed_contracts_settles_final_ones_and_leaves_open_ones(stub_registry, stub_db):
    _seed_event(stub_registry)
    _seed(stub_registry, 10, 100, "DONE:yes", active=True)
    _seed(stub_registry, 11, 101, "DONE:no", active=True)
    _seed(stub_registry, 12, 102, "WAIT:yes", active=True, event_id=2)
    _seed(stub_registry, 13, 103, "BLIP:yes", active=True, event_id=3)
    adapter = _adapter(fetch_market_states=MagicMock(return_value={
        "DONE:yes": settled(PRICE_SCALE), "DONE:no": settled(0), "WAIT:yes": CLOSED, "BLIP:yes": OPEN}))
    left_feed = {"DONE:yes", "DONE:no", "WAIT:yes", "BLIP:yes", "GONE:yes"}

    result = check_closures([adapter], KALSHI_EXCHANGES, {2: left_feed}, stub_registry, stub_db, 500)

    assert adapter.fetch_market_states.call_args.args[0] == left_feed
    assert {l.exchange_security_id: l.active for l in stub_registry._listings} == {
        "DONE:yes": False, "DONE:no": False, "WAIT:yes": False, "BLIP:yes": True}
    assert stub_registry._settlement_by_event_contract == {100: PRICE_SCALE, 101: 0}
    assert (result["settlements_recorded"], result["listings_deactivated"], result["resolved_event_ids"]) == (2, 3, [1])


def test_closure_check_leaves_a_mass_disappearance_to_the_resolve_cycle(stub_registry, stub_db, caplog):
    adapter = _adapter(fetch_market_states=MagicMock(return_value={}))
    result = check_closures([adapter], KALSHI_EXCHANGES, {2: {f"T{i}:yes" for i in range(6)}}, stub_registry,
                            stub_db, 5)
    assert not adapter.fetch_market_states.called and result["settlements_recorded"] == 0
    assert "not checking them this cycle" in caplog.text


def test_closure_check_reports_a_venue_it_could_not_ask(stub_registry, stub_db):
    adapter = _adapter(fetch_market_states=MagicMock(side_effect=RuntimeError("down")))
    result = check_closures([adapter], KALSHI_EXCHANGES, {2: {"T:yes"}}, stub_registry, stub_db, 500)
    assert result["failed_exchanges"] == ["KALSHI"]
