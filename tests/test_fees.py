import dataclasses
from datetime import datetime, timezone

from classifier.adapters.polymarket_intl import PolymarketIntlAdapter
from classifier.adapters.polymarket_us import _sport_slug
from classifier.adapters.types import AdapterContract
from classifier.stages.entities import create_entities
from classifier.stages.fees import (
    kalshi_definition,
    kalshi_fee_scopes,
    kalshi_schedules,
    polymarket_intl_definition,
    polymarket_intl_schedules,
    sync_venue_fee_schedules,
)
from gnomepy.registry.types import AssetClass, ContractType, Exchange, SecurityType

BEFORE_ACCUMULATOR = datetime(2025, 11, 15, tzinfo=timezone.utc)
AFTER_ACCUMULATOR = datetime(2026, 8, 7, tzinfo=timezone.utc)


def _taker_maker(definition: dict) -> tuple[float, float]:
    component = definition["components"][0]
    return component["taker"], component["maker"]


def test_kalshi_definition_by_fee_type_and_multiplier():
    assert _taker_maker(kalshi_definition("quadratic", 1, AFTER_ACCUMULATOR)) == (0.07, 0)
    assert _taker_maker(kalshi_definition("quadratic_with_maker_fees", 1, AFTER_ACCUMULATOR)) == (0.07, 0.0175)
    assert _taker_maker(kalshi_definition("quadratic_with_combo_maker_fees", 1, AFTER_ACCUMULATOR)) == (0.07, 0.035)
    # Exactly half, not 0.035000000000000004: schedules are compared by value to decide whether anything changed.
    assert _taker_maker(kalshi_definition("quadratic_with_maker_fees", 0.5, AFTER_ACCUMULATOR)) == (0.035, 0.00875)
    assert kalshi_definition("flat", 1, AFTER_ACCUMULATOR) is None


def test_kalshi_rounding_follows_the_cutover():
    assert kalshi_definition("quadratic", 1, BEFORE_ACCUMULATOR)["rounding"] == {"mode": "CEIL", "increment": 0.01, "scope": "FILL"}
    assert kalshi_definition("quadratic", 1, AFTER_ACCUMULATOR)["rounding"] == {
        "mode": "CEIL", "increment": 0.0001, "scope": "ORDER_ACCUMULATOR"}


def test_kalshi_series_history_gains_a_row_at_the_rounding_cutover():
    series = [
        {"id": "b", "series_ticker": "KXMLBGAME", "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 0.5,
         "scheduled_ts": "2026-08-07T04:59:45.131Z"},
        {"id": "a", "series_ticker": "KXMLBGAME", "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1,
         "scheduled_ts": "2025-10-04T07:00:00Z"},
    ]
    rows = kalshi_schedules(5, [], series, [])
    assert [(r["effective_from"], r["source_ref"], _taker_maker(r["definition"]), r["definition"]["rounding"]["scope"])
            for r in rows] == [
        ("2025-10-04T07:00:00Z", "kalshi:series_fee_change:a", (0.07, 0.0175), "FILL"),
        ("2026-03-03T00:00:00Z", "kalshi:series_fee_change:a:accumulator", (0.07, 0.0175), "ORDER_ACCUMULATOR"),
        ("2026-08-07T04:59:45.131000Z", "kalshi:series_fee_change:b", (0.035, 0.00875), "ORDER_ACCUMULATOR"),
    ]
    assert all(r["exchange_id"] == 5 and r["fee_group"] == "KXMLBGAME" for r in rows)


def test_kalshi_event_override_fills_in_its_series_and_a_lifted_override_clears():
    series = [{"id": "s", "series_ticker": "KXMLBHIT", "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 0.5,
               "scheduled_ts": "2026-08-07T00:00:00Z"}]
    events = [
        {"id": "e1", "event_ticker": "KXMLBHIT-G", "series_ticker": "KXMLBHIT", "fee_type_override": None,
         "fee_multiplier_override": 1, "scheduled_ts": "2026-10-16T00:00:00Z"},
        {"id": "e2", "event_ticker": "KXMLBHIT-G", "series_ticker": "KXMLBHIT", "fee_type_override": None,
         "fee_multiplier_override": None, "scheduled_ts": "2026-10-20T00:00:00Z"},
    ]
    rows = [r for r in kalshi_schedules(5, [], series, events) if "fee_event" in r]
    assert [(r["effective_from"], r["definition"] and _taker_maker(r["definition"])) for r in rows] == [
        ("2026-10-16T00:00:00Z", (0.07, 0.0175)),
        ("2026-10-20T00:00:00Z", None),
    ]
    assert all(r["fee_event"] == "KXMLBHIT-G" for r in rows)


def test_kalshi_series_charging_other_than_the_default_without_history_holds_from_july_2025():
    series = [
        {"ticker": "KXNBA", "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1},
        {"ticker": "KXMLBGAME", "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 0.5},
        {"ticker": "KXPLAIN", "fee_type": "quadratic", "fee_multiplier": 1},
        {"ticker": "KXBARE"},
    ]
    changes = [{"id": "a", "series_ticker": "KXMLBGAME", "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 0.5,
                "scheduled_ts": "2026-08-07T00:00:00Z"}]
    rows = kalshi_schedules(5, series, changes, [])
    assert [(r["fee_group"], r["effective_from"], r["source_ref"], _taker_maker(r["definition"])) for r in rows] == [
        ("KXMLBGAME", "2026-08-07T00:00:00Z", "kalshi:series_fee_change:a", (0.035, 0.00875)),
        ("KXNBA", "2025-07-01T00:00:00Z", "kalshi:series_fee_change:unrecorded-KXNBA", (0.07, 0.0175)),
        ("KXNBA", "2026-03-03T00:00:00Z", "kalshi:series_fee_change:unrecorded-KXNBA:accumulator", (0.07, 0.0175)),
    ]


def test_kalshi_fee_types_no_schedule_expresses_are_recorded_without_one():
    changes = [{"id": "a", "series_ticker": "KXFLAT", "fee_type": "flat", "fee_multiplier": 1,
                "scheduled_ts": "2026-08-07T00:00:00Z"}]
    events = [{"id": "e", "event_ticker": "KXFLAT-G", "series_ticker": "KXFLAT", "fee_type_override": "flat",
               "fee_multiplier_override": 1, "scheduled_ts": "2026-10-18T00:00:00Z"}]
    rows = kalshi_schedules(5, [], changes, events)
    assert [(r.get("fee_group") or r.get("fee_event"), r["definition"], r["source_ref"]) for r in rows] == [
        ("KXFLAT", None, "kalshi:series_fee_change:a:unsupported:flat"),
        ("KXFLAT-G", None, "kalshi:event_fee_change:e:unsupported:flat"),
    ]


def test_kalshi_fee_scopes_are_the_series_and_events_with_schedules():
    series = [
        {"ticker": "KXNBA", "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1},
        {"ticker": "KXINX", "fee_type": "quadratic", "fee_multiplier": 1},
        {"ticker": "KXPLAIN", "fee_type": "quadratic", "fee_multiplier": 1},
    ]
    changes = [{"id": "a", "series_ticker": "KXINX", "fee_type": "quadratic", "fee_multiplier": 1,
                "scheduled_ts": "2026-07-03T00:00:00Z"}]
    events = [{"id": "e", "event_ticker": "KXMLBRFI-G", "series_ticker": "KXMLBRFI", "fee_type_override": "quadratic",
               "fee_multiplier_override": 1, "scheduled_ts": "2026-10-18T00:00:00Z"}]
    assert kalshi_fee_scopes(series, changes, events) == ({"KXNBA", "KXINX"}, {"KXMLBRFI-G"})


def test_polymarket_intl_definition_from_the_market_schedule():
    market = {"feesEnabled": True, "feeSchedule": {"exponent": 1, "rate": 0.05, "takerOnly": True, "rebateRate": 0.15}}
    assert polymarket_intl_definition(market) == {
        "currency": "USD",
        "components": [{"kind": "PRICE_CURVE", "taker": 0.05, "maker": 0, "exponent": 1}],
        "rounding": {"mode": "HALF_UP", "increment": 0.00001, "scope": "FILL"},
    }
    assert polymarket_intl_definition({**market, "feesEnabled": False}) is None
    assert polymarket_intl_definition({"feesEnabled": True, "feeSchedule": {"rate": 0, "exponent": 1}}) is None
    assert polymarket_intl_definition({"feesEnabled": False}) is None


def _intl_definition(rate: float, exponent: int = 1) -> dict:
    return polymarket_intl_definition(
        {"feesEnabled": True, "feeSchedule": {"exponent": exponent, "rate": rate, "takerOnly": True}})


def test_polymarket_intl_fee_types_hold_from_their_rollout_unless_their_rates_moved():
    now = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
    rows = polymarket_intl_schedules(3, {
        "crypto_15_min": _intl_definition(0.25, 2),
        "politics_fees": _intl_definition(0.04),
        "economics_fees": _intl_definition(0.06),
        "new_fees": _intl_definition(0.05),
    }, now)
    assert [(r["fee_group"], r["effective_from"]) for r in rows] == [
        ("crypto_15_min", "2026-01-05T00:00:00Z"),
        ("politics_fees", "2026-03-30T00:00:00Z"),
        ("economics_fees", "2026-10-16T12:00:00Z"),
        ("new_fees", "2026-10-16T12:00:00Z"),
    ]
    assert all(r["exchange_id"] == 3 and r["source_ref"].startswith(f"polymarket_intl:fee_type:{r['fee_group']}:")
               for r in rows)
    moved = polymarket_intl_schedules(3, {"politics_fees": _intl_definition(0.05)}, now)
    assert moved[0]["source_ref"] != rows[1]["source_ref"]


def test_polymarket_intl_markets_are_grouped_by_fee_type():
    adapter = PolymarketIntlAdapter()
    schedule = {"exponent": 1, "rate": 0.04, "takerOnly": True, "rebateRate": 0.25}
    assert adapter._fee_group({"feeType": "politics_fees", "feesEnabled": True, "feeSchedule": schedule}) == "politics_fees"
    assert adapter._fee_group({"feeType": None, "feesEnabled": False}) is None
    assert adapter._fee_group({"feeType": "zero_fees", "feesEnabled": True,
                               "feeSchedule": {**schedule, "rate": 0}}) is None
    assert [r["fee_group"] for r in adapter.fetch_fee_schedules(3)] == ["politics_fees"]


def test_polymarket_us_fee_group_is_the_sport():
    event = {"tags": [{"slug": "sports"}, {"slug": "wtt", "league": {"slug": "wtt"}},
                      {"slug": "table-tennis", "sport": {"slug": "table-tennis"}}]}
    assert _sport_slug(event) == "table-tennis"
    assert _sport_slug({"tags": [{"slug": "politics"}]}) is None


def _contract(outcome: str, **fees) -> AdapterContract:
    return AdapterContract(
        exchange_id=2, exchange_security_id=f"T-1:{outcome.lower()}", exchange_security_symbol=f"T -- {outcome}",
        base_currency="USDC", quote_currency="USDC", settle_currency="USDC",
        security_type=SecurityType.EVENT_CONTRACT, contract_type=ContractType.BINARY, asset_class=AssetClass.PREDICTION,
        inverse=False, is_quanto=False, tick_size=1.0, lot_size=1.0, min_notional=0.0, contract_multiplier=1.0,
        event_title="T?", outcome_label=outcome, exchange_event_native_id="T-1", security_symbol=f"KX-T-1-{outcome}".upper(),
        **fees,
    )


def test_listings_carry_fee_keys_and_follow_the_venue(stub_registry, stub_db, mock_converse):
    contracts = [_contract(side, fee_group="KXT", fee_event="KXT-A") for side in ("Yes", "No")]
    create_entities(stub_registry, mock_converse, contracts, db=stub_db)
    assert set(stub_registry._fee_keys_by_listing.values()) == {("KXT", "KXT-A")}

    moved = [dataclasses.replace(c, fee_event="KXT-B") for c in contracts]
    create_entities(stub_registry, mock_converse, moved, db=stub_db)
    assert set(stub_registry._fee_keys_by_listing.values()) == {("KXT", "KXT-B")}


def test_a_lifted_event_override_keeps_its_fee_event(stub_registry, stub_db, mock_converse):
    create_entities(stub_registry, mock_converse, [_contract("Yes", fee_group="KXT", fee_event="KXT-A")], db=stub_db)
    create_entities(stub_registry, mock_converse, [_contract("Yes")], db=stub_db)
    assert set(stub_registry._fee_keys_by_listing.values()) == {(None, "KXT-A")}


class _FeeAdapter:
    exchange_code = "KALSHI"

    def fetch_fee_schedules(self, exchange_id):
        return [{"exchange_id": exchange_id, "fee_group": "KXT", "effective_from": "2026-01-01T00:00:00Z",
                 "definition": kalshi_definition("quadratic", 1, AFTER_ACCUMULATOR), "source_ref": "kalshi:series_fee_change:x"}]


def test_venue_schedules_are_written_once(stub_registry, stub_db):
    exchanges = {"KALSHI": Exchange(exchange_id=2, exchange_code="KALSHI", exchange_name="Kalshi", region="",
                                    schema_type="", date_modified="", date_created="")}
    first = sync_venue_fee_schedules([_FeeAdapter()], exchanges, stub_registry, stub_db)
    second = sync_venue_fee_schedules([_FeeAdapter()], exchanges, stub_registry, stub_db)
    assert (first["fee_schedules_written"], second["fee_schedules_written"]) == (1, 0)
    assert stub_registry._fee_schedules[0]["source"] == "VENUE_API"
