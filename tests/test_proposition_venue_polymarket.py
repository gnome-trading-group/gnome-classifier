import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from classifier.propositions.derive import EQUIVALENT, FLAG_SOURCE_DIFFERS, Contract, derive
from classifier.propositions.types import Comparator, Kind, Observation
from classifier.propositions.venue import claims, extract_venue

# Sources as the Polymarket adapters build them, from the live venues (2026-10-29).
SOURCES = json.loads((Path(__file__).parent / "fixtures/relationships/polymarket_crypto_sources.json").read_text())


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def read(key: str):
    source = SOURCES[key]
    assert claims(source)
    proposition = extract_venue(source)
    assert proposition is not None
    return proposition


class TestPolymarketUs:
    @pytest.mark.parametrize("key, start, end", [
        ("cpc-btc-updown-15m-2026-10-29-2315z", utc(2026, 10, 29, 23, 15), utc(2026, 10, 29, 23, 30)),
        ("cpc-btc-updown-5m-2026-10-29-2320z", utc(2026, 10, 29, 23, 20), utc(2026, 10, 29, 23, 25)),
        ("cpc-btc-updown-1h-2026-10-29-2300z", utc(2026, 10, 29, 23), utc(2026, 10, 30, 0)),
        ("cpceth-eth-updown-1d-2026-10-29-2100z", utc(2026, 10, 29, 21), utc(2026, 10, 30, 21)),
    ])
    def test_up_or_down_is_the_change_over_the_window(self, key, start, end):
        p = read(key)
        assert (p.measure, p.comparator, p.value, p.observation) == ("change", Comparator.GTE, 0.0, Observation.AT_TIME)
        assert (p.window_start, p.window_end) == (start, end)

    def test_the_short_leg_is_the_negation(self):
        assert extract_venue({**SOURCES["cpc-btc-updown-15m-2026-10-29-2315z"], "leg": "short"}).negated

    def test_price_at_a_time(self):
        p = read("cpc-btc-above-hr-2026-10-30-0000z-78200")
        assert (p.subject, p.comparator, p.value, p.window_end) == ("btc-usd", Comparator.GTE, 78200.0, utc(2026, 10, 30, 0))
        assert p.resolution_source == "cf-benchmarks-brti"

    @pytest.mark.parametrize("key, subject, low, high", [
        ("cpc-btc-range-hr-2026-10-30-0000z-78700", "btc-usd", 78700.0, 78799.99),
        ("cpcxrp-xrp-range-wk-2026-10-30-1p2", "xrp-usd", 1.2, 1.24999),
        ("cpc-btc-pricerange-yr-12-31-2026-82500", "btc-usd", 80000.0, 84999.99),
    ])
    def test_ranges_include_both_bounds(self, key, subject, low, high):
        p = read(key)
        assert (p.subject, p.kind, p.low, p.high, p.low_inclusive, p.high_inclusive) == (subject, Kind.RANGE, low, high,
                                                                                         True, True)

    def test_year_end_edge_brackets(self):
        above, below = read("cpc-btc-pricerange-yr-12-31-2026-above-150k"), read("cpc-btc-pricerange-yr-12-31-2026-below-20k")
        assert (above.comparator, above.value, below.comparator, below.value) == (Comparator.GTE, 150000.0,
                                                                                Comparator.LTE, 19999.99)
        assert above.window_end == utc(2027, 1, 1, 5)

    def test_trimmed_mean_markets_name_a_different_source(self):
        assert read("cpc-btc-pricerange-yr-12-31-2026-82500").resolution_source == "cf-benchmarks-brti-trimmed-mean"

    @pytest.mark.parametrize("key, comparator, value, start", [
        ("cpc-btc-hit-h-w-2026-10-30-100000", Comparator.GTE, 100000.0, utc(2026, 10, 23, 21)),
        ("cpc-btc-hit-l-m-2026-10-77500", Comparator.LTE, 77500.0, utc(2026, 10, 1, 4)),
        ("cpc-btc-150k-12-31-2026", Comparator.GT, 149999.99, None),
        ("cpc-btc-above-yr-12-31-2026-200k", Comparator.GT, 200000.0, None),
        ("cpc-btc-hitprice-high-yr-12-31-2026-90k", Comparator.GT, 89999.99, None),
    ])
    def test_touching_a_level_within_a_window(self, key, comparator, value, start):
        p = read(key)
        assert (p.observation, p.comparator, p.value, p.window_start) == (Observation.ANY_TIME, comparator, value, start)

    def test_a_reworded_market_falls_back(self):
        source = SOURCES["cpc-btc-above-hr-2026-10-30-0000z-78200"]
        reworded = {**source, "market": {**source["market"], "description": "Settles Yes when Bitcoin clears $78,200 (BRTI)."}}
        assert claims(reworded) and extract_venue(reworded) is None

    def test_the_slug_and_the_index_must_name_the_same_asset(self):
        source = SOURCES["cpc-btc-above-hr-2026-10-30-0000z-78200"]
        mislabelled = {**source, "market": {**source["market"], "slug": "cpceth-eth-above-hr-2026-10-30-0000z-78200"}}
        assert extract_venue(mislabelled) is None


class TestPolymarketIntl:
    @pytest.mark.parametrize("label, fields", [
        ("<72,000", {"kind": Kind.THRESHOLD, "comparator": Comparator.LT, "value": 72000.0}),
        ("88,000-90,000", {"kind": Kind.RANGE, "low": 88000.0, "high": 90000.0, "high_inclusive": False}),
        (">90,000", {"kind": Kind.THRESHOLD, "comparator": Comparator.GTE, "value": 90000.0}),
    ])
    def test_brackets_give_a_boundary_price_to_the_higher_bracket(self, label, fields):
        p = read(f"bitcoin-price-on-november-3-2026:{label}")
        assert {k: getattr(p, k) for k in fields} == fields
        # Noon Eastern after the clocks change.
        assert (p.subject, p.window_end, p.resolution_source) == ("btc-usd", utc(2026, 11, 3, 17), "binance-btcusdt-1m-close")

    def test_brackets_below_a_dollar(self):
        assert read("xrp-price-on-october-30-2026:<0.90").value == 0.9

    def test_short_up_or_down_compares_a_twap_with_the_start(self):
        up, down = read("eth-updown-15m-1793367000:Up"), read("eth-updown-15m-1793367000:Down")
        assert (up.measure, up.window_start, up.window_end) == ("twap_change", utc(2026, 10, 30, 13, 30),
                                                                utc(2026, 10, 30, 13, 45))
        assert up.resolution_source == "chainlink-ethusd-twap-stream"
        assert down == up.negate()

    def test_short_up_or_down_in_other_wording_is_not_read(self):
        source = SOURCES["eth-updown-15m-1793367000:Up"]
        reworded = {**source, "market": {**source["market"], "description": source["market"]["description"].replace(
            "time-weighted average price (TWAP)", "price at the end")}}
        assert claims(reworded) and extract_venue(reworded) is None

    def test_hourly_binance_candle(self):
        p = read("bitcoin-up-or-down-october-29-2026-7pm-et:Up")
        assert (p.window_start, p.window_end, p.resolution_source) == (utc(2026, 10, 29, 23), utc(2026, 10, 30, 0),
                                                                       "binance-btcusdt-1h-candle")

    @pytest.mark.parametrize("key", [
        # A tie settles 50-50, so Down is not the negation of Up.
        "bitcoin-up-or-down-on-october-30-2026:Up",
        # Compares a TWAP over the window with the opening price.
        "eth-updown-4h-1793304000:Up",
        "meta-up-or-down-on-october-29-2026:Up",
    ])
    def test_families_left_to_the_llm(self, key):
        assert not claims(SOURCES[key]) and extract_venue(SOURCES[key]) is None


class TestAcrossVenues:
    def test_the_same_up_or_down_window_on_the_same_index_is_equivalent(self):
        kalshi = extract_venue({"venue": "KALSHI", "side": "yes", "market": {
            "ticker": "KXBTC15M-26OCT291930-30",
            "rules_primary": "If the simple average of the sixty seconds of CF Benchmarks' BRTI before 7:30 PM EDT on Oct 29, "
                             "2026 is at least the simple average of the sixty seconds of CF Benchmarks' BRTI before 7:15 PM "
                             "EDT on October 29, 2026, then the market resolves to Yes."}})
        us = read("cpc-btc-updown-15m-2026-10-29-2315z")
        [found] = derive([Contract(1, "KALSHI", kalshi), Contract(2, "POLYMARKET_US", us)])
        assert (found.relationship_type, found.flags) == (EQUIVALENT, ())

    def test_a_twap_window_never_matches_a_close_over_the_same_window(self):
        twap = read("eth-updown-15m-1793367000:Up")
        close = dataclasses.replace(twap, measure="change", resolution_source="chainlink-ethusd-stream")
        assert derive([Contract(1, "POLYMARKET_INTL", twap), Contract(2, "KALSHI", close)]) == []

    def test_the_same_hour_on_different_sources_is_equivalent_with_a_flag(self):
        us = read("cpc-btc-updown-1h-2026-10-29-2300z")
        intl = read("bitcoin-up-or-down-october-29-2026-7pm-et:Up")
        [found] = derive([Contract(1, "POLYMARKET_US", us), Contract(2, "POLYMARKET_INTL", intl)])
        assert (found.relationship_type, found.flags) == (EQUIVALENT, (FLAG_SOURCE_DIFFERS,))

    def test_windows_that_end_together_but_start_apart_are_unrelated(self):
        hour = read("cpc-btc-updown-1h-2026-10-29-2300z")
        last_quarter = dataclasses.replace(hour, window_start=utc(2026, 10, 29, 23, 45))
        assert derive([Contract(1, "POLYMARKET_US", hour), Contract(2, "KALSHI", last_quarter)]) == []
