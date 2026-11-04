import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from classifier.propositions.derive import (
    EQUIVALENT,
    FLAG_SOURCE_DIFFERS,
    IMPLIES,
    Contract,
    derivation_units,
    derive,
    derive_unit,
)
from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability
from classifier.propositions.venue import claims, correct_venue, extract_venue

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


# Rate-decision sources from both venues (prod registry, 2026-11-03).
RATE_SOURCES = json.loads((Path(__file__).parent / "fixtures/relationships/rate_decision_sources.json").read_text())


def rate(key: str):
    source = RATE_SOURCES[key]
    assert claims(source)
    proposition = extract_venue(source)
    assert proposition is not None
    return proposition


class TestPolymarketRateDecisions:
    def test_most_banks_round_to_the_nearest_25(self):
        cut, big_cut = rate("bank-of-england-decision:november:25 bps decrease"), rate(
            "bank-of-england-decision:november:50+ bps decrease")
        assert (cut.kind, cut.low, cut.high, cut.low_inclusive, cut.high_inclusive) == (Kind.RANGE, -37.5, 0.0, False, False)
        assert (big_cut.comparator, big_cut.value) == (Comparator.LTE, -37.5)
        hike = rate("bank-of-england-decision:november:50+ bps increase")
        assert (hike.comparator, hike.value) == (Comparator.GTE, 37.5)
        assert (cut.subject, cut.measure, cut.stability) == ("boe-bank-rate:2026-11-05", "bank_rate_change_bps",
                                                             Stability.SETTLED)
        assert cut.window_end == utc(2026, 11, 5, 23, 59)

    def test_the_fed_rounds_up_to_the_next_25(self):
        cut, big_cut = rate("fomc:january:25 bps decrease"), rate("fomc:january:50+ bps decrease")
        assert (cut.low, cut.high, cut.low_inclusive, cut.high_inclusive) == (-25.0, 0.0, True, False)
        assert (big_cut.comparator, big_cut.value) == (Comparator.LT, -25.0)
        assert (cut.subject, cut.window_end) == ("fomc:2027-01-meeting", utc(2027, 1, 28, 4, 59))

    def test_no_change_is_holding(self):
        p = rate("fomc:january:No change")
        assert (p.kind, p.low, p.high) == (Kind.RANGE, 0.0, 0.0)

    def test_cut_and_hike_labels(self):
        assert rate("bank-of-korea-decision:november:25 bps cut").high == 0.0
        assert rate("bank-of-korea-decision:november:25 bps hike").low == 0.0

    @pytest.mark.parametrize("key, subject", [
        ("ecb-interest-rates:december:25 bps decrease", "ecb-deposit-rate:2026-12-17"),
        ("reserve-bank-of-australia-decision:december:25 bps decrease", "rba-cash-rate:2026-12-08"),
        ("reserve-bank-of-india-decision:december:25 bps decrease", "rbi-policy-repo:2026-12-04"),
    ])
    def test_the_meeting_is_its_last_day(self, key, subject):
        assert rate(key).subject == subject

    def test_unknown_rounding_goes_to_the_llm(self):
        source = RATE_SOURCES["bank-of-england-decision:november:25 bps decrease"]
        reworded = {**source, "market": {**source["market"], "description": source["market"]["description"].replace(
            "rounded away from 0", "rounded toward 0")}}
        assert extract_venue(reworded) is None and claims(reworded)

    def test_a_meeting_date_in_another_month_than_the_event_goes_to_the_llm(self):
        source = RATE_SOURCES["bank-of-england-decision:november:25 bps decrease"]
        moved = {**source, "market": {**source["market"], "description": source["market"]["description"].replace(
            "November 5, 2026", "December 17, 2026")}}
        assert extract_venue(moved) is None

    @pytest.mark.parametrize("kalshi, polymarket, expected", [
        ("KXFEDDECISION-27JAN", "fomc:january", [
            ("50+ bps decrease", EQUIVALENT, "Cut >25bps"), ("Cut 25bps", IMPLIES, "25 bps decrease"),
            ("No change", EQUIVALENT, "Fed maintains rate"), ("Hike 25bps", IMPLIES, "25 bps increase"),
            ("50+ bps increase", EQUIVALENT, "Hike >25bps"),
        ]),
        ("KXCBDECISIONENGLAND-26NOV05", "bank-of-england-decision:november", [
            ("50+ bps decrease", IMPLIES, "Cut more than 25bps"), ("Cut 1-25bps", IMPLIES, "25 bps decrease"),
            ("No change", EQUIVALENT, "Maintain current rate"), ("Hike 1-25bps", IMPLIES, "25 bps increase"),
            ("50+ bps increase", IMPLIES, "Hike more than 25bps"),
        ]),
    ])
    def test_the_venues_meetings_relate_without_caveats(self, kalshi, polymarket, expected):
        keys = [k for k in RATE_SOURCES if k.startswith((kalshi, polymarket + ":"))]
        label = {i: RATE_SOURCES[k]["market"].get("yes_sub_title") or RATE_SOURCES[k]["outcome"] for i, k in enumerate(keys)}
        contracts = [Contract(i, RATE_SOURCES[k]["venue"], rate(k), None, 1 if k.startswith("KX") else 2)
                     for i, k in enumerate(keys)]
        rows = [d for unit in derivation_units(contracts) for d in derive_unit(unit)]

        def unordered(a, kind, b):
            # An equivalence is stored lower security first; which side that is doesn't matter here.
            return (*sorted((a, b)), kind) if kind == EQUIVALENT else (a, b, kind)

        got = sorted(unordered(label[d.security_id_a], d.relationship_type, label[d.security_id_b]) for d in rows)
        assert got == sorted(unordered(*row) for row in expected)
        assert all(d.flags == () for d in rows)


RULES_FOR_EVERY_LEVEL = ("This market will resolve to \"Yes\" if Anthropic's private market valuation, as measured by the "
                         "NPM Price reported by Nasdaq Private Market, LLC (NPM) for any date between market creation and "
                         "December 31, 2026, reaches or exceeds the listed amount.")


def level_market(title: str, question: str) -> dict:
    return {"venue": "POLYMARKET_INTL", "outcome": "Yes",
            "event": {"slug": "will-anthropics-valuation-hit-by-december-31"},
            "market": {"question": question, "groupItemTitle": title, "outcomes": '["Yes", "No"]',
                       "description": RULES_FOR_EVERY_LEVEL}}


def valuation(comparator: Comparator, value: float) -> Proposition:
    return Proposition(subject="anthropic:valuation-2026", kind=Kind.THRESHOLD, measure="valuation", unit="usd_bn",
                       comparator=comparator, value=value, observation=Observation.ANY_TIME,
                       window_end=utc(2027, 1, 1))


class TestLevelDirections:
    LOW = level_market("↓$700B", "Will Anthropic's valuation hit (LOW) $700B by December 31?")
    HIGH = level_market("↑$1.5T", "Will Anthropic's valuation hit (HIGH) $1.5T by December 31?")

    def test_a_low_level_read_as_reaching_it_from_below_is_a_fall_to_it(self):
        fixed = correct_venue(self.LOW, valuation(Comparator.GTE, 700))
        assert (fixed.comparator, fixed.value) == (Comparator.LTE, 700)

    def test_a_high_level_is_a_rise_to_it(self):
        assert correct_venue(self.HIGH, valuation(Comparator.GT, 1500)).comparator == Comparator.GTE

    def test_a_direction_the_reading_left_unknown_is_filled_in(self):
        unknown = dataclasses.replace(valuation(Comparator.GTE, 700), comparator=None, unknown=frozenset({"comparator"}))
        fixed = correct_venue(self.LOW, unknown)
        assert fixed.comparator == Comparator.LTE and fixed.unknown == frozenset()

    def test_the_no_side_keeps_its_negation(self):
        fixed = correct_venue(self.LOW, valuation(Comparator.GTE, 700).negate())
        assert fixed.negated and fixed.comparator == Comparator.LTE

    def test_a_title_and_question_that_disagree_change_nothing(self):
        mixed = level_market("↑$700B", "Will Anthropic's valuation hit (LOW) $700B by December 31?")
        assert correct_venue(mixed, valuation(Comparator.GTE, 700)).comparator == Comparator.GTE

    def test_markets_without_a_direction_and_other_kinds_are_left_alone(self):
        plain = level_market("$700B", "Will Anthropic's valuation reach $700B by December 31?")
        assert correct_venue(plain, valuation(Comparator.GTE, 700)).comparator == Comparator.GTE
        occurs = Proposition(subject="anthropic:ipo", kind=Kind.OCCURS_BY, window_end=utc(2027, 1, 1))
        assert correct_venue(self.LOW, occurs) == occurs


BY_PARTY_RULES = ("This market will resolve according to the party of the candidate who wins the TX-17 congressional "
                  "district seat in the U.S. House of Representatives in the 2026 midterm elections.")


class TestElectionsSettleOnTheParty:
    def source(self, title, outcome="Yes", rules=BY_PARTY_RULES):
        return {"venue": "POLYMARKET_INTL", "outcome": outcome, "event": {"slug": "tx-17-house-election-winner"},
                "market": {"question": f"TX-17 House Election Winner: {title}", "groupItemTitle": title,
                           "outcomes": '["Yes", "No"]', "description": rules}}

    def candidate(self, entity="pete-sessions", negated=False):
        return Proposition(subject="tx-17-house-2026", kind=Kind.WINNER, measure="winner", entity=entity, negated=negated)

    def test_the_letter_after_the_candidate_names_the_party(self):
        assert correct_venue(self.source("Pete Sessions (R)"), self.candidate()).entity == "republican"
        no_side = correct_venue(self.source("Pete Sessions (R)", outcome="No"), self.candidate(negated=True))
        assert (no_side.entity, no_side.negated) == ("republican", True)

    def test_a_market_that_settles_on_the_person_is_left_alone(self):
        rules = "This market will resolve to Yes if Pete Sessions wins the TX-17 seat."
        assert correct_venue(self.source("Pete Sessions (R)", rules=rules), self.candidate()).entity == "pete-sessions"


class TestPolymarketUsSpreads:
    """The title names the team the line favours; the market asks whether the event's first team covers it."""

    def source(self, question, slug, leg="long"):
        return {"venue": "POLYMARKET_US", "leg": leg,
                "market": {"question": question, "slug": slug, "sportsMarketType": "football_team_full_game_spread"}}

    def margin(self, key, comparator=Comparator.GT, value=10.5):
        return Proposition(subject="game:ncaaf:2026-10-31:calncst", kind=Kind.THRESHOLD, measure=f"margin:{key}",
                           comparator=comparator, value=value, unit="points", observation=Observation.CUMULATIVE)

    PLUS = ("Will the California cover 10.5 vs the NC State in California vs. NC State?",
            "asc-cfb-cah-ncst-2026-10-31-pos-10pt5")

    def test_the_long_side_is_the_first_team_covering(self):
        # "NC State wins by over 10.5 points": California +10.5 covers when NC State wins by less.
        fixed = correct_venue(self.source(*self.PLUS), self.margin("ncst"))
        assert (fixed.measure, fixed.comparator, fixed.value, fixed.negated) == ("margin:ncst", Comparator.LT, 10.5, False)

    def test_the_short_side_is_its_no(self):
        fixed = correct_venue(self.source(*self.PLUS, leg="short"), self.margin("ncst"))
        assert (fixed.comparator, fixed.value, fixed.negated) == (Comparator.LT, 10.5, True)

    def test_a_margin_keyed_to_the_first_team(self):
        minus = ("Will the California cover -3.5 vs the NC State in California vs. NC State?",
                 "asc-cfb-cah-ncst-2026-10-31-neg-3pt5")
        fixed = correct_venue(self.source(*minus), self.margin("cal", Comparator.LT, -3.5))
        assert (fixed.measure, fixed.comparator, fixed.value) == ("margin:cal", Comparator.GT, 3.5)

    def test_questions_without_the_article_and_keys_by_initials(self):
        source = self.source("Will Hampton cover +14.5 vs Georgia Tech in Hampton vs. Georgia Tech?",
                             "asc-cbb-hamp-gtech-2026-11-03-pos-14pt5")
        fixed = correct_venue(source, self.margin("gt", value=14.5))
        assert (fixed.comparator, fixed.value) == (Comparator.LT, 14.5)

    def test_a_name_beats_an_abbreviation(self):
        source = self.source("Will the Baltimore Ravens cover -2.5 vs the Buffalo Bills in Ravens vs. Bills?",
                             "aec-nfl-bal-buf-2026-11-02-neg-2pt5")
        assert correct_venue(source, self.margin("bal")).comparator == Comparator.GT

    def test_a_key_for_neither_team_is_left_unknown(self):
        source = self.source("Will the Florida cover 7.5 vs the Georgia in Florida vs. Georgia?",
                             "asc-cfb-fl-ga-2026-10-31-pos-7pt5")
        assert "measure" in correct_venue(source, self.margin("uga")).unknown

    def test_other_markets_are_left_alone(self):
        total = dataclasses.replace(self.margin("ncst"), measure="total_points")
        assert correct_venue(self.source(*self.PLUS), total) == total
