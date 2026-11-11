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
from classifier.propositions.venue.common import names_team

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


class TestPolymarketIntlSpreads:
    """`Spread: Rams (-16.5)` resolves to the Rams if they win by 17 or more, otherwise to the Commanders."""

    RULES = 'This market will resolve to "Rams" if the Rams win the game by 17 or more points.\n\nOtherwise, this ' \
            'market will resolve to "Commanders".'

    def source(self, outcome, description=RULES):
        return {"venue": "POLYMARKET_INTL", "outcome": outcome,
                "event": {"slug": "nfl-la-was-2026-11-08", "title": "Rams vs. Commanders"},
                "market": {"question": "Spread: Rams (-16.5)", "sportsMarketType": "spreads",
                           "outcomes": '["Rams", "Commanders"]', "description": description}}

    def reading(self, **fields):
        base = dict(subject="game:nfl:2026-11-08:lar-was", kind=Kind.THRESHOLD, unit="points",
                    observation=Observation.CUMULATIVE, scope="match")
        return Proposition(**{**base, **fields})

    def winner(self, entity):
        return self.reading(kind=Kind.WINNER, measure="winner", entity=entity, unit=None, observation=None)

    def test_a_spread_read_as_the_moneyline_becomes_the_margin(self):
        fixed = correct_venue(self.source("Rams"), self.winner("rams"))
        assert (fixed.kind, fixed.measure, fixed.comparator, fixed.value, fixed.unit) == (
            Kind.THRESHOLD, "margin:rams", Comparator.GTE, 17, "points")

    def test_the_other_outcome_keyed_by_its_own_team(self):
        fixed = correct_venue(self.source("Commanders"), self.winner("commanders"))
        assert (fixed.measure, fixed.comparator, fixed.value) == ("margin:commanders", Comparator.GT, -17)

    def test_a_wrong_line_is_set_from_the_rules(self):
        wrong = self.reading(measure="margin:rams", comparator=Comparator.GTE, value=4)
        assert correct_venue(self.source("Rams"), wrong).value == 17

    def test_a_wrong_side_is_put_right(self):
        # Commanders cover when the Rams win by less than 17: the Commanders' margin is over -17, not under 17.
        wrong = self.reading(measure="margin:commanders", comparator=Comparator.LT, value=17)
        fixed = correct_venue(self.source("Commanders"), wrong)
        assert (fixed.comparator, fixed.value) == (Comparator.GT, -17)

    def test_a_right_reading_is_left_alone(self):
        right = self.reading(measure="margin:rams", comparator=Comparator.LT, value=17)
        assert correct_venue(self.source("Commanders"), right) == right

    def test_a_key_the_name_does_not_place_keeps_a_reading_in_the_rules_form(self):
        # `la` names neither team by name; the reading itself says it's keyed to the favourite.
        right = self.reading(measure="margin:la", comparator=Comparator.GTE, value=17)
        assert correct_venue(self.source("Rams"), right) == right

    def test_whole_number_forms_count_as_the_rules_form(self):
        right = self.reading(measure="margin:la", comparator=Comparator.LTE, value=16)
        fixed = correct_venue(self.source("Commanders"), right)
        assert (fixed.measure, fixed.comparator, fixed.value) == ("margin:la", Comparator.LT, 17)

    def test_a_moneyline_misread_the_name_does_not_place_is_left_unknown(self):
        fixed = correct_venue(self.source("Rams"), self.winner("la"))
        assert "measure" in fixed.unknown

    def test_goals_in_soccer(self):
        rules = 'This market will resolve to "AD Ceuta FC" if AD Ceuta FC win the game by 3 or more goals.'
        source = {"venue": "POLYMARKET_INTL", "outcome": "AD Ceuta FC", "event": {},
                  "market": {"question": "Spread: AD Ceuta FC (-2.5)", "sportsMarketType": "spreads",
                             "outcomes": '["AD Ceuta FC", "CD Tenerife"]', "description": rules}}
        fixed = correct_venue(source, self.reading(kind=Kind.WINNER, measure="winner", entity="ceuta"))
        assert (fixed.measure, fixed.comparator, fixed.value, fixed.unit) == ("margin:ceuta", Comparator.GTE, 3, "goals")

    def test_other_markets_are_left_alone(self):
        source = {**self.source("Rams"), "market": {**self.source("Rams")["market"], "sportsMarketType": "moneyline"}}
        assert correct_venue(source, self.winner("rams")) == self.winner("rams")


class TestTeamNames:
    def test_an_accented_letter_does_not_split_a_word(self):
        # "Varaždin" once read as the words `vara` and `din`, so `din` named Varaždin surely.
        assert names_team("NK Varaždin", None, "din") == 0
        assert names_team("NK Varaždin", None, "varazdin") == 2

    def test_the_start_of_a_later_word(self):
        assert names_team("GNK Dinamo Zagreb", None, "din") == 2
        assert names_team("IF Elfsborg", None, "elf") == 2
        assert names_team("Carlos Alcaraz", None, "alc") == 2

    def test_two_letters_of_a_later_word_are_not_enough(self):
        assert names_team("GNK Dinamo Zagreb", None, "za") == 0

    def test_accents_in_the_name_and_the_key(self):
        assert names_team("QUINTESSÊNCIA", None, "quintessencia") == 2
        assert names_team("Bodø/Glimt", None, "bodo") == 2


class TestPolymarketIntlSpreadsByName:
    RULES = ('This market will resolve to "GNK Dinamo Zagreb" if GNK Dinamo Zagreb win the game by 2 or more goals. '
             'Otherwise, this market will resolve to "NK Varaždin".')

    def correct(self, outcome, reading):
        source = {"venue": "POLYMARKET_INTL", "outcome": outcome, "event": {},
                  "market": {"question": "Spread: GNK Dinamo Zagreb (-1.5)", "sportsMarketType": "spreads",
                             "outcomes": '["GNK Dinamo Zagreb", "NK Varaždin"]', "description": self.RULES}}
        return correct_venue(source, reading)

    def margin(self, comparator, value):
        return Proposition(subject="game:hnl:2026-11-08:vardin", kind=Kind.THRESHOLD, measure="margin:din",
                           comparator=comparator, value=value, unit="goals", observation=Observation.CUMULATIVE,
                           scope="regulation")

    def test_a_key_starting_a_later_word_puts_a_wrong_side_right(self):
        # The model read Dinamo covering as `din` losing by 2 or more: in the rules' form for the other team, but `din`
        # is Dinamo.
        fixed = self.correct("GNK Dinamo Zagreb", self.margin(Comparator.LTE, -2))
        assert (fixed.measure, fixed.comparator, fixed.value) == ("margin:din", Comparator.GTE, 2)
        fixed = self.correct("NK Varaždin", self.margin(Comparator.GT, -2))
        assert (fixed.measure, fixed.comparator, fixed.value) == ("margin:din", Comparator.LT, 2)


class TestPolymarketIntlPartSpreads:
    """`1H Spread: Lions (-2.5)` resolves to the Lions if they outscore the Dolphins by 3 or more points in the first
    half, otherwise to the Dolphins."""

    def source(self, outcome, family="first_half_spreads", part="first half", short="1H"):
        rules = (f'This market will resolve to "Lions" if the Lions outscore the Dolphins by 3 or more points in the '
                 f'{part}. Otherwise, this market will resolve to "Dolphins".')
        return {"venue": "POLYMARKET_INTL", "outcome": outcome, "event": {},
                "market": {"question": f"{short} Spread: Lions (-2.5)", "sportsMarketType": family,
                           "outcomes": '["Lions", "Dolphins"]', "description": rules}}

    def reading(self, **fields):
        base = dict(subject="game:nfl:2026-11-08:detmia", kind=Kind.THRESHOLD, unit="points",
                    observation=Observation.CUMULATIVE, scope="first_half")
        return Proposition(**{**base, **fields})

    def test_the_models_measure_becomes_the_margin_scoped_to_the_half(self):
        fixed = correct_venue(self.source("Lions"), self.reading(measure="margin_1h:det", comparator=Comparator.GTE,
                                                                 value=3))
        assert (fixed.measure, fixed.comparator, fixed.value, fixed.scope) == (
            "margin:det", Comparator.GTE, 3, "first_half")

    def test_the_other_outcome(self):
        fixed = correct_venue(self.source("Dolphins"), self.reading(measure="margin_1h:det", comparator=Comparator.LT,
                                                                    value=3))
        assert (fixed.measure, fixed.comparator, fixed.value) == ("margin:det", Comparator.LT, 3)

    def test_quarters_and_the_second_half(self):
        for family, part, short, scope in (("q1_spreads", "first quarter", "1Q", "first_quarter"),
                                           ("second_half_spreads", "second half", "2H", "second_half"),
                                           ("q4_spreads", "fourth quarter", "4Q", "fourth_quarter")):
            fixed = correct_venue(self.source("Lions", family, part, short),
                                  self.reading(measure="margin:lions", comparator=Comparator.GTE, value=3, scope=None))
            assert (fixed.measure, fixed.scope) == ("margin:lions", scope)

    def test_a_spread_misread_as_the_halfs_winner_the_name_does_not_place_is_left_unknown(self):
        # "Lions" doesn't name `det`, so it can't be put on a side; as the half's winner it would equal Kalshi's.
        fixed = correct_venue(self.source("Lions"), self.reading(kind=Kind.WINNER, measure="winner", entity="det",
                                                                 unit=None, observation=None))
        assert "measure" in fixed.unknown


class TestPolymarketIntlMapWinners:
    def correct(self, measure, scope):
        source = {"venue": "POLYMARKET_INTL", "outcome": "maybe", "event": {},
                  "market": {"question": "Counter-Strike: KUUSAMO.gg vs maybe - Map 2 Winner",
                             "sportsMarketType": "child_moneyline", "outcomes": '["KUUSAMO.gg", "maybe"]'}}
        reading = Proposition(subject="game:cs2:2026-11-05:ksm-maybe", kind=Kind.WINNER, entity="maybe",
                              measure=measure, scope=scope)
        return correct_venue(source, reading)

    def test_a_map_winner_read_as_the_series_winner_is_scoped_to_its_map(self):
        fixed = self.correct("winner", "match")
        assert (fixed.measure, fixed.scope) == ("winner", "map:2")

    def test_every_spelling_of_the_measure_becomes_winner(self):
        assert {self.correct(m, "map:2").measure for m in (None, "map_winner", "map_2_winner")} == {"winner"}


class TestPolymarketIntlMapHandicaps:
    """`Map Handicap: SAW.Y (-1.5) vs Four Magic (+1.5)` resolves to SAW Youngsters if it wins 2 or more maps than Four
    Magic, otherwise to Four Magic."""

    def source(self, outcome):
        rules = ('This market will resolve to "SAW Youngsters" if SAW Youngsters wins 2 or more maps than Four Magic in '
                 'this match. Otherwise, this market will resolve to "Four Magic".')
        return {"venue": "POLYMARKET_INTL", "outcome": outcome, "event": {},
                "market": {"question": "Map Handicap: SAW.Y (-1.5) vs Four Magic (+1.5)", "sportsMarketType": "map_handicap",
                           "outcomes": '["SAW Youngsters", "Four Magic"]', "description": rules}}

    def test_a_handicap_read_as_the_series_winner_becomes_the_maps_margin(self):
        wrong = Proposition(subject="game:cs2:2026-11-06:sawy-4m1", kind=Kind.WINNER, measure="winner",
                            entity="saw-youngsters", scope="match")
        fixed = correct_venue(self.source("SAW Youngsters"), wrong)
        assert (fixed.kind, fixed.measure, fixed.comparator, fixed.value, fixed.unit, fixed.scope) == (
            Kind.THRESHOLD, "margin:saw-youngsters", Comparator.GTE, 2, "maps", "match")

    def test_the_underdog_side(self):
        wrong = Proposition(subject="game:cs2:2026-11-06:sawy-4m1", kind=Kind.THRESHOLD, measure="map_margin:four-magic",
                            comparator=Comparator.GTE, value=2, unit="maps", observation=Observation.CUMULATIVE,
                            scope="match")
        fixed = correct_venue(self.source("Four Magic"), wrong)
        assert (fixed.measure, fixed.comparator, fixed.value) == ("margin:four-magic", Comparator.GT, -2)


class TestPolymarketIntlRoundHandicaps:
    """`Map 3 Rounds Handicap: BetBoom Team (-3.5) vs Liquid (+3.5)` resolves to BetBoom Team if it wins map 3 by 4 or
    more rounds, otherwise to Liquid."""

    RULES = ('This market will resolve to "BetBoom Team" if BetBoom Team wins Map 3 by 4 or more rounds, including any '
             "overtime (i.e. BetBoom Team's round total on Map 3 exceeds Liquid's by 4 or more). Otherwise, this market "
             'will resolve to "Liquid".')

    def source(self, outcome):
        return {"venue": "POLYMARKET_INTL", "outcome": outcome, "event": {},
                "market": {"question": "Map 3 Rounds Handicap: BetBoom Team (-3.5) vs Liquid (+3.5)",
                           "sportsMarketType": "round_handicap_game_3", "outcomes": '["BetBoom Team", "Liquid"]',
                           "description": self.RULES}}

    def reading(self, **fields):
        base = dict(subject="game:cs2:2026-11-04:liquid-betboom", kind=Kind.THRESHOLD, unit="rounds",
                    observation=Observation.CUMULATIVE, scope="map:3")
        return Proposition(**{**base, **fields})

    def test_the_underdog_side_read_as_winning_by_the_line_is_put_right(self):
        wrong = self.reading(measure="margin:liquid", comparator=Comparator.GTE, value=4)
        fixed = correct_venue(self.source("Liquid"), wrong)
        assert (fixed.measure, fixed.comparator, fixed.value, fixed.scope) == ("margin:liquid", Comparator.GT, -4, "map:3")

    def test_a_handicap_read_as_the_match_winner_becomes_the_maps_margin(self):
        wrong = self.reading(kind=Kind.WINNER, measure="winner", entity="betboom", unit=None, observation=None,
                             scope="match")
        fixed = correct_venue(self.source("BetBoom Team"), wrong)
        assert (fixed.kind, fixed.measure, fixed.comparator, fixed.value, fixed.unit, fixed.scope) == (
            Kind.THRESHOLD, "margin:betboom", Comparator.GTE, 4, "rounds", "map:3")

    def test_the_models_margin_spellings_become_one(self):
        odd = self.reading(measure="round_margin_map3:betboom", comparator=Comparator.GTE, value=4)
        assert correct_venue(self.source("BetBoom Team"), odd).measure == "margin:betboom"

    def test_a_right_reading_is_left_alone(self):
        right = self.reading(measure="margin:betboom", comparator=Comparator.LT, value=4)
        assert correct_venue(self.source("Liquid"), right) == right
