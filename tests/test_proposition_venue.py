from datetime import datetime, timezone

import pytest

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability
from classifier.propositions.venue import claims, extract_venue

# Rules text as Kalshi and Hyperliquid wrote it (prod registry, 2026-10-29).
BTC_RULES = ("If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time Index (BRTI) before 3 PM EDT "
             "is above 79899.99 at 3 PM EDT on Oct 29, 2026, then the market resolves to Yes.")
SPX_RULES = "If the end-of-day S&P 500 index value on October 29, 2026 is above 7679.9999, then the market resolves to Yes."
MLB_TOTAL_RULES = ("If Milwaukee and Tampa Bay collectively score more 10.5 runs in the Milwaukee vs Tampa Bay professional "
                   "baseball game originally scheduled for Oct 28, 2026 at 8:00 PM EDT, then the market resolves to Yes.")
MLB_SPREAD_RULES = ("If Milwaukee wins by more than 5.5 runs in the Milwaukee vs Tampa Bay professional baseball game "
                    "originally scheduled for Oct 28, 2026 at 8:00 PM EDT, then the market resolves to Yes.")
TEAM_TOTAL_RULES = ("If North Texas scores over 43.5 points in the Florida Atlantic vs North Texas college football game "
                    "originally scheduled for Oct 29, 2026, then the market resolves to Yes.")
FIRST_HALF_RULES = ("If the total goals scored by Everton and Newcastle  is more than 1.5 goals in the 1st Half of the Everton "
                    "vs Newcastle professional EFL Cup soccer game originally scheduled for Oct 29, 2026, then the market "
                    "resolves to Yes.")
EPL_RULES_SECONDARY = ("The following market refers to the Manchester United vs Aston Villa professional EPL soccer game "
                       "originally scheduled for Nov 8, 2026 after 90 minutes plus stoppage time (does not include extra "
                       "time or penalties).")


def kalshi(ticker: str, rules: str = "", side: str = "yes", **market) -> dict:
    return {"venue": "KALSHI", "event": {}, "market": {"ticker": ticker, "rules_primary": rules, **market}, "side": side}


class TestKalshiPrices:
    def test_btc_snapshot(self):
        p = extract_venue(kalshi("KXBTCD-26OCT2915-T79899.99", BTC_RULES))
        assert p == Proposition(
            subject="btc-usd", kind=Kind.THRESHOLD, measure="price", comparator=Comparator.GT, value=79899.99,
            unit="usd", observation=Observation.AT_TIME, window_end=datetime(2026, 10, 29, 19, tzinfo=timezone.utc),
            stability=Stability.LIVE, resolution_source="cf-benchmarks-brti",
        )

    def test_the_no_side_is_the_negation(self):
        assert extract_venue(kalshi("KXBTCD-26OCT2915-T79899.99", BTC_RULES, side="no")).negated

    def test_a_strike_field_that_agrees_keeps_the_value(self):
        p = extract_venue(kalshi("KXBTCD-26OCT2915-T79899.99", BTC_RULES, strike_type="greater", floor_strike=79899.99))
        assert p.value == 79899.99 and not p.unknown

    def test_a_strike_field_that_disagrees_leaves_the_threshold_unknown(self):
        p = extract_venue(kalshi("KXBTCD-26OCT2915-T79899.99", BTC_RULES, strike_type="greater", floor_strike=80000))
        assert p.unknown == {"comparator", "value"}

    def test_sp500_close_is_4pm_eastern_in_utc(self):
        p = extract_venue(kalshi("KXINXU-26OCT29H1600-T7679.9999", SPX_RULES))
        assert (p.subject, p.measure, p.value) == ("spx", "close", 7679.9999)
        assert p.window_end == datetime(2026, 10, 29, 20, tzinfo=timezone.utc)

    def test_winter_time_shifts_the_utc_hour(self):
        p = extract_venue(kalshi("KXBTCD-26DEC0115-T90000", BTC_RULES.replace("EDT", "EST").replace("Oct 29", "Dec 1")
                                 .replace("79899.99", "90000")))
        assert p.window_end == datetime(2026, 12, 1, 20, tzinfo=timezone.utc)

    def test_unparsed_rules_go_to_the_llm(self):
        assert extract_venue(kalshi("KXBTCD-26OCT2915-T79899.99", "Some new wording.")) is None


class TestKalshiGames:
    def test_total_runs(self):
        p = extract_venue(kalshi("KXMLBTOTAL-26OCT282000MILTB-11", MLB_TOTAL_RULES))
        assert (p.subject, p.measure, p.comparator, p.value, p.scope) == (
            "game:mlb:2026-10-28:miltb-2000", "total_runs", Comparator.GT, 10.5, "match")
        assert p.observation == Observation.CUMULATIVE

    def test_spread_and_total_share_the_game_subject(self):
        total = extract_venue(kalshi("KXMLBTOTAL-26OCT282000MILTB-11", MLB_TOTAL_RULES))
        spread = extract_venue(kalshi("KXMLBSPREAD-26OCT282000MILTB-MIL6", MLB_SPREAD_RULES))
        assert spread.subject == total.subject
        assert (spread.measure, spread.value) == ("margin:mil", 5.5)

    def test_team_total(self):
        p = extract_venue(kalshi("KXNCAAFTEAMTOTAL-26OCT29FAUUNT-UNT44", TEAM_TOTAL_RULES))
        assert (p.subject, p.measure, p.value) == ("game:ncaaf:2026-10-29:fauunt", "team_points:unt", 43.5)

    def test_first_half_total_is_its_own_scope(self):
        p = extract_venue(kalshi("KXEFLCUP1HTOTAL-26OCT29EVENEW-2", FIRST_HALF_RULES))
        assert (p.subject, p.measure, p.value, p.scope) == ("game:eflcup:2026-10-29:evenew", "total_goals", 1.5, "first_half")

    def test_regulation_time_total(self):
        p = extract_venue(kalshi("KXEFLTOTAL-26AUG09WOLPV-3", MLB_TOTAL_RULES.replace("runs", "goals"),
                                 title="Wolverhampton vs Port Vale: Regulation Time Total Goals: Reg Time: Over 2.5"))
        assert p.scope == "regulation"

    def test_soccer_winner_is_decided_in_regulation(self):
        p = extract_venue(kalshi("KXEPLGAME-26NOV08MUNAVL-TIE", "If the game ends in a tie, Tie resolves to Yes.",
                                 side="yes", rules_secondary=EPL_RULES_SECONDARY))
        assert (p.kind, p.entity, p.scope) == (Kind.WINNER, "tie", "regulation")

    def test_map_winner_scope(self):
        p = extract_venue(kalshi("KXCS2MAP-26OCT301400TSAFAL-1-FAL", "map 1"))
        assert (p.subject, p.entity, p.scope) == ("game:cs2:2026-10-30:tsafal-1400", "fal", "map:1")

    def test_other_kalshi_series_go_to_the_llm(self):
        assert extract_venue(kalshi("KXELONMARS-99", "If Elon Musk visits Mars before 2099.")) is None


class TestHyperliquid:
    def source(self, label="Yes"):
        return {"venue": "HYPERLIQUID", "label": label, "outcome": {
            "description": "class:priceBinary|underlying:BTC|expiry:20261002-0600|targetPrice:84277|period:1d"}}

    def test_recurring_price_binary(self):
        p = extract_venue(self.source())
        assert (p.subject, p.comparator, p.value, p.resolution_source) == ("btc-usd", Comparator.GT, 84277.0,
                                                                            "hyperliquid-twap-300s")
        assert p.window_end == datetime(2026, 10, 2, 6, tzinfo=timezone.utc)

    def test_no_label_is_negated(self):
        assert extract_venue(self.source("No")).negated

    def test_free_text_outcomes_go_to_the_llm(self):
        assert extract_venue({"venue": "HYPERLIQUID", "outcome": {"description": "other"}}) is None


def test_polymarket_always_goes_to_the_llm():
    assert extract_venue({"venue": "POLYMARKET_INTL", "market": {"question": "Will BTC hit 100k?"}}) is None


class TestFields:
    def test_round_trip(self):
        p = extract_venue(kalshi("KXBTCD-26OCT2915-T79899.99", BTC_RULES, side="no"))
        assert Proposition.from_fields(p.subject, p.to_fields()) == p

    @pytest.mark.parametrize("kwargs", [
        {"kind": Kind.THRESHOLD, "value": 1.0},
        {"kind": Kind.WINNER},
        {"kind": Kind.RANGE, "low": 1.0},
        {"kind": Kind.OTHER, "unknown": frozenset({"made_up"})},
    ])
    def test_refuses_incomplete_propositions(self, kwargs):
        with pytest.raises(ValueError):
            Proposition(subject="btc-usd", **kwargs)


def test_soccer_totals_are_regulation_time():
    rules = ("If over 4.5 goals are scored in the AIK vs Hammarby professional Allsvenskan soccer game originally scheduled "
             "for Aug 30, 2026 after 90 minutes plus stoppage time (does not include extra time or penalties), then the "
             "market resolves to Yes.")
    p = extract_venue(kalshi("KXALLSVENSKANTOTAL-26AUG30AIKHIF-5", rules))
    assert (p.subject, p.measure, p.value, p.scope) == ("game:allsvenskan:2026-08-30:aikhif", "total_goals", 4.5, "regulation")


def test_xrp_index_written_without_the_hyphen():
    rules = ("If the simple average of the sixty seconds of CF Benchmarks' Ripple-Dollar Real Time Index (XRPUSD_RTI) before "
             "5 PM EDT is above 1.4999 at 5 PM EDT on Oct 3, 2026, then the market resolves to Yes.")
    assert extract_venue(kalshi("KXXRPD-26OCT0317-T1.4999", rules)).subject == "xrp-usd"


class TestLivePayloads:
    """Market fields and rules as Kalshi's API returned them on 2026-10-29."""

    BTC_RULE = ("If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time Index (BRTI) before 7 PM "
                "EDT is {} at 7 PM EDT on Oct 29, 2026, then the market resolves to Yes.")

    def test_below_uses_the_cap_strike(self):
        p = extract_venue(kalshi("KXBTC-26OCT2919-T71700", self.BTC_RULE.format("below 71700"),
                                 strike_type="less", cap_strike=71700))
        assert (p.comparator, p.value, p.unknown) == (Comparator.LT, 71700.0, frozenset())

    def test_dashed_ranges(self):
        p = extract_venue(kalshi("KXBTC-26OCT2919-B71750", self.BTC_RULE.format("between 71700-71799.99"),
                                 strike_type="between", floor_strike=71700, cap_strike=71799.99))
        assert (p.kind, p.low, p.high, p.unknown) == (Kind.RANGE, 71700.0, 71799.99, frozenset())

    def test_sp500_intraday_and_a_strike_naming_the_boundary_from_the_other_side(self):
        rules = "If the S&P 500 index value on Oct 30, 2026 at 10am EDT is above 7574.9999, then the market resolves to Yes."
        p = extract_venue(kalshi("KXINXU-26OCT30H1000-T7574.9999", rules, strike_type="greater_or_equal", floor_strike=7575))
        assert (p.subject, p.comparator, p.value, p.unknown) == ("spx", Comparator.GT, 7574.9999, frozenset())
        assert p.window_end == datetime(2026, 10, 30, 14, tzinfo=timezone.utc)

    def test_a_strike_in_the_other_direction_still_leaves_the_threshold_unknown(self):
        p = extract_venue(kalshi("KXBTCD-26OCT2919-T71599.99", self.BTC_RULE.format("above 71599.99"),
                                 strike_type="less", cap_strike=71599.99))
        assert p.unknown == {"comparator", "value"}


class TestFamiliesFromTheOpenMarketDryRun:
    """Wording found when every open Kalshi market in prod went through the readers (2026-10-29)."""

    def test_basketball_totals_say_total_points(self):
        p = extract_venue(kalshi("KXNBATOTAL-26OCT03MIATOR-232", "If over 231.5 total points are scored in the Miami vs "
                                 "Toronto Pro Basketball game originally scheduled for Oct 3, 2026, then the market resolves to Yes."))
        assert (p.subject, p.measure, p.value, p.scope) == ("game:nba:2026-10-03:miator", "total_points", 231.5, "match")

    def test_basketball_first_half_lowercase(self):
        p = extract_venue(kalshi("KXNBA1HTOTAL-26OCT03MIATOR-115", "If over 114.5 total points are scored in the 1st half of "
                                 "the Miami vs Toronto Pro Basketball game originally scheduled for Oct 3, 2026."))
        assert (p.subject, p.scope) == ("game:nba:2026-10-03:miator", "first_half")

    def test_soccer_team_totals_say_records_above(self):
        p = extract_venue(kalshi("KXLALIGATEAMTOTAL-26AUG23ELCBAR-BAR4", "If Barcelona records above 3.5 goals in the Elche vs "
                                 "Barcelona professional La Liga soccer game originally scheduled for Aug 23, 2026 after 90 "
                                 "minutes plus stoppage time"))
        assert (p.measure, p.value, p.scope) == ("team_goals:bar", 3.5, "regulation")

    def test_sol_names_its_index_by_code(self):
        p = extract_venue(kalshi("KXSOLD-26OCT0220-T118.4999", "If the simple average of the sixty seconds of CF Benchmarks' "
                                 "SOLUSD_RTI before 8 PM EDT is above $118.4999 at 8 PM EDT on Oct 2, 2026, then the market resolves to Yes."))
        assert (p.subject, p.comparator, p.value) == ("sol-usd", Comparator.GT, 118.4999)

    def test_fifteen_minute_markets_are_the_change_over_the_window(self):
        p = extract_venue(kalshi("KXBTC15M-26OCT021130-30", "If the simple average of the sixty seconds of CF Benchmarks' BRTI "
                                 "before 11:30 AM EDT on Oct 2, 2026 is at least the simple average of the sixty seconds of "
                                 "CF Benchmarks' BRTI before 11:15 AM EDT on October 2, 2026, then the market resolves to Yes."))
        assert (p.subject, p.measure, p.comparator, p.value) == ("btc-usd", "change", Comparator.GTE, 0.0)
        assert (p.window_start, p.window_end) == (datetime(2026, 10, 2, 15, 15, tzinfo=timezone.utc),
                                                  datetime(2026, 10, 2, 15, 30, tzinfo=timezone.utc))
        assert p.resolution_source == "cf-benchmarks-brti"

    @pytest.mark.parametrize("ticker, rules, measure, value", [
        ("KXNFLREC-26SEP09NESEA-NEMHOLLINS13-2", "If Mack Hollins records 2+ receptions in the New England vs Seattle Pro "
         "Football game originally scheduled for Sep 9, 2026", "receptions:mack-hollins", 2.0),
        ("KXNFLTD-26SEP13NYJTEN-TENWROBINSON4-1", "If Wan'Dale Robinson scores at least 1+ touchdowns in the New York J vs "
         "Tennessee Pro Football game originally scheduled for Sep 13, 2026", "touchdowns:wan-dale-robinson", 1.0),
        ("KXMLBHR-26AUG181940ATHKC-ATHTWHITE47-1", "If Tommy White records 1+ home runs in A's vs Kansas City professional "
         "baseball game originally scheduled for Aug 18, 2026 at 7:40 PM EDT", "home_runs:tommy-white", 1.0),
        ("KXNFLRECYDS-26SEP20CINHOU-HOUXHUTCHINSON19-60", "If Xavier Hutchinson records 60+ receiving yards in the Cincinnati "
         "vs Houston Pro Football game originally scheduled for Sep 20, 2026", "receiving_yards:xavier-hutchinson", 60.0),
    ])
    def test_player_props(self, ticker, rules, measure, value):
        p = extract_venue(kalshi(ticker, rules))
        assert (p.measure, p.comparator, p.value, p.observation) == (measure, Comparator.GTE, value, Observation.CUMULATIVE)
        assert p.subject.startswith("game:")

    def test_a_player_prop_and_its_game_total_share_a_subject(self):
        prop = extract_venue(kalshi("KXNBAPTS-26OCT29MEMPOR-PORJMORANT1-20", "If Ja Morant records 20+ points in the Memphis "
                                    "vs Portland Pro Basketball game originally scheduled for Oct 29, 2026"))
        total = extract_venue(kalshi("KXNBATOTAL-26OCT29MEMPOR-220", "If over 219.5 total points are scored in the Memphis vs "
                                     "Portland Pro Basketball game originally scheduled for Oct 29, 2026"))
        assert prop.subject == total.subject == "game:nba:2026-10-29:mempor"

    def test_tennis_games(self):
        total = extract_venue(kalshi("KXATPGTOTAL-26SEP08TIAMIC-52", "If the number of completed games in the full match is "
                                     "above 51.5 in the Frances Tiafoe vs Alex Michelsen professional tennis match"))
        spread = extract_venue(kalshi("KXATPGSPREAD-26SEP08TIAMIC-TIA6", "If the game differential in favor of Frances Tiafoe "
                                      "across the full match is above 5.5 games in the Frances Tiafoe vs Alex Michelsen match"))
        assert total.subject == spread.subject == "game:atp:2026-09-08:tiamic"
        assert (total.measure, total.value, spread.measure, spread.value) == ("total_games", 51.5, "margin:tia", 5.5)


class TestLongerSeriesNames:
    def test_series_that_only_end_in_a_stat_code_are_not_props(self):
        for ticker in ("KXNFLFIRSTTD-26SEP13DALNYG-NYGIABANIKANDA25", "KXNCAAFDSTTD-26SEP12ALAMSST-1",
                       "KXNFLFFPTS-26OCT29CARGB-GBJLOVE10-17"):
            source = kalshi(ticker, "If Israel Abanikanda scores the first touchdown in Dallas vs New York G Pro Football game")
            assert extract_venue(source) is None and not claims(source)

    def test_hyphenated_stats_and_doubleheaders(self):
        three = extract_venue(kalshi("KXNBA3PT-26OCT29MEMPOR-MEMJMORANT12-2", "If Ja Morant records 2+ made three-pointers in "
                                     "the Memphis vs Portland Pro Basketball game originally scheduled for Oct 29, 2026"))
        assert three.measure == "made_three_pointers:ja-morant"
        game_two = extract_venue(kalshi("KXMLBHIT-26SEP231840TORBAL-TORJSMITH4-1", "If Josh Smith records 1+ hits in the Toronto "
                                        "vs Baltimore professional baseball game 2 of the double header originally scheduled for "
                                        "Sep 23, 2026 at 6:40 PM EDT"))
        assert game_two.subject == "game:mlb:2026-09-23:torbal-1840"

    def test_series_spreads_count_games(self):
        p = extract_venue(kalshi("KXLOLSPREAD-26OCT301600LOSTL-TL2", "If Team Liquid wins by over 1.5 games in the Worlds "
                                 "2026: LOS vs. Team Liquid League of Legends match originally scheduled for Oct 30, 2026 at "
                                 "4:00 PM EDT, then the market resolves to Yes."))
        assert (p.measure, p.value, p.unit) == ("margin:tl", 1.5, "games")

    def test_first_half_spreads_say_win_by(self):
        p = extract_venue(kalshi("KXINTLFRIENDLY1HSPREAD-26OCT10ARGBFA-ARG2", "If Argentina win by more than 1.5 goals in the 1st "
                                 "Half of the Argentina vs Burkina Faso professional Intl Friendlies soccer game"))
        assert (p.subject, p.measure, p.value, p.scope) == ("game:intlfriendly:2026-10-10:argbfa", "margin:arg", 1.5, "first_half")
