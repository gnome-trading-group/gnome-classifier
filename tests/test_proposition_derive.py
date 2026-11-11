import dataclasses
import time
from datetime import datetime, timedelta, timezone

import pytest

from classifier.propositions.derive import (
    DERIVATION_VERSION,
    EQUIVALENT,
    FLAG_DEADLINE_DIFFERS,
    FLAG_DEADLINE_ZONE,
    FLAG_SNAPSHOT_TIME,
    FLAG_SOURCE_DIFFERS,
    FLAG_SOURCE_UNKNOWN,
    FLAG_VOID_RULES,
    FLAG_VOTE_SCOPE,
    IMPLIES,
    Contract,
    derivation_units,
    derive,
    derive_unit,
    minimal_rows,
)
from gnomepy.registry.relations import Edge, close

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability

T = datetime(2026, 10, 29, 19, tzinfo=timezone.utc)
GT, GTE, LT, LTE = Comparator.GT, Comparator.GTE, Comparator.LT, Comparator.LTE


def threshold(comparator, value, measure="price", observation=Observation.AT_TIME, end=T, **kw):
    return Proposition(subject="s", kind=Kind.THRESHOLD, measure=measure, comparator=comparator, value=value,
                       observation=observation, window_end=end, **kw)


def goals(comparator, value, **kw):
    return threshold(comparator, value, measure="total_goals", unit="goals", observation=Observation.CUMULATIVE,
                     end=None, scope="regulation", **kw)


def by(end, measure="ipo", **kw):
    return Proposition(subject="s", kind=Kind.OCCURS_BY, measure=measure, window_end=end, **kw)


def winner(entity, scope="match", measure="winner", **kw):
    return Proposition(subject="s", kind=Kind.WINNER, entity=entity, scope=scope, measure=measure, **kw)


def relation(a: Proposition, b: Proposition, venue_a="K", venue_b="K", event_a=None, event_b=None):
    found = derive([Contract(1, venue_a, a, event_id=event_a), Contract(2, venue_b, b, event_id=event_b)])
    assert len(found) <= 1
    if not found:
        return None
    d = found[0]
    if d.relationship_type == EQUIVALENT:
        return EQUIVALENT, d.flags
    return ("1=>2" if d.security_id_a == 1 else "2=>1"), d.flags


class TestValueSets:
    @pytest.mark.parametrize("a, b, expected", [
        (threshold(GT, 85100), threshold(GT, 84500), "1=>2"),             # a higher strike implies a lower one
        (threshold(GT, 84500), threshold(GT, 85100), "2=>1"),
        (threshold(GT, 100), threshold(GTE, 100), "1=>2"),                # continuous: > is stricter than >=
        (threshold(GT, 100), threshold(GT, 100), EQUIVALENT),
        (threshold(LT, 33), threshold(LT, 34), "1=>2"),                   # lowest approval below 33 implies below 34
        (threshold(GT, 100, negated=True), threshold(LTE, 100), EQUIVALENT),
        (threshold(GT, 100), threshold(LT, 50), None),                    # disjoint
    ])
    def test_continuous(self, a, b, expected):
        assert (relation(a, b) or (None,))[0] == expected

    @pytest.mark.parametrize("a, b, expected", [
        (goals(GT, 4.5), goals(GTE, 5), EQUIVALENT),                      # over 4.5 goals == 5+ goals
        (goals(GT, 4.5, negated=True), goals(LT, 4.5), EQUIVALENT),       # not over 4.5 == under 4.5
        (goals(GT, 5.5), goals(GT, 4.5), "1=>2"),
        (goals(LT, 1.5), goals(LT, 3.5), "1=>2"),                         # under 1.5 implies under 3.5
        (goals(GT, 1.5, negated=True), goals(GT, 5.5, negated=True), "1=>2"),
    ])
    def test_integer_measures(self, a, b, expected):
        assert (relation(a, b) or (None,))[0] == expected

    def test_a_range_inside_a_threshold(self):
        inside = Proposition(subject="s", kind=Kind.RANGE, measure="price", low=80000, high=80500,
                             observation=Observation.AT_TIME, window_end=T)
        assert relation(inside, threshold(GT, 79000))[0] == "1=>2"

    def test_percentage_margins_are_not_integers(self):
        a = threshold(GT, 1.5, measure="margin:republican", unit="percentage_points")
        b = threshold(GTE, 2, measure="margin:republican", unit="percentage_points")
        assert relation(a, b)[0] == "2=>1"


class TestReadings:
    def test_different_snapshot_times_are_unrelated_for_live_measures(self):
        assert relation(threshold(GT, 100), threshold(GT, 90, end=T + timedelta(hours=2))) is None

    def test_settled_measures_on_the_same_day_compare_with_a_flag(self):
        a = threshold(GT, 50, measure="rt_score", stability=Stability.SETTLED)
        b = threshold(GTE, 35, measure="rt_score", stability=Stability.SETTLED, end=T + timedelta(hours=10))
        assert relation(a, b) == ("1=>2", (FLAG_SNAPSHOT_TIME,))

    def test_different_measures_or_scopes_are_unrelated(self):
        assert relation(threshold(GT, 1, measure="price"), threshold(GT, 1, measure="volume")) is None
        assert relation(goals(GT, 1.5), goals(GT, 1.5, ).__class__(**{**goals(GT, 1.5).__dict__, "scope": "first_half"})) is None

    def test_a_missing_scope_does_not_block(self):
        a = goals(GT, 2.5)
        b = Proposition(**{**goals(GT, 1.5).__dict__, "scope": None})
        assert relation(a, b)[0] == "1=>2"

    def test_unknown_fields_block_the_rule(self):
        unknown = threshold(GT, 1, unknown=frozenset({"value"}))
        assert relation(unknown, threshold(GT, 0)) is None

    def test_an_unknown_subject_never_relates(self):
        a = threshold(GT, 1, unknown=frozenset({"subject"}))
        assert relation(a, threshold(GT, 0)) is None


class TestDeadlines:
    def test_earlier_deadline_implies_later(self):
        assert relation(by(T), by(T + timedelta(days=21)))[0] == "1=>2"

    def test_not_by_the_later_deadline_implies_not_by_the_earlier(self):
        assert relation(by(T + timedelta(days=21), negated=True), by(T, negated=True))[0] == "1=>2"

    def test_before_vs_by_a_day_apart_stay_distinct(self):
        # Kalshi "before Aug 9" (Aug 9 04:00Z) vs Polymarket "by August 9" (Aug 10 03:59Z).
        kalshi = by(datetime(2026, 8, 9, 4, tzinfo=timezone.utc))
        polymarket = by(datetime(2026, 8, 10, 3, 59, tzinfo=timezone.utc))
        assert relation(kalshi, polymarket)[0] == "1=>2"

    def test_the_same_boundary_in_two_time_zones_is_equivalent_with_a_flag(self):
        a = by(datetime(2027, 1, 1, 5, tzinfo=timezone.utc), resolution_source="x")
        b = by(datetime(2027, 1, 1, tzinfo=timezone.utc), resolution_source="x")
        assert relation(a, b, "K", "PMI") == (EQUIVALENT, (FLAG_DEADLINE_ZONE,))

    def test_one_venues_deadlines_hours_apart_are_a_ladder(self):
        # Kalshi "before Nov 4" (05:00Z) and "before 4AM Nov 4" (09:00Z): the AP called it in between.
        before_midnight = by(datetime(2026, 11, 4, 5, tzinfo=timezone.utc))
        before_4am = by(datetime(2026, 11, 4, 9, tzinfo=timezone.utc))
        assert relation(before_4am, before_midnight, "K", "K")[0] == "2=>1"


    def test_different_events_on_one_subject_are_unrelated(self):
        assert relation(by(T, measure="make_playoffs"), by(T + timedelta(days=30), measure="reach_nfc_championship")) is None

    def test_a_deadline_question_without_its_event_is_not_used(self):
        assert relation(by(T, measure=None), by(T + timedelta(days=1), measure=None)) is None


class TestPeriods:
    def test_a_year_and_a_quarter_are_different_readings(self):
        year = threshold(GT, 2.2, measure="real_gdp_growth", stability=Stability.SETTLED,
                         window_start=datetime(2026, 1, 1, tzinfo=timezone.utc))
        quarter = threshold(GT, 3.5, measure="real_gdp_growth", stability=Stability.SETTLED,
                            window_start=datetime(2026, 10, 1, tzinfo=timezone.utc))
        assert relation(year, quarter) is None


class TestWinners:
    def test_same_entity_same_scope(self):
        assert relation(winner("clem"), winner("clem"))[0] == EQUIVALENT

    def test_map_one_and_the_match_are_unrelated(self):
        assert relation(winner("fal", scope="map:1"), winner("fal", scope="match")) is None

    def test_one_win_implies_the_other_entity_loses(self):
        assert relation(winner("unc"), winner("clem", negated=True), event_a=1, event_b=1)[0] == "1=>2"

    def test_entities_no_event_lists_together_are_not_told_apart(self):
        # Two venues' keys for one team: excluding one from the other would say the team can't beat itself.
        assert relation(winner("ggo", scope="regulation"), winner("goias", scope="regulation", negated=True),
                        "PMI", "PMUS", event_a=1, event_b=2) is None

    @pytest.mark.parametrize("measure", ["place:2", "nomination", "first_round", "race_to_35"])
    def test_winning_something_else_is_unrelated_to_winning(self, measure):
        assert relation(winner("psoe", scope=None), winner("psoe", scope=None, measure=measure)) is None

    def test_places_are_exclusive_but_a_top_three_is_not(self):
        assert relation(winner("pp", measure="place:2"), winner("vox", measure="place:2", negated=True),
                        event_a=1, event_b=1)[0] == "1=>2"
        assert relation(winner("pp", measure="top:3"), winner("vox", measure="top:3", negated=True),
                        event_a=1, event_b=1) is None

    def test_two_different_winners_are_not_stored(self):
        assert relation(winner("unc"), winner("clem")) is None

    def test_a_winning_margin_says_nothing_about_a_side_contest(self):
        margin = threshold(GT, 10.5, measure="margin:sea", unit="points", observation=Observation.CUMULATIVE, end=None)
        assert relation(margin, winner("sea", measure="race_to_35")) is None

    def test_a_winning_margin_implies_the_win(self):
        margin = threshold(GTE, 1, measure="margin:republican", unit="percentage_points", observation=Observation.AT_TIME,
                           end=None, scope="popular_vote")
        assert relation(margin, winner("republican", scope="popular_vote"))[0] == "1=>2"

    def test_a_team_total_implies_the_game_total(self):
        team = threshold(GT, 1.5, measure="team_goals:columbus", unit="goals", observation=Observation.CUMULATIVE, end=None)
        total = threshold(GT, 1.5, measure="total_goals", unit="goals", observation=Observation.CUMULATIVE, end=None)
        assert relation(team, total)[0] == "1=>2"


class TestFlags:
    def test_cross_venue_pairs_carry_source_flags(self):
        assert relation(threshold(GT, 1, resolution_source="a"), threshold(GT, 1, resolution_source="b"), "K", "PMI") == \
            (EQUIVALENT, (FLAG_SOURCE_DIFFERS,))
        assert relation(threshold(GT, 1, resolution_source="a"), threshold(GT, 1), "K", "PMI") == (EQUIVALENT, (FLAG_SOURCE_UNKNOWN,))

    def test_same_venue_pairs_carry_none(self):
        assert relation(threshold(GT, 2), threshold(GT, 1), "K", "K") == ("1=>2", ())

    def test_same_venue_pairs_with_two_named_sources_that_differ_carry_one(self):
        mark = threshold(GT, 2, resolution_source="hyperliquid-xyz-cl-mark-twap-60s")
        candle = threshold(GT, 1, resolution_source="hyperliquid-xyz-cl-1m-candle-twap-300s")
        assert relation(mark, candle, "HL", "HL") == ("1=>2", (FLAG_SOURCE_DIFFERS,))
        assert relation(mark, threshold(GT, 1), "HL", "HL") == ("1=>2", ())


def prime_minister(end, **kw):
    return Proposition(subject="israel-pm-after-2026-election", kind=Kind.WINNER, measure="holder:prime-minister",
                       entity="gadi-eizenkot", window_end=end, **kw)


class TestContingencyFlags:
    KALSHI_DEADLINE = datetime(2027, 10, 27, 4, tzinfo=timezone.utc)
    POLYMARKET_DEADLINE = datetime(2028, 1, 1, 5, tzinfo=timezone.utc)

    def test_one_off_contests_whose_venues_give_up_at_different_deadlines(self):
        flags = relation(prime_minister(self.KALSHI_DEADLINE), prime_minister(self.POLYMARKET_DEADLINE), "K", "PMI")
        assert flags == (EQUIVALENT, (FLAG_DEADLINE_DIFFERS, FLAG_SOURCE_UNKNOWN))

    def test_one_deadline_unknown_differs_too(self):
        assert FLAG_DEADLINE_DIFFERS in relation(prime_minister(self.KALSHI_DEADLINE), prime_minister(None), "K", "PMU")[1]

    def test_the_same_deadline_or_none_on_both_is_not_flagged(self):
        assert FLAG_DEADLINE_DIFFERS not in relation(prime_minister(self.KALSHI_DEADLINE),
                                                     prime_minister(self.KALSHI_DEADLINE + timedelta(hours=5)), "K", "PMI")[1]
        assert FLAG_DEADLINE_DIFFERS not in relation(prime_minister(None), prime_minister(None), "K", "PMI")[1]

    def test_games_on_different_venues_carry_void_rules_not_deadlines(self):
        tie = lambda end: dataclasses.replace(winner("tie", scope="regulation"), subject="game:chi1:2026-11-05:uco-ohf",
                                              window_end=end)
        assert relation(tie(None), tie(datetime(2026, 11, 5, tzinfo=timezone.utc)), "K", "PMI")[1] == \
            (FLAG_SOURCE_UNKNOWN, FLAG_VOID_RULES)
        assert relation(tie(None), tie(None), "PMI", "PMI")[1] == ()


def test_one_relationship_per_pair_with_evidence():
    found = derive([Contract(5, "K", threshold(GT, 1)), Contract(3, "K", threshold(GT, 1))])
    assert [(d.security_id_a, d.security_id_b, d.relationship_type) for d in found] == [(3, 5, EQUIVALENT)]
    assert found[0].evidence["rule"] == "value_sets" and found[0].evidence["version"] == DERIVATION_VERSION


def test_a_ladder_derives_every_implication():
    ladder = [Contract(i, "K", threshold(GT, v)) for i, v in enumerate((100, 200, 300), 1)]
    found = {(d.security_id_a, d.security_id_b) for d in derive(ladder)}
    assert found == {(2, 1), (3, 1), (3, 2)}
    assert all(d.relationship_type == IMPLIES for d in derive(ladder))


class TestReviewedMistakes:
    def test_touching_a_high_and_dipping_to_a_low_are_unrelated(self):
        touch_high = threshold(GTE, 120000, observation=Observation.ANY_TIME)
        never_dips = threshold(LTE, 100000, observation=Observation.ANY_TIME, negated=True)
        assert relation(touch_high, never_dips) is None
        assert relation(touch_high, threshold(GTE, 110000, observation=Observation.ANY_TIME))[0] == "1=>2"
        assert relation(threshold(LTE, 90000, observation=Observation.ANY_TIME),
                        threshold(LTE, 100000, observation=Observation.ANY_TIME))[0] == "1=>2"

    def test_a_deadline_question_without_its_deadline_is_skipped(self):
        assert relation(by(None), by(T)) is None
        assert relation(threshold(GT, 1, observation=None), threshold(GT, 0, observation=None)) is None

    def test_an_index_quoted_in_points_is_not_a_whole_number(self):
        a = threshold(GT, 5024.5, measure="close", unit="points")
        b = threshold(GTE, 5024.75, measure="close", unit="points")
        assert relation(a, b)[0] == "2=>1"

    def test_words_inside_other_words_are_not_counts(self):
        a = threshold(GT, 1.5, measure="assets_under_management", observation=Observation.CUMULATIVE, end=None)
        b = threshold(GTE, 2, measure="assets_under_management", observation=Observation.CUMULATIVE, end=None)
        assert relation(a, b)[0] == "2=>1"

    def test_a_tennis_games_margin_says_nothing_about_the_winner(self):
        games = threshold(GT, 3.5, measure="margin:sinner", unit="games", observation=Observation.CUMULATIVE, end=None)
        assert relation(games, winner("sinner")) is None
        maps = threshold(GT, 1.5, measure="margin:navi", unit="maps", observation=Observation.CUMULATIVE, end=None)
        assert relation(maps, winner("navi"))[0] == "1=>2"

    def test_a_popular_vote_margin_says_nothing_about_a_winner_that_names_no_vote(self):
        margin = threshold(GT, 0, measure="margin:dem", unit="percentage_points", observation=Observation.AT_TIME,
                           end=None, scope="popular_vote")
        assert relation(margin, winner("dem", scope=None)) is None

    def test_a_vote_margin_with_no_vote_named_on_either_side_implies_the_winner_flagged(self):
        margin = threshold(GT, 0, measure="margin:dem", unit="percentage_points", observation=Observation.AT_TIME, end=None)
        assert relation(margin, winner("dem", scope=None)) == ("1=>2", (FLAG_VOTE_SCOPE,))

    def test_a_reading_whose_date_is_unknown_matches_nothing(self):
        # A fiscal year's customer count and one quarter's, both read without their dates.
        fiscal_year = threshold(GT, 910, measure="customers", end=None, unknown=frozenset({"window_end"}))
        quarter = threshold(GT, 825, measure="customers", end=None, unknown=frozenset({"window_end"}))
        assert relation(fiscal_year, quarter) is None

    def test_values_in_different_unit_sizes_compare_in_one_size(self):
        billions = threshold(GTE, 600, measure="market_cap", unit="usd-billions")
        trillions = threshold(GTE, 1.8, measure="market_cap", unit="T")
        assert relation(billions, trillions)[0] == "2=>1"
        assert relation(threshold(GTE, 600, measure="market_cap"), trillions) is None

    def test_deadlines_compare_only_from_the_same_start(self):
        from_listing = by(T)
        from_july = by(T + timedelta(days=180), window_start=T + timedelta(days=1))
        assert relation(from_listing, from_july) is None
        assert relation(by(T, window_start=T - timedelta(days=30)),
                        by(T + timedelta(days=180), window_start=T - timedelta(days=30)))[0] == "1=>2"


class TestUnits:
    def tennis(self, unit, value):
        return threshold(GTE, value, measure="margin:djokovic", unit=unit, observation=Observation.CUMULATIVE, end=None,
                         scope="match")

    def test_a_margin_in_games_and_one_in_sets_are_unrelated(self):
        # Djokovic won 3 more games than Vacherot and lost a set: 2.5+ games, under 2 sets.
        assert relation(self.tennis("games", 3), self.tennis("sets", 2)) is None

    def test_an_esports_series_in_games_is_one_in_maps(self):
        assert relation(self.tennis("games", 2), self.tennis("maps", 2))[0] == EQUIVALENT

    def test_other_units_spelled_differently_still_compare(self):
        a = threshold(GT, 3, measure="cpi", unit="percent")
        b = threshold(GT, 2, measure="cpi", unit="percentage points")
        assert relation(a, b)[0] == "1=>2"

    def ladder(self, start, hours, strikes):
        out = []
        sid = start
        for h in range(hours):
            end = T + timedelta(hours=h)
            for k in range(strikes):
                for negated in (False, True):
                    out.append(Contract(sid, "K", threshold(GT, 100000 + 500 * k, end=end, window_start=None,
                                                            negated=negated)))
                    sid += 1
        return out

    def test_snapshots_at_different_times_are_derived_apart(self):
        units = derivation_units(self.ladder(1, hours=3, strikes=4))
        assert len(units) == 3
        assert all(len(u.contracts) == 8 for u in units)

    def test_a_margin_joins_its_spread_ladder_and_the_winners_into_one_unit(self):
        contracts = [
            Contract(1, "K", threshold(GT, 1.5, measure="margin:navi", unit="maps", observation=Observation.CUMULATIVE, end=None)),
            Contract(2, "K", threshold(GT, 0.5, measure="margin:navi", unit="maps", observation=Observation.CUMULATIVE, end=None)),
            Contract(3, "K", winner("navi")),
            Contract(4, "K", by(T)),
        ]
        units = derivation_units(contracts)
        assert sorted(sorted(c.security_id for c in u.contracts) for u in units) == [[1, 2, 3], [4]]
        assert {(d.security_id_a, d.security_id_b) for d in derive(contracts)} == {(1, 2), (1, 3), (2, 3)}

    def test_a_week_of_hourly_ladders_derives_quickly_and_only_within_each_hour(self):
        contracts = self.ladder(1, hours=24 * 7, strikes=40)
        end_of = {c.security_id: c.proposition.window_end for c in contracts}
        started = time.monotonic()
        found = derive(contracts)
        assert time.monotonic() - started < 20
        assert found and all(end_of[d.security_id_a] == end_of[d.security_id_b] for d in found)


class TestMinimalRows:
    def two_venue_ladder(self, strikes=10):
        """Yes/No strike ladders on two venues settled by different sources: ids 1.. Kalshi Yes, then its No sides,
        then the same for Polymarket from 1001."""
        contracts = []
        for venue, base, source in (("K", 1, "cf-benchmarks-brti"), ("PMU", 1001, "binance")):
            for k in range(strikes):
                yes, no = base + k, base + strikes + k
                for sid, comp, negated in ((yes, no, False), (no, yes, True)):
                    contracts.append(Contract(sid, venue, threshold(GT, 100000 + 500 * k, negated=negated,
                                                                    resolution_source=source), comp))
        return contracts

    def test_a_flagged_two_venue_ladder_stores_steps_and_one_row_per_strike(self):
        contracts = self.two_venue_ladder()
        (unit,) = derivation_units(contracts)
        derived = derive_unit(unit)
        rows, reduced = minimal_rows(unit, derived)
        assert reduced
        assert len(derived) > 300
        assert len(rows) == 9 + 9 + 10
        assert all(r.flags == (FLAG_SOURCE_DIFFERS,) for r in rows if r.relationship_type == EQUIVALENT)
        complements = {c.security_id: c.complement for c in contracts}
        stored = close([Edge(r.security_id_a, r.security_id_b, r.relationship_type, r.flags) for r in rows], complements)
        assert stored == close([Edge(d.security_id_a, d.security_id_b, d.relationship_type, d.flags) for d in derived],
                               complements)

    def test_kept_rows_carry_their_derived_evidence(self):
        (unit,) = derivation_units(self.two_venue_ladder(3))
        rows, _ = minimal_rows(unit, derive_unit(unit))
        assert all(r.evidence["rule"] in ("value_sets", "composed") for r in rows)
        assert any(r.evidence["rule"] == "value_sets" for r in rows)


class TestEntities:
    def test_deadlines_for_different_candidates_are_unrelated(self):
        mottley = by(T, measure="confirm_secretary_general", entity="mia-mottley")
        grossi = by(T, measure="confirm_secretary_general", entity="rafael-grossi")
        assert relation(mottley, grossi) is None
        assert relation(mottley, by(T + timedelta(days=30), measure="confirm_secretary_general",
                                    entity="mia-mottley"))[0] == "1=>2"

    def test_the_same_measure_of_different_entities_is_unrelated(self):
        a = threshold(GT, 40, measure="vote_share", entity="dem", observation=Observation.AT_TIME)
        b = threshold(GT, 40, measure="vote_share", entity="rep", observation=Observation.AT_TIME)
        assert relation(a, b) is None
        assert relation(a, dataclasses.replace(a))[0] == EQUIVALENT

    def test_an_entity_repeated_on_one_side_matches_a_measure_that_names_it(self):
        a = threshold(GT, 6.5, measure="wins:texas-st", entity="texas-st", observation=Observation.CUMULATIVE, end=None)
        b = threshold(GT, 6.5, measure="wins:texas-st", observation=Observation.CUMULATIVE, end=None)
        assert relation(a, b)[0] == EQUIVALENT

    def test_one_candidate_is_narrower_than_anyone(self):
        anyone = by(T, measure="confirm_secretary_general")
        assert relation(by(T, measure="confirm_secretary_general", entity="mia-mottley"), anyone) is None


class TestGameTotals:
    def test_a_total_without_a_window_end_compares_with_one_that_has_it(self):
        kalshi = threshold(GT, 0.5, measure="total_goals", unit="goals", observation=Observation.CUMULATIVE, end=None,
                           scope="regulation")
        pm = threshold(GTE, 1, measure="total_goals", unit="goals", observation=Observation.CUMULATIVE,
                       end=T, scope="regulation")
        assert relation(kalshi, pm) == (EQUIVALENT, ())
        assert len(derivation_units([Contract(1, "K", kalshi), Contract(2, "PMI", pm)])) == 1


class TestWinnerTiming:
    def leader(self, entity, observation, **kw):
        return winner(entity, scope=None, measure="holder:leader", observation=observation, window_end=T, **kw)

    def test_holding_at_the_end_implies_holding_at_some_point(self):
        at_end = self.leader("reza-pahlavi", Observation.AT_TIME)
        any_time = self.leader("reza-pahlavi", Observation.ANY_TIME)
        assert relation(at_end, any_time)[0] == "1=>2"
        assert relation(any_time.negate(), at_end.negate())[0] == "1=>2"
        assert relation(at_end, self.leader("reza-pahlavi", Observation.AT_TIME))[0] == EQUIVALENT

    def test_different_people_can_each_hold_it_at_some_point(self):
        a = self.leader("reza-pahlavi", Observation.ANY_TIME)
        b = self.leader("mojtaba-khamenei", Observation.ANY_TIME, negated=True)
        assert relation(a, b) is None
        assert relation(self.leader("reza-pahlavi", Observation.AT_TIME),
                        self.leader("mojtaba-khamenei", Observation.AT_TIME, negated=True),
                        event_a=1, event_b=1)[0] == "1=>2"

    def test_a_de_facto_leader_is_not_the_leader(self):
        official = self.leader("nicolas-maduro", Observation.AT_TIME)
        de_facto = dataclasses.replace(official, measure="holder:de-facto-leader")
        assert relation(official, de_facto) is None

    def test_a_contest_decided_once_still_matches_without_timing(self):
        assert relation(winner("navi"), winner("navi"))[0] == EQUIVALENT
        assert relation(winner("navi"), self.leader("navi", Observation.AT_TIME)) is None


class TestOneEventsOutcomes:
    """Two candidates on one ballot given one name (the Alaska Senate race lists both "Dan Sullivan" and "Daniel J.
    Sullivan Jr.") must not become one contract."""

    def contracts(self):
        return [
            Contract(1, "PMI", winner("dan-sullivan"), event_id=10),
            Contract(2, "PMI", winner("dan-sullivan"), event_id=10),
            Contract(3, "PMI", winner("mary-peltola"), event_id=10),
            Contract(4, "K", winner("dan-sullivan"), event_id=20),
            Contract(7, "K", winner("mary-peltola"), complement=8, event_id=30),
            Contract(8, "K", winner("mary-peltola", negated=True), complement=7, event_id=30),
        ]

    def pairs(self, found):
        return {(d.security_id_a, d.security_id_b, d.relationship_type) for d in found}

    def test_colliding_outcomes_get_no_relationships_and_the_rest_are_kept(self, caplog):
        found = self.pairs(derive(self.contracts()))
        assert not any({1, 2} & {a, b} for a, b, _ in found)
        assert (3, 7, EQUIVALENT) in found
        assert (4, 8, IMPLIES) in found
        assert "proposition collision: event=10 securities=[1, 2] subject=s" in caplog.text

    def test_a_binary_events_two_sides_are_not_a_collision(self, caplog):
        found = self.pairs(derive(self.contracts()[2:]))
        assert (7, 8, EQUIVALENT) not in found
        assert (3, 7, EQUIVALENT) in found
        assert "proposition collision" not in caplog.text

    def test_a_binary_side_read_as_its_other_side_is_a_collision(self, caplog):
        # A No side read as its Yes side.
        found = derive([Contract(1, "HL", by(T), complement=2, event_id=5),
                        Contract(2, "HL", by(T), complement=1, event_id=5),
                        Contract(3, "K", by(T), event_id=6)])
        assert not any({1, 2} & {d.security_id_a, d.security_id_b} for d in found)
        assert "proposition collision: event=5 securities=[1, 2]" in caplog.text

    def test_the_same_reading_in_different_events_still_equates(self):
        found = self.pairs(derive([Contract(1, "PMI", winner("dan-sullivan"), event_id=10),
                                   Contract(4, "K", winner("dan-sullivan"), event_id=20)]))
        assert found == {(1, 4, EQUIVALENT)}


def score(aek, rma, scope="regulation", unit="goals", **kw):
    return Proposition(subject="s", kind=Kind.SCORE, scores=(("aek", aek), ("rma", rma)), unit=unit, scope=scope, **kw)


def goals_market(comparator, value, measure="total_goals", **kw):
    return threshold(comparator, value, measure=measure, unit="goals", observation=Observation.CUMULATIVE, end=None,
                     scope="regulation", **kw)


class TestScores:
    def test_a_score_implies_its_winner_and_rules_out_the_others(self):
        assert relation(score(0, 1), winner("rma", scope="regulation"))[0] == "1=>2"
        assert relation(score(0, 1), winner("aek", scope="regulation", negated=True))[0] == "1=>2"
        assert relation(score(0, 1), winner("tie", scope="regulation", negated=True))[0] == "1=>2"
        assert relation(score(1, 1), winner("tie", scope="regulation"))[0] == "1=>2"
        assert relation(score(0, 1), winner("aek", scope="regulation")) is None

    def test_a_score_decides_totals_team_totals_and_margins(self):
        assert relation(score(2, 1), goals_market(GT, 2.5))[0] == "1=>2"
        assert relation(score(2, 1), goals_market(GT, 3.5, negated=True))[0] == "1=>2"
        assert relation(score(2, 1), goals_market(GTE, 2, measure="team_goals:aek"))[0] == "1=>2"
        assert relation(score(2, 1), goals_market(GT, 0.5, measure="margin:aek"))[0] == "1=>2"
        assert relation(score(2, 1), goals_market(GT, 1.5, measure="margin:aek", negated=True))[0] == "1=>2"

    def test_the_same_score_on_two_venues_is_equivalent(self):
        assert relation(score(0, 1), score(0, 1), "K", "PMI")[0] == EQUIVALENT
        assert relation(score(0, 1), score(1, 0), "K", "PMI") is None

    def test_a_canceled_game_settles_polymarket_intls_winners_and_spreads_differently(self):
        # The moneyline and exact scores settle as the Draw (0-0), the spreads and totals 50-50.
        game = lambda p: dataclasses.replace(p, subject="game:epl:2026-11-14:aek-rma")
        draw, two_one = game(winner("tie", scope="regulation")), game(score(2, 1))
        total, margin_line = game(goals_market(GT, 2.5)), game(goals_market(GT, 0.5, measure="margin:aek"))
        assert relation(two_one, total, "POLYMARKET_INTL", "POLYMARKET_INTL") == ("1=>2", (FLAG_VOID_RULES,))
        assert relation(two_one, margin_line, "POLYMARKET_INTL", "POLYMARKET_INTL")[1] == (FLAG_VOID_RULES,)
        assert relation(game(score(1, 1)), draw, "POLYMARKET_INTL", "POLYMARKET_INTL") == ("1=>2", ())
        assert relation(two_one, total, "KALSHI", "KALSHI") == ("1=>2", ())

    def test_a_score_says_nothing_in_another_unit_scope_or_on_its_no_side(self):
        assert relation(score(2, 1, scope="first_half"), goals_market(GT, 2.5)) is None
        assert relation(score(2, 1, unit="points"), goals_market(GT, 2.5)) is None
        assert relation(score(2, 1, negated=True), winner("aek", scope="regulation")) is None


class TestMarginRanges:
    def test_a_margin_range_above_zero_implies_its_side_wins(self):
        margin = Proposition(subject="s", kind=Kind.RANGE, measure="margin:republican", low=9, high=12,
                             unit="percent", observation=Observation.AT_TIME)
        assert relation(margin, winner("republican", scope=None)) == ("1=>2", (FLAG_VOTE_SCOPE,))

    def test_unit_spellings_are_the_same_unit(self):
        margin = Proposition(subject="s", kind=Kind.RANGE, measure="margin:republican", low=9, high=12,
                             unit="percentage points", observation=Observation.CUMULATIVE)
        assert relation(margin, winner("republican", scope=None)) == ("1=>2", (FLAG_VOTE_SCOPE,))

    def test_a_range_that_includes_zero_does_not(self):
        margin = Proposition(subject="s", kind=Kind.RANGE, measure="margin:republican", low=0, high=3,
                             unit="percent", observation=Observation.AT_TIME)
        assert relation(margin, winner("republican", scope=None)) is None


def margin(entity, comparator, value, observation=Observation.AT_TIME, end=T, unit="percentage points", **kw):
    return Proposition(subject="nevada-governor-2026", kind=Kind.THRESHOLD, measure=f"margin:{entity}",
                       comparator=comparator, value=value, observation=observation, window_end=end, unit=unit, **kw)


class TestContestResults:
    def test_at_the_time_and_cumulative_are_one_reading(self):
        found = relation(margin("democratic", GTE, 3), margin("democratic", GTE, 1, observation=Observation.CUMULATIVE))
        assert found == ("1=>2", ())

    def test_a_later_deadline_compares_flagged(self):
        five = Proposition(subject="michigan-us-house-2026", kind=Kind.RANGE, measure="seats:democratic", low=5, high=5,
                           unit="seats", observation=Observation.AT_TIME, window_end=T)
        below = Proposition(subject="michigan-us-house-2026", kind=Kind.THRESHOLD, measure="seats:democratic",
                            comparator=LT, value=5, unit="seats", observation=Observation.AT_TIME,
                            window_end=T + timedelta(days=60), negated=True)
        assert relation(five, below) == ("1=>2", (FLAG_DEADLINE_DIFFERS,))

    def test_senate_seats_held_and_won_are_not_taken_as_one(self):
        def seats(value, end):
            return Proposition(subject="us-senate-2026-seats", kind=Kind.THRESHOLD, measure="seats:republican",
                               comparator=GTE, value=value, unit="seats", observation=Observation.AT_TIME, window_end=end)
        assert relation(seats(53, T), seats(50, T + timedelta(days=60))) is None

    def test_a_named_round_does_not_meet_a_margin_with_none(self):
        final_round = Proposition(subject="maine-senate-2026", kind=Kind.RANGE, measure="margin:democratic", low=3,
                                  high=6, unit="percent", observation=Observation.CUMULATIVE, scope="final_round")
        assert relation(final_round, margin("democratic", GTE, 1, unit="percent")) is None

    def test_a_price_is_still_read_at_its_own_time(self):
        assert relation(threshold(GT, 100), threshold(GT, 90, end=T + timedelta(days=2))) is None


class TestNamedRounds:
    def test_topping_a_first_round_is_not_winning(self):
        assert relation(winner("kicillof", scope="first_round"), winner("kicillof", scope=None)) is None

    def test_one_round_on_two_venues_is_one_reading(self):
        assert relation(winner("kicillof", scope="first_round"), winner("kicillof", scope="first_round")) is not None

    def test_old_spellings_of_a_round_compare(self):
        assert relation(winner("hall", scope="round_1"), winner("hall", scope="first_round")) is not None

    def test_a_first_round_margin_does_not_imply_the_winner(self):
        first_round = margin("mary-peltola", GTE, 3, observation=Observation.CUMULATIVE, scope="first_round")
        assert relation(first_round, winner("mary-peltola", scope=None)) is None

    def test_a_margin_naming_no_round_still_implies_the_winner_flagged(self):
        found = relation(margin("mary-peltola", GTE, 3, observation=Observation.CUMULATIVE), winner("mary-peltola", scope=None))
        assert found == ("1=>2", (FLAG_VOTE_SCOPE,))


class TestCanonicalKeys:
    def test_measure_spellings_compare(self):
        a = threshold(GT, 8.5, measure="rushing_tds:christian-mccaffrey", observation=Observation.CUMULATIVE, end=None)
        b = threshold(GT, 9.5, measure="rushing_touchdowns:christian-mccaffrey", observation=Observation.CUMULATIVE,
                      end=None)
        assert relation(b, a)[0] == "1=>2"

    def test_a_truncated_last_word_is_the_same_entity(self):
        a = threshold(GT, 5.5, measure="wins:new-york-jets", observation=Observation.CUMULATIVE, end=None)
        b = threshold(GT, 6.5, measure="wins:new-york-j", observation=Observation.CUMULATIVE, end=None)
        assert relation(b, a)[0] == "1=>2"

    def test_a_key_is_never_extended_by_a_word(self):
        a = threshold(GT, 8.5, measure="wins:iowa-state", observation=Observation.CUMULATIVE, end=None)
        b = threshold(GT, 9.5, measure="wins:iowa", observation=Observation.CUMULATIVE, end=None)
        assert relation(b, a) is None

    def test_a_team_code_in_a_game(self):
        hull = Proposition(subject="game:epl:2026-11-07:arshul", kind=Kind.THRESHOLD, measure="margin:hul",
                           comparator=GTE, value=4, unit="goals", observation=Observation.CUMULATIVE, scope="regulation")
        arsenal = Proposition(subject="game:epl:2026-11-07:arshul", kind=Kind.WINNER, measure="winner",
                              entity="hull-city", scope="regulation")
        assert relation(hull, arsenal)[0] == "1=>2"

    def test_keys_an_event_lists_apart_stay_apart(self):
        found = derive([Contract(1, "K", game_winner("man"), event_id=10),
                        Contract(2, "K", game_winner("manchester-united"), event_id=10),
                        Contract(3, "P", game_winner("manchester-united"), event_id=20)])
        assert not [d for d in found if {d.security_id_a, d.security_id_b} == {1, 3}]

    def test_a_maps_winner_spelled_per_map_is_its_winner(self):
        kalshi = winner("saw-youngsters", scope="map:2", measure="map_2_winner")
        polymarket = winner("saw-youngsters", scope="map:2", measure="winner")
        assert relation(kalshi, polymarket)[0] == EQUIVALENT

    def test_a_map_winner_spelling_naming_another_map_is_left_alone(self):
        assert relation(winner("x", scope="map:1", measure="map_2_winner"), winner("x", scope="map:1")) is None

    def test_an_ambiguous_code_is_left_alone(self):
        found = derive([Contract(1, "K", game_winner("man"), event_id=1),
                        Contract(2, "P", game_winner("manchester-city"), event_id=2),
                        Contract(3, "P", game_winner("manchester-united"), event_id=3)])
        assert not [d for d in found if 1 in (d.security_id_a, d.security_id_b)]


def game_winner(entity):
    return Proposition(subject="game:epl:2026-11-07:x", kind=Kind.WINNER, measure="winner", entity=entity,
                       scope="regulation")


GAME = "game:cod:2026-11-06:faze-m80"
KICKOFF = datetime(2026, 11, 6, 21, tzinfo=timezone.utc)


def team_wins(entity, subject=GAME, scope="match", **kw):
    return Proposition(subject=subject, kind=Kind.WINNER, measure="winner", entity=entity, scope=scope, **kw)


def listed(sid, venue, p, game, start, event=None, label=None):
    return Contract(sid, venue, p, event_id=event if event is not None else sid, game=game, start=start, label=label)


def pairs(found):
    return {frozenset((d.security_id_a, d.security_id_b)) for d in found}


class TestFixtures:
    def test_two_matches_of_one_day_never_compare(self):
        early, late = KICKOFF - timedelta(hours=3, minutes=30), KICKOFF + timedelta(minutes=40)
        found = derive([
            listed(1, "POLYMARKET_INTL", team_wins("faze"), "codmw-faz-m80-2026-11-06", early),
            listed(2, "POLYMARKET_INTL", team_wins("faze"), "codmw-faz-m80-2026-11-06", late),
            listed(3, "KALSHI", team_wins("faze"), "26NOV061600FAZEM80", KICKOFF),
        ])
        assert pairs(found) == {frozenset((2, 3))}

    def test_venues_far_apart_on_one_match_still_compare(self):
        tennis = "game:atp:2026-11-07:canfel"
        found = derive([
            listed(1, "KALSHI", team_wins("caniato", tennis), "26NOV07CANFEL", KICKOFF),
            listed(2, "POLYMARKET_INTL", team_wins("caniato", tennis), "atp-caniato-feldbau-2026-11-07",
                   KICKOFF + timedelta(hours=1, minutes=50)),
        ])
        assert pairs(found) == {frozenset((1, 2))}

    def test_one_venues_events_starting_together_are_one_game(self):
        found = derive([
            listed(1, "HYPERLIQUID", team_wins("faze"), None, KICKOFF, event=10),
            listed(2, "HYPERLIQUID", team_wins("faze"), None, KICKOFF + timedelta(minutes=10), event=11),
            listed(3, "KALSHI", team_wins("faze"), "26NOV061600FAZEM80", KICKOFF),
            listed(4, "KALSHI", team_wins("faze"), "26NOV062000FAZEM80", KICKOFF + timedelta(hours=4)),
        ])
        assert frozenset((1, 3)) in pairs(found) and frozenset((2, 3)) in pairs(found)
        assert not {p for p in pairs(found) if 4 in p}

    def test_a_game_between_two_matches_is_left_out(self):
        found = derive([
            listed(1, "KALSHI", team_wins("faze"), "A", KICKOFF),
            listed(2, "KALSHI", team_wins("faze"), "B", KICKOFF + timedelta(hours=4)),
            listed(3, "POLYMARKET_INTL", team_wins("faze"), "p", KICKOFF + timedelta(hours=2)),
        ])
        assert found == []

    def test_contracts_with_no_start_time_join_the_one_match(self):
        found = derive([
            listed(1, "KALSHI", team_wins("faze"), "26NOV061600FAZEM80", None),
            listed(2, "POLYMARKET_INTL", team_wins("faze"), "codmw-faz-m80-2026-11-06", KICKOFF),
        ])
        assert pairs(found) == {frozenset((1, 2))}

    def test_other_subjects_are_never_split(self):
        a = threshold(GT, 80000, measure="price")
        b = threshold(GT, 79000, measure="price")
        found = derive([Contract(1, "KALSHI", a, event_id=1, game="X", start=KICKOFF),
                        Contract(2, "KALSHI", b, event_id=2, game="Y", start=KICKOFF + timedelta(hours=9))])
        assert pairs(found) == {frozenset((1, 2))}


def game_total(comparator, value, scope="match", start=None, measure="total_points", unit="points", **kw):
    return Proposition(subject="game:ncaaf:2026-11-07:texmizz", kind=Kind.THRESHOLD, measure=measure,
                       comparator=comparator, value=value, unit=unit, observation=Observation.CUMULATIVE, scope=scope,
                       window_start=start, **kw)


class TestFullGameReadings:
    def test_a_window_start_on_a_game_total_is_not_a_period(self):
        kalshi = game_total(GT, 50.5)
        llm = game_total(GT, 48.5, start=datetime(2026, 11, 7, 5, tzinfo=timezone.utc))
        assert relation(kalshi, llm)[0] == "1=>2"

    def test_a_game_reading_naming_no_scope_is_the_whole_game(self):
        quarter = game_total(GT, 57.5, scope="second_quarter")
        assert relation(game_total(GT, 57.5, scope=None), game_total(GT, 50.5))[0] == "1=>2"
        assert relation(game_total(GT, 57.5, scope=None), quarter) is None

    def test_other_subjects_keep_unscoped_readings_unscoped(self):
        season = threshold(GT, 9.5, measure="wins:texas", observation=Observation.CUMULATIVE, end=None)
        first_round = threshold(GT, 8.5, measure="wins:texas", observation=Observation.CUMULATIVE, end=None,
                                scope="first_half")
        assert relation(season, first_round) is not None


TEX = "game:ncaaf:2026-11-07:texmizz"


def side_wins(entity, **kw):
    return Proposition(subject=TEX, kind=Kind.WINNER, measure="winner", entity=entity, scope="match", **kw)


def spread(team, comparator, value, **kw):
    return Proposition(subject=TEX, kind=Kind.THRESHOLD, measure=f"margin:{team}", comparator=comparator, value=value,
                       unit="points", observation=Observation.CUMULATIVE, scope="match", **kw)


class TestTeamNames:
    def test_keys_listed_under_one_name_are_one_team(self):
        found = derive([
            Contract(1, "KALSHI", side_wins("mizz"), event_id=1, label="Missouri"),
            Contract(2, "KALSHI", side_wins("tex"), event_id=1, label="Texas"),
            Contract(3, "POLYMARKET_INTL", side_wins("missouri"), event_id=2, label="Missouri"),
            Contract(4, "KALSHI", spread("mizz", GT, 3.5), event_id=3),
        ])
        assert frozenset((1, 3)) in pairs(found)
        assert frozenset((3, 4)) in pairs(found)

    def test_keys_an_event_lists_as_opponents_stay_apart(self):
        found = derive([
            Contract(1, "KALSHI", side_wins("texas-a"), event_id=1, label="Texas"),
            Contract(2, "KALSHI", side_wins("texas-b"), event_id=1, label="Texas"),
            Contract(3, "POLYMARKET_INTL", side_wins("texas-b"), event_id=2, label="Texas"),
        ])
        assert frozenset((1, 3)) not in pairs(found)

    def test_a_binary_sides_name_is_no_team(self):
        found = derive([
            Contract(1, "KALSHI", side_wins("mizz"), event_id=1, label="Yes"),
            Contract(2, "POLYMARKET_INTL", side_wins("missouri-tigers"), event_id=2, label="Yes"),
        ])
        assert found == []


class TestOppositeMargins:
    RIVALS = [Contract(10, "POLYMARKET_INTL", side_wins("texas"), event_id=9),
              Contract(11, "POLYMARKET_INTL", side_wins("missouri"), event_id=9)]

    def derived(self, a, b):
        found = derive([Contract(1, "KALSHI", a, event_id=1), Contract(2, "POLYMARKET_US", b, event_id=2), *self.RIVALS])
        return {(d.security_id_a, d.security_id_b, d.relationship_type, d.evidence["rule"]) for d in found
                if {d.security_id_a, d.security_id_b} <= {1, 2}}

    def test_one_team_by_more_is_the_other_by_less(self):
        # Texas by more than 7.5 is Missouri by less than -7.5, which is inside Missouri by less than 3.5.
        assert self.derived(spread("texas", GT, 7.5), spread("missouri", LT, 3.5)) == {(1, 2, IMPLIES, "opposite_margins")}

    def test_mirrored_lines_are_equivalent(self):
        assert self.derived(spread("texas", GT, 7.5), spread("missouri", LT, -7.5)) == {(1, 2, EQUIVALENT,
                                                                                         "opposite_margins")}

    def test_ranges_mirror(self):
        texas = dataclasses.replace(spread("texas", GT, 0), kind=Kind.RANGE, comparator=None, value=None, low=1, high=7,
                                    low_inclusive=True, high_inclusive=True)
        missouri = spread("missouri", LT, 0)
        assert self.derived(texas, missouri) == {(1, 2, IMPLIES, "opposite_margins")}

    def test_teams_no_event_lists_as_opponents_never_mirror(self):
        found = derive([Contract(1, "KALSHI", spread("texas", GT, 7.5)),
                        Contract(2, "POLYMARKET_US", spread("missouri", LT, 3.5))])
        assert found == []

    def test_a_margin_below_zero_implies_the_opponent_wins(self):
        found = derive([Contract(1, "KALSHI", spread("texas", LT, -3.5), event_id=1), *self.RIVALS])
        assert (1, 11, IMPLIES, "margin_implies_winner") in {
            (d.security_id_a, d.security_id_b, d.relationship_type, d.evidence["rule"]) for d in found}

    def test_a_draw_is_no_opponent(self):
        rivals = [Contract(10, "POLYMARKET_INTL", side_wins("texas"), event_id=9),
                  Contract(11, "POLYMARKET_INTL", side_wins("draw"), event_id=9)]
        found = derive([Contract(1, "KALSHI", spread("texas", LT, -3.5), event_id=1), *rivals])
        assert not [d for d in found if {d.security_id_a, d.security_id_b} == {1, 11}]


CS2 = "game:cs2:2026-11-04:lowchama-1400"


def map_wins(entity, scope="map:1", **kw):
    return Proposition(subject=CS2, kind=Kind.WINNER, measure="winner", entity=entity, scope=scope, **kw)


class TestTwoTeamContests:
    """Kalshi lists a CS2 map as one event with both teams' markets; Polymarket US as one market per team, "BORRACHEIROS
    wins map 1", whose No side is BORRACHEIROS not winning it."""

    def derived(self, *contracts):
        return {(d.security_id_a, d.security_id_b, d.relationship_type) for d in derive(list(contracts))}

    def kalshi_map(self, scope="map:1", size=2):
        return [Contract(1, "KALSHI", map_wins("low", scope), event_id=1, event_size=size),
                Contract(2, "KALSHI", map_wins("chama", scope), event_id=1, event_size=size)]

    def test_one_team_not_winning_is_the_other_winning(self):
        no_side = Contract(3, "POLYMARKET_US", map_wins("chama", negated=True), event_id=2, event_size=2)
        assert (1, 3, EQUIVALENT) in self.derived(*self.kalshi_map(), no_side)

    def test_not_where_an_outcome_of_the_event_is_missing(self):
        # A third outcome (a Draw market) the unit doesn't hold: the two listed teams needn't cover every result.
        no_side = Contract(3, "POLYMARKET_US", map_wins("chama", negated=True), event_id=2, event_size=2)
        assert (1, 3, EQUIVALENT) not in self.derived(*self.kalshi_map(size=3), no_side)

    def test_not_where_the_event_lists_a_draw(self):
        contracts = [Contract(1, "KALSHI", map_wins("low"), event_id=1, event_size=3),
                     Contract(2, "KALSHI", map_wins("chama"), event_id=1, event_size=3),
                     Contract(4, "KALSHI", map_wins("tie"), event_id=1, event_size=3),
                     Contract(3, "POLYMARKET_US", map_wins("chama", negated=True), event_id=2, event_size=2)]
        assert (1, 3, EQUIVALENT) not in self.derived(*contracts)

    def test_not_at_another_scope(self):
        no_side = Contract(3, "POLYMARKET_US", map_wins("chama", "map:2", negated=True), event_id=2, event_size=2)
        assert not {r for r in self.derived(*self.kalshi_map(), no_side) if 3 in r[:2]}

    def test_two_teams_not_winning_say_nothing_of_each_other(self):
        contracts = [*self.kalshi_map(),
                     Contract(3, "POLYMARKET_US", map_wins("chama", negated=True), event_id=2, event_size=2),
                     Contract(4, "POLYMARKET_US", map_wins("low", negated=True), event_id=3, event_size=2)]
        assert not {r for r in self.derived(*contracts) if {r[0], r[1]} == {3, 4}}

    def test_only_in_games(self):
        election = "election:2026-11-03:us-senate-tx"
        found = self.derived(
            Contract(1, "KALSHI", Proposition(subject=election, kind=Kind.WINNER, measure="winner", entity="allred"),
                     event_id=1, event_size=2),
            Contract(2, "KALSHI", Proposition(subject=election, kind=Kind.WINNER, measure="winner", entity="cruz"),
                     event_id=1, event_size=2),
            Contract(3, "POLYMARKET_US", Proposition(subject=election, kind=Kind.WINNER, measure="winner",
                                                     entity="cruz", negated=True), event_id=2, event_size=2))
        assert (1, 3, EQUIVALENT) not in found
