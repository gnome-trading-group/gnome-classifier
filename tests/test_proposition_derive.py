import dataclasses
import time
from datetime import datetime, timedelta, timezone

import pytest

from classifier.propositions.derive import (
    DERIVATION_VERSION,
    EQUIVALENT,
    FLAG_DEADLINE_ZONE,
    FLAG_SNAPSHOT_TIME,
    FLAG_SOURCE_DIFFERS,
    FLAG_SOURCE_UNKNOWN,
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


def winner(entity, scope="match", **kw):
    return Proposition(subject="s", kind=Kind.WINNER, entity=entity, scope=scope, **kw)


def relation(a: Proposition, b: Proposition, venue_a="K", venue_b="K"):
    found = derive([Contract(1, venue_a, a), Contract(2, venue_b, b)])
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
        assert relation(winner("unc"), winner("clem", negated=True))[0] == "1=>2"

    def test_two_different_winners_are_not_stored(self):
        assert relation(winner("unc"), winner("clem")) is None

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

    def test_a_vote_margin_needs_the_same_named_vote(self):
        margin = threshold(GT, 0, measure="margin:dem", unit="percentage_points", observation=Observation.AT_TIME, end=None)
        assert relation(margin, winner("dem", scope=None)) is None

    def test_deadlines_compare_only_from_the_same_start(self):
        from_listing = by(T)
        from_july = by(T + timedelta(days=180), window_start=T + timedelta(days=1))
        assert relation(from_listing, from_july) is None
        assert relation(by(T, window_start=T - timedelta(days=30)),
                        by(T + timedelta(days=180), window_start=T - timedelta(days=30)))[0] == "1=>2"


class TestUnits:
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
