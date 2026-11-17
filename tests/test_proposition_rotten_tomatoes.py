import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from classifier.propositions.derive import IMPLIES, Contract, derive
from classifier.propositions.types import Comparator, Observation, Proposition
from classifier.propositions.venue import correct_venue

# Stored Claude readings and their venue sources from the prod registry (2026-11-17). KXRT-VIO-60 was read with its
# window start unknown, VIO-55 with the start at the window's end, which kept the two rungs from comparing.
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "propositions" / "rotten_tomatoes.json").read_text())
VIO_SETTLES = datetime(2026, 12, 7, 15, tzinfo=timezone.utc)
RELEASE_MONDAY = datetime(2026, 12, 21, 15, tzinfo=timezone.utc)


def stored(name: str) -> tuple[dict, Proposition]:
    row = FIXTURE[name]
    return copy.deepcopy(row["source"]), Proposition.from_fields(row["subject"], row["fields"])


def corrected(name: str) -> Proposition:
    source, reading = stored(name)
    return correct_venue(source, reading)


def implies(a: Proposition, b: Proposition, venue_a="KALSHI", venue_b="KALSHI") -> bool:
    found = derive([Contract(1, venue_a, a), Contract(2, venue_b, b)])
    return any(d.relationship_type == IMPLIES and (d.security_id_a, d.security_id_b) == (1, 2) for d in found)


class TestKalshi:
    def test_a_rung_read_without_its_window_start_is_a_snapshot(self):
        fixed = corrected("kalshi_vio_60_yes")
        assert (fixed.measure, fixed.comparator, fixed.value, fixed.observation) == (
            "rt_score", Comparator.GT, 60, Observation.AT_TIME)
        assert (fixed.window_start, fixed.window_end, fixed.unknown) == (None, VIO_SETTLES, frozenset())
        assert fixed.subject == "violent-night-2"

    def test_every_rung_of_a_film_gets_the_same_window(self):
        assert corrected("kalshi_vio_55_yes").window_start == corrected("kalshi_vio_60_yes").window_start is None

    def test_the_higher_rung_implies_the_lower_once_corrected(self):
        assert implies(corrected("kalshi_vio_60_yes"), corrected("kalshi_vio_55_yes"))

    def test_the_stored_readings_compare_even_uncorrected(self):
        # One gives the snapshot a start at its end, the other none: both are one moment.
        _, higher = stored("kalshi_vio_60_yes")
        _, lower = stored("kalshi_vio_55_yes")
        assert implies(higher, lower)

    def test_the_no_side_stays_negated(self):
        fixed = corrected("kalshi_vio_60_no")
        assert fixed.negated and fixed.unknown == frozenset()

    def test_the_monday_after_release_is_the_markets_close(self):
        fixed = corrected("kalshi_dune_monday")
        assert (fixed.window_end, fixed.unknown) == (RELEASE_MONDAY, frozenset())

    def test_a_close_that_is_not_the_stated_monday_time_leaves_the_time_unknown(self):
        source, reading = stored("kalshi_dune_monday")
        source["market"]["close_time"] = "2026-12-22T15:00:00Z"
        assert "window_end" in correct_venue(source, reading).unknown

    def test_a_strike_that_disagrees_with_the_rules_leaves_the_value_unknown(self):
        source, reading = stored("kalshi_vio_60_yes")
        source["market"]["floor_strike"] = 65
        assert "value" in correct_venue(source, reading).unknown

    def test_other_kalshi_series_are_left_alone(self):
        source, reading = stored("kalshi_vio_60_yes")
        source["market"]["ticker"] = "KXRTCOMPARE-VIO-60"
        assert correct_venue(source, reading) == reading


class TestPolymarket:
    def test_polymarket_us_reads_or_more_as_at_least(self):
        fixed = corrected("pm_us_ipl_80_long")
        assert (fixed.comparator, fixed.value, fixed.window_start, fixed.unknown) == (
            Comparator.GTE, 80, None, frozenset())

    def test_polymarket_us_monday_takes_the_slugs_date(self):
        fixed = corrected("pm_us_monday")
        assert (fixed.window_end, fixed.value, fixed.unknown) == (RELEASE_MONDAY, 95, frozenset())

    def test_polymarket_us_slug_strike_must_agree(self):
        source, reading = stored("pm_us_ipl_80_long")
        source["market"]["slug"] = source["market"]["slug"].replace("gte80", "gte85")
        assert "value" in correct_venue(source, reading).unknown

    def test_polymarket_intl_monday_takes_the_date_of_the_markets_end(self):
        fixed = corrected("pm_intl_monday")
        assert (fixed.comparator, fixed.value, fixed.window_end, fixed.unknown) == (
            Comparator.GTE, 80, RELEASE_MONDAY, frozenset())

    def test_polymarket_intl_label_must_agree(self):
        source, reading = stored("pm_intl_monday")
        source["market"]["groupItemTitle"] = "85+"
        assert "value" in correct_venue(source, reading).unknown


def test_kalshi_above_implies_polymarket_at_least_on_one_film():
    kalshi, polymarket = corrected("kalshi_ipl_80_yes"), corrected("pm_us_ipl_80_long")
    assert kalshi.subject == polymarket.subject
    assert implies(kalshi, polymarket, "KALSHI", "POLYMARKET_US")


@pytest.mark.parametrize("name", [k for k in FIXTURE if not k.startswith("_")])
def test_every_fixture_reading_is_usable_after_correction(name):
    fixed = corrected(name)
    assert fixed.window_start is None and fixed.unknown == frozenset()
