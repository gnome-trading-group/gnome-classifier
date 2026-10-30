import json

import pytest

from classifier.relationships.evaluation import (
    A_IMPLIES_B,
    B_IMPLIES_A,
    EQUIVALENT,
    GOLDEN_DIR,
    LABELS,
    NONE,
    GoldenPair,
    Prediction,
    load_golden,
    load_predictions,
    score,
)

A = ("KALSHI", "A:yes")
B = ("POLYMARKET_INTL", "B")
C = ("KALSHI", "C:yes")


def golden(expected: str, a=A, b=B, flags=()) -> GoldenPair:
    return GoldenPair(id=f"{a[1]}-{b[1]}", a=a, b=b, expected=expected, flags=tuple(flags), review=False, note="",
                      source="test", raw={})


class TestPredictedLabel:
    def test_an_implication_matches_its_direction(self):
        report = score([golden(A_IMPLIES_B)], [Prediction(A, B, "IMPLIES")])
        assert report.implies.precision == 1.0
        assert report.implies.recall == 1.0

    def test_a_reversed_implication_is_wrong_and_missed(self):
        report = score([golden(A_IMPLIES_B)], [Prediction(B, A, "IMPLIES")])
        assert report.implies.precision == 0.0
        assert report.implies.recall == 0.0
        assert [s.predicted for s in report.wrong()] == [B_IMPLIES_A]

    def test_implies_both_ways_reads_as_equivalent(self):
        report = score([golden(EQUIVALENT)], [Prediction(A, B, "IMPLIES"), Prediction(B, A, "IMPLIES")])
        assert report.equivalent.precision == 1.0
        assert report.equivalent.recall == 1.0
        assert report.implies.predicted == 0

    def test_equivalent_is_found_in_either_stored_direction(self):
        report = score([golden(EQUIVALENT)], [Prediction(B, A, EQUIVALENT)])
        assert report.equivalent.found == 1


class TestCorrectness:
    def test_an_implication_between_equivalents_is_true_but_misses_the_equivalence(self):
        report = score([golden(EQUIVALENT)], [Prediction(A, B, "IMPLIES")])
        assert report.implies.precision == 1.0
        assert report.equivalent.recall == 0.0
        assert len(report.missed()) == 1
        assert report.wrong() == []

    def test_equivalent_on_an_implication_is_wrong(self):
        report = score([golden(A_IMPLIES_B)], [Prediction(A, B, EQUIVALENT)])
        assert report.equivalent.precision == 0.0
        assert report.implies.recall == 0.0

    def test_any_relationship_on_an_unrelated_pair_is_wrong(self):
        report = score([golden(NONE)], [Prediction(A, B, "IMPLIES")])
        assert report.implies.precision == 0.0
        assert report.missed() == []

    def test_rows_outside_golden_pairs_are_ignored(self):
        report = score([golden(NONE)], [Prediction(A, C, EQUIVALENT)])
        assert report.equivalent.predicted == 0
        assert report.passes(0.99)


class TestGate:
    def test_no_predictions_pass_vacuously(self):
        assert score([golden(EQUIVALENT)], []).passes(0.99)

    def test_one_wrong_in_a_hundred_passes_two_fail(self):
        pairs = [golden(A_IMPLIES_B, a=("KALSHI", f"x{i}"), b=("KALSHI", f"y{i}")) for i in range(100)]
        right = [Prediction(p.a, p.b, "IMPLIES") for p in pairs[:99]]
        assert score(pairs, right + [Prediction(pairs[99].b, pairs[99].a, "IMPLIES")]).passes(0.99)
        assert not score(pairs, right[:98] + [Prediction(p.b, p.a, "IMPLIES") for p in pairs[98:]]).passes(0.99)


def test_flag_mismatches_compare_predicted_and_expected_flags():
    report = score([golden(EQUIVALENT, flags=["RESOLUTION_SOURCE_DIFFERS"])], [Prediction(A, B, EQUIVALENT)])
    assert len(report.flag_mismatches()) == 1
    report = score([golden(EQUIVALENT, flags=["RESOLUTION_SOURCE_DIFFERS"])],
                   [Prediction(A, B, EQUIVALENT, ("RESOLUTION_SOURCE_DIFFERS",))])
    assert report.flag_mismatches() == []


def test_load_round_trips_jsonl(tmp_path):
    side_a = {"exchange": A[0], "exchange_security_id": A[1], "title": "t", "outcome_label": "Yes"}
    side_b = {"exchange": B[0], "exchange_security_id": B[1], "title": "t", "outcome_label": "Yes"}
    (tmp_path / "pairs.jsonl").write_text(json.dumps(
        {"id": "p1", "source": "test", "expected": A_IMPLIES_B, "a": side_a, "b": side_b}) + "\n")
    (tmp_path / "preds.jsonl").write_text(json.dumps({"a": side_a, "b": side_b, "type": "IMPLIES"}) + "\n")
    report = score(load_golden(tmp_path / "pairs.jsonl"), load_predictions(tmp_path / "preds.jsonl"))
    assert report.implies.correct == 1


def test_load_refuses_an_unknown_label(tmp_path):
    (tmp_path / "pairs.jsonl").write_text(json.dumps(
        {"id": "p1", "expected": "MUTUALLY_EXCLUSIVE", "a": {"exchange": "K", "exchange_security_id": "a"},
         "b": {"exchange": "K", "exchange_security_id": "b"}}) + "\n")
    with pytest.raises(ValueError, match="unknown label"):
        load_golden(tmp_path / "pairs.jsonl")


class TestGoldenSet:
    """The checked-in golden pairs stay well formed."""

    def test_pairs_are_unique_and_labelled(self):
        pairs = load_golden()
        assert len({p.id for p in pairs}) == len(pairs)
        assert len({frozenset((p.a, p.b)) for p in pairs}) == len(pairs)
        assert all(p.expected in LABELS and p.a != p.b for p in pairs)

    def test_baseline_rows_only_name_golden_securities(self):
        securities = {s for p in load_golden() for s in (p.a, p.b)}
        assert all({p.a, p.b} <= securities for p in load_predictions(GOLDEN_DIR / "baseline_judge.jsonl"))
