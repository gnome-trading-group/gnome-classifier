from datetime import datetime, timezone

from gnomepy.registry.types import ContractType

from classifier.propositions.evaluation import LabelledProposition, load_labelled, score
from classifier.propositions.inputs import ExtractionInput
from classifier.propositions.types import Comparator, Extraction, Kind, Method, Proposition


def labelled(sid, group=None, entity_group=None, expected=None):
    item = ExtractionInput(sid, "KALSHI", sid, ContractType.BINARY, "t", "Yes", None, None, {"venue": "KALSHI"}, "h")
    return LabelledProposition(f"k{sid}", item, group, entity_group, expected, False)


def read(subject, **kwargs):
    defaults = {"kind": Kind.THRESHOLD, "comparator": Comparator.GT, "value": 1.0}
    return Extraction(Proposition(subject=subject, **{**defaults, **kwargs}), Method.LLM, "h")


def test_subject_agreement_counts_pairs_within_a_group():
    report = score([labelled(1, "g"), labelled(2, "g"), labelled(3, "g")],
                   {1: read("a"), 2: read("a"), 3: read("b")})
    assert (report.subject_agreed, report.subject_pairs) == (1, 3)
    assert report.split_groups == [("g", [("k1", "a"), ("k2", "a"), ("k3", "b")])]


def test_fields_compare_exactly_and_windows_within_twelve_hours():
    end = datetime(2027, 1, 1, 5, tzinfo=timezone.utc)
    expected = read("a", window_end=end).proposition.to_fields()
    near = read("a", window_end=datetime(2027, 1, 1, tzinfo=timezone.utc))
    far = read("a", window_end=datetime(2027, 1, 2, 5, tzinfo=timezone.utc), value=2.0)
    report = score([labelled(1, expected=expected), labelled(2, expected=expected)], {1: near, 2: far})
    assert report.field_accuracy("window_end") == 0.5
    assert report.field_accuracy("value") == 0.5


def test_missing_reads_are_listed():
    assert score([labelled(1)], {}).missing == ["k1"]


def test_the_labelled_set_loads_and_binary_sides_share_an_event():
    rows = load_labelled()
    assert len(rows) > 300
    by_event = {}
    for lp in rows:
        by_event.setdefault(lp.item.event_id, []).append(lp)
    binaries = [group for group in by_event.values() if group[0].item.contract_type == ContractType.BINARY]
    assert binaries and all(len(group) <= 2 for group in binaries)
