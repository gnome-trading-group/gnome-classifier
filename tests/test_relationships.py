from classifier.relationships.structural import find_complement_pairs, find_mutually_exclusive_pairs
from classifier.types import RelationshipMatch, RelationshipType
from tests.conftest import make_event_contract


def test_find_complement_pairs_binary():
    ecs = [
        make_event_contract(1, event_id=10, security_id=100, outcome_label="Yes"),
        make_event_contract(2, event_id=10, security_id=101, outcome_label="No"),
    ]
    pairs = find_complement_pairs(ecs)
    assert len(pairs) == 2
    assert {(p.security_id_a, p.security_id_b) for p in pairs} == {(100, 101), (101, 100)}
    assert all(p.relationship_type == RelationshipType.COMPLEMENT for p in pairs)


def test_find_complement_pairs_multi_outcome():
    ecs = [
        make_event_contract(1, event_id=10, security_id=100, outcome_label="A"),
        make_event_contract(2, event_id=10, security_id=101, outcome_label="B"),
        make_event_contract(3, event_id=10, security_id=102, outcome_label="C"),
    ]
    pairs = find_complement_pairs(ecs)
    assert len(pairs) == 0


def test_find_mutually_exclusive_pairs():
    ecs = [
        make_event_contract(1, event_id=10, security_id=100, outcome_label="A"),
        make_event_contract(2, event_id=10, security_id=101, outcome_label="B"),
        make_event_contract(3, event_id=10, security_id=102, outcome_label="C"),
        make_event_contract(4, event_id=20, security_id=200, outcome_label="Yes"),
        make_event_contract(5, event_id=20, security_id=201, outcome_label="No"),
    ]
    pairs = find_mutually_exclusive_pairs(ecs)
    # event 10: 3 contracts → 3×2=6 directed pairs; event 20: 2 contracts → 2×1=2 directed pairs → total 8
    assert len(pairs) == 8
    assert all(p.relationship_type == RelationshipType.MUTUALLY_EXCLUSIVE for p in pairs)


