from unittest.mock import patch

from classifier.stages.classify import classify_structural, prepare_semantic_batch, process_semantic_results
from tests.conftest import make_event, make_event_contract


def test_classify_structural_produces_complement_relationships(stub_registry, stub_db, mock_anthropic):
    event = make_event(event_id=10, title="Will X happen?")
    stub_registry._events.append(event)

    ec_yes = make_event_contract(1, event_id=10, security_id=100, outcome_label="Yes")
    ec_no = make_event_contract(2, event_id=10, security_id=101, outcome_label="No")
    stub_registry._event_contracts.extend([ec_yes, ec_no])

    result = classify_structural(stub_registry, new_security_ids=[100, 101], db=stub_db)

    assert result["relationships_written"] == 2
    rels = stub_registry._contract_relationships
    assert {(r.security_id_a, r.security_id_b) for r in rels} == {(100, 101), (101, 100)}
    assert all(r.relationship_type == "COMPLEMENT" for r in rels)


def test_prepare_semantic_batch_returns_requests_and_context(stub_registry, stub_db):
    event = make_event(event_id=10, title="Will X happen?")
    stub_registry._events.append(event)

    ec_yes = make_event_contract(1, event_id=10, security_id=100, outcome_label="Yes")
    ec_no = make_event_contract(2, event_id=10, security_id=101, outcome_label="No")
    stub_registry._event_contracts.extend([ec_yes, ec_no])

    stub_db._embeddings[10] = [0.1] * 1536

    with patch("classifier.stages.classify.find_semantic_candidates") as mock_candidates:
        mock_candidates.return_value = ([], [])
        api_requests, pending_context, cached_results = prepare_semantic_batch(
            new_security_ids=[100, 101], db=stub_db,
        )

    assert api_requests == []
    assert pending_context == []
    assert cached_results == []


def test_process_semantic_results_writes_relationships(stub_registry, stub_db):
    event = make_event(event_id=10, title="Will X happen?")
    stub_registry._events.append(event)

    ec_yes = make_event_contract(1, event_id=10, security_id=100, outcome_label="Yes")
    ec_no = make_event_contract(2, event_id=10, security_id=101, outcome_label="No")
    stub_registry._event_contracts.extend([ec_yes, ec_no])

    from classifier.types import RelationshipType
    cached_results = [
        {"security_id_a": 100, "security_id_b": 101, "relationship_type": RelationshipType.EQUIVALENT, "confidence": 0.95},
    ]

    result = process_semantic_results(
        stub_registry,
        responses={},
        pending_context=[],
        cached_results=cached_results,
        new_security_ids=[100, 101],
        db=stub_db,
    )

    # EQUIVALENT(100→101) + complement derivation produces (100,101) and (101,100)
    assert result["relationships_written"] == 2
