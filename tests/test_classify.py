from unittest.mock import patch

from classifier.stages.classify import classify_relationships
from tests.conftest import make_event, make_event_contract


def test_skip_semantic_produces_structural_relationships(stub_registry, mock_voyage, mock_anthropic):
    event = make_event(event_id=10, title="Will X happen?")
    stub_registry._events.append(event)

    ec_yes = make_event_contract(1, event_id=10, security_id=100, outcome_label="Yes")
    ec_no = make_event_contract(2, event_id=10, security_id=101, outcome_label="No")
    stub_registry._event_contracts.extend([ec_yes, ec_no])

    with patch("classifier.stages.classify.embed_events_voyage") as mock_embed, \
         patch("classifier.stages.classify.find_semantic_matches") as mock_semantic:

        result = classify_relationships(
            stub_registry, mock_anthropic, mock_voyage,
            new_security_ids=[100, 101],

            skip_semantic=True,
        )

    mock_embed.assert_not_called()
    mock_semantic.assert_not_called()
    assert result["relationships_written"] == 2
    rels = stub_registry._contract_relationships
    assert {(r.security_id_a, r.security_id_b) for r in rels} == {(100, 101), (101, 100)}
    assert all(r.relationship_type == "COMPLEMENT" for r in rels)


def test_skip_semantic_false_calls_voyage(stub_registry, mock_voyage, mock_anthropic):
    event = make_event(event_id=10, title="Will X happen?")
    stub_registry._events.append(event)

    ec_yes = make_event_contract(1, event_id=10, security_id=100, outcome_label="Yes")
    ec_no = make_event_contract(2, event_id=10, security_id=101, outcome_label="No")
    stub_registry._event_contracts.extend([ec_yes, ec_no])

    with patch("classifier.stages.classify.embed_events_voyage", return_value={}) as mock_embed, \
         patch("classifier.stages.classify.find_semantic_matches", return_value=[]) as mock_semantic:

        classify_relationships(
            stub_registry, mock_anthropic, mock_voyage,
            new_security_ids=[100, 101],

            skip_semantic=False,
        )

    mock_embed.assert_called_once()
    mock_semantic.assert_called_once()
