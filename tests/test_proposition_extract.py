import dataclasses
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from classifier.propositions.extract import extract_propositions, family
from classifier.propositions.inputs import ExtractionInput, is_primary_side
from classifier.propositions.llm import RESPONSE_SCHEMA, PropositionExtractor, parse_answer
from classifier.propositions.subjects import InMemorySubjectIndex, KnownSubject
from classifier.propositions.types import Comparator, Kind, Method, Observation
from classifier.runtime_config import Models
from gnomepy.registry.types import ContractType

BTC_RULES = ("If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time Index (BRTI) before 3 PM EDT "
             "is above 79899.99 at 3 PM EDT on Oct 29, 2026, then the market resolves to Yes.")


def item(security_id, event_id, source, contract_type=ContractType.BINARY, outcome="Yes", exchange="KALSHI"):
    return ExtractionInput(security_id=security_id, exchange=exchange, event_id=event_id, contract_type=contract_type,
                           event_title="t", outcome_label=outcome, description=None, expiry=None, source=source,
                           source_hash=f"h{security_id}")


def kalshi_side(side, ticker="KXBTCD-26OCT2915-T79899.99", rules=BTC_RULES):
    return {"venue": "KALSHI", "market": {"ticker": ticker, "rules_primary": rules}, "event": {}, "side": side}


def pm_source(outcome, outcomes=("Yes", "No")):
    return {"venue": "POLYMARKET_INTL", "market": {"question": "Will the Fed cut in December?", "outcomes": list(outcomes)},
            "outcome": outcome}


def entry(contract_id, **overrides):
    base = {
        "contract_id": str(contract_id), "subject": "fed-funds-rate-dec-2026", "subject_description": "The Fed's December 2026 decision",
        "kind": "OCCURS_BY", "measure": None, "comparator": None, "value": None, "unit": None, "low": None, "high": None,
        "low_inclusive": True, "high_inclusive": True, "observation": None, "window_start": None,
        "window_end": "2026-12-10T19:00:00Z", "entity": None, "scope": None, "stability": "LIVE",
        "resolution_source": "federal-reserve", "unknown": [],
    }
    return {**base, **overrides}


def extractor_for(client, **kwargs):
    return PropositionExtractor(lambda provider: client, ["anthropic"], {"anthropic": "claude-sonnet-5-5"}, **kwargs)


def fake_client(answers_by_call):
    client = MagicMock()
    client.messages.create.side_effect = [
        SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps({"propositions": a}))])
        for a in answers_by_call
    ]
    return client


class TestPrimarySide:
    def test_binary_primary_sides_per_venue(self):
        assert is_primary_side(item(1, 1, kalshi_side("yes")))
        assert not is_primary_side(item(2, 1, kalshi_side("no")))
        assert is_primary_side(item(3, 2, pm_source("Yes")))
        assert not is_primary_side(item(4, 2, pm_source("No")))
        assert is_primary_side(item(5, 3, {"venue": "POLYMARKET_US", "leg": "long"}))
        assert not is_primary_side(item(6, 3, {"venue": "POLYMARKET_US", "leg": "short"}))

    def test_polymarket_outcomes_given_as_a_json_string(self):
        source = pm_source("Notre Dame")
        source["market"]["outcomes"] = json.dumps(["Notre Dame", "Navy"])
        assert is_primary_side(item(1, 1, source))

    def test_multi_outcome_contracts_are_each_read(self):
        assert is_primary_side(item(1, 1, kalshi_side("yes", ticker="KXNBAGAME-26OCT30DENPHX-PHX"),
                                    contract_type=ContractType.MULTI_OUTCOME))


class TestParseAnswer:
    def test_valid_entries_become_propositions(self):
        out = parse_answer(json.dumps({"propositions": [entry(7)]}), [item(7, 1, pm_source("Yes"))])
        proposition, description = out[7]
        assert proposition.kind == Kind.OCCURS_BY
        assert proposition.window_end == datetime(2026, 12, 10, 19, tzinfo=timezone.utc)
        assert description == "The Fed's December 2026 decision"

    def test_an_entry_that_does_not_validate_stays_pending(self):
        bad = entry(7, kind="THRESHOLD")  # a threshold without comparator or value, and not listed unknown
        assert parse_answer(json.dumps({"propositions": [bad]}), [item(7, 1, pm_source("Yes"))]) == {7: None}

    def test_a_local_time_without_zone_is_refused(self):
        out = parse_answer(json.dumps({"propositions": [entry(7, window_end="2026-12-10T14:00:00")]}),
                           [item(7, 1, pm_source("Yes"))])
        assert out == {7: None}

    def test_unknown_fields_are_kept_not_guessed(self):
        threshold = entry(7, kind="THRESHOLD", measure="rate", comparator=None, value=None,
                          unknown=["comparator", "value"])
        proposition, _ = parse_answer(json.dumps({"propositions": [threshold]}), [item(7, 1, pm_source("Yes"))])[7]
        assert proposition.unknown == {"comparator", "value"}

    def test_ids_not_asked_about_are_ignored_and_missing_ones_pending(self):
        out = parse_answer(json.dumps({"propositions": [entry(99)]}), [item(7, 1, pm_source("Yes"))])
        assert out == {7: None}

    def test_garbage_leaves_everything_pending(self):
        assert parse_answer("not json", [item(7, 1, pm_source("Yes"))]) == {7: None}


class TestExtractPropositions:
    def test_venue_first_llm_for_the_rest_and_negated_second_sides(self):
        items = [
            item(1, 10, kalshi_side("yes")), item(2, 10, kalshi_side("no")),
            item(3, 20, pm_source("Yes"), exchange="POLYMARKET_INTL"), item(4, 20, pm_source("No"), exchange="POLYMARKET_INTL"),
        ]
        client = fake_client([[entry(3)]])
        results, report = extract_propositions(items, extractor_for(client))

        assert results[1].method == Method.VENUE and results[1].proposition.comparator == Comparator.GT
        assert results[2].proposition == results[1].proposition.negate()
        assert results[3].method == Method.LLM and results[4].proposition == results[3].proposition.negate()
        assert client.messages.create.call_count == 1
        prompt = json.loads(client.messages.create.call_args.kwargs["messages"][0]["content"])
        assert [c["contract_id"] for c in prompt["contracts"]] == ["3"]
        assert report.by_family == {("KALSHI:KXBTCD", "VENUE"): 1, ("POLYMARKET_INTL", "LLM"): 1}
        assert report.subject_descriptions == {"fed-funds-rate-dec-2026": "The Fed's December 2026 decision"}

    def test_requests_ask_for_the_schema_and_a_cached_system_prompt(self):
        client = fake_client([[entry(3)]])
        extract_propositions([item(3, 20, pm_source("Yes"), exchange="POLYMARKET_INTL")],
                             extractor_for(client))
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5-5"
        assert kwargs["output_config"]["format"] == {"type": "json_schema", "schema": RESPONSE_SCHEMA}
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_a_failed_request_leaves_its_contracts_and_their_partners_pending(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("throttled")
        items = [item(3, 20, pm_source("Yes"), exchange="POLYMARKET_INTL"), item(4, 20, pm_source("No"), exchange="POLYMARKET_INTL")]
        results, report = extract_propositions(items, extractor_for(client))
        assert results == {} and sorted(report.failed) == [3, 4]

    def test_without_an_extractor_only_venue_reads_happen(self):
        results, report = extract_propositions([item(3, 20, pm_source("Yes"), exchange="POLYMARKET_INTL")], None)
        assert results == {} and report.failed == [3]

    def test_large_events_are_chunked(self):
        items = [item(i, 30, kalshi_side("yes", ticker=f"KXNEW-{i}", rules="free text"),
                      contract_type=ContractType.MULTI_OUTCOME) for i in range(1, 31)]
        client = fake_client([[entry(i) for i in range(1, 26)], [entry(i) for i in range(26, 31)]])
        results, _ = extract_propositions(items, extractor_for(client, max_workers=1))
        assert client.messages.create.call_count == 2 and len(results) == 30


class TestProviders:
    def answer(self):
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps({"propositions": [entry(3)]}))])

    def test_falls_back_when_the_first_provider_is_unavailable(self):
        anthropic_client, bedrock_client = MagicMock(), MagicMock()
        anthropic_client.messages.create.side_effect = anthropic.AuthenticationError(
            "bad key", response=httpx.Response(401, request=httpx.Request("POST", "https://x")), body=None)
        bedrock_client.messages.create.return_value = self.answer()
        clients = {"anthropic": anthropic_client, "bedrock": bedrock_client}
        extractor = PropositionExtractor(clients.__getitem__, ["anthropic", "bedrock"],
                                         {"anthropic": "claude-sonnet-5-5", "bedrock": "anthropic.claude-sonnet-5-5"})
        results, _ = extract_propositions([item(3, 20, pm_source("Yes"), exchange="POLYMARKET_INTL")], extractor)
        assert results[3].method == Method.LLM
        assert bedrock_client.messages.create.call_args.kwargs["model"] == "anthropic.claude-sonnet-5-5"

    def test_a_bad_request_does_not_fall_back(self):
        anthropic_client, bedrock_client = MagicMock(), MagicMock()
        anthropic_client.messages.create.side_effect = anthropic.BadRequestError(
            "bad schema", response=httpx.Response(400, request=httpx.Request("POST", "https://x")), body=None)
        clients = {"anthropic": anthropic_client, "bedrock": bedrock_client}
        extractor = PropositionExtractor(clients.__getitem__, ["anthropic", "bedrock"],
                                         {"anthropic": "a", "bedrock": "b"})
        extract_propositions([item(3, 20, pm_source("Yes"), exchange="POLYMARKET_INTL")], extractor)
        bedrock_client.messages.create.assert_not_called()

    def test_every_provider_needs_a_model(self):
        with pytest.raises(ValueError):
            PropositionExtractor(lambda p: None, ["anthropic", "bedrock"], {"anthropic": "a"})

    def test_config_defaults_to_the_anthropic_api_and_lists_a_fallback(self):
        models = Models()
        assert models.proposition_providers() == ["anthropic"]
        assert models.proposition_models["anthropic"] == "claude-haiku-5-5"
        assert Models(proposition_fallback_provider="bedrock").proposition_providers() == ["anthropic", "bedrock"]


def test_family_names_kalshi_series_and_hyperliquid_classes():
    assert family(item(1, 1, kalshi_side("yes"))) == "KALSHI:KXBTCD"
    assert family(item(1, 1, {"venue": "HYPERLIQUID", "outcome": {"description": "class:priceBinary|x:1"}})) == "HYPERLIQUID:class:priceBinary"
    assert family(item(1, 1, pm_source("Yes"))) == "POLYMARKET_INTL"


def test_a_venue_read_carries_the_observation():
    results, _ = extract_propositions([item(1, 10, kalshi_side("yes"))], None)
    assert results[1].proposition.observation == Observation.AT_TIME


def test_a_kalshi_ladder_is_read_in_one_request():
    def strike(sid, v):
        source = {"venue": "KALSHI", "side": "yes", "event": {"event_ticker": "KXAAAGASED-26NOV03"},
                  "market": {"ticker": f"KXAAAGASED-26NOV03-{v}", "rules_primary": f"gas above {v}"}}
        return item(sid, sid, source)
    client = fake_client([[entry(1), entry(2)]])
    extract_propositions([strike(1, "2.75"), strike(2, "4.75")], extractor_for(client))
    assert client.messages.create.call_count == 1


def test_later_venues_see_subjects_earlier_venues_were_given():
    kalshi_item = item(1, 10, {"venue": "KALSHI", "side": "yes", "event": {"event_ticker": "KXRECOG-27"},
                               "market": {"ticker": "KXRECOG-27-NZ", "rules_primary": "If New Zealand recognizes Palestine"}})
    pm_item = item(2, 20, pm_source("Yes"), exchange="POLYMARKET_INTL")
    pm_item = dataclasses.replace(pm_item, event_title="Will New Zealand recognize Palestine before 2027?")
    client = fake_client([
        [entry(1, subject="new-zealand-recognizes-palestine", subject_description="New Zealand recognizing Palestine")],
        [entry(2, subject="new-zealand-recognizes-palestine")],
    ])
    extract_propositions([pm_item, kalshi_item], extractor_for(client, max_workers=1))
    first, second = (json.loads(c.kwargs["messages"][0]["content"]) for c in client.messages.create.call_args_list)
    assert [c["contract_id"] for c in first["contracts"]] == ["1"]
    assert [s["subject"] for s in second["candidate_subjects"]] == ["new-zealand-recognizes-palestine"]


def test_fields_without_meaning_for_the_kind_are_cleared():
    noisy = entry(7, kind="OCCURS_BY", measure="announce_ipo", observation="ANY_TIME", unknown=["observation"])
    proposition, _ = parse_answer(json.dumps({"propositions": [noisy]}), [item(7, 1, pm_source("Yes"))])[7]
    assert (proposition.measure, proposition.observation, proposition.unknown) == ("announce_ipo", None, frozenset())
    winner = entry(8, kind="WINNER", entity="clem", measure="moneyline")
    assert parse_answer(json.dumps({"propositions": [winner]}), [item(8, 1, pm_source("Yes"))])[8][0].measure is None


def test_embedding_neighbours_become_candidates():
    seahawks = dataclasses.replace(item(2, 20, pm_source("Seahawks", outcomes=("Seahawks", "Titans")), exchange="POLYMARKET_INTL"),
                                   event_title="Seahawks vs. Titans")
    index = InMemorySubjectIndex(lambda texts, kind: [[1.0, 0.0] if "sea" in t.lower() else [0.0, 1.0] for t in texts])
    index.add([KnownSubject("game:nfl:2026-08-23:seaten", "Seattle vs Tennessee", {"sea", "ten"})])
    client = fake_client([[entry(2, subject="game:nfl:2026-08-23:seaten")]])
    extract_propositions([seahawks], extractor_for(client), index=index)
    prompt = json.loads(client.messages.create.call_args.kwargs["messages"][0]["content"])
    assert prompt["candidate_subjects"][0] == {"subject": "game:nfl:2026-08-23:seaten", "description": "Seattle vs Tennessee",
                                               "entities": ["sea", "ten"], "measures": []}


def test_a_side_labelled_with_a_name_is_read_even_without_side_names_in_the_source():
    source = {"venue": "HYPERLIQUID", "label": "CLE", "outcome": {"description": "This Basketball market has two sides"}}
    assert is_primary_side(item(1, 1, source, outcome="CLE", exchange="HYPERLIQUID"))
