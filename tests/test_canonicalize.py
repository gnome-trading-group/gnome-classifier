import json

import pytest

from classifier.client import BedrockConverseClient, ConverseResult
from classifier.stages.canonicalize import canonicalize_events, _parse_canonical_result, _parse_response, _title_key
from classifier.types import CanonicalizeInput
from scripts.testing import FakeConverseClient, MemoryClassifierCache

MODEL = "us.amazon.nova-lite-v1:0"


class ScriptedConverseClient(BedrockConverseClient):
    """Returns queued payloads in order: chunk responses first, then individual retries."""

    def __init__(self, payloads):
        super().__init__(client=None, max_workers=1)
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    def converse(self, model, prompt, max_tokens):
        self.prompts.append(prompt)
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return ConverseResult(text=json.dumps(payload), input_tokens=10, output_tokens=5)


def test_key_mismatch_triggers_individual_retry():
    """Wrong key in a chunk response (ID swap) is detected and the event is retried on its own."""
    swapped_key = _title_key("Federal Funds Rate Decision")
    client = ScriptedConverseClient([
        [{"id": 1, "key": swapped_key, "category": "ECONOMICS", "tags": ["fed"]}],
        {"category": "POLITICS", "tags": ["election"]},
    ])

    results = canonicalize_events(client, [CanonicalizeInput("AL-02 House Election Winner", None, None, 1, "al-02")], model=MODEL)

    assert len(client.prompts) == 2
    assert "Exchange-provided title: AL-02 House Election Winner" in client.prompts[1]
    assert results[(1, "al-02")] == {"category": "POLITICS", "tags": ["election"]}


def test_correct_key_accepts_without_retry():
    raw_title = "AL-02 House Election Winner"
    client = ScriptedConverseClient([
        [{"id": 1, "key": _title_key(raw_title), "category": "POLITICS", "tags": ["election"]}],
    ])

    results = canonicalize_events(client, [CanonicalizeInput(raw_title, None, None, 1, "al-02")], model=MODEL)

    assert len(client.prompts) == 1
    assert results[(1, "al-02")] == {"category": "POLITICS", "tags": ["election"]}


def test_unparseable_chunk_retries_every_event():
    events = [
        CanonicalizeInput("Will BTC hit 100k?", None, None, 1, "btc"),
        CanonicalizeInput("Who wins the election?", None, None, 1, "election"),
    ]
    client = ScriptedConverseClient([
        "not a list",
        {"category": "CRYPTO", "tags": ["btc"]},
        {"category": "POLITICS", "tags": ["election"]},
    ])

    results = canonicalize_events(client, events, model=MODEL)

    assert len(client.prompts) == 3
    assert {r["category"] for r in results.values()} == {"CRYPTO", "POLITICS"}


def test_failed_retry_raises():
    client = ScriptedConverseClient([RuntimeError("throttled"), RuntimeError("throttled")])

    with pytest.raises(RuntimeError, match="1 events failed canonicalization"):
        canonicalize_events(client, [CanonicalizeInput("Will BTC hit 100k?", None, None, 1, "btc")], model=MODEL)


def test_events_split_into_chunks():
    events = [CanonicalizeInput(f"Event {i}", None, None, 1, f"e{i}") for i in range(5)]
    client = FakeConverseClient(category="SPORTS")

    results = canonicalize_events(client, events, model=MODEL, batch_size=2)

    assert client.calls == 3
    assert len(results) == 5


def test_parse_canonical_result_valid():
    item = {"category": "POLITICS", "tags": ["a", "b", "c"]}
    assert _parse_canonical_result(item) == {"category": "POLITICS", "tags": ["a", "b", "c"]}


def test_parse_canonical_result_invalid_category():
    item = {"title": "T", "category": "INVALID", "tags": ["a", "b", "c"]}
    result = _parse_canonical_result(item)
    assert result["category"] == "OTHER"


def test_parse_canonical_result_category_case_insensitive():
    assert _parse_canonical_result({"category": " Politics ", "tags": []})["category"] == "POLITICS"


def test_parse_canonical_result_non_string_category():
    assert _parse_canonical_result({"category": None, "tags": []})["category"] == "OTHER"


def test_parse_canonical_result_bad_tags():
    item = {"title": "T", "category": "POLITICS", "tags": "not-a-list"}
    result = _parse_canonical_result(item)
    assert result["tags"] == []


def test_parse_canonical_result_short_tags_kept():
    item = {"title": "T", "category": "POLITICS", "tags": ["a"]}
    result = _parse_canonical_result(item)
    assert result["tags"] == ["a"]


def test_parse_canonical_result_tags_capped_at_six():
    item = {"title": "T", "category": "POLITICS", "tags": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]}
    result = _parse_canonical_result(item)
    assert result["tags"] == ["a", "b", "c", "d", "e", "f"]


@pytest.mark.parametrize("tag", [
    "86232", "$700m", "1b", "4.0%", "2026", "2026-27", "year-end 2031", "2026-10-19", "08:00",
    "above 8.0%", "at least 3", "price", "prediction market", "twap", "economics",
])
def test_clean_tags_drops_event_specific_and_filler(tag):
    assert _parse_canonical_result({"category": "ECONOMICS", "tags": [tag]})["tags"] == []


@pytest.mark.parametrize("tag", ["u-3", "s&p 500", "covid-19", "mi-08", "gpt-5"])
def test_clean_tags_keeps_entities(tag):
    assert _parse_canonical_result({"category": "ECONOMICS", "tags": [tag]})["tags"] == [tag]


def test_clean_tags_normalizes_and_dedupes():
    result = _parse_canonical_result({"category": "SPORTS", "tags": ["NBA", " nba ", "#Steph  Curry", 7]})
    assert result["tags"] == ["nba", "steph curry"]


def test_canonicalize_events_batch(mock_converse):
    events = [
        CanonicalizeInput("Will BTC hit 100k?", None, None, 1, "native-1"),
        CanonicalizeInput("Who wins the election?", "US presidential race", "POLITICS", 1, "native-2"),
    ]
    result = canonicalize_events(mock_converse, events, model=MODEL)
    assert (1, "native-1") in result
    assert (1, "native-2") in result
    for r in result.values():
        assert set(r) == {"category", "tags"}


def test_canonicalize_events_cache_hit(mock_converse):
    cache = MemoryClassifierCache()
    cached_result = {"category": "CRYPTO", "tags": ["btc", "price", "crypto"]}
    cache.put_canonicalization(MODEL, 1, "native-abc", cached_result)

    result = canonicalize_events(mock_converse, [CanonicalizeInput("raw title", None, None, 1, "native-abc")], cache=cache, model=MODEL)

    assert mock_converse.calls == 0
    assert result[(1, "native-abc")] == {"category": "CRYPTO", "tags": ["bitcoin"]}


def test_canonicalize_events_cache_miss_then_store(mock_converse):
    cache = MemoryClassifierCache()

    canonicalize_events(mock_converse, [CanonicalizeInput("raw title 2", None, None, 2, "native-xyz")], cache=cache, model=MODEL)

    assert mock_converse.calls == 1
    cached = cache.get_canonicalization(MODEL, 2, "native-xyz")
    assert set(cached) == {"category", "tags"}


# ── Exchange labels ───────────────────────────────────────────────────────────

def test_mapped_category_and_enough_tags_skip_the_model(mock_converse):
    ev = CanonicalizeInput("FURIA vs. Legacy", None, "Sports", 2, "KXCS2GAME-1", "SPORTS", ("cs2", "esports"))

    result = canonicalize_events(mock_converse, [ev], model=MODEL)

    assert mock_converse.calls == 0
    assert result[(2, "KXCS2GAME-1")] == {"category": "SPORTS", "tags": ["cs2", "esports"]}


def test_model_fills_thin_tags_but_keeps_mapped_category():
    ev = CanonicalizeInput("Nevada's 4th District margin of victory", None, "Elections", 2, "nv-04", "POLITICS", ("us elections",))
    client = ScriptedConverseClient([[{"id": 1, "key": _title_key(ev.raw_title), "category": "OTHER", "tags": ["nevada", "house", "us elections"]}]])

    result = canonicalize_events(client, [ev], model=MODEL)

    assert "Exchange tags: us elections | Category: POLITICS (fixed)" in client.prompts[0]
    assert result[(2, "nv-04")] == {"category": "POLITICS", "tags": ["us elections", "nevada", "house"]}


def test_model_decides_unmapped_category_and_keeps_exchange_tags():
    ev = CanonicalizeInput("Will OpenAI IPO?", None, "Science and Technology", 2, "ipo", None, ("openai", "ipo"))
    client = ScriptedConverseClient([[{"id": 1, "key": _title_key(ev.raw_title), "category": "FINANCE", "tags": ["openai"]}]])

    result = canonicalize_events(client, [ev], model=MODEL)

    assert "Exchange tags: openai, ipo | Category: choose" in client.prompts[0]
    assert result[(2, "ipo")] == {"category": "FINANCE", "tags": ["openai", "ipo"]}


def test_parse_response_finds_json_after_preamble():
    text = 'Here is the categorization:\n\n```json\n[{"id": 1, "key": "abc", "category": "SPORTS", "tags": []}]\n```'
    assert _parse_response(text) == [{"id": 1, "key": "abc", "category": "SPORTS", "tags": []}]


def test_multiline_titles_stay_on_one_prompt_line(mock_converse):
    ev = CanonicalizeInput(" XRP price at Oct 15?\n: $1.69991 or above", "line one\nline two", "Crypto", 2, "xrp", None, ())

    result = canonicalize_events(mock_converse, [ev], model=MODEL)

    assert mock_converse.calls == 1
    assert (2, "xrp") in result
