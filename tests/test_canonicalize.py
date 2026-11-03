import json

import pytest

from classifier.client.claude import Answer, ClaudeAsker
from classifier.stages.canonicalize import RESPONSE_SCHEMA, canonicalize_events, _parse_canonical_result, _title_key
from classifier.types import CanonicalizeInput
from scripts.testing import FakeCanonicalizer, MemoryClassifierCache

MODEL = "fake-model"


class ScriptedAsker(ClaudeAsker):
    """Returns queued answers in order: chunk answers first, then individual retries. A dict is an answer's events
    list, a string is returned as the raw text, an exception is raised."""

    def __init__(self, payloads):
        super().__init__(lambda provider: None, ["anthropic"], {"anthropic": MODEL})
        self._payloads = list(payloads)
        self.requests: list[dict] = []

    def ask_with_usage(self, system, prompt, schema, max_tokens, effort):
        self.requests.append({"system": system, "prompt": prompt, "schema": schema, "effort": effort})
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        text = payload if isinstance(payload, str) else json.dumps({"events": payload})
        return Answer(text=text, input_tokens=10, output_tokens=5)

    @property
    def prompts(self) -> list[str]:
        return [r["prompt"] for r in self.requests]


def entry(ev: CanonicalizeInput, category: str, tags: list[str], id_: int = 1, key: str | None = None) -> dict:
    return {"id": id_, "key": key if key is not None else _title_key(ev.raw_title), "category": category, "tags": tags}


def test_key_mismatch_triggers_individual_retry():
    """Wrong key in a chunk answer (ID swap) is detected and the event is retried on its own."""
    ev = CanonicalizeInput("AL-02 House Election Winner", None, None, 1, "al-02")
    client = ScriptedAsker([
        [entry(ev, "ECONOMICS", ["fed"], key=_title_key("Federal Funds Rate Decision"))],
        [entry(ev, "POLITICS", ["election"])],
    ])

    results = canonicalize_events(client, [ev], max_workers=1)

    assert len(client.prompts) == 2
    assert "Title: AL-02 House Election Winner" in client.prompts[1]
    assert results[(1, "al-02")] == {"category": "POLITICS", "tags": ["election"]}


def test_correct_key_accepts_without_retry():
    ev = CanonicalizeInput("AL-02 House Election Winner", None, None, 1, "al-02")
    client = ScriptedAsker([[entry(ev, "POLITICS", ["election"])]])

    results = canonicalize_events(client, [ev], max_workers=1)

    assert len(client.prompts) == 1
    assert results[(1, "al-02")] == {"category": "POLITICS", "tags": ["election"]}


def test_requests_use_the_schema_and_cached_instructions_at_low_effort():
    ev = CanonicalizeInput("AL-02 House Election Winner", None, None, 1, "al-02")
    client = ScriptedAsker([[entry(ev, "POLITICS", [])]])

    canonicalize_events(client, [ev], max_workers=1)

    request = client.requests[0]
    assert request["schema"] is RESPONSE_SCHEMA and request["effort"] == "low"
    assert request["system"].startswith("You are categorizing prediction market events")
    assert "Categories:" not in request["prompt"]


def test_unparseable_chunk_retries_every_event():
    events = [
        CanonicalizeInput("Will BTC hit 100k?", None, None, 1, "btc"),
        CanonicalizeInput("Who wins the election?", None, None, 1, "election"),
    ]
    client = ScriptedAsker([
        "not json",
        [entry(events[0], "CRYPTO", ["btc"])],
        [entry(events[1], "POLITICS", ["election"])],
    ])

    results = canonicalize_events(client, events, max_workers=1)

    assert len(client.prompts) == 3
    assert {r["category"] for r in results.values()} == {"CRYPTO", "POLITICS"}


def test_failed_retry_raises():
    client = ScriptedAsker([RuntimeError("throttled"), RuntimeError("throttled")])

    with pytest.raises(RuntimeError, match="1 events failed canonicalization"):
        canonicalize_events(client, [CanonicalizeInput("Will BTC hit 100k?", None, None, 1, "btc")], max_workers=1)


def test_events_split_into_chunks():
    events = [CanonicalizeInput(f"Event {i}", None, None, 1, f"e{i}") for i in range(5)]
    client = FakeCanonicalizer(category="SPORTS")

    results = canonicalize_events(client, events, batch_size=2)

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


def test_canonicalize_events_batch(mock_canonicalizer):
    events = [
        CanonicalizeInput("Will BTC hit 100k?", None, None, 1, "native-1"),
        CanonicalizeInput("Who wins the election?", "US presidential race", "POLITICS", 1, "native-2"),
    ]
    result = canonicalize_events(mock_canonicalizer, events)
    assert (1, "native-1") in result
    assert (1, "native-2") in result
    for r in result.values():
        assert set(r) == {"category", "tags"}


def test_canonicalize_events_cache_hit(mock_canonicalizer):
    cache = MemoryClassifierCache()
    cached_result = {"category": "CRYPTO", "tags": ["btc", "price", "crypto"]}
    cache.put_canonicalization(MODEL, 1, "native-abc", cached_result)

    result = canonicalize_events(mock_canonicalizer, [CanonicalizeInput("raw title", None, None, 1, "native-abc")], cache=cache)

    assert mock_canonicalizer.calls == 0
    assert result[(1, "native-abc")] == {"category": "CRYPTO", "tags": ["bitcoin"]}


def test_canonicalize_events_cache_miss_then_store(mock_canonicalizer):
    cache = MemoryClassifierCache()

    canonicalize_events(mock_canonicalizer, [CanonicalizeInput("raw title 2", None, None, 2, "native-xyz")], cache=cache)

    assert mock_canonicalizer.calls == 1
    cached = cache.get_canonicalization(MODEL, 2, "native-xyz")
    assert set(cached) == {"category", "tags"}


# ── Exchange labels ───────────────────────────────────────────────────────────

def test_mapped_category_and_enough_tags_skip_the_model(mock_canonicalizer):
    ev = CanonicalizeInput("FURIA vs. Legacy", None, "Sports", 2, "KXCS2GAME-1", "SPORTS", ("cs2", "esports"))

    result = canonicalize_events(mock_canonicalizer, [ev])

    assert mock_canonicalizer.calls == 0
    assert result[(2, "KXCS2GAME-1")] == {"category": "SPORTS", "tags": ["cs2", "esports"]}


def test_model_fills_thin_tags_but_keeps_mapped_category():
    ev = CanonicalizeInput("Nevada's 4th District margin of victory", None, "Elections", 2, "nv-04", "POLITICS", ("us elections",))
    client = ScriptedAsker([[entry(ev, "OTHER", ["nevada", "house", "us elections"])]])

    result = canonicalize_events(client, [ev], max_workers=1)

    assert "Exchange tags: us elections | Category: POLITICS (fixed)" in client.prompts[0]
    assert result[(2, "nv-04")] == {"category": "POLITICS", "tags": ["us elections", "nevada", "house"]}


def test_model_decides_unmapped_category_and_keeps_exchange_tags():
    ev = CanonicalizeInput("Will OpenAI IPO?", None, "Science and Technology", 2, "ipo", None, ("openai", "ipo"))
    client = ScriptedAsker([[entry(ev, "FINANCE", ["openai"])]])

    result = canonicalize_events(client, [ev], max_workers=1)

    assert "Exchange tags: openai, ipo | Category: choose" in client.prompts[0]
    assert result[(2, "ipo")] == {"category": "FINANCE", "tags": ["openai", "ipo"]}


def test_multiline_titles_stay_on_one_prompt_line(mock_canonicalizer):
    ev = CanonicalizeInput(" XRP price at Oct 15?\n: $1.69991 or above", "line one\nline two", "Crypto", 2, "xrp", None, ())

    result = canonicalize_events(mock_canonicalizer, [ev])

    assert mock_canonicalizer.calls == 1
    assert (2, "xrp") in result
