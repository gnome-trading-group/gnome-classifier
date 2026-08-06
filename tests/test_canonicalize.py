import hashlib
import json
from unittest.mock import MagicMock

import pytest

from classifier.stages.canonicalize import canonicalize_events, parse_canon_results, _parse_canonical_result, _title_key
from classifier.types import CanonicalizeInput


def _make_response(payload):
    mock_content = MagicMock()
    mock_content.text = json.dumps(payload)
    mock_resp = MagicMock()
    mock_resp.content = [mock_content]
    return mock_resp


def test_key_mismatch_triggers_individual_retry():
    """Wrong key in batch response (ID swap) is detected and event is retried individually."""
    swapped_key = _title_key("Federal Funds Rate Decision")
    swapped_response = _make_response([
        {"id": 1, "key": swapped_key, "title": "Federal Funds Rate Decision",
         "category": "ECONOMICS", "tags": ["fed"]}
    ])
    canon_context = [{"custom_id": "canon_0", "events": [
        {"raw_title": "AL-02 House Election Winner", "description": None, "category": None,
         "exchange_id": 1, "native_id": "al-02-house-election"}
    ]}]

    retry_response = _make_response({"title": "AL-02 House Election Winner", "category": "POLITICS", "tags": ["election"]})
    mock_client = MagicMock()
    mock_client.messages.create.return_value = retry_response

    results = parse_canon_results({"canon_0": swapped_response}, canon_context, None, mock_client)

    mock_client.messages.create.assert_called_once()
    assert (1, "al-02-house-election") in results
    assert results[(1, "al-02-house-election")]["title"] == "AL-02 House Election Winner"


def test_correct_key_accepts_without_retry():
    """Correct key passes validation without falling back to individual retry."""
    raw_title = "AL-02 House Election Winner"
    correct_response = _make_response([
        {"id": 1, "key": _title_key(raw_title), "title": raw_title,
         "category": "POLITICS", "tags": ["election"]}
    ])
    canon_context = [{"custom_id": "canon_0", "events": [
        {"raw_title": raw_title, "description": None, "category": None,
         "exchange_id": 1, "native_id": "al-02-house-election"}
    ]}]
    mock_client = MagicMock()

    results = parse_canon_results({"canon_0": correct_response}, canon_context, None, mock_client)

    mock_client.messages.create.assert_not_called()
    assert (1, "al-02-house-election") in results
    assert results[(1, "al-02-house-election")]["title"] == raw_title




def test_parse_canonical_result_valid():
    item = {"title": "Clean Title", "category": "POLITICS", "tags": ["a", "b", "c"]}
    result = _parse_canonical_result(item, "raw")
    assert result["title"] == "Clean Title"
    assert result["category"] == "POLITICS"
    assert result["tags"] == ["a", "b", "c"]


def test_parse_canonical_result_invalid_category():
    item = {"title": "T", "category": "INVALID", "tags": ["a", "b", "c"]}
    result = _parse_canonical_result(item, "raw")
    assert result["category"] == "OTHER"


def test_parse_canonical_result_bad_tags():
    item = {"title": "T", "category": "POLITICS", "tags": "not-a-list"}
    result = _parse_canonical_result(item, "raw")
    assert result["tags"] == []


def test_parse_canonical_result_short_tags_kept():
    item = {"title": "T", "category": "POLITICS", "tags": ["a"]}
    result = _parse_canonical_result(item, "raw")
    assert result["tags"] == ["a"]


def test_parse_canonical_result_tags_capped_at_eight():
    item = {"title": "T", "category": "POLITICS", "tags": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]}
    result = _parse_canonical_result(item, "raw")
    assert result["tags"] == ["a", "b", "c", "d", "e", "f", "g", "h"]


def test_canonicalize_events_batch(mock_anthropic):
    events = [
        CanonicalizeInput("Will BTC hit 100k?", None, None, 1, "native-1"),
        CanonicalizeInput("Who wins the election?", "US presidential race", "POLITICS", 1, "native-2"),
    ]
    result = canonicalize_events(mock_anthropic, events)
    assert (1, "native-1") in result
    assert (1, "native-2") in result
    for r in result.values():
        assert "title" in r
        assert "category" in r
        assert "tags" in r


def test_canonicalize_events_cache_hit(mock_anthropic):
    from scripts.testing import MemoryClassifierCache
    cache = MemoryClassifierCache()
    cached_result = {"title": "Cached Title", "category": "CRYPTO", "tags": ["btc", "price", "crypto"]}
    cache.put_canonicalization("claude-haiku-4-5-20251001", 1, "native-abc", cached_result)

    result = canonicalize_events(mock_anthropic, [CanonicalizeInput("raw title", None, None, 1, "native-abc")], cache=cache)

    mock_anthropic._client.messages.create.assert_not_called()
    assert result[(1, "native-abc")] == cached_result


def test_canonicalize_events_cache_miss_then_store(mock_anthropic):
    from scripts.testing import MemoryClassifierCache
    cache = MemoryClassifierCache()

    canonicalize_events(mock_anthropic, [CanonicalizeInput("raw title 2", None, None, 2, "native-xyz")], cache=cache)

    assert mock_anthropic._client.messages.create.called
    cached = cache.get_canonicalization("claude-haiku-4-5-20251001", 2, "native-xyz")
    assert cached is not None
    assert "title" in cached
