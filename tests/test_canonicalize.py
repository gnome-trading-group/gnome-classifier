import json
from unittest.mock import MagicMock

import pytest

from classifier.stages.canonicalize import canonicalize_events, _parse_canonical_result
from classifier.types import CanonicalizeInput


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


def test_canonicalize_events_cache_hit(mock_anthropic, s3_bucket):
    from classifier.cache import S3ClassifierCache
    cache = S3ClassifierCache(s3_bucket)
    cached_result = {"title": "Cached Title", "category": "CRYPTO", "tags": ["btc", "price", "crypto"]}
    cache.put_canonicalization("claude-haiku-4-5-20251001", 1, "native-abc", cached_result)

    result = canonicalize_events(mock_anthropic, [CanonicalizeInput("raw title", None, None, 1, "native-abc")], cache=cache)

    mock_anthropic._client.messages.create.assert_not_called()
    assert result[(1, "native-abc")] == cached_result


def test_canonicalize_events_cache_miss_then_store(mock_anthropic, s3_bucket):
    from classifier.cache import S3ClassifierCache
    cache = S3ClassifierCache(s3_bucket)

    canonicalize_events(mock_anthropic, [CanonicalizeInput("raw title 2", None, None, 2, "native-xyz")], cache=cache)

    assert mock_anthropic._client.messages.create.called
    cached = cache.get_canonicalization("claude-haiku-4-5-20251001", 2, "native-xyz")
    assert cached is not None
    assert "title" in cached
