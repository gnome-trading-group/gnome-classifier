import pytest
from moto import mock_aws

from classifier.cache import ClassifierCache


@pytest.fixture
def cache(s3_bucket):
    return ClassifierCache(s3_bucket)


def test_canonicalization_miss(cache):
    assert cache.get_canonicalization("model", 1, "native-id") is None


def test_canonicalization_roundtrip(cache):
    result = {"title": "Canonical Title", "category": "POLITICS", "tags": ["a", "b", "c"]}
    cache.put_canonicalization("model", 1, "native-abc", result)
    assert cache.get_canonicalization("model", 1, "native-abc") == result


def test_canonicalization_model_in_key(cache):
    result = {"title": "T", "category": "OTHER", "tags": ["x", "y", "z"]}
    cache.put_canonicalization("model-v1", 1, "native-abc", result)
    assert cache.get_canonicalization("model-v2", 1, "native-abc") is None


def test_judgment_miss(cache):
    assert cache.get_judgment("model", "A", ["Yes"], "B", ["Yes"]) is None


def test_judgment_roundtrip(cache):
    items = [{"first_label": "Yes", "second_label": "Yes", "type": "EQUIVALENT", "confidence": 0.95}]
    cache.put_judgment("model", "Event A", ["Yes"], "Event B", ["Yes"], items, a_is_first=True)
    result = cache.get_judgment("model", "Event A", ["Yes"], "Event B", ["Yes"])
    assert result is not None
    cached_items, a_is_first = result
    assert cached_items == items
    assert a_is_first is True


def test_judgment_key_is_symmetric(cache):
    items = [{"first_label": "Yes", "second_label": "Yes", "type": "EQUIVALENT", "confidence": 0.9}]
    cache.put_judgment("model", "Alpha", ["Yes"], "Beta", ["Yes"], items, a_is_first=True)
    result = cache.get_judgment("model", "Beta", ["Yes"], "Alpha", ["Yes"])
    assert result is not None
    cached_items, a_is_first = result
    assert cached_items == items
    assert a_is_first is False


def test_judgment_model_in_key(cache):
    items = [{"first_label": "Yes", "second_label": "Yes", "type": "EQUIVALENT", "confidence": 0.9}]
    cache.put_judgment("model-v1", "A", ["Yes"], "B", ["Yes"], items, a_is_first=True)
    assert cache.get_judgment("model-v2", "A", ["Yes"], "B", ["Yes"]) is None
