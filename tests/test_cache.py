import pytest

from scripts.testing import MemoryClassifierCache


@pytest.fixture
def cache():
    return MemoryClassifierCache()


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
