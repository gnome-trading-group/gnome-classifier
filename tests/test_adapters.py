from classifier.adapters import ADAPTERS


def test_symbol_prefixes_are_distinct_and_non_empty():
    prefixes = [adapter.symbol_prefix for adapter in ADAPTERS]
    assert all(prefixes)
    assert len(set(prefixes)) == len(prefixes)
