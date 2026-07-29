import pytest

from classifier.utils import generate_security_symbol


def test_no_expiry():
    assert generate_security_symbol("Will BTC hit $100k?", "Yes") == "WILL-BTC-HIT-100K-YES"


def test_with_expiry_includes_date_and_hour():
    symbol = generate_security_symbol(
        "Seattle vs Los Angeles D: Total Runs: Over 9.5 runs scored",
        "Yes",
        "2026-07-28T23:59:00Z",
    )
    assert symbol == "SEATTLE-VS-LOS-ANGELES-D-TOTAL-RUNS-OVER-95-RUNS-SCORED-20260728T23-YES"


def test_different_days_produce_different_symbols():
    symbol_a = generate_security_symbol("Team A vs Team B", "Yes", "2026-07-28T19:00:00Z")
    symbol_b = generate_security_symbol("Team A vs Team B", "Yes", "2026-07-29T19:00:00Z")
    assert symbol_a != symbol_b


def test_different_hours_produce_different_symbols():
    symbol_a = generate_security_symbol("Team A vs Team B", "Yes", "2026-07-28T13:00:00Z")
    symbol_b = generate_security_symbol("Team A vs Team B", "Yes", "2026-07-28T19:00:00Z")
    assert symbol_a != symbol_b


def test_same_hour_produces_same_symbol():
    symbol_a = generate_security_symbol("Team A vs Team B", "Yes", "2026-07-28T23:00:00Z")
    symbol_b = generate_security_symbol("Team A vs Team B", "Yes", "2026-07-28T23:59:00Z")
    assert symbol_a == symbol_b


def test_malformed_expiry_falls_back():
    symbol_with_bad_expiry = generate_security_symbol("Some Event", "Yes", "not-a-date")
    symbol_no_expiry = generate_security_symbol("Some Event", "Yes")
    assert symbol_with_bad_expiry == symbol_no_expiry


def test_none_expiry_behaves_like_no_expiry():
    assert generate_security_symbol("Some Event", "Yes", None) == generate_security_symbol("Some Event", "Yes")
