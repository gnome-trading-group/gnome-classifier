import pytest
from datetime import timedelta

from classifier.utils import expiry_close, generate_security_symbol


def test_expiry_close_both_none():
    assert expiry_close(None, None, timedelta(hours=1)) is True


def test_expiry_close_a_none_returns_false():
    assert expiry_close(None, "2026-11-03T00:00:00Z", timedelta(hours=1)) is False


def test_expiry_close_b_none_returns_false():
    assert expiry_close("2026-11-03T00:00:00Z", None, timedelta(hours=1)) is False


def test_expiry_close_malformed_date_returns_false():
    assert expiry_close("bad-date", "2026-11-03T00:00:00Z", timedelta(hours=1)) is False


def test_expiry_close_within_tolerance():
    assert expiry_close("2026-11-03T00:00:00Z", "2026-11-03T00:30:00Z", timedelta(hours=1)) is True


def test_expiry_close_outside_tolerance():
    assert expiry_close("2026-11-03T00:00:00Z", "2026-11-04T12:00:00Z", timedelta(hours=1)) is False


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
