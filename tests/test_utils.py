import pytest

from classifier.utils import format_security_symbol


def test_kalshi_binary_symbol():
    assert format_security_symbol("KX", "KXWTI15M-26AUG240015-15", "Yes") == "KX-KXWTI15M-26AUG240015-15-YES"


def test_kalshi_ticker_keeps_decimal_point():
    assert format_security_symbol("KX", "KXBTCD-26AUG0517-T117999.99") == "KX-KXBTCD-26AUG0517-T117999.99"


def test_polymarket_slug_and_outcome():
    assert format_security_symbol("PM_I", "will-btc-hit-100k-in-2026", "Yes") == "PM_I-WILL-BTC-HIT-100K-IN-2026-YES"


def test_special_characters_collapse_to_single_dash():
    assert format_security_symbol("HL", "123", "Over 5.5 / Under") == "HL-123-OVER-5.5-UNDER"


def test_prefix_is_not_sanitized():
    assert format_security_symbol("PM_I", "abc").startswith("PM_I-")


@pytest.mark.parametrize("window", ["26AUG240015-15", "26AUG240045-45"])
def test_distinct_kalshi_windows_get_distinct_symbols(window):
    other = "26AUG240045-45" if window == "26AUG240015-15" else "26AUG240015-15"
    assert format_security_symbol("KX", f"KXWTI15M-{window}", "Yes") != format_security_symbol("KX", f"KXWTI15M-{other}", "Yes")
