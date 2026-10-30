"""Parsing helpers the venue readers share."""
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability

EASTERN = ZoneInfo("America/New_York")
CMP_WORDS = {"above": Comparator.GT, "below": Comparator.LT, "at or above": Comparator.GTE, "at or below": Comparator.LTE}

# Asset -> (subject, resolution source) for the CF Benchmarks real-time indices Kalshi and Polymarket US settle on.
CF_INDICES = {
    "BTC": ("btc-usd", "cf-benchmarks-brti"),
    "ETH": ("eth-usd", "cf-benchmarks-erti"),
    "SOL": ("sol-usd", "cf-benchmarks-solusd-rti"),
    "XRP": ("xrp-usd", "cf-benchmarks-xrpusd-rti"),
    "DOGE": ("doge-usd", "cf-benchmarks-dogeusd-rti"),
    "HYPE": ("hype-usd", "cf-benchmarks-hypeusd-rti"),
}
# Venues name the same index several ways: BRTI, ERTI, ETHUSDRTI, ETHUSD_RTI.
_INDEX_CODE_ASSETS = {"BRTI": "BTC", "ERTI": "ETH"}


def cf_index_asset(code: str) -> str | None:
    code = code.replace("_", "").upper()
    asset = _INDEX_CODE_ASSETS.get(code) or (code.removesuffix("USDRTI") if code.endswith("USDRTI") else None)
    return asset if asset in CF_INDICES else None


def number(text: str) -> float:
    return float(text.replace(",", ""))


def eastern(date_text: str, time_text: str) -> datetime:
    """A US Eastern date ("Oct 29, 2026" / "October 29, 2026") and time ("3 PM", "3:30PM") in UTC, summer or winter."""
    time_text = re.sub(r"\s*([ap])\.?m\.?$", r" \1M", time_text.strip(), flags=re.IGNORECASE).upper()
    time_format = "%I:%M %p" if ":" in time_text else "%I %p"
    for date_format in ("%b %d, %Y", "%B %d, %Y"):
        try:
            local = datetime.strptime(f"{date_text} {time_text}", f"{date_format} {time_format}")
        except ValueError:
            continue
        return local.replace(tzinfo=EASTERN).astimezone(timezone.utc)
    raise ValueError(f"not an Eastern date and time: {date_text!r} {time_text!r}")


def comparison(match: re.Match) -> dict:
    """Proposition fields from a match with `cmp` (above, below, at or above, at or below, between), `v1` and `v2`."""
    words = match.group("cmp")
    if words == "between":
        return {"kind": Kind.RANGE, "low": number(match.group("v1")), "high": number(match.group("v2"))}
    return {"kind": Kind.THRESHOLD, "comparator": CMP_WORDS[words], "value": number(match.group("v1"))}


def price_rise(subject: str, start: datetime, end: datetime, resolution_source: str) -> Proposition:
    """"Up or down" markets: the price at `end` is at least the price at `start`. Neither price is known when the market
    lists, so the question is the change over the window, not a threshold on the price."""
    return Proposition(
        subject=subject, kind=Kind.THRESHOLD, measure="change", unit="usd", comparator=Comparator.GTE, value=0.0,
        observation=Observation.AT_TIME, window_start=start, window_end=end, stability=Stability.LIVE,
        resolution_source=resolution_source,
    )
