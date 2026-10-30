"""Hyperliquid's recurring daily price binaries, whose descriptions are structured `key:value|...` text
(`class:priceBinary|underlying:BTC|expiry:20261002-0600|targetPrice:84277|period:1d`). Everything else is free text
for the LLM."""
import re
from datetime import datetime, timezone

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability, subject_key
from classifier.propositions.venue.common import number

_VALUES = re.compile(r"(\w+):([^|]*)")


def _values(source: dict) -> dict[str, str]:
    return dict(_VALUES.findall((source.get("outcome") or {}).get("description", "")))


def claims(source: dict) -> bool:
    return _values(source).get("class") == "priceBinary"


def is_no_side(source: dict) -> bool:
    return source.get("label") == "No"


def read(source: dict) -> Proposition | None:
    values = _values(source)
    if values.get("class") != "priceBinary" or not {"underlying", "targetPrice", "expiry"} <= set(values):
        return None
    expiry = datetime.strptime(values["expiry"], "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
    return Proposition(
        subject=subject_key(f"{values['underlying']}-usd"), kind=Kind.THRESHOLD, measure="price", unit="usd",
        comparator=Comparator.GT, value=number(values["targetPrice"]), observation=Observation.AT_TIME,
        window_end=expiry, stability=Stability.LIVE, resolution_source="hyperliquid-twap-300s",
    )
