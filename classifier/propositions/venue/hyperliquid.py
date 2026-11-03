"""Hyperliquid's structured outcomes, whose descriptions are `key:value|...` text: the recurring daily price binaries
(`class:priceBinary|underlying:BTC|expiry:20261002-0600|targetPrice:84277|period:1d`) and the price touches
(`template:priceTouch`, `perp:BTC|...|seconds:3|target:65000|time:20261201-0000`). Everything else is free text for
the LLM.

A touch resolves Yes once the perp's mark TWAP "equals or crosses" the target, from whichever side, so the rules
don't say which way it points. A touch settles as soon as it happens, so while one is open the price is still on the
side it started from: a target above the current mid is a rise to it, one below a fall. A target within
TOUCH_MIN_DISTANCE of the mid, or a perp without a mid, leaves the direction unknown.
"""
import re
from collections.abc import Mapping
from datetime import datetime, timezone

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability, subject_key
from classifier.propositions.venue.common import number

_VALUES = re.compile(r"(\w+):([^|]*)")
TOUCH_TEMPLATE = "template:priceTouch"
TOUCH_MIN_DISTANCE = 0.001


def _values(source: dict) -> dict[str, str]:
    return dict(_VALUES.findall((source.get("outcome") or {}).get("description", "")))


def is_touch(source: dict) -> bool:
    return (source.get("outcome") or {}).get("name") == TOUCH_TEMPLATE


def claims(source: dict) -> bool:
    return _values(source).get("class") == "priceBinary" or is_touch(source)


def is_no_side(source: dict) -> bool:
    return source.get("label") == "No"


def _expiry(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)


def _touch(values: dict[str, str], prices: Mapping[str, float]) -> Proposition | None:
    if not {"perp", "target", "time"} <= set(values):
        return None
    target, mid = number(values["target"]), prices.get(values["perp"])
    if target is None:
        return None
    direction = None
    if mid and abs(target - mid) > TOUCH_MIN_DISTANCE * mid:
        direction = Comparator.GTE if target > mid else Comparator.LTE
    return Proposition(
        subject=subject_key(f"{values['perp']}-usd"), kind=Kind.THRESHOLD, measure="price", unit="usd",
        comparator=direction, value=target, observation=Observation.ANY_TIME, window_end=_expiry(values["time"]),
        stability=Stability.LIVE, resolution_source=f"hyperliquid-mark-twap-{values.get('seconds', '3')}s",
        unknown=frozenset() if direction else frozenset({"comparator"}),
    )


def read(source: dict, prices: Mapping[str, float]) -> Proposition | None:
    values = _values(source)
    if is_touch(source):
        return _touch(values, prices)
    if values.get("class") != "priceBinary" or not {"underlying", "targetPrice", "expiry"} <= set(values):
        return None
    return Proposition(
        subject=subject_key(f"{values['underlying']}-usd"), kind=Kind.THRESHOLD, measure="price", unit="usd",
        comparator=Comparator.GT, value=number(values["targetPrice"]), observation=Observation.AT_TIME,
        window_end=_expiry(values["expiry"]), stability=Stability.LIVE, resolution_source="hyperliquid-twap-300s",
    )
