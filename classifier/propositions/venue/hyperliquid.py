"""Hyperliquid's structured outcomes, whose descriptions are `key:value|...` text: the recurring daily price binaries
(`class:priceBinary|underlying:BTC|expiry:20261002-0600|targetPrice:84277|period:1d`), the price touches
(`template:priceTouch`, `perp:BTC|...|seconds:3|target:65000|time:20261201-0000`) and the prices at a time
(`template:binaryPrice`, `perp:xyz:CL|priceDescription:1-minute candle|seconds:300|threshold:102|time:20261106-2100`,
"above" the threshold). Everything else is free text for the LLM.

A touch or a price at a time settles on a TWAP of what `priceDescription` names, over `seconds`: the perp's mark, its
1-minute candles, or an index. Markets on one perp can settle on different ones, so each is its own resolution
source.

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
BINARY_PRICE_TEMPLATE = "template:binaryPrice"
_ABOVE = re.compile(r"\bis above\b")
_SLUG = re.compile(r"[^a-z0-9]+")
TOUCH_MIN_DISTANCE = 0.001


def _values(source: dict) -> dict[str, str]:
    return dict(_VALUES.findall((source.get("outcome") or {}).get("description", "")))


def is_touch(source: dict) -> bool:
    return _template(source) == TOUCH_TEMPLATE


def _template(source: dict) -> str:
    return (source.get("outcome") or {}).get("name") or ""


def claims(source: dict) -> bool:
    return _values(source).get("class") == "priceBinary" or _template(source) in (TOUCH_TEMPLATE, BINARY_PRICE_TEMPLATE)


def _resolution_source(values: dict[str, str]) -> str:
    perp = _SLUG.sub("-", values["perp"].lower()).strip("-")
    described = values.get("priceDescription", "").lower()
    if described.endswith("perp mark"):
        what = f"hyperliquid-{perp}-mark"
    elif described == "1-minute candle":
        what = f"hyperliquid-{perp}-1m-candle"
    else:
        what = _SLUG.sub("-", described.removeprefix("the ").removesuffix(" price")).strip("-")
    return f"{what}-twap-{values.get('seconds', '?')}s"


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
        stability=Stability.LIVE, resolution_source=_resolution_source(values),
        unknown=frozenset() if direction else frozenset({"comparator"}),
    )


def _binary_price(values: dict[str, str], rules: str) -> Proposition | None:
    if not {"perp", "threshold", "time"} <= set(values) or not _ABOVE.search(rules):
        return None
    threshold = number(values["threshold"])
    if threshold is None:
        return None
    return Proposition(
        subject=subject_key(f"{values['perp']}-usd"), kind=Kind.THRESHOLD, measure="price", unit="usd",
        comparator=Comparator.GT, value=threshold, observation=Observation.AT_TIME, window_end=_expiry(values["time"]),
        stability=Stability.LIVE, resolution_source=_resolution_source(values),
    )


def read(source: dict, prices: Mapping[str, float]) -> Proposition | None:
    values = _values(source)
    if is_touch(source):
        return _touch(values, prices)
    if _template(source) == BINARY_PRICE_TEMPLATE:
        return _binary_price(values, source.get("description") or "")
    if values.get("class") != "priceBinary" or not {"underlying", "targetPrice", "expiry"} <= set(values):
        return None
    return Proposition(
        subject=subject_key(f"{values['underlying']}-usd"), kind=Kind.THRESHOLD, measure="price", unit="usd",
        comparator=Comparator.GT, value=number(values["targetPrice"]), observation=Observation.AT_TIME,
        window_end=_expiry(values["expiry"]), stability=Stability.LIVE, resolution_source="hyperliquid-twap-300s",
    )
