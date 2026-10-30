"""Polymarket International crypto markets written from fixed templates: the weekly price brackets on a Binance noon
candle (`bitcoin-price-on-november-3-2026`), and up or down over a short window on a Chainlink stream
(`sol-updown-15m-...`) or an hourly Binance candle (`bitcoin-up-or-down-october-29-2026-7pm-et`). The daily up or down
markets are left to the LLM: a tie there settles 50-50, so Down is not the negation of Up. So are the 4-hour ones,
which compare a TWAP with the opening price.
"""
import re
from datetime import datetime, timedelta, timezone

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability
from classifier.propositions.venue.common import EASTERN, number, price_rise

_ASSETS = {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH", "solana": "SOL", "sol": "SOL",
           "xrp": "XRP", "dogecoin": "DOGE", "doge": "DOGE", "hype": "HYPE", "bnb": "BNB"}
_SUBJECTS = {asset: f"{asset.lower()}-usd" for asset in set(_ASSETS.values())}

_BRACKETS_SLUG = re.compile(r"^(?P<asset>[a-z]+)-price-on-(?P<month>[a-z]+)-(?P<day>\d{1,2})-(?P<year>\d{4})$")
_SHORT_UPDOWN_SLUG = re.compile(r"^(?P<asset>[a-z]+)-updown-(?:5m|15m)-\d+$")
_HOURLY_UPDOWN_SLUG = re.compile(r"^(?P<asset>[a-z]+)-up-or-down-[a-z]+-\d{1,2}-\d{4}-\d{1,2}(?:am|pm)-et$")

_BRACKETS_RULE = re.compile(
    r'final "Close" price of the Binance 1 minute candle for (?P<pair>[A-Z]+)/USDT closing at 12:00 in the ET timezone '
    r"\(noon\) on the date specified in the title"
)
_HIGHER_BRACKET = "falls exactly between two brackets, then this market will resolve to the higher range bracket"
_CHAINLINK_RULE = re.compile(
    r'resolve to "Up" if the [\w ]+? price at the end of the time range specified in the title is greater than or equal '
    r"to the price at the beginning of that range\..*?Chainlink, specifically the (?P<pair>[A-Z]+)/USD data stream"
)
_BINANCE_HOURLY_RULE = re.compile(
    r'resolve to "Up" if the close price is greater than or equal to the open price for the (?P<pair>[A-Z]+)/USDT 1 hour '
    r"candle that begins on the time and date specified in the title"
)
_TITLE_WINDOW = re.compile(
    r"(?P<start>\d{1,2}(?::\d{2})?[AP]M)(?:-(?P<end>\d{1,2}(?::\d{2})?[AP]M))? ET$"
)
_BELOW = re.compile(r"^<\s?(?P<v>[\d,.]+)$")
_ABOVE = re.compile(r"^>\s?(?P<v>[\d,.]+)$")
_BETWEEN = re.compile(r"^(?P<low>[\d,.]+)\s?-\s?(?P<high>[\d,.]+)$")


def _slug(source: dict) -> str:
    return (source.get("event") or {}).get("slug") or ""


def _family(slug: str) -> tuple[str, re.Match] | None:
    for name, pattern in (("brackets", _BRACKETS_SLUG), ("short", _SHORT_UPDOWN_SLUG), ("hourly", _HOURLY_UPDOWN_SLUG)):
        if (m := pattern.match(slug)) and m.group("asset") in _ASSETS:
            return name, m
    return None


def claims(source: dict) -> bool:
    return _family(_slug(source)) is not None


def is_no_side(source: dict) -> bool:
    return source.get("outcome") in ("No", "Down")


def _utc(text: str | None) -> datetime | None:
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _is_eastern_clock(at: datetime, clock: str | None) -> bool:
    """Whether `at` reads `clock` ("7PM", "10:45AM") on an Eastern wall clock."""
    if clock is None:
        return False
    fmt = "%I:%M%p" if ":" in clock else "%I%p"
    wanted = datetime.strptime(clock, fmt)
    local = at.astimezone(EASTERN)
    return (local.hour, local.minute) == (wanted.hour, wanted.minute)


def _bracket(label: str) -> dict | None:
    """A bracket label as proposition fields. A price exactly on a boundary belongs to the higher bracket."""
    if m := _BELOW.match(label):
        return {"kind": Kind.THRESHOLD, "comparator": Comparator.LT, "value": number(m.group("v"))}
    if m := _ABOVE.match(label):
        return {"kind": Kind.THRESHOLD, "comparator": Comparator.GTE, "value": number(m.group("v"))}
    if m := _BETWEEN.match(label):
        return {"kind": Kind.RANGE, "low": number(m.group("low")), "high": number(m.group("high")), "high_inclusive": False}
    return None


def read(source: dict) -> Proposition | None:
    found = _family(_slug(source))
    if found is None:
        return None
    name, slug = found
    asset = _ASSETS[slug.group("asset")]
    market = source.get("market") or {}
    rules = re.sub(r"\s+", " ", market.get("description") or "")
    subject = _SUBJECTS[asset]

    if name == "brackets":
        m = _BRACKETS_RULE.search(rules)
        fields = _bracket(market.get("groupItemTitle") or source.get("outcome") or "")
        if m is None or m.group("pair") != asset or _HIGHER_BRACKET not in rules or fields is None:
            return None
        noon = datetime.strptime(f"{slug.group('month')} {slug.group('day')} {slug.group('year')} 12", "%B %d %Y %H")
        return Proposition(
            subject=subject, measure="price", unit="usd", observation=Observation.AT_TIME,
            window_end=noon.replace(tzinfo=EASTERN).astimezone(timezone.utc), stability=Stability.LIVE,
            resolution_source=f"binance-{asset.lower()}usdt-1m-close", **fields,
        )

    # The window is in the title in Eastern time; the market's own start and end times must say the same.
    start, end = _utc(market.get("eventStartTime")), _utc(market.get("endDate"))
    window = _TITLE_WINDOW.search((source.get("event") or {}).get("title") or "")
    if start is None or window is None or not _is_eastern_clock(start, window.group("start")):
        return None
    if name == "hourly":
        m = _BINANCE_HOURLY_RULE.search(rules)
        if m is None or m.group("pair") != asset or window.group("end"):
            return None
        return price_rise(subject, start, start + timedelta(hours=1), f"binance-{asset.lower()}usdt-1h-candle")
    m = _CHAINLINK_RULE.search(rules)
    if m is None or m.group("pair") != asset or end is None or not _is_eastern_clock(end, window.group("end")):
        return None
    return price_rise(subject, start, end, f"chainlink-{asset.lower()}usd-stream")
