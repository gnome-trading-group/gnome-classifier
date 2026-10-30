"""Polymarket US crypto markets, which settle on the CF Benchmarks real-time indices Kalshi uses and are written from
fixed templates: up or down over a window, the price at a time (at or above, or in a range), and touching a level
within a window. Market slugs name the family (`cpc-btc-updown-15m-...`, `cpceth-eth-above-hr-...`); everything else on
the venue is free text for the LLM.
"""
import re

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability
from classifier.propositions.venue.common import CF_INDICES, CMP_WORDS, cf_index_asset, eastern, number, price_rise

_FAMILY_SLUG = re.compile(
    r"^cpc[a-z]*-(?P<asset>[a-z]+)-(?:updown-\d+[mhd]|above-(?:hr|wk|yr)|range-(?:hr|wk)|pricerange-yr|"
    r"hit-[hl]-[wm]|hitprice-(?:high|low)-yr|\d+k)-"
)
_TIME = r"\d{1,2}(?::\d{2})?\s?[AP]M"
_DATE = r"[A-Z][a-z]+ \d{1,2}, \d{4}"
_INDEX = re.compile(r"\((?P<code>BRTI|ERTI|[A-Z]+USD_?RTI)\)")
_UPDOWN = re.compile(
    rf"settle to Up if the [\w ]+? price at (?P<end_time>{_TIME}) ET on (?P<end_date>{_DATE}) is greater than or equal to "
    rf"the price at (?P<start_time>{_TIME}) ET on (?P<start_date>{_DATE})\."
)
_AT_TIME = re.compile(
    rf"settle to Yes if the price of [\w ]+? is (?P<cmp>at or above|at or below|above|below) \$(?P<v>[\d,.]+) "
    rf"at (?P<time>{_TIME}) ET on (?P<date>{_DATE})\."
)
_EDGE_BRACKET = re.compile(
    rf"settle to Yes if the price of [\w ]+? is \$?(?P<v>[\d,.]+) or (?P<side>above|below) at (?P<time>{_TIME}) ET on "
    rf"(?P<date>{_DATE})\."
)
_RANGE = re.compile(
    rf"settle to Yes if the price of [\w ]+? is between \$?(?P<low>[\d,.]+) (?:and|to) \$?(?P<high>[\d,.]+)"
    rf"(?:, inclusive of both bounds)? at (?P<time>{_TIME}) ET on (?P<date>{_DATE})\."
)
_TOUCH = re.compile(
    rf"settle to Yes if the price of [\w ]+? is (?P<cmp>at or above|at or below|above|below) "
    rf"(?:\$(?P<v>[\d,.]+)|the price specified in the title) at any point "
    rf"(?:from (?:the later of this market's listing and )?(?P<start_time>{_TIME}) ET on (?P<start_date>{_DATE}) until|"
    rf"from the creation of this market until|after the creation of this market and before|before) "
    rf"(?P<end_time>{_TIME}) ET on (?P<end_date>{_DATE})"
)
_TITLE_PRICE = re.compile(r"\$(?P<v>[\d,.]+)")


def _slug(source: dict) -> str:
    return (source.get("market") or {}).get("slug") or ""


def claims(source: dict) -> bool:
    family = _FAMILY_SLUG.match(_slug(source))
    return bool(family and family.group("asset").upper() in CF_INDICES)


def is_no_side(source: dict) -> bool:
    return source.get("leg") == "short"


def read(source: dict) -> Proposition | None:
    family = _FAMILY_SLUG.match(_slug(source))
    market = source.get("market") or {}
    rules = re.sub(r"\s+", " ", market.get("description") or "")
    index = _INDEX.search(rules)
    if family is None or index is None:
        return None
    asset = cf_index_asset(index.group("code"))
    # The slug and the rules must agree on the asset, or one of them is not what this reader thinks it is.
    if asset is None or asset != family.group("asset").upper():
        return None
    subject, resolution = CF_INDICES[asset]
    # The yearly markets settle on a trimmed mean of the index rather than its simple average.
    if "trimmed mean" in rules:
        resolution = f"{resolution}-trimmed-mean"

    if m := _UPDOWN.search(rules):
        return price_rise(subject, eastern(m.group("start_date"), m.group("start_time")),
                          eastern(m.group("end_date"), m.group("end_time")), resolution)
    common = {"subject": subject, "measure": "price", "unit": "usd", "stability": Stability.LIVE,
              "resolution_source": resolution}
    if m := _TOUCH.search(rules):
        title_price = _TITLE_PRICE.search(market.get("title") or "")
        value = m.group("v") or (title_price.group("v") if title_price else None)
        start = eastern(m.group("start_date"), m.group("start_time")) if m.group("start_date") else None
        return Proposition(
            kind=Kind.THRESHOLD, comparator=CMP_WORDS[m.group("cmp")], value=number(value) if value else None,
            observation=Observation.ANY_TIME, window_start=start,
            window_end=eastern(m.group("end_date"), m.group("end_time")),
            unknown=frozenset() if value else frozenset({"value"}), **common,
        )
    if m := _RANGE.search(rules):
        return Proposition(kind=Kind.RANGE, low=number(m.group("low")), high=number(m.group("high")),
                           observation=Observation.AT_TIME, window_end=eastern(m.group("date"), m.group("time")), **common)
    if m := _EDGE_BRACKET.search(rules):
        comparator = Comparator.GTE if m.group("side") == "above" else Comparator.LTE
        return Proposition(kind=Kind.THRESHOLD, comparator=comparator, value=number(m.group("v")),
                           observation=Observation.AT_TIME, window_end=eastern(m.group("date"), m.group("time")), **common)
    if m := _AT_TIME.search(rules):
        return Proposition(kind=Kind.THRESHOLD, comparator=CMP_WORDS[m.group("cmp")], value=number(m.group("v")),
                           observation=Observation.AT_TIME, window_end=eastern(m.group("date"), m.group("time")), **common)
    return None
