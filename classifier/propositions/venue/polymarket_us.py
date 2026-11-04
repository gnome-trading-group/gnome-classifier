"""Polymarket US crypto markets, which settle on the CF Benchmarks real-time indices Kalshi uses and are written from
fixed templates: up or down over a window, the price at a time (at or above, or in a range), and touching a level
within a window. Market slugs name the family (`cpc-btc-updown-15m-...`, `cpceth-eth-above-hr-...`); everything else on
the venue is free text for the LLM.

Game spreads are read by the LLM (only it can key the teams the way the game's other venues do) and corrected here:
their title names whichever team the line favours ("NC State wins by over 10.5 points") while the market asks whether
the event's first team covers it ("Will the California cover 10.5 vs the NC State"), so the model reads about half of
them on the wrong side.
"""
import dataclasses
import re
from collections.abc import Mapping

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
_SPREAD_QUESTION = re.compile(
    r"^Will (?:the )?(?P<first>.+?) cover (?P<line>[+-]?\d+(?:\.\d+)?) vs (?:the )?(?P<second>.+?) in "
)
# `asc-cfb-cah-ncst-2026-10-31-neg-3pt5`: the event's two team codes, first team first.
_SPREAD_SLUG = re.compile(r"^[a-z]+-[a-z0-9]+-(?P<first>[a-z0-9]+)-(?P<second>[a-z0-9]+)-\d{4}-")


def _slug(source: dict) -> str:
    return (source.get("market") or {}).get("slug") or ""


def claims(source: dict) -> bool:
    family = _FAMILY_SLUG.match(_slug(source))
    return bool(family and family.group("asset").upper() in CF_INDICES)


def is_no_side(source: dict) -> bool:
    return source.get("leg") == "short"


def read(source: dict, prices: Mapping[str, float]) -> Proposition | None:
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


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _abbreviates(key: str, name: str) -> bool:
    """Whether a key is an abbreviation of a name: its letters in order, starting with the name's first (`orst`, Oregon
    State), a trailing u for University dropped (`bsu`, Boise State)."""
    for short in {key, key.removesuffix("u") if len(key) > 2 else key}:
        rest = iter(name)
        if short[0] == name[0] and all(ch in rest for ch in short):
            return True
    return False


def _names(team: str, code: str | None, key: str) -> int:
    """How surely an entity key the model chose stands for a team: 2 for the start of its venue code (`gt`, `gtech`)
    or of its name, its initials (`gt`, Georgia Tech) or one of its words (`lakers`, LA Lakers); 1 for an
    abbreviation of it; 0 for neither."""
    flat, name = _flat(key), _flat(team)
    words = re.findall(r"[a-z0-9]+", team.lower())
    if len(flat) < 2 or not name:
        return 0
    if (code and (code.startswith(flat) or flat.startswith(code)) or name.startswith(flat) or flat.startswith(name)
            or flat == "".join(w[0] for w in words) or flat in words):
        return 2
    return 1 if _abbreviates(flat, name) else 0


def _spread_side(market: dict, key: str) -> tuple[str, float] | None:
    """("first" or "second", the line) for the team a spread's margin key names, or None when it names neither, or
    both equally surely."""
    question = _SPREAD_QUESTION.match(market.get("question") or "")
    if question is None or not (market.get("sportsMarketType") or "").endswith("_spread"):
        return None
    slug = _SPREAD_SLUG.match(market.get("slug") or "")
    first = _names(question.group("first"), slug and slug.group("first"), key)
    second = _names(question.group("second"), slug and slug.group("second"), key)
    if first == second:
        return None
    return ("first" if first > second else "second"), float(question.group("line"))


def correct(source: dict, proposition: Proposition) -> Proposition:
    """A game spread as its market asks it: Yes (the long side) when the event's first team covers the line, that is
    wins by more than minus the line; the short side is the No. The margin stays keyed by the team the model chose."""
    if proposition.kind != Kind.THRESHOLD or not (proposition.measure or "").startswith("margin:"):
        return proposition
    market = source.get("market") or {}
    if _SPREAD_QUESTION.match(market.get("question") or "") is None:
        return proposition
    side = _spread_side(market, proposition.measure.partition(":")[2])
    if side is None:
        # A margin the model keyed to neither team can't be put on a side: no relationship rather than a wrong one.
        return dataclasses.replace(proposition, unknown=proposition.unknown | {"measure"})
    team, line = side
    comparator, value = (Comparator.GT, -line) if team == "first" else (Comparator.LT, line)
    return dataclasses.replace(proposition, comparator=comparator, value=value, low=None, high=None,
                               negated=source.get("leg") == "short",
                               unknown=proposition.unknown - {"comparator", "value"})
