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
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability
from classifier.propositions.venue.common import (
    CF_INDICES,
    CMP_WORDS,
    cf_index_asset,
    eastern,
    esports_game_subject,
    esports_winner,
    names_team,
    number,
    price_rise,
    squash,
)

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


# Esports series whose match and map winners are read here, and the game tag their subjects take.
_ESPORTS_SERIES = {"cs2": ("CS2", "cs2")}
_ESPORTS_MATCH = "esports_match_winner"
_ESPORTS_MAP = re.compile(r"^esports_map_winner_(?P<map>\d+)$")
_SOURCED_FROM = re.compile(r"\bOutcome sourced from (?P<source>[A-Za-z]+)\.")


def _slug(source: dict) -> str:
    return (source.get("market") or {}).get("slug") or ""


def _esports_family(source: dict) -> tuple[str, str] | None:
    """(the rules' game name, the subject tag) for an esports match or map winner market."""
    series = _ESPORTS_SERIES.get((source.get("event") or {}).get("seriesSlug") or "")
    family = (source.get("market") or {}).get("sportsMarketType") or ""
    if series is None or not (family == _ESPORTS_MATCH or _ESPORTS_MAP.match(family)):
        return None
    return series


def claims(source: dict) -> bool:
    family = _FAMILY_SLUG.match(_slug(source))
    return bool(family and family.group("asset").upper() in CF_INDICES) or _esports_family(source) is not None


def is_no_side(source: dict) -> bool:
    # A match winner's two legs are its two teams, each read as its own team winning.
    return source.get("leg") == "short" and (source.get("market") or {}).get("sportsMarketType") != _ESPORTS_MATCH


def scoped_by_rules(source: dict) -> bool:
    """Whether the reader's scope comes from the rules, which name the match or the map exactly; an event title naming a
    map ("WRAITH PCIFIC vs. Morrow Esports  Map 2" over a match winner) then doesn't re-scope it."""
    return _esports_family(source) is not None


def _esports_winner(source: dict) -> Proposition | None:
    """A match winner ("Isurus vs RED Canids Academy", one leg per team) or a map winner ("PsychoFace wins map 1", Yes
    on the long leg), each only when the rules name the same match ("the winner of [Map 1 in] the A vs B CS2 match")
    and the outcomes or the title name one of its two teams. Anything else is left to the LLM."""
    game, tag = _esports_family(source)
    market = source.get("market") or {}
    rules = squash(market.get("description"))
    family = market.get("sportsMarketType")
    title = squash(market.get("title"))
    if family == _ESPORTS_MATCH:
        sides = source.get("sides") or []
        names = [squash(s.get("description")) for s in sides]
        try:
            outcomes = [squash(o) for o in json.loads(market.get("outcomes") or "[]")]
        except (TypeError, ValueError, AttributeError):
            return None
        long = [squash(s.get("description")) for s in sides if s.get("long") is True]
        if len(sides) != 2 or names != outcomes or names[0] == names[1] or len(long) != 1 \
                or title != f"{names[0]} vs {names[1]}":
            return None
        first, second = names
        if not rules.startswith(f"This market will settle to the winner of the {first} vs {second} {game} match "):
            return None
        team = long[0] if source.get("leg") == "long" else next(n for n in names if n != long[0])
        scope = "match"
    else:
        map_number = _ESPORTS_MAP.match(family).group("map")
        lead = re.match(rf"^This market will settle to the winner of Map {map_number} in the (?P<first>.+?) vs "
                        rf"(?P<second>.+?) {re.escape(game)} match scheduled for ", rules)
        if lead is None:
            return None
        teams = (lead.group("first"), lead.group("second"))
        team = next((t for t in teams if title == f"{t} wins map {map_number}"), None)
        if team is None or teams[0] == teams[1]:
            return None
        scope = f"map:{map_number}"
    slug = (source.get("event") or {}).get("slug") or ""
    start = _utc(market.get("gameStartTime"))
    subject = esports_game_subject(tag, slug)
    if subject is None or start is None:
        return None
    sourced = _SOURCED_FROM.search(rules)
    return esports_winner(subject, team, scope, start, sourced.group("source").lower() if sourced else None)


def _utc(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def read(source: dict, prices: Mapping[str, float]) -> Proposition | None:
    if _esports_family(source):
        return _esports_winner(source)
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


def _spread_side(market: dict, key: str) -> tuple[str, float] | None:
    """("first" or "second", the line) for the team a spread's margin key names, or None when it names neither, or
    both equally surely."""
    question = _SPREAD_QUESTION.match(market.get("question") or "")
    if question is None or not (market.get("sportsMarketType") or "").endswith("_spread"):
        return None
    slug = _SPREAD_SLUG.match(market.get("slug") or "")
    first = names_team(question.group("first"), slug and slug.group("first"), key)
    second = names_team(question.group("second"), slug and slug.group("second"), key)
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
