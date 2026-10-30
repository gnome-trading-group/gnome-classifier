"""Kalshi markets read by code: crypto and S&P snapshots (CF Benchmarks indices, 15-minute windows), and per-game
markets keyed by the ticker's game segment (totals, first halves, team totals, spreads, winners, map winners, player
props, tennis games). A game's markets share one subject, `game:<league>:<date>:<teams>[-<hhmm>]`, so its ladders
compare with each other.
"""
import re

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability, subject_key
from classifier.propositions.venue.common import CF_INDICES, comparison, eastern, number, price_rise

_MONTHS = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}

_KALSHI_CRYPTO_SERIES = re.compile(r"^KX(?P<asset>[A-Z]+?)D?$")
_KALSHI_CRYPTO_15M_SERIES = re.compile(r"^KX(?P<asset>[A-Z]+)15M$")
_KALSHI_INDEX = {"KXINXU": ("spx", "sp-dow-jones-indices"), "KXINX": ("spx", "sp-dow-jones-indices")}

_CRYPTO_RULE = re.compile(
    r"CF Benchmarks' (?:[\w ()'-]*?Real[- ]Time Index \(\w+\)|\w+) before (?P<time>\d{1,2}(?::\d{2})? [AP]M) E[DS]T is "
    r"(?P<cmp>above|below|at or above|at or below|between) \$?(?P<v1>[\d,.]+)(?:(?: and |-)\$?(?P<v2>[\d,.]+))? "
    r"at \d{1,2}(?::\d{2})? [AP]M E[DS]T on (?P<date>\w{3} \d{1,2}, \d{4})"
)
_INDEX_RULE = re.compile(
    r"end-of-day S&P 500 index value on (?P<date>\w+ \d{1,2}, \d{4}) is "
    r"(?P<cmp>above|below|at or above|at or below|between) (?P<v1>[\d,.]+)(?:(?: and |-)(?P<v2>[\d,.]+))?"
)
_INDEX_INTRADAY_RULE = re.compile(
    r"S&P 500 index value on (?P<date>\w{3} \d{1,2}, \d{4}) at (?P<time>\d{1,2}(?::\d{2})? ?[ap]m) E[DS]T is "
    r"(?P<cmp>above|below|at or above|at or below|between) (?P<v1>[\d,.]+)(?:(?: and |-)(?P<v2>[\d,.]+))?"
)
# Kalshi's strike fields and its rules can name the same boundary from either side ("above 7574.9999" vs ">= 7575").
_STRIKE_TOLERANCE = 0.011
_TOTAL_RULE = re.compile(
    r"(?:collectively score more(?: than)?|is more than|If over) (?P<v>[\d.]+) (?:total )?(?P<unit>runs|points|goals|maps)"
)
_TENNIS_TOTAL_RULE = re.compile(r"number of completed games in the full match is above (?P<v>[\d.]+)")
_TENNIS_SPREAD_RULE = re.compile(r"game differential in favor of .+? across the full match is above (?P<v>[\d.]+) games")
_TEAM_TOTAL_RULE = re.compile(r"(?:scores (?:over|more than)|records above) (?P<v>[\d.]+) (?P<unit>runs|points|goals)")
# Kalshi's 15-minute markets: the index at the window's end is at least the index at its start.
_CRYPTO_15M_RULE = re.compile(
    r"CF Benchmarks' \w+ before (?P<end_time>\d{1,2}(?::\d{2})? [AP]M) E[DS]T on (?P<end_date>\w{3} \d{1,2}, \d{4}) is at "
    r"least the simple average of the sixty seconds of CF Benchmarks' \w+ before (?P<start_time>\d{1,2}(?::\d{2})? [AP]M) "
    r"E[DS]T on (?P<start_date>\w+ \d{1,2}, \d{4})"
)
# Per-game player props: "If Mack Hollins records 2+ receptions in the New England vs Seattle ... game".
_PROP_RULE = re.compile(
    r"If (?P<player>.+?) (?:records|scores at least) (?P<v>\d+)\+ (?P<stat>[A-Za-z +-]+?) in (?:the )?.+? "
    r"(?:game|match)(?: \d of the double header)? originally scheduled"
)
# A prop series is <league><stat>; anything else ending in a stat code (KXNFLFIRSTTD, KXNFLFFPTS) is another question.
_PROP_LEAGUES = {"NFL", "NBA", "WNBA", "MLB", "NHL", "NCAAF", "NCAAB", "NCAAMB", "NCAAWB", "MLS", "EPL"}
_PROP_SUFFIXES = ("RECYDS", "RSHYDS", "PASSYDS", "PASSTD", "REC", "TD", "HRR", "HR", "HIT", "PTS", "REB", "AST", "KS", "SOG",
                  "RBI", "SB", "TB", "3PT")
# Tennis games markets name the tour with a trailing G (KXATPGTOTAL); their match winners don't.
_TENNIS_GAMES = re.compile(r"^(?P<tour>ATP|WTA|ATPCHALLENGER|WTACHALLENGER)G$")
_SPREAD_RULE = re.compile(r"wins? by (?:more than|over) (?P<v>[\d.]+) (?P<unit>runs|points|goals|maps)")
_GAME_SEGMENT = re.compile(r"^(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hhmm>\d{4})?(?P<teams>[A-Z][A-Z0-9]+)$")
_FAMILY_SUFFIXES = ("TEAMTOTAL", "1HTOTAL", "TOTAL", "SPREAD", "GAME", "MAP") + _PROP_SUFFIXES

_STRIKE_TYPES = {"greater": Comparator.GT, "greater_or_equal": Comparator.GTE, "less": Comparator.LT,
                 "less_or_equal": Comparator.LTE}


def _close(a, b) -> bool:
    return a is not None and abs(float(a) - b) <= _STRIKE_TOLERANCE


def _agrees_with_strikes(market: dict, parts: dict) -> bool:
    """A strike field the venue sent must say what the rules say: the same direction and the same boundary."""
    strike_type = market.get("strike_type")
    if parts["kind"] == Kind.THRESHOLD and strike_type in _STRIKE_TYPES:
        strike = _STRIKE_TYPES[strike_type]
        upward = strike in (Comparator.GT, Comparator.GTE)
        if upward != (parts["comparator"] in (Comparator.GT, Comparator.GTE)):
            return False
        return _close(market.get("floor_strike" if upward else "cap_strike"), parts["value"])
    if parts["kind"] == Kind.RANGE and strike_type == "between":
        return _close(market.get("floor_strike"), parts["low"]) and _close(market.get("cap_strike"), parts["high"])
    return True


def _with_strike_check(market: dict, parts: dict) -> dict:
    if _agrees_with_strikes(market, parts):
        return parts
    unknown = {"comparator", "value"} if parts["kind"] == Kind.THRESHOLD else {"low", "high"}
    return {**parts, "unknown": frozenset(unknown)}


def read(source: dict) -> Proposition | None:
    market = source.get("market") or {}
    ticker = market.get("ticker") or ""
    rules = market.get("rules_primary") or ""
    series, _, rest = ticker.partition("-")

    crypto = _KALSHI_CRYPTO_SERIES.match(series)
    if crypto and crypto.group("asset") in CF_INDICES and (m := _CRYPTO_RULE.search(rules)):
        subject, resolution = CF_INDICES[crypto.group("asset")]
        at = eastern(m.group("date"), m.group("time"))
        return Proposition(
            subject=subject, measure="price", unit="usd", observation=Observation.AT_TIME, window_end=at,
            stability=Stability.LIVE, resolution_source=resolution, **_with_strike_check(market, comparison(m)),
        )
    crypto_15m = _KALSHI_CRYPTO_15M_SERIES.match(series)
    if crypto_15m and crypto_15m.group("asset") in CF_INDICES and (m := _CRYPTO_15M_RULE.search(rules)):
        subject, resolution = CF_INDICES[crypto_15m.group("asset")]
        return price_rise(subject, eastern(m.group("start_date"), m.group("start_time")),
                          eastern(m.group("end_date"), m.group("end_time")), resolution)
    if series in _KALSHI_INDEX and (m := _INDEX_RULE.search(rules)):
        subject, resolution = _KALSHI_INDEX[series]
        close = eastern(m.group("date"), "4 PM")
        return Proposition(
            subject=subject, measure="close", unit="points", observation=Observation.AT_TIME, window_end=close,
            stability=Stability.LIVE, resolution_source=resolution,
            **_with_strike_check(market, comparison(m)),
        )
    if series in _KALSHI_INDEX and (m := _INDEX_INTRADAY_RULE.search(rules)):
        subject, resolution = _KALSHI_INDEX[series]
        at = eastern(m.group("date"), m.group("time"))
        return Proposition(
            subject=subject, measure="price", unit="points", observation=Observation.AT_TIME, window_end=at,
            stability=Stability.LIVE, resolution_source=resolution, **_with_strike_check(market, comparison(m)),
        )
    return _kalshi_game(series, rest, market, rules)


def _kalshi_crypto_claimed(series: str) -> bool:
    crypto = _KALSHI_CRYPTO_SERIES.match(series) or _KALSHI_CRYPTO_15M_SERIES.match(series)
    return bool(crypto and crypto.group("asset") in CF_INDICES)


def _game_subject(series: str, event_segment: str) -> tuple[str, str] | None:
    family = next((s for s in _FAMILY_SUFFIXES if series.endswith(s)), None)
    m = _GAME_SEGMENT.match(event_segment)
    if family is None or m is None or m.group("mon") not in _MONTHS:
        return None
    sport = series.removeprefix("KX").removesuffix(family)
    if family in _PROP_SUFFIXES and sport not in _PROP_LEAGUES:
        return None
    # A first-half market (KX<league>1HSPREAD) is the same game; its scope comes from the rules.
    sport = sport.removesuffix("1H")
    if tennis := _TENNIS_GAMES.match(sport):
        sport = tennis.group("tour")
    date = f"20{m.group('yy')}-{_MONTHS[m.group('mon')]:02d}-{m.group('dd')}"
    parts = ["game", sport, date, m.group("teams") + (f"-{m.group('hhmm')}" if m.group("hhmm") else "")]
    return subject_key(*parts), family


def _team_code(segment: str) -> str:
    return re.sub(r"\d+$", "", segment).lower()


def _kalshi_game(series: str, rest: str, market: dict, rules: str) -> Proposition | None:
    segments = rest.split("-")
    found = _game_subject(series, segments[0]) if segments and segments[0] else None
    if found is None:
        return None
    subject, family = found
    if family == "1HTOTAL" or "1st half" in rules.lower():
        scope = "first_half"
    elif "regulation" in (rules + " " + (market.get("title") or "")).lower() or "after 90 minutes" in rules:
        scope = "regulation"
    else:
        scope = "match"

    if family == "TOTAL" and (m := _TENNIS_TOTAL_RULE.search(rules)):
        return Proposition(subject=subject, kind=Kind.THRESHOLD, measure="total_games", comparator=Comparator.GT,
                           value=float(m.group("v")), unit="games", observation=Observation.CUMULATIVE, scope=scope)
    if family == "SPREAD" and len(segments) > 1 and (m := _TENNIS_SPREAD_RULE.search(rules)):
        return Proposition(subject=subject, kind=Kind.THRESHOLD, measure=f"margin:{_team_code(segments[1])}",
                           comparator=Comparator.GT, value=float(m.group("v")), unit="games",
                           observation=Observation.CUMULATIVE, scope=scope)
    if family in _PROP_SUFFIXES and (m := _PROP_RULE.search(rules)):
        stat = re.sub(r"[^a-z0-9]+", "_", m.group("stat").lower().removeprefix("total ")).strip("_")
        player = re.sub(r"[^a-z0-9.]+", "-", m.group("player").lower()).strip("-")
        return Proposition(subject=subject, kind=Kind.THRESHOLD, measure=f"{stat}:{player}", comparator=Comparator.GTE,
                           value=float(m.group("v")), unit=stat, observation=Observation.CUMULATIVE, scope=scope)
    if family in ("TOTAL", "1HTOTAL") and (m := _TOTAL_RULE.search(rules)):
        return Proposition(subject=subject, kind=Kind.THRESHOLD, measure=f"total_{m.group('unit')}",
                           comparator=Comparator.GT, value=float(m.group("v")), unit=m.group("unit"),
                           observation=Observation.CUMULATIVE, scope=scope)
    if family == "TEAMTOTAL" and len(segments) > 1 and (m := _TEAM_TOTAL_RULE.search(rules)):
        return Proposition(subject=subject, kind=Kind.THRESHOLD, measure=f"team_{m.group('unit')}:{_team_code(segments[1])}",
                           comparator=Comparator.GT, value=float(m.group("v")), unit=m.group("unit"),
                           observation=Observation.CUMULATIVE, scope=scope)
    if family == "SPREAD" and len(segments) > 1 and (m := _SPREAD_RULE.search(rules)):
        return Proposition(subject=subject, kind=Kind.THRESHOLD, measure=f"margin:{_team_code(segments[1])}",
                           comparator=Comparator.GT, value=float(m.group("v")), unit=m.group("unit"),
                           observation=Observation.CUMULATIVE, scope=scope)
    if family == "GAME" and len(segments) > 1:
        return Proposition(subject=subject, kind=Kind.WINNER, entity=segments[1].lower(),
                           scope="regulation" if "90 minutes" in (rules + " " + (market.get("rules_secondary") or "")) else "match")
    if family == "MAP" and len(segments) > 2 and segments[1].isdigit():
        return Proposition(subject=subject, kind=Kind.WINNER, entity=segments[2].lower(), scope=f"map:{segments[1]}")
    return None


def is_no_side(source: dict) -> bool:
    return source.get("side") == "no"


def claims(source: dict) -> bool:
    series, _, rest = ((source.get("market") or {}).get("ticker") or "").partition("-")
    if _kalshi_crypto_claimed(series) or series in _KALSHI_INDEX:
        return True
    return _game_subject(series, rest.split("-")[0]) is not None if rest else False
