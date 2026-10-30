"""Relationships derived from propositions that share a subject: a pure function, so the same propositions always give
the same relationships, each with the rule and fields that produced it.

A threshold or range is the set of values of its measure that pay out (integer measures normalized, so "over 4.5
goals" and "5+ goals" are the same set); a negated proposition is the complement. One implies another when its set
is inside the other's, and they are equivalent when the sets are equal. By-deadline questions imply later deadlines,
winners are equal on the same entity and exclude other entities, and a short list of cross-measure rules relates
margins to winners and team totals to game totals. Anything with a field the rule needs left unknown is skipped:
no relationship is better than a wrong one.
"""
import dataclasses
import math
from datetime import timedelta
from itertools import combinations

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability

DERIVATION_VERSION = "d1"

EQUIVALENT = "EQUIVALENT"
IMPLIES = "IMPLIES"

# Deadlines this close are the same date boundary read in different time zones ("before 2027" as 00:00 or 05:00 UTC).
DEADLINE_TOLERANCE = timedelta(hours=12)
# Snapshots of a settled measure on the same day are the same reading (a review score after release).
SETTLED_SNAPSHOT_TOLERANCE = timedelta(hours=24)

_INTEGER_WORDS = ("goals", "runs", "points", "maps", "wins", "touchdowns", "tds", "receptions", "yards", "hits",
                  "strikeouts", "rbis", "hurricanes", "seats", "games", "sets", "assists", "rebounds", "position")

FLAG_SOURCE_DIFFERS = "RESOLUTION_SOURCE_DIFFERS"
FLAG_SOURCE_UNKNOWN = "RESOLUTION_SOURCE_UNKNOWN"
FLAG_DEADLINE_ZONE = "DEADLINE_TIMEZONE_DIFFERS"
FLAG_SNAPSHOT_TIME = "OBSERVATION_TIME_DIFFERS"


@dataclasses.dataclass(frozen=True)
class Contract:
    security_id: int
    venue: str
    proposition: Proposition


@dataclasses.dataclass(frozen=True)
class Derived:
    security_id_a: int
    security_id_b: int
    relationship_type: str
    flags: tuple[str, ...]
    evidence: dict


# ── Value sets ────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class _Interval:
    low: float
    low_closed: bool
    high: float
    high_closed: bool

    def empty(self) -> bool:
        return self.low > self.high or (self.low == self.high and not (self.low_closed and self.high_closed))


_INF = math.inf


def _is_integer(p: Proposition) -> bool:
    text = f"{p.measure or ''} {p.unit or ''}".lower()
    return "percent" not in text and any(w in text for w in _INTEGER_WORDS)


def _whole(interval: _Interval) -> _Interval:
    """The same set of whole numbers with closed whole-number bounds, so integer sets compare exactly."""
    low, high = interval.low, interval.high
    if low != -_INF:
        low = math.ceil(low) if interval.low_closed else math.floor(low) + 1
    if high != _INF:
        high = math.floor(high) if interval.high_closed else math.ceil(high) - 1
    return _Interval(low, low != -_INF, high, high != _INF)


def _threshold_interval(comparator: Comparator, value: float) -> _Interval:
    return {
        Comparator.GT: _Interval(value, False, _INF, False),
        Comparator.GTE: _Interval(value, True, _INF, False),
        Comparator.LT: _Interval(-_INF, False, value, False),
        Comparator.LTE: _Interval(-_INF, False, value, True),
    }[comparator]





def _complement(interval: _Interval) -> list[_Interval]:
    parts = [
        _Interval(-_INF, False, interval.low, not interval.low_closed),
        _Interval(interval.high, not interval.high_closed, _INF, False),
    ]
    return [p for p in parts if not p.empty() and not (p.low == -_INF and p.high == -_INF)
            and not (p.low == _INF and p.high == _INF)]


def value_set(p: Proposition) -> list[_Interval] | None:
    if p.kind == Kind.THRESHOLD:
        base = _threshold_interval(p.comparator, p.value)
    elif p.kind == Kind.RANGE:
        base = _Interval(p.low, p.low_inclusive, p.high, p.high_inclusive)
    else:
        return None
    parts = _complement(base) if p.negated else [base]
    if _is_integer(p):
        parts = [w for w in map(_whole, parts) if not w.empty()]
    return parts


def _inside(inner: _Interval, outer: _Interval) -> bool:
    low_ok = inner.low > outer.low or (inner.low == outer.low and (outer.low_closed or not inner.low_closed))
    high_ok = inner.high < outer.high or (inner.high == outer.high and (outer.high_closed or not inner.high_closed))
    return low_ok and high_ok


def _subset(a: list[_Interval], b: list[_Interval]) -> bool:
    return all(any(_inside(x, y) for y in b) for x in a)


# ── Comparability ─────────────────────────────────────────────────────

_NEEDED = {
    Kind.THRESHOLD: {"measure", "comparator", "value", "observation"},
    Kind.RANGE: {"measure", "low", "high", "observation"},
    Kind.OCCURS_BY: {"window_end", "measure"},
    Kind.WINNER: {"entity"},
}


def usable(p: Proposition) -> bool:
    if p.kind not in _NEEDED or "subject" in p.unknown or _NEEDED[p.kind] & p.unknown:
        return False
    # A deadline question must name the event that has to happen, or two different events would compare.
    return p.kind != Kind.OCCURS_BY or bool(p.measure)


def _scope_matches(a: Proposition, b: Proposition) -> bool:
    return a.scope is None or b.scope is None or a.scope == b.scope


def _same_reading(a: Proposition, b: Proposition) -> tuple[bool, list[str]]:
    """Whether two measurements of the same measure are taken at the same time, and any caveat."""
    if a.observation != b.observation:
        return False, []
    live_snapshots = a.observation == Observation.AT_TIME and not (a.stability == b.stability == Stability.SETTLED)
    if live_snapshots and a.window_start != b.window_start:
        # A live change over a window (up or down in 15 minutes vs in the hour ending at the same time) depends on
        # exactly when it starts.
        return False, []
    if a.window_start != b.window_start and not (
            a.window_start and b.window_start and abs(a.window_start - b.window_start) <= DEADLINE_TOLERANCE):
        # Different measurement periods (a year's growth vs one quarter's) are different readings.
        return False, []
    if a.observation == Observation.CUMULATIVE and (a.window_end is None or b.window_end is None):
        # A game's or season's total: the subject and scope already fix the window.
        return True, []
    if a.window_end is None or b.window_end is None:
        return a.window_end is None and b.window_end is None, []
    gap = abs(a.window_end - b.window_end)
    if gap == timedelta(0):
        return True, []
    if a.observation == Observation.AT_TIME:
        if a.stability == b.stability == Stability.SETTLED and gap <= SETTLED_SNAPSHOT_TOLERANCE:
            return True, [FLAG_SNAPSHOT_TIME]
        return False, []
    if gap <= DEADLINE_TOLERANCE:
        return True, [FLAG_DEADLINE_ZONE]
    return False, []


# ── Rules ─────────────────────────────────────────────────────────────

def _values(a: Proposition, b: Proposition) -> tuple[str, str, list[str]] | None:
    """(forward, backward) truth of a ⇒ b and b ⇒ a for two thresholds/ranges on the same measure."""
    if a.measure != b.measure or not _scope_matches(a, b):
        return None
    same, flags = _same_reading(a, b)
    if not same:
        return None
    sa, sb = value_set(a), value_set(b)
    return ("y" if _subset(sa, sb) else "n", "y" if _subset(sb, sa) else "n", flags)


def _deadlines(a: Proposition, b: Proposition) -> tuple[str, str, list[str]] | None:
    if a.measure != b.measure:
        return None
    if a.window_start != b.window_start and a.window_start is not None and b.window_start is not None:
        return None
    gap = b.window_end - a.window_end
    if abs(gap) <= DEADLINE_TOLERANCE:
        if a.negated != b.negated:
            return None
        return "y", "y", ([FLAG_DEADLINE_ZONE] if gap else [])
    earlier_first = gap > timedelta(0)
    # Happening by an earlier deadline implies happening by a later one; not happening by a later one implies not
    # happening by an earlier one.
    if not a.negated and not b.negated:
        return ("y", "n", []) if earlier_first else ("n", "y", [])
    if a.negated and b.negated:
        return ("n", "y", []) if earlier_first else ("y", "n", [])
    return None


def _winners(a: Proposition, b: Proposition) -> tuple[str, str, list[str]] | None:
    if not _scope_matches(a, b):
        return None
    if a.entity == b.entity:
        return ("y", "y", []) if a.negated == b.negated else None
    # Different entities can't both win: a win by one implies the other's loss.
    if not a.negated and b.negated:
        return "y", "n", []
    if a.negated and not b.negated:
        return "n", "y", []
    return None


def _margin_implies_winner(margin: Proposition, winner: Proposition) -> bool:
    if winner.negated or margin.negated or not (margin.measure or "").startswith("margin:"):
        return False
    if margin.measure.split(":", 1)[1] != winner.entity or not _scope_matches(margin, winner):
        return False
    interval = value_set(margin)[0]
    return interval.low > 0 or (interval.low == 0 and not interval.low_closed)


def _team_total_implies_total(team: Proposition, total: Proposition) -> bool:
    """One team scoring at least v means the game total is at least v (the other team scores zero or more)."""
    if team.negated or total.negated or not team.measure or not total.measure or ":" not in team.measure:
        return False
    base = team.measure.split(":", 1)[0]
    if not base.startswith("team_") or total.measure != "total_" + base.removeprefix("team_"):
        return False
    if not _scope_matches(team, total) or team.observation != total.observation:
        return False
    team_set, total_set = value_set(team), value_set(total)
    if len(team_set) != 1 or len(total_set) != 1 or team_set[0].high != _INF or total_set[0].high != _INF:
        return False
    return _inside(team_set[0], total_set[0])


def _relate(a: Proposition, b: Proposition) -> tuple[str, str, list[str], str] | None:
    """(a ⇒ b, b ⇒ a, flags, rule) or None when the rules say nothing."""
    if a.kind in (Kind.THRESHOLD, Kind.RANGE) and b.kind in (Kind.THRESHOLD, Kind.RANGE):
        found = _values(a, b)
        if found is None and a.kind == b.kind == Kind.THRESHOLD:
            if _team_total_implies_total(a, b):
                return "y", "n", [], "team_total_within_total"
            if _team_total_implies_total(b, a):
                return "n", "y", [], "team_total_within_total"
        return None if found is None else (*found, "value_sets")
    if a.kind == b.kind == Kind.OCCURS_BY:
        found = _deadlines(a, b)
        return None if found is None else (*found, "deadlines")
    if a.kind == b.kind == Kind.WINNER:
        found = _winners(a, b)
        return None if found is None else (*found, "winners")
    if a.kind == Kind.THRESHOLD and b.kind == Kind.WINNER and _margin_implies_winner(a, b):
        return "y", "n", [], "margin_implies_winner"
    if b.kind == Kind.THRESHOLD and a.kind == Kind.WINNER and _margin_implies_winner(b, a):
        return "n", "y", [], "margin_implies_winner"
    return None


def _source_flags(a: Contract, b: Contract) -> list[str]:
    if a.venue == b.venue:
        return []
    sa, sb = a.proposition.resolution_source, b.proposition.resolution_source
    if sa and sb:
        return [] if sa == sb else [FLAG_SOURCE_DIFFERS]
    return [FLAG_SOURCE_UNKNOWN]


def _evidence(rule: str, a: Proposition, b: Proposition) -> dict:
    keep = ("kind", "measure", "comparator", "value", "low", "high", "observation", "window_end", "entity", "scope",
            "negated", "resolution_source")
    pick = lambda p: {k: v for k, v in p.to_fields().items() if k in keep and v not in (None, False)}
    return {"rule": rule, "version": DERIVATION_VERSION, "a": pick(a), "b": pick(b)}


def derive(contracts: list[Contract]) -> list[Derived]:
    """All relationships among one subject's contracts, at most one per pair: EQUIVALENT (lower security first) or
    IMPLIES (antecedent first)."""
    out = []
    usable_contracts = sorted((c for c in contracts if usable(c.proposition)), key=lambda c: c.security_id)
    for a, b in combinations(usable_contracts, 2):
        found = _relate(a.proposition, b.proposition)
        if found is None:
            continue
        forward, backward, flags, rule = found
        all_flags = tuple(sorted(set(flags + _source_flags(a, b))))
        if forward == backward == "y":
            out.append(Derived(a.security_id, b.security_id, EQUIVALENT, all_flags, _evidence(rule, a.proposition, b.proposition)))
        elif forward == "y":
            out.append(Derived(a.security_id, b.security_id, IMPLIES, all_flags, _evidence(rule, a.proposition, b.proposition)))
        elif backward == "y":
            out.append(Derived(b.security_id, a.security_id, IMPLIES, all_flags, _evidence(rule, b.proposition, a.proposition)))
    return out


def derive_all(contracts: list[Contract]) -> list[Derived]:
    """Derives within each subject; contracts on different subjects are never compared."""
    by_subject: dict[str, list[Contract]] = {}
    for c in contracts:
        if "subject" not in c.proposition.unknown:
            by_subject.setdefault(c.proposition.subject, []).append(c)
    return [d for group in by_subject.values() for d in derive(group)]
