"""Relationships derived from propositions that share a subject: a pure function, so the same propositions always give
the same relationships, each with the rule and fields that produced it.

A threshold or range is the set of values of its measure that pay out (integer measures normalized, so "over 4.5
goals" and "5+ goals" are the same set); a negated proposition is the complement. One implies another when its set
is inside the other's, and they are equivalent when the sets are equal. By-deadline questions imply later deadlines,
winners are equal on the same entity and exclude other entities, and a short list of cross-measure rules relates
margins to winners and team totals to game totals. Anything with a field the rule needs left unknown is skipped:
no relationship is better than a wrong one.

A subject can hold thousands of contracts (every BTC strike on every venue and expiry), so contracts are first split
into groups that no rule can relate across (snapshots at different times, deadlines on different events), and only
pairs inside a group are compared. Groups that share a contract form one derivation unit, the set the classifier
re-derives and replaces together.
"""
import dataclasses
import logging
import math
import re
from datetime import timedelta
from itertools import combinations

from gnomepy.registry.relations import Edge, close, reduce

from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability

logger = logging.getLogger(__name__)

DERIVATION_VERSION = "d5"

EQUIVALENT = "EQUIVALENT"
IMPLIES = "IMPLIES"

# Deadlines this close are the same date boundary read in different time zones ("before 2027" as 00:00 or 05:00 UTC).
DEADLINE_TOLERANCE = timedelta(hours=12)
# Snapshots of a settled measure on the same day are the same reading (a review score after release).
SETTLED_SNAPSHOT_TOLERANCE = timedelta(hours=24)

# Words in a measure that count whole things. Matched as whole words of the measure, never its unit: an index is
# quoted in "points" but trades in fractions of one.
_INTEGER_WORDS = frozenset({
    "goals", "runs", "points", "maps", "wins", "touchdowns", "tds", "receptions", "yards", "hits", "strikeouts",
    "rbis", "hurricanes", "seats", "games", "sets", "assists", "rebounds", "position",
})
# Sports units: a margin in them is whole, and the side with more of them wins. Not games: a tennis player can win
# more games and lose the match (6-0 4-6 4-6).
_DECIDING_MARGIN_UNITS = frozenset({"points", "goals", "runs", "maps", "sets", "rounds"})
# A vote-share margin decides the vote it is counted in, which must be named on both sides: winning the popular vote
# is not winning the electoral college.
_VOTE_MARGIN_UNITS = frozenset({"percentage_points", "percent", "votes"})

FLAG_SOURCE_DIFFERS = "RESOLUTION_SOURCE_DIFFERS"
FLAG_SOURCE_UNKNOWN = "RESOLUTION_SOURCE_UNKNOWN"
FLAG_DEADLINE_ZONE = "DEADLINE_TIMEZONE_DIFFERS"
FLAG_SNAPSHOT_TIME = "OBSERVATION_TIME_DIFFERS"


@dataclasses.dataclass(frozen=True)
class Contract:
    security_id: int
    venue: str
    proposition: Proposition
    # The other security of its BINARY event (its No side, or Yes side), when the event has exactly two.
    complement: int | None = None
    event_id: int | None = None


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
    words = set(re.split(r"[^a-z0-9]+", (p.measure or "").lower()))
    if "margin" in words:
        return (p.unit or "").lower() in _DECIDING_MARGIN_UNITS | {"games"}
    return "percent" not in words and bool(words & _INTEGER_WORDS)


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
    """Whether every field its kind's rules read is known. A field left empty counts as unknown even when extraction
    didn't say so: a deadline question without its deadline, or two different events, would otherwise compare."""
    if p.kind not in _NEEDED or "subject" in p.unknown or _NEEDED[p.kind] & p.unknown:
        return False
    return all(getattr(p, name) not in (None, "") for name in _NEEDED[p.kind])


def _same_entity(a: Proposition, b: Proposition) -> bool:
    """Whether two propositions on one measure are about the same entity. A measure that already names its entity
    (wins:texas-st) settles it when only one side repeats the entity; two different entities never match, and an
    entity on one side alone is otherwise a narrower question (one candidate confirmed vs anyone confirmed)."""
    if a.entity == b.entity:
        return True
    if a.entity and b.entity:
        return False
    named = a.entity or b.entity
    return (a.measure or "").endswith(f":{named}")


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

def _extreme(p: Proposition) -> str:
    return "max" if p.comparator in (Comparator.GT, Comparator.GTE) else "min"


def _values(a: Proposition, b: Proposition) -> tuple[str, str, list[str]] | None:
    """(forward, backward) truth of a ⇒ b and b ⇒ a for two thresholds/ranges on the same measure."""
    if a.measure != b.measure or not _same_entity(a, b) or not _scope_matches(a, b):
        return None
    if a.observation == Observation.ANY_TIME:
        # Touching a level at any time is about the window's maximum ("reaches 120k") or its minimum ("dips to 100k"),
        # two different quantities: the price can do both. Only touches on the same side compare.
        if a.kind != Kind.THRESHOLD or b.kind != Kind.THRESHOLD or _extreme(a) != _extreme(b):
            return None
    same, flags = _same_reading(a, b)
    if not same:
        return None
    sa, sb = value_set(a), value_set(b)
    return ("y" if _subset(sa, sb) else "n", "y" if _subset(sb, sa) else "n", flags)


def _deadlines(a: Proposition, b: Proposition) -> tuple[str, str, list[str]] | None:
    if a.measure != b.measure or not _same_entity(a, b) or not _scope_matches(a, b):
        return None
    # The earlier deadline's window must sit inside the later one's, so both have to start together: "by June, from
    # listing" says nothing about "by December, from July" (a March event makes the first true and the second false).
    if (a.window_start is None) != (b.window_start is None):
        return None
    if a.window_start is not None and abs(a.window_start - b.window_start) > DEADLINE_TOLERANCE:
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


def _winner_timing(a: Proposition, b: Proposition) -> tuple[str, list[str]] | None:
    """How two winner propositions' timings compare: "same", "a_at_end" (a holds it at the window's end, b at any time
    in the window, so a implies b), "b_at_end", or None when they are about different times."""
    if a.observation is None and b.observation is None:
        # A contest decided once (a game, an election): its time is fixed by the subject.
        return "same", []
    if a.observation is None or b.observation is None or a.window_end is None or b.window_end is None:
        return None
    gap = abs(a.window_end - b.window_end)
    if gap > DEADLINE_TOLERANCE:
        return None
    flags = [FLAG_DEADLINE_ZONE] if gap else []
    if a.observation == b.observation:
        return "same", flags
    if a.observation == Observation.AT_TIME and b.observation == Observation.ANY_TIME:
        return "a_at_end", flags
    if a.observation == Observation.ANY_TIME and b.observation == Observation.AT_TIME:
        return "b_at_end", flags
    return None


def _winners(a: Proposition, b: Proposition) -> tuple[str, str, list[str]] | None:
    if not _scope_matches(a, b):
        return None
    timing = _winner_timing(a, b)
    if timing is None:
        return None
    how, flags = timing
    if a.entity == b.entity:
        if a.negated != b.negated:
            return None
        if how == "same":
            return "y", "y", flags
        # Holding a position at the end of a window implies having held it at some point in it; not having held it at
        # all implies not holding it at the end.
        a_first = (how == "a_at_end") != a.negated
        return ("y", "n", flags) if a_first else ("n", "y", flags)
    # Different entities can't both win a contest or hold a position at the same moment, but each can hold it at some
    # point in a window.
    if how != "same" or a.observation == Observation.ANY_TIME:
        return None
    if not a.negated and b.negated:
        return "y", "n", flags
    if a.negated and not b.negated:
        return "n", "y", flags
    return None


def _margin_implies_winner(margin: Proposition, winner: Proposition) -> bool:
    if winner.negated or margin.negated or not (margin.measure or "").startswith("margin:"):
        return False
    unit = (margin.unit or "").lower()
    if unit in _VOTE_MARGIN_UNITS:
        if margin.scope is None or margin.scope != winner.scope:
            return False
    elif unit not in _DECIDING_MARGIN_UNITS:
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


def _relate_pair(a: Contract, b: Contract) -> Derived | None:
    found = _relate(a.proposition, b.proposition)
    if found is None:
        return None
    forward, backward, flags, rule = found
    all_flags = tuple(sorted(set(flags + _source_flags(a, b))))
    if forward == backward == "y":
        return Derived(a.security_id, b.security_id, EQUIVALENT, all_flags, _evidence(rule, a.proposition, b.proposition))
    if forward == "y":
        return Derived(a.security_id, b.security_id, IMPLIES, all_flags, _evidence(rule, a.proposition, b.proposition))
    if backward == "y":
        return Derived(b.security_id, a.security_id, IMPLIES, all_flags, _evidence(rule, b.proposition, a.proposition))
    return None


# ── Groups ────────────────────────────────────────────────────────────

def _chains(contracts: list[Contract], tolerance: timedelta) -> list[list[Contract]]:
    """Runs of contracts whose window ends are each within `tolerance` of the previous one, so any two within it share
    a run."""
    runs: list[list[Contract]] = []
    for c in sorted(contracts, key=lambda c: c.proposition.window_end):
        if runs and c.proposition.window_end - runs[-1][-1].proposition.window_end <= tolerance:
            runs[-1].append(c)
        else:
            runs.append([c])
    return runs


def _candidate_groups(contracts: list[Contract]) -> list[list[Contract]]:
    """Groups no rule relates across; a margin threshold sits in both its value group and the winners."""
    keyed: dict[tuple, list[Contract]] = {}
    chained: dict[tuple, list[Contract]] = {}
    for c in contracts:
        p = c.proposition
        if p.kind == Kind.WINNER:
            keyed.setdefault(("winner",), []).append(c)
            continue
        if p.kind == Kind.OCCURS_BY:
            keyed.setdefault(("deadline", p.measure), []).append(c)
            continue
        if p.kind == Kind.THRESHOLD and p.measure.startswith("margin:"):
            keyed.setdefault(("winner",), []).append(c)
        if p.observation == Observation.CUMULATIVE:
            # A game's or season's total compares across window ends, missing ones included (`_same_reading`): the
            # subject and scope fix the window, so all of a subject's totals share one group.
            keyed.setdefault(("value", p.observation), []).append(c)
        elif p.window_end is None:
            keyed.setdefault(("value", p.observation, None), []).append(c)
        elif p.observation == Observation.AT_TIME and p.stability != Stability.SETTLED:
            # A live snapshot only matches one taken over exactly the same window.
            keyed.setdefault(("value", p.observation, p.window_start, p.window_end), []).append(c)
        else:
            chained.setdefault((p.observation, p.stability == Stability.SETTLED), []).append(c)
    groups = list(keyed.values())
    for (observation, settled), members in chained.items():
        tolerance = SETTLED_SNAPSHOT_TOLERANCE if observation == Observation.AT_TIME and settled else DEADLINE_TOLERANCE
        groups.extend(_chains(members, tolerance))
    return groups


@dataclasses.dataclass(frozen=True)
class Unit:
    """Contracts derived and replaced together: groups joined by a contract they share."""
    contracts: tuple[Contract, ...]
    groups: tuple[tuple[Contract, ...], ...]


def derivation_units(contracts: list[Contract]) -> list[Unit]:
    usable_contracts = [c for c in contracts if usable(c.proposition)]
    groups = _candidate_groups(usable_contracts)
    parent = list(range(len(groups)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_group: dict[int, int] = {}
    for i, group in enumerate(groups):
        for c in group:
            if c.security_id in first_group:
                parent[root(i)] = root(first_group[c.security_id])
            else:
                first_group[c.security_id] = i
    joined: dict[int, list[int]] = {}
    for i in range(len(groups)):
        joined.setdefault(root(i), []).append(i)
    units = []
    for indexes in joined.values():
        members = {c.security_id: c for i in indexes for c in groups[i]}
        units.append(Unit(
            contracts=tuple(sorted(members.values(), key=lambda c: c.security_id)),
            groups=tuple(tuple(sorted(groups[i], key=lambda c: c.security_id)) for i in indexes),
        ))
    return sorted(units, key=lambda u: u.contracts[0].security_id)


def derive_unit(unit: Unit) -> list[Derived]:
    """The relationships among a unit's contracts, at most one per pair: EQUIVALENT (lower security first) or
    IMPLIES (antecedent first).

    A venue never lists the same contract twice in one event, so two of an event's outcomes read as equivalent are a
    misreading (two candidates given one name). Neither reading can be trusted against the other venues' contracts, so
    the colliding contracts are left out of the unit: a missed relationship rather than a wrong one."""
    left_out: set[int] = set()
    while True:
        derived = _derive_pairs(unit, left_out)
        collided = _collisions(unit, derived)
        if not collided:
            return derived
        for event_id, ids in sorted(collided.items()):
            # Counted by the PropositionCollisions metric filter (cdk/lib/stacks/classifier-stack.ts); keep the wording.
            logger.warning("proposition collision: event=%d securities=%s subject=%s",
                           event_id, sorted(ids), unit.contracts[0].proposition.subject)
            left_out |= ids


def _derive_pairs(unit: Unit, left_out: set[int]) -> list[Derived]:
    out = []
    seen: set[tuple[int, int]] = set()
    for group in unit.groups:
        for a, b in combinations([c for c in group if c.security_id not in left_out], 2):
            if (a.security_id, b.security_id) in seen:
                continue
            seen.add((a.security_id, b.security_id))
            derived = _relate_pair(a, b)
            if derived is not None:
                out.append(derived)
    return out


def _collisions(unit: Unit, derived: list[Derived]) -> dict[int, set[int]]:
    """{event_id: securities} of equivalences between two different outcomes of one event."""
    by_id = {c.security_id: c for c in unit.contracts}
    out: dict[int, set[int]] = {}
    for d in derived:
        a, b = by_id[d.security_id_a], by_id[d.security_id_b]
        if (d.relationship_type == EQUIVALENT and a.event_id is not None and a.event_id == b.event_id
                and a.complement != b.security_id):
            out.setdefault(a.event_id, set()).update((a.security_id, b.security_id))
    return out


def minimal_rows(unit: Unit, derived: list[Derived]) -> tuple[list[Derived], bool]:
    """The fewest rows from which consumers rebuild everything `derived` says about the unit (gnomepy's
    `relations.close` holds the rules: chains, contrapositives through complements, flagged rows composed once).
    Returns the rows and whether the round trip held; when it doesn't, which would be a bug in the reduction, all of
    `derived` comes back so nothing is lost."""
    ids = {c.security_id for c in unit.contracts}
    complements = {c.security_id: c.complement for c in unit.contracts if c.complement in ids}
    relation = close([Edge(d.security_id_a, d.security_id_b, d.relationship_type, d.flags) for d in derived],
                     complements)
    rows = reduce(relation, complements)
    if close(rows, complements) != relation:
        return derived, False
    by_pair = {(d.security_id_a, d.security_id_b, d.relationship_type): d for d in derived}
    out = []
    for row in rows:
        same = by_pair.get((row.security_id_a, row.security_id_b, row.relationship_type))
        # A kept row usually is a derived pair; one that only follows from others (the No side of a ladder step, or a
        # pair joined through equivalents) records that it was composed.
        evidence = same.evidence if same is not None else {"rule": "composed", "version": DERIVATION_VERSION}
        out.append(Derived(row.security_id_a, row.security_id_b, row.relationship_type, row.flags, evidence))
    return out, True


def derive(contracts: list[Contract]) -> list[Derived]:
    """All relationships among one subject's contracts."""
    return [d for unit in derivation_units(contracts) for d in derive_unit(unit)]


def derive_all(contracts: list[Contract]) -> list[Derived]:
    """Derives within each subject; contracts on different subjects are never compared."""
    by_subject: dict[str, list[Contract]] = {}
    for c in contracts:
        if "subject" not in c.proposition.unknown:
            by_subject.setdefault(c.proposition.subject, []).append(c)
    return [d for group in by_subject.values() for d in derive(group)]
