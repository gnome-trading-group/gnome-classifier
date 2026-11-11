"""Finds subjects that name the same thing under two keys, so contracts that ask about the same thing get compared.
Derivation only compares propositions that share a subject, and extraction sometimes coins a second key for a subject
that already has one: another venue's (`us-house-2026` and `us-house-2026-control`, `ga-governor-2026` and
`georgia-governor-2026`), or the same venue's (`a-j-brown:nfl-2026` with a 1,400.5+ receiving-yards market and
`a-j-brown:nfl-2026-27` with a 1,000.5+ one, which merged imply each other).

Candidates are cheap and recall-oriented: embedding neighbours, keys that normalize alike, and subjects whose
propositions share a deadline date and an entity or measure. A Claude judge then decides each pair from the subjects'
sample contracts. Embedding distance alone can't: the nearest
subject to `us-house-2026-control` is `us-house-2028-control`, and merging those would make two different elections'
contracts equivalent.
"""
import dataclasses
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from classifier.client.claude import ClaudeAsker, is_unavailable
from classifier.propositions.canonical import ENTITY_ALIAS, MEASURE_ALIAS, StoredAlias
from classifier.propositions.derive import Contract, Derived, preview
from classifier.propositions.subjects import entity_of
from classifier.propositions.types import Kind, Proposition

logger = logging.getLogger(__name__)

SAMPLES_PER_SUBJECT = 6
RULES_CHARS = 600
PAIRS_PER_REQUEST = 6
MAX_TOKENS = 8000
MAX_WORKERS = 8
EFFORT = "medium"
# A deadline date and entity shared by more subjects than this is too common to suggest a duplicate (election day and
# `democratic`, a season's last day and `winner`).
SHARED_BUCKET_LIMIT = 6
# Where extraction puts contracts whose subject it could not determine; never a merge candidate.
UNKNOWN_SUBJECT = "unknown"

_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new-hampshire": "nh", "new-jersey": "nj",
    "new-mexico": "nm", "new-york": "ny", "north-carolina": "nc", "north-dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode-island": "ri", "south-carolina": "sc",
    "south-dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west-virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
# Longest names first, so west-virginia is replaced before virginia.
_STATE_PATTERN = re.compile(r"(?<![a-z])(" + "|".join(sorted(_STATES, key=len, reverse=True)) + r")(?![a-z])")
_SEPARATORS = re.compile(r"[:._/-]+")
# Words that say how a market is asked rather than what about.
_FILLER = {"control", "winner", "winners", "win", "race", "election", "elections", "result", "results", "outcome", "the"}
_GENERIC_ENTITIES = {"yes", "no", "tie", "draw", "over", "under", "other", "field", "none"}
# A game key's date and, when it carries one, its start time (`game:cs2:2026-11-03:mayfal-1300`). The same teams can
# meet twice in a day in different tournaments, and the judge, reading two near-identical match markets, calls them one.
_GAME_START = re.compile(r"^game:[^:]+:(\d{4}-\d{2}-\d{2}):.*-([01]\d|2[0-3])([0-5]\d)$")
_GAME_KEY = re.compile(r"^game:[^:]+:(\d{4}-\d{2}-\d{2}):(.+)$")
# A game key's scheduled start (`mouznnavij-0400`, US Eastern, from Kalshi's event ticker), which other venues' keys
# don't carry.
_START_SUFFIX = re.compile(r"-([01]\d|2[0-3])[0-5]\d$")
_EASTERN = ZoneInfo("America/New_York")


class Verdict(StrEnum):
    # The same contest, quantity or occasion: their contracts can be compared.
    SAME = "SAME"
    # The same topic but a different contest: another year, office, round, region or publisher.
    RELATED = "RELATED"
    DIFFERENT = "DIFFERENT"


@dataclasses.dataclass(frozen=True)
class SubjectContract:
    venue: str
    event_title: str
    outcome: str
    rules: str
    proposition: Proposition


@dataclasses.dataclass(frozen=True)
class SubjectProfile:
    """What the judge sees of a subject, and what candidates are found from."""
    key: str
    description: str
    venues: frozenset[str]
    entities: frozenset[str]
    measures: frozenset[str]
    deadlines: frozenset[date]
    proposition_count: int
    samples: tuple[dict, ...]

    def to_json(self) -> dict:
        return {
            "key": self.key, "description": self.description, "venues": sorted(self.venues),
            "entities": sorted(self.entities), "measures": sorted(self.measures),
            "deadlines": sorted(d.isoformat() for d in self.deadlines), "proposition_count": self.proposition_count,
            "samples": list(self.samples),
        }

    @classmethod
    def from_json(cls, row: dict) -> "SubjectProfile":
        return cls(
            key=row["key"], description=row.get("description") or "", venues=frozenset(row["venues"]),
            entities=frozenset(row.get("entities") or ()), measures=frozenset(row.get("measures") or ()),
            deadlines=frozenset(date.fromisoformat(d) for d in row.get("deadlines") or ()),
            proposition_count=row.get("proposition_count", 0), samples=tuple(row.get("samples") or ()),
        )


def _compact(p: Proposition) -> dict:
    fields = {k: v for k, v in p.to_fields().items()
              if v not in (None, [], False) and k not in ("window_start", "stability", "low_inclusive", "high_inclusive")}
    if "window_end" in fields:
        fields["window_end"] = fields["window_end"][:16]
    return fields


def _sample(c: SubjectContract) -> dict:
    return {"venue": c.venue, "event_title": c.event_title, "outcome": c.outcome, "rules": c.rules[:RULES_CHARS],
            "proposition": _compact(c.proposition)}


def _spread(contracts: list[SubjectContract], limit: int) -> list[SubjectContract]:
    """Up to `limit` contracts, taking each venue in turn and, within a venue, a different kind and measure first,
    so the judge sees what each venue asks."""
    by_venue: dict[str, list[SubjectContract]] = defaultdict(list)
    for c in contracts:
        by_venue[c.venue].append(c)
    queues = []
    for venue in sorted(by_venue):
        seen, first, rest = set(), [], []
        for c in by_venue[venue]:
            shape = (c.proposition.kind, c.proposition.measure)
            (rest if shape in seen else first).append(c)
            seen.add(shape)
        queues.append(first + rest)
    out = []
    while len(out) < limit and any(queues):
        for q in queues:
            if q and len(out) < limit:
                out.append(q.pop(0))
    return out


def profile(key: str, description: str, contracts: list[SubjectContract],
            samples: int = SAMPLES_PER_SUBJECT) -> SubjectProfile:
    entities, measures, deadlines = set(), set(), set()
    for c in contracts:
        p = c.proposition
        if p.kind == Kind.WINNER and p.entity and "entity" not in p.unknown:
            entities.add(p.entity)
        if (entity := entity_of(p.measure)) is not None:
            entities.add(entity)
        if p.measure and "measure" not in p.unknown:
            measures.add(p.measure)
        if p.window_end and "window_end" not in p.unknown:
            deadlines.add(p.window_end.date())
    return SubjectProfile(
        key=key, description=description, venues=frozenset(c.venue for c in contracts), entities=frozenset(entities),
        measures=frozenset(measures), deadlines=frozenset(deadlines), proposition_count=len(contracts),
        samples=tuple(_sample(c) for c in _spread(contracts, samples)),
    )


def key_tokens(key: str) -> tuple[str, ...]:
    """A key's words in order, with state names as their codes and filler words dropped: `georgia-governor-2026` and
    `ga-governor-2026` give the same tokens, as do `us-house-2026` and `us-house-2026-control`. Order matters: the
    combination keys `combo:r-ak-ia-d-oh` and `combo:r-ak-oh-d-ia` have the same words but differ."""
    text = _STATE_PATTERN.sub(lambda m: _STATES[m.group(1)], key.lower())
    return tuple(t for t in _SEPARATORS.split(text) if t and t not in _FILLER)


def different_start_times(a: str, b: str) -> bool:
    """Whether both keys are games on one date that name different start times."""
    ma, mb = _GAME_START.match(a), _GAME_START.match(b)
    return bool(ma and mb and ma.group(1) == mb.group(1) and ma.groups()[1:] != mb.groups()[1:])


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _token_pairs(profiles: list[SubjectProfile]) -> set[tuple[str, str]]:
    """Keys with the same tokens, or where one key's tokens are the other's with one more number (`fomc:2026-12` and
    `fomc:2026-12-09`). An extra word usually narrows the subject (`us-senate-2026` and `us-senate-2026-texas`)."""
    by_tokens: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for p in profiles:
        if tokens := key_tokens(p.key):
            by_tokens[tokens].append(p.key)
    out = set()
    for tokens, keys in by_tokens.items():
        out.update(_pair(a, b) for i, a in enumerate(keys) for b in keys[i + 1:])
        if len(tokens) < 3:
            continue
        for i in range(len(tokens)):
            if not tokens[i].isdigit():
                continue
            for other in by_tokens.get(tokens[:i] + tokens[i + 1:], ()):
                out.update(_pair(k, other) for k in keys)
    return out


def _game_pairs(profiles: list[SubjectProfile]) -> set[tuple[str, str]]:
    """Games on one date whose teams read the same without hyphens or a start time, whatever the league is called:
    venues spell the league several ways (`bun`, `bundesliga`), the teams with or without a hyphen (`nap-laz`,
    `naplaz`) and with or without the start (`mouzn-navij`, `mouznnavij-0400`), and the token pairs never meet them.

    A key naming no start pairs with one naming a start only when that is the only start any key names for those teams
    that day: where two are named (a rematch, a doubleheader, the many daily games of virtual sports), which one it is
    can't be told from the keys. Two keys naming different starts never pair (`different_start_times`). Whatever pairs,
    a merge of two games also needs their scheduled starts to agree (`start_gap`)."""
    by_game: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in profiles:
        if m := _GAME_KEY.match(p.key):
            by_game[(m.group(1), _START_SUFFIX.sub("", m.group(2)).replace("-", ""))].append(p.key)
    out = set()
    for keys in by_game.values():
        starts = {m.group(0) for k in keys if (m := _START_SUFFIX.search(k))}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if len(starts) > 1 and bool(_START_SUFFIX.search(a)) != bool(_START_SUFFIX.search(b)):
                    continue
                out.add(_pair(a, b))
    return out


def _shared_pairs(profiles: list[SubjectProfile], limit: int) -> set[tuple[str, str]]:
    """Subjects with a deadline date in common (a day apart counts: venues disagree about a date's time zone) and an
    entity in common, when few enough subjects share both for it to mean something. Measures are too generic for
    this (`price` on one day is every asset's)."""
    buckets: dict[tuple[date, str], set[str]] = defaultdict(set)
    for p in profiles:
        for day in p.deadlines:
            for entity in p.entities - _GENERIC_ENTITIES:
                buckets[(day, entity)].add(p.key)
    out = set()
    for (day, entity), keys in buckets.items():
        near = keys | buckets.get((day + timedelta(days=1), entity), set())
        if len(near) > limit:
            continue
        ordered = sorted(near)
        out.update(_pair(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:])
    return out


def key_start(key: str) -> datetime | None:
    """The scheduled start a game key names (`game:cs2:2026-11-12:mouznnavij-0400` is 04:00 US Eastern), else None."""
    m = _GAME_START.match(key)
    if m is None:
        return None
    day = date.fromisoformat(m.group(1))
    return datetime(day.year, day.month, day.day, int(m.group(2)), int(m.group(3)), tzinfo=_EASTERN)


def start_gap(a_key: str, a: list[Contract], b_key: str, b: list[Contract]) -> tuple[timedelta, datetime, datetime] | None:
    """How far apart two game subjects are scheduled: the nearest two of each side's starts, as (gap, a's, b's), or
    None when either side has none. A side's starts are its contracts' event start times and the start its key names:
    venues move a delayed match's start while Kalshi's ticker keeps the first one, and either is when that match was
    scheduled."""
    starts_a = {c.start for c in a if c.start is not None} | {t for t in [key_start(a_key)] if t is not None}
    starts_b = {c.start for c in b if c.start is not None} | {t for t in [key_start(b_key)] if t is not None}
    if not starts_a or not starts_b:
        return None
    return min(((abs(x - y), x, y) for x in starts_a for y in starts_b), key=lambda t: (t[0], t[1], t[2]))


def candidate_pairs(profiles: Iterable[SubjectProfile], neighbours: dict[str, list[str]],
                    bucket_limit: int = SHARED_BUCKET_LIMIT) -> list[tuple[str, str]]:
    """Pairs of subject keys worth judging, (smaller key, larger key), sorted. `neighbours` are each subject's nearest
    subjects by embedding."""
    by_key = {p.key: p for p in profiles if p.key != UNKNOWN_SUBJECT}
    subjects = list(by_key.values())
    pairs = _token_pairs(subjects) | _game_pairs(subjects) | _shared_pairs(subjects, bucket_limit)
    pairs |= {_pair(k, n) for k, ns in neighbours.items() for n in ns if k != n}
    return sorted((a, b) for a, b in pairs if a in by_key and b in by_key and not different_start_times(a, b))


@dataclasses.dataclass(frozen=True)
class Judgment:
    verdict: Verdict
    reason: str
    # (entity of a, entity of b) that name the same team, player, party or outcome.
    entity_map: tuple[tuple[str, str], ...] = ()
    measure_map: tuple[tuple[str, str], ...] = ()

    def to_json(self) -> dict:
        return {"verdict": self.verdict.value, "reason": self.reason,
                "entity_map": [list(m) for m in self.entity_map], "measure_map": [list(m) for m in self.measure_map]}

    def flipped(self) -> "Judgment":
        """The same judgment with subjects a and b swapped."""
        return dataclasses.replace(self, entity_map=tuple((b, a) for a, b in self.entity_map),
                                   measure_map=tuple((b, a) for a, b in self.measure_map))

    @classmethod
    def from_json(cls, row: dict) -> "Judgment":
        return cls(Verdict(row["verdict"]), row.get("reason", ""),
                   tuple(tuple(m) for m in row.get("entity_map") or ()),
                   tuple(tuple(m) for m in row.get("measure_map") or ()))


_SYSTEM = """You decide whether two subject keys from a prediction-market database name the same thing.

Every contract was read into a proposition on a subject: the entity or event the question is about (an asset, one \
game, one election, a team's or player's season, an award in one year, one central-bank meeting). The measure, \
threshold, entity and deadline are separate fields, so one subject holds many questions: "Democrats win the House", \
"Republicans win 220+ seats" and "House control after the 2026 midterms" all belong on one subject for that \
election. A program compares contracts only when they share a subject, and real money trades on those comparisons, \
so a wrong SAME is far worse than a missed one.

For each pair, answer:
- SAME: both subjects are the same contest, occasion or quantity, so every contract on either could be compared with \
the other's. The same election (office, jurisdiction and cycle), the same game (teams and date), the same asset, the \
same team's or player's same season, the same award in the same year, the same meeting or data release. Different \
key styles, a missing date in one key, or one subject holding more kinds of question than the other do not matter. A \
season may be named by its start year or by both years (ncaaf-2026 and ncaaf-2026-27, nfl-2026 and nfl-2026-27 are \
the same season). A deadline is not part of a subject: "a pandemic declared in 2026" and "a pandemic declared by the \
end of 2027" are the same subject with different deadlines.
- RELATED: the same topic but a different contest or occasion: another edition of a recurring contest (the 2026 and \
2028 elections, two seasons), another office, chamber, district or county, another round or stage (a primary or \
nomination and the general election, a first round and the runoff), a different meeting or data release, another \
award, a player's season and the league's, a game and a team's season, a chart or leaderboard in another region or \
by another publisher. Also RELATED: keys that each name a different outcome of one contest (combination keys such as \
combo:r-ak-ia-d-oh and combo:r-ak-d-ia-oh), and subjects where a question on one would read exactly like a different \
question on the other once merged (winning the national popular vote and winning the House majority are both \
"Democrats win").
- DIFFERENT: anything else.
Judge from the contracts (titles, outcomes, rules, deadlines), not from the keys alone. When unsure, it is not SAME.

For SAME, also list entity_map: each entity key of subject a with the key subject b uses for the same team, player, \
party or outcome where they differ (a "ggo", b "goias"), and measure_map likewise for measures that are the same \
quantity under different names. Leave both empty otherwise.

reason: one short sentence a reviewer can check. Return one entry per pair id given, echoing the pair's a_key and b_key
exactly as given."""

_MAP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    },
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "string"},
                    "a_key": {"type": "string"},
                    "b_key": {"type": "string"},
                    "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
                    "entity_map": _MAP_SCHEMA,
                    "measure_map": _MAP_SCHEMA,
                    "reason": {"type": "string"},
                },
                "required": ["pair_id", "a_key", "b_key", "verdict", "entity_map", "measure_map", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["judgments"],
    "additionalProperties": False,
}


def _shown(p: SubjectProfile) -> dict:
    return {"key": p.key, "description": p.description, "venues": sorted(p.venues), "entities": sorted(p.entities),
            "measures": sorted(p.measures), "deadlines": sorted(d.isoformat() for d in p.deadlines),
            "contracts": list(p.samples)}


def build_prompt(pairs: list[tuple[SubjectProfile, SubjectProfile]]) -> str:
    return json.dumps({"pairs": [{"pair_id": str(i), "a_key": a.key, "b_key": b.key, "a": _shown(a), "b": _shown(b)}
                                 for i, (a, b) in enumerate(pairs)]},
                      ensure_ascii=False, default=str)


def _mapping(rows: list[dict]) -> tuple[tuple[str, str], ...]:
    return tuple((r["a"], r["b"]) for r in rows if r.get("a") and r.get("b") and r["a"] != r["b"])


def parse_answer(text: str, pairs: list[tuple[SubjectProfile, SubjectProfile]]) -> dict[tuple[str, str], Judgment | None]:
    out: dict[tuple[str, str], Judgment | None] = {(a.key, b.key): None for a, b in pairs}
    try:
        entries = json.loads(text)["judgments"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Subject merge answer is not the schema's JSON: %s", e)
        return out
    for entry in entries:
        try:
            a, b = pairs[int(entry["pair_id"])]
            # An answer filed under another pair's id would merge two subjects the judge never compared.
            if (entry.get("a_key"), entry.get("b_key")) != (a.key, b.key):
                logger.warning("Subject merge judgment for pair %s names %s/%s, not %s/%s; dropped",
                               entry.get("pair_id"), entry.get("a_key"), entry.get("b_key"), a.key, b.key)
                continue
            out[(a.key, b.key)] = Judgment(Verdict(entry["verdict"]), entry.get("reason") or "",
                                           _mapping(entry.get("entity_map") or []),
                                           _mapping(entry.get("measure_map") or []))
        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning("Subject merge judgment %r does not validate: %s", entry, e)
    return out


class SubjectMergeJudge:
    def __init__(self, asker: ClaudeAsker, max_workers: int = MAX_WORKERS, pairs_per_request: int = PAIRS_PER_REQUEST):
        self._asker = asker
        self._max_workers = max_workers
        self._pairs_per_request = pairs_per_request

    def judge(self, pairs: list[tuple[SubjectProfile, SubjectProfile]]) -> dict[tuple[str, str], Judgment | None]:
        """{(a key, b key): judgment}. A pair the model answered without a valid entry maps to None; one whose request
        no provider could take is left out, so it is asked again next pass."""
        chunks = [pairs[i:i + self._pairs_per_request] for i in range(0, len(pairs), self._pairs_per_request)]
        results: dict[tuple[str, str], Judgment | None] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._asker.ask, _SYSTEM, build_prompt(chunk), RESPONSE_SCHEMA, MAX_TOKENS, EFFORT):
                       chunk for chunk in chunks}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    results.update(parse_answer(future.result(), chunk))
                except Exception as e:
                    logger.warning("Subject merge request for %d pairs failed: %s", len(chunk), e)
                    if not is_unavailable(e):
                        results.update({(a.key, b.key): None for a, b in chunk})
        return results


@dataclasses.dataclass(frozen=True)
class MergeCheck:
    """What merging two subjects would change, before it is made."""
    # Relationships the merge adds: between a contract of each subject, on any venues.
    new_relationships: tuple[Derived, ...]
    # {event_id: securities} two outcomes of one event that would read as equivalent once merged; derivation would
    # drop them, so a merge that makes any is wrong somewhere.
    new_collisions: dict[int, frozenset[int]]

    @property
    def gain(self) -> int:
        return len(self.new_relationships)


def key_maps(judgment: Judgment | None) -> tuple[dict[str, str], dict[str, str]]:
    """The judge's maps as {subject b's key: subject a's key}, for entities and for measures."""
    if judgment is None:
        return {}, {}
    return {b: a for a, b in judgment.entity_map}, {b: a for a, b in judgment.measure_map}


def _joined_outcomes(contracts: list[Contract], aliases: tuple[StoredAlias, ...]) -> dict[int, set[int]]:
    """{event: its securities} for each event listing both keys of an entity alias as outcomes."""
    out: dict[int, set[int]] = {}
    for a in aliases:
        if a.kind != ENTITY_ALIAS:
            continue
        by_event: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        for c in contracts:
            if c.event_id is not None and not c.proposition.negated and c.proposition.entity in (a.key, a.canonical_key):
                by_event[c.event_id][c.proposition.entity].add(c.security_id)
        for event_id, keys in by_event.items():
            if len(keys) == 2:
                out.setdefault(event_id, set()).update(*keys.values())
    return out


def merge_aliases(judgment: Judgment | None) -> tuple[StoredAlias, ...]:
    """The key aliases a merge stores: subject b's keys read as the ones subject a uses for the same entity or
    quantity, as the judge mapped them."""
    entities, measures = key_maps(judgment)
    # A measure alias is looked up by the measure's name before any `:`; what follows (`margin:nxs`) is an entity
    # key, joined by the entity aliases, so an alias of the whole `margin:nxs` would never apply.
    return (tuple(StoredAlias(ENTITY_ALIAS, b, a) for b, a in sorted(entities.items()))
            + tuple(StoredAlias(MEASURE_ALIAS, b, a) for b, a in sorted(measures.items())
                    if ":" not in a and ":" not in b))


def with_samples(p: SubjectProfile, contracts: list[SubjectContract], samples: int = SAMPLES_PER_SUBJECT) -> SubjectProfile:
    """A profile summarised elsewhere (from every contract) with sample contracts for the judge to read."""
    return dataclasses.replace(p, samples=tuple(_sample(c) for c in _spread(contracts, samples)))


def check_merge(a: list[Contract], b: list[Contract], judgment: Judgment | None = None,
                a_aliases: Iterable[StoredAlias] = (), b_aliases: Iterable[StoredAlias] = ()) -> MergeCheck:
    """Derives subject b's contracts together with subject a's, b's entity and measure keys read as a's where the judge
    mapped them (the aliases the merge stores), each subject's own key aliases applied too, and keeps what neither
    subject derives alone."""
    subject = a[0].proposition.subject
    a_aliases, b_aliases = tuple(a_aliases), tuple(b_aliases)
    merged = list(a) + [dataclasses.replace(c, proposition=dataclasses.replace(c.proposition, subject=subject)) for c in b]
    mapped = merge_aliases(judgment)
    derived, collided = preview(merged, a_aliases + b_aliases + mapped)
    # Derivation leaves out an alias between two outcomes of one event; a judge mapping one is wrong about the merge.
    for event_id, ids in _joined_outcomes(merged, mapped).items():
        collided.setdefault(event_id, set()).update(ids)
    before: set[int] = set()
    for side, aliases in ((a, a_aliases), (b, b_aliases)):
        for ids in preview(list(side), aliases)[1].values():
            before |= ids

    a_ids = {c.security_id for c in a}
    new = tuple(d for d in derived if (d.security_id_a in a_ids) != (d.security_id_b in a_ids))
    return MergeCheck(new, {e: frozenset(ids) for e, ids in collided.items() if not ids <= before})
