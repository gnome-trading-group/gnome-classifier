"""The subject merge pass: finds subjects that name the same thing under two keys, has Claude judge them, checks what a
merge would change, and merges the pairs that pass (`subject_auto_merge_enabled`) or queues them for a person.

Each pass:
1. Summarises every canonical subject with open contracts and finds candidate pairs (`propositions/merges.py`), keeping
   pairs that touch a subject with at least `subject_merge_min_volume` of 24h volume (none by default). Embedding
   neighbours, the one costly source, are looked up only for subjects that changed since the last complete sweep.
2. Judges up to `subject_merge_judge_limit` pairs not judged before, busiest first. A sweep is complete once nothing
   is left to judge; until then the next pass starts from the same point, so pairs past the limit are found again.
3. Handles each pair judged SAME once: merges it into the subject with more propositions unless
   - exactly one key is a team's or player's season (`josh-allen:nfl-2026-27`): judged the same as an award or league
     subject only because that's all it holds yet, so merging would put the whole key on the other subject;
   - merging would make two outcomes of one event read alike (queued, COLLISION);
   - it adds no cross-venue relationship (skipped: nothing to gain);
   - two venues' prices for a relationship it adds disagree by more than `subject_merge_max_price_gap` (queued, PRICE).
     The price check is a second opinion, not a gate: a relationship with no price on either side (a venue that
     couldn't be read, an empty or closed book, Hyperliquid) isn't checked, and a merge with none priced goes ahead.
4. After a merge, renames the merged subject's entity and measure keys to the ones the kept subject uses, as the judge
   mapped them, so its propositions compare with the kept subject's (`ggo` and `goias`).

What each pair came to is remembered in Redis with what it rested on. A pair is judged again when either subject's
venues, entities or measures change; a pair that gained nothing is checked again when either subject's contracts
change; one only logged while auto-merge was off is handled again once it is on. A queued pair waits for a person, and
one a person merged, dismissed or unmerged is never merged or queued again. RELATED and DIFFERENT pairs are never merged or queued: a
RELATED pair that would gain relationships is a different contest on the same topic (an award and the MVP, a
nomination and the election), exactly the merge that must not happen.
"""
import dataclasses
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Protocol

import redis as redis_lib
from gnomepy.registry import RegistryClient

from classifier.db import ClassifierDB
from classifier.propositions.derive import EQUIVALENT, Contract, Derived
from classifier.propositions.merges import (
    Judgment,
    MergeCheck,
    SubjectContract,
    SubjectMergeJudge,
    SubjectProfile,
    Verdict,
    align,
    candidate_pairs,
    check_merge,
    key_maps,
    with_samples,
)
from classifier.propositions.quotes import ListingRef, Quote, read_quotes
from classifier.propositions.subjects import entity_of
from classifier.propositions.types import Proposition

logger = logging.getLogger(__name__)

MERGE_ACTOR = "classifier:subject-merger"
NEIGHBOURS = 5
MIN_NEIGHBOUR_SIMILARITY = 0.75
MEMORY_TTL_SECONDS = 30 * 86400
# A subject is embedded after its propositions are written, so one that changed just before the last sweep may have
# had no embedding to find neighbours with.
SWEEP_OVERLAP = timedelta(hours=1)
_SEASON_KEY = re.compile(r"^[a-z0-9.-]+:[a-z0-9]+-\d{4}(-\d{2})?$")
_PAIR_KEY = "merge:judged:{}|{}"
_SWEEP_KEY = "merge:last-sweep"

QuoteReader = Callable[[list[ListingRef]], dict[int, Quote]]


class Outcome(StrEnum):
    MERGED = "MERGED"
    # Passed every check while auto-merge was off.
    WOULD_MERGE = "WOULD_MERGE"
    QUEUED_PRICE = "QUEUED_PRICE"
    QUEUED_COLLISION = "QUEUED_COLLISION"
    NO_GAIN = "NO_GAIN"
    SEASON_KEY = "SEASON_KEY"


@dataclasses.dataclass(frozen=True)
class Remembered:
    # What the judge's answer rests on (`fingerprint`).
    fingerprint: str
    judgment: Judgment
    # Both subjects' proposition counts when the pair was last handled.
    counts: str
    # None until a SAME pair is handled.
    outcome: Outcome | None = None

    def to_json(self) -> dict:
        return {"fingerprint": self.fingerprint, "judgment": self.judgment.to_json(), "counts": self.counts,
                "outcome": self.outcome.value if self.outcome else None}

    @classmethod
    def from_json(cls, row: dict) -> "Remembered":
        return cls(row["fingerprint"], Judgment.from_json(row["judgment"]), row.get("counts", ""),
                   Outcome(row["outcome"]) if row.get("outcome") else None)


class MergeMemory(Protocol):
    def recall(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], Remembered]:
        """What is remembered of these pairs."""

    def remember(self, pair: tuple[str, str], remembered: Remembered) -> None:
        """Keeps what a pair came to (expiring, as every Redis key here must)."""

    def last_sweep(self) -> datetime | None:
        """When the last complete sweep started, in the database's `subject_clock` terms."""

    def set_last_sweep(self, at: datetime) -> None:
        ...


class RedisMergeMemory:
    def __init__(self, redis_url: str):
        self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def recall(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], Remembered]:
        if not pairs:
            return {}
        values = self._redis.mget([_PAIR_KEY.format(*pair) for pair in pairs])
        return {pair: Remembered.from_json(json.loads(v)) for pair, v in zip(pairs, values) if v is not None}

    def remember(self, pair: tuple[str, str], remembered: Remembered) -> None:
        self._redis.set(_PAIR_KEY.format(*pair), json.dumps(remembered.to_json()), ex=MEMORY_TTL_SECONDS)

    def last_sweep(self) -> datetime | None:
        value = self._redis.get(_SWEEP_KEY)
        return datetime.fromisoformat(value) if value else None

    def set_last_sweep(self, at: datetime) -> None:
        self._redis.set(_SWEEP_KEY, at.isoformat(), ex=MEMORY_TTL_SECONDS)


@dataclasses.dataclass
class MergePassResult:
    subjects: int = 0
    neighbour_lookups: int = 0
    neighbour_seconds: float = 0.0
    candidates: int = 0
    judged: int = 0
    left_to_judge: int = 0
    handled: int = 0
    merged: int = 0
    would_merge: int = 0
    queued_price: int = 0
    queued_collision: int = 0
    no_gain: int = 0
    season_key: int = 0
    relationships_added: int = 0
    keys_aligned: int = 0


def fingerprint(a: SubjectProfile, b: SubjectProfile) -> str:
    """Changes when either subject's venues, entities or measures do: what the judge's answer rests on."""
    parts = [[p.key, sorted(p.venues), sorted(p.entities), sorted(p.measures)] for p in (a, b)]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def is_season_key(key: str) -> bool:
    return bool(_SEASON_KEY.match(key))


def is_due(remembered: Remembered, counts: str, auto_merge: bool) -> bool:
    """Whether a pair judged SAME should be handled this pass: never yet, or something that could change its outcome
    has changed since."""
    if remembered.judgment.verdict != Verdict.SAME:
        return False
    if remembered.outcome is None:
        return True
    if remembered.outcome == Outcome.NO_GAIN:
        return remembered.counts != counts
    if remembered.outcome == Outcome.WOULD_MERGE:
        return auto_merge
    return False


def _date(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


@dataclasses.dataclass
class _Subjects:
    profiles: dict[str, SubjectProfile]
    ids: dict[str, int]
    volume: dict[str, float]
    changed_at: dict[str, datetime]


def _summaries(db: ClassifierDB) -> _Subjects:
    s = _Subjects({}, {}, {}, {})
    for subject_id, key, description, venues, winners, measures, deadlines, count, vol, changed in db.subject_summaries():
        if key == "unknown":
            continue
        entities = set(winners) | {e for m in measures if (e := entity_of(m)) is not None}
        s.profiles[key] = SubjectProfile(
            key=key, description=description, venues=frozenset(venues), entities=frozenset(entities),
            measures=frozenset(measures), deadlines=frozenset(d for text in deadlines if (d := _date(text))),
            proposition_count=count, samples=(),
        )
        s.ids[key] = subject_id
        s.volume[key] = vol
        s.changed_at[key] = changed
    return s


def _sampled(db: ClassifierDB, s: _Subjects, keys: set[str]) -> dict[str, SubjectProfile]:
    rows = db.subject_samples([s.ids[k] for k in keys])
    out = {}
    for key in keys:
        contracts = []
        for venue, title, description, outcome, subject_key, fields in rows.get(s.ids[key], []):
            try:
                contracts.append(SubjectContract(venue, title, outcome, description or "",
                                                 Proposition.from_fields(subject_key, fields)))
            except (ValueError, KeyError):
                continue
        out[key] = with_samples(s.profiles[key], contracts)
    return out


def _price_flags(check: MergeCheck, quotes: dict[int, Quote], max_gap: float) -> list[tuple[Derived, float, float]]:
    """New relationships two venues' mid prices contradict: equivalents more than `max_gap` apart, or an implication
    whose antecedent is priced more than `max_gap` above its consequent."""
    flags = []
    for d in check.new_relationships:
        qa, qb = quotes.get(d.security_id_a), quotes.get(d.security_id_b)
        ma, mb = (qa.mid if qa else None), (qb.mid if qb else None)
        if ma is None or mb is None:
            continue
        gap = abs(ma - mb) if d.relationship_type == EQUIVALENT else ma - mb
        if gap > max_gap:
            flags.append((d, ma, mb))
    return flags


def _listing_refs(db: ClassifierDB, contracts: list[Contract], security_ids: set[int]) -> list[ListingRef]:
    venue = {c.security_id: c.venue for c in contracts}
    refs, seen = [], set()
    for _, security_id, _, exchange_security_id in db.get_active_listings_for_securities(sorted(security_ids)):
        if security_id not in seen and exchange_security_id:
            seen.add(security_id)
            refs.append(ListingRef(security_id, venue[security_id], exchange_security_id))
    return refs


def _align_keys(db: ClassifierDB, registry: RegistryClient, alias_id: int, judgment: Judgment) -> int:
    """Renames the merged subject's entity and measure keys to the kept subject's, as the judge mapped them; returns
    how many propositions changed. The rows keep the subject they were read onto, so an unmerge still moves them back."""
    entities, measures = key_maps(judgment)
    if not entities and not measures:
        return 0
    rows = []
    for security_id, extracted_id, key, fields, method, version, source_hash in db.propositions_extracted_under(alias_id):
        proposition = Proposition.from_fields(key, fields)
        aligned = align(proposition, entities, measures)
        if aligned != proposition:
            rows.append(dict(security_id=security_id, extracted_subject_id=extracted_id, fields=aligned.to_fields(),
                             method=method, extractor_version=version, source_hash=source_hash))
    if rows:
        registry.upsert_contract_propositions(rows)
    return len(rows)


@dataclasses.dataclass
class _Context:
    db: ClassifierDB
    registry: RegistryClient
    quotes: QuoteReader
    auto_merge: bool
    max_price_gap: float
    result: MergePassResult
    queue: list[dict]


def _handle(ctx: _Context, s: _Subjects, a: str, b: str, judgment: Judgment) -> Outcome | None:
    """Merges, queues or skips one pair judged SAME; None when neither subject has open contracts left."""
    r = ctx.result
    if is_season_key(a) != is_season_key(b):
        r.season_key += 1
        return Outcome.SEASON_KEY
    keep, alias = (a, b) if s.profiles[a].proposition_count >= s.profiles[b].proposition_count else (b, a)
    judgment = judgment if keep == a else judgment.flipped()
    keep_contracts, _, _ = ctx.db.get_subject_contracts(s.ids[keep])
    alias_contracts, _, _ = ctx.db.get_subject_contracts(s.ids[alias])
    if not keep_contracts or not alias_contracts:
        return None
    check = check_merge(keep_contracts, alias_contracts, judgment)
    base = dict(alias_subject_id=s.ids[alias], canonical_subject_id=s.ids[keep], reason=judgment.reason, gain=check.gain)
    entity_map = [list(m) for m in judgment.entity_map]
    if check.new_collisions:
        r.queued_collision += 1
        ctx.queue.append(dict(base, hold="COLLISION", details={
            "collisions": {str(e): sorted(ids) for e, ids in check.new_collisions.items()}, "entity_map": entity_map}))
        return Outcome.QUEUED_COLLISION
    if check.gain == 0:
        r.no_gain += 1
        return Outcome.NO_GAIN
    contracts = keep_contracts + alias_contracts
    involved = {x for d in check.new_relationships for x in (d.security_id_a, d.security_id_b)}
    flags = _price_flags(check, ctx.quotes(_listing_refs(ctx.db, contracts, involved)), ctx.max_price_gap)
    if flags:
        r.queued_price += 1
        ctx.queue.append(dict(base, hold="PRICE", details={
            "prices": [{"security_id_a": d.security_id_a, "security_id_b": d.security_id_b,
                        "relationship_type": d.relationship_type, "mid_a": round(ma, 4), "mid_b": round(mb, 4)}
                       for d, ma, mb in flags[:20]],
            "entity_map": entity_map}))
        return Outcome.QUEUED_PRICE
    if not ctx.auto_merge:
        r.would_merge += 1
        logger.info("Subject merge (auto-merge off): %s -> %s, %d relationships: %s",
                    alias, keep, check.gain, judgment.reason)
        return Outcome.WOULD_MERGE
    ctx.registry.merge_proposition_subjects(from_subject_id=s.ids[alias], into_subject_id=s.ids[keep], actor=MERGE_ACTOR)
    r.merged += 1
    r.relationships_added += check.gain
    r.keys_aligned += _align_keys(ctx.db, ctx.registry, s.ids[alias], judgment)
    logger.info("Subject merged: %s -> %s, %d relationships: %s", alias, keep, check.gain, judgment.reason)
    return Outcome.MERGED


def run_merge_pass(
    *,
    db: ClassifierDB,
    registry: RegistryClient,
    judge: SubjectMergeJudge,
    memory: MergeMemory,
    auto_merge: bool,
    min_volume: float,
    judge_limit: int,
    max_price_gap: float,
    quotes: QuoteReader = read_quotes,
) -> MergePassResult:
    result = MergePassResult()
    started = db.subject_clock()
    last_sweep = memory.last_sweep()
    s = _summaries(db)
    result.subjects = len(s.profiles)

    fresh = [k for k in s.profiles if s.volume[k] >= min_volume
             and (last_sweep is None or s.changed_at[k] >= last_sweep - SWEEP_OVERLAP)]
    result.neighbour_lookups = len(fresh)
    by_id = {v: k for k, v in s.ids.items()}
    timer = time.monotonic()
    neighbours = {by_id[i]: [by_id[n] for n in ns if n in by_id]
                  for i, ns in db.subject_neighbours([s.ids[k] for k in fresh], NEIGHBOURS,
                                                     MIN_NEIGHBOUR_SIMILARITY).items()}
    result.neighbour_seconds = round(time.monotonic() - timer, 1)
    pairs = [(a, b) for a, b in candidate_pairs(s.profiles.values(), neighbours)
             if max(s.volume[a], s.volume[b]) >= min_volume]
    # A person's decision is final: a queued pair they merged or dismissed, or a merge they undid.
    decided = {frozenset((c.alias_subject_id, c.canonical_subject_id))
               for c in registry.get_subject_merge_candidates(status="ALL") if c.status != "PENDING"}
    decided |= {frozenset((m.alias_subject_id, m.canonical_subject_id))
                for m in registry.get_subject_merge_log(action="UNMERGE")}
    pairs = [p for p in pairs if frozenset((s.ids[p[0]], s.ids[p[1]])) not in decided]
    pairs.sort(key=lambda pair: -max(s.volume[pair[0]], s.volume[pair[1]]))
    result.candidates = len(pairs)

    prints = {p: fingerprint(s.profiles[p[0]], s.profiles[p[1]]) for p in pairs}
    counts = {p: f"{s.profiles[p[0]].proposition_count}:{s.profiles[p[1]].proposition_count}" for p in pairs}
    known = {p: m for p, m in memory.recall(pairs).items() if m.fingerprint == prints[p]}
    unjudged = [p for p in pairs if p not in known]
    todo = unjudged[:judge_limit]
    if todo:
        sampled = _sampled(db, s, {k for pair in todo for k in pair})
        for pair, judgment in judge.judge([(sampled[a], sampled[b]) for a, b in todo]).items():
            if judgment is not None:
                known[pair] = Remembered(prints[pair], judgment, counts[pair])
                memory.remember(pair, known[pair])
                result.judged += 1
    # Pairs past the limit, and any the model failed to answer, keep the sweep open so the next pass finds them again.
    result.left_to_judge = len(unjudged) - result.judged

    ctx = _Context(db, registry, quotes, auto_merge, max_price_gap, result, [])
    merged_now: set[str] = set()
    for pair in pairs:
        remembered = known.get(pair)
        if remembered is None or not is_due(remembered, counts[pair], auto_merge):
            continue
        if pair[0] in merged_now or pair[1] in merged_now:
            # Its contracts just changed; the next pass handles it.
            continue
        outcome = _handle(ctx, s, pair[0], pair[1], remembered.judgment)
        result.handled += 1
        memory.remember(pair, dataclasses.replace(remembered, counts=counts[pair], outcome=outcome))
        if outcome == Outcome.MERGED:
            merged_now |= set(pair)

    if ctx.queue:
        registry.upsert_subject_merge_candidates(ctx.queue)
    if result.left_to_judge == 0:
        memory.set_last_sweep(started)
    return result
