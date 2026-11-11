"""The key reconcile pass: finds one entity or quantity spelled two ways on a subject (`propositions/key_reconcile.py`),
has Claude say which keys name the same thing, checks what joining them would change, and stores the joins that pass as
the subject's key aliases, which derivation reads.

Each pass looks at the canonical subjects in `relationship_scope` that more than one venue lists and whose contracts
changed since the last complete sweep, and judges up to `key_reconcile_judge_limit` of their unmatched slots, busiest
subjects first. A slot is judged again only when its unmatched keys change. A join is held back (logged, not stored)
when it would make two outcomes of one event read alike, or when two venues' books contradict a relationship it adds by
more than `subject_merge_max_price_gap`, the merge pass's checks.
"""
import dataclasses
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol

import redis as redis_lib
import requests
from gnomepy.registry import RegistryClient

from classifier.db import ClassifierDB
from classifier.propositions.canonical import StoredAlias, apply_stored, canonical_readings
from classifier.propositions.derive import preview
from classifier.propositions.key_reconcile import KeyJudge, Reading, Slot, aliases, resolved, unmatched_slots
from classifier.propositions.merges import MergeCheck
from classifier.propositions.quotes import read_quotes
from classifier.propositions.scope import EVERYTHING, LLMScope
from classifier.stages.subject_merges import QuoteReader, listing_refs, price_flags

logger = logging.getLogger(__name__)

RECONCILE_ACTOR = "classifier:key-reconcile"
MEMORY_TTL_SECONDS = 30 * 86400
# Raised when what a slot is, or how a judged one is handled, changes: every slot is judged again once.
RECONCILE_RULES_VERSION = 1
SWEEP_OVERLAP = timedelta(hours=1)
_SLOT_KEY = f"keys:judged:{RECONCILE_RULES_VERSION}:{{}}"
_SWEEP_KEY = f"keys:last-sweep:{RECONCILE_RULES_VERSION}:{{}}"


class Outcome(StrEnum):
    STORED = "STORED"
    # Judged, and no key in it names another's entity.
    NOTHING_JOINED = "NOTHING_JOINED"
    HELD_COLLISION = "HELD_COLLISION"
    HELD_PRICE = "HELD_PRICE"
    # The registry refused the aliases (409).
    REFUSED = "REFUSED"


class SlotMemory(Protocol):
    def judged(self, prints: list[str]) -> set[str]:
        """The slot fingerprints already judged."""

    def remember(self, print_: str, outcome: Outcome) -> None:
        ...

    def last_sweep(self, scope: str) -> datetime | None:
        ...

    def set_last_sweep(self, at: datetime, scope: str) -> None:
        ...


class RedisSlotMemory:
    def __init__(self, redis_url: str):
        self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def judged(self, prints: list[str]) -> set[str]:
        if not prints:
            return set()
        values = self._redis.mget([_SLOT_KEY.format(p) for p in prints])
        return {p for p, v in zip(prints, values) if v is not None}

    def remember(self, print_: str, outcome: Outcome) -> None:
        self._redis.set(_SLOT_KEY.format(print_), outcome.value, ex=MEMORY_TTL_SECONDS)

    def last_sweep(self, scope: str) -> datetime | None:
        value = self._redis.get(_SWEEP_KEY.format(_scope_hash(scope)))
        return datetime.fromisoformat(value) if value else None

    def set_last_sweep(self, at: datetime, scope: str) -> None:
        self._redis.set(_SWEEP_KEY.format(_scope_hash(scope)), at.isoformat(), ex=MEMORY_TTL_SECONDS)


def _scope_hash(scope: str) -> str:
    return hashlib.sha256(scope.encode()).hexdigest()[:16]


@dataclasses.dataclass
class KeyReconcileResult:
    subjects: int = 0
    slots: int = 0
    judged: int = 0
    left_to_judge: int = 0
    stored: int = 0
    aliases_stored: int = 0
    nothing_joined: int = 0
    held_collision: int = 0
    held_price: int = 0
    refused: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def slot_print(slot: Slot) -> str:
    """Changes when the slot's unmatched keys, or the venues using them, do: what a judgment rests on."""
    parts = [slot.id, sorted([k.key, list(k.venues)] for k in slot.keys)]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:20]


@dataclasses.dataclass
class _Subject:
    subject_id: int
    key: str
    contracts: list
    stored: list[StoredAlias]
    slots: list[Slot]
    uses: Counter


def _subject(db: ClassifierDB, subject_id: int, key: str) -> _Subject | None:
    contracts, _, _ = db.get_subject_contracts(subject_id)
    if len({c.venue for c in contracts}) < 2:
        return None
    stored = db.get_subject_key_aliases(subject_id)
    titles = db.event_titles([c.security_id for c in contracts])
    raw = [(c.event_id, c.proposition, c.label) for c in contracts]
    applied, _ = apply_stored(raw, stored)
    readings = [Reading(c.security_id, c.venue, c.event_id, p, c.label, titles.get(c.security_id))
                for c, p in zip(contracts, canonical_readings(applied))]
    uses = Counter(k for r in readings for k in (r.proposition.entity, (r.proposition.measure or "").partition(":")[2])
                   if k)
    return _Subject(subject_id, key, contracts, stored, unmatched_slots(key, readings), uses)


def _check(subject: _Subject, new: list[StoredAlias], quotes: QuoteReader, max_gap: float, db: ClassifierDB) -> Outcome | None:
    """None when the joins pass; else why they are held back."""
    before_rows, before_collided = preview(subject.contracts, subject.stored)
    after_rows, after_collided = preview(subject.contracts, list(subject.stored) + new)
    collided = {e: frozenset(ids) for e, ids in after_collided.items() if not ids <= before_collided.get(e, set())}
    if collided:
        return Outcome.HELD_COLLISION
    known = {(d.security_id_a, d.security_id_b, d.relationship_type) for d in before_rows}
    added = tuple(d for d in after_rows if (d.security_id_a, d.security_id_b, d.relationship_type) not in known)
    involved = {x for d in added for x in (d.security_id_a, d.security_id_b)}
    if added and price_flags(MergeCheck(added, {}), quotes(listing_refs(db, subject.contracts, involved)), max_gap):
        return Outcome.HELD_PRICE
    return None


def run_key_reconcile(
    *,
    db: ClassifierDB,
    registry: RegistryClient,
    judge: KeyJudge,
    memory: SlotMemory,
    judge_limit: int,
    max_price_gap: float,
    quotes: QuoteReader = read_quotes,
    scope: LLMScope = EVERYTHING,
) -> KeyReconcileResult:
    result = KeyReconcileResult()
    started = db.subject_clock()
    last_sweep = memory.last_sweep(scope.fingerprint())
    summaries = [row for row in db.subject_summaries() if len(row[3]) >= 2 and row[1] != "unknown"]
    if not scope.unlimited:
        ids = db.subjects_in_scope(sorted(scope.categories), sorted(scope.tags))
        summaries = [row for row in summaries if row[0] in ids]
    summaries = [row for row in summaries if last_sweep is None or row[9] >= last_sweep - SWEEP_OVERLAP]
    summaries.sort(key=lambda row: -row[8])
    result.subjects = len(summaries)

    todo: list[tuple[_Subject, Slot, str]] = []
    for subject_id, key, *_ in summaries:
        subject = _subject(db, subject_id, key)
        if subject is None:
            continue
        prints = {slot_print(slot): slot for slot in subject.slots}
        result.slots += len(prints)
        done = memory.judged(list(prints))
        todo += [(subject, slot, p) for p, slot in prints.items() if p not in done]
    result.left_to_judge = max(0, len(todo) - judge_limit)
    todo = todo[:judge_limit]

    judgments = judge.judge([slot for _, slot, _ in todo]) if todo else {}
    result.input_tokens, result.output_tokens = getattr(judge, "input_tokens", 0), getattr(judge, "output_tokens", 0)
    for subject, slot, print_ in todo:
        judgment = judgments.get(slot.id)
        if judgment is None:
            # Unanswered or unavailable: the sweep stays open and it is asked again next pass.
            result.left_to_judge += 1
            continue
        result.judged += 1
        joins = aliases(slot, judgment, subject.uses)
        new = resolved(tuple(StoredAlias("ENTITY", key, target, flags) for key, (target, flags) in sorted(joins.items())),
                       subject.stored)
        if not new:
            result.nothing_joined += 1
            memory.remember(print_, Outcome.NOTHING_JOINED)
            continue
        held = _check(subject, new, quotes, max_price_gap, db)
        if held is not None:
            if held == Outcome.HELD_COLLISION:
                result.held_collision += 1
            else:
                result.held_price += 1
            logger.info("Key aliases held back (%s): subject=%s %s: %s", held.value, subject.key,
                        [(a.key, a.canonical_key) for a in new], judgment.reason)
            memory.remember(print_, held)
            continue
        try:
            registry.change_subject_key_aliases(subject_id=subject.subject_id, actor=RECONCILE_ACTOR, set=[
                dict(kind=a.kind, key=a.key, canonical_key=a.canonical_key, flags=list(a.flags), origin="RECONCILE",
                     reason=judgment.reason) for a in new])
        except requests.HTTPError as exc:
            # A 409 (the aliases would chain) holds this slot back, not the rest of the pass.
            if exc.response is None or exc.response.status_code != 409:
                raise
            result.refused += 1
            logger.error("key aliases refused: subject=%s %s: %s", subject.key, [(a.key, a.canonical_key) for a in new],
                         exc.response.text)
            memory.remember(print_, Outcome.REFUSED)
            continue
        subject.stored = list(subject.stored) + new
        result.stored += 1
        result.aliases_stored += len(new)
        logger.info("Key aliases stored: subject=%s %s: %s", subject.key, [(a.key, a.canonical_key, a.flags) for a in new],
                    judgment.reason)
        memory.remember(print_, Outcome.STORED)
    if result.left_to_judge == 0:
        memory.set_last_sweep(started, scope.fingerprint())
    return result
