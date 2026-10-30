"""The relationships worker's pass: read the propositions of new or changed securities, store them with their
subjects, then derive and reconcile the relationships of every subject that is due (its propositions changed, or a
person merged or unmerged it).

Propositions are only read for securities whose venue source or the extractor changed (source hash, extractor
version), so a redelivered message costs nothing, and only those securities are read: the rest of their event is left
alone (a second side is negated from its first side's stored proposition). A security that failed to read, or that
only the LLM can read while it is off, waits (see `attempts`) instead of being asked again every pass. A due subject re-derives only the derivation units holding a
contract that changed since its last derivation, and each unit's relationships are replaced as one set, which makes
redelivery and re-derivation idempotent. Only the fewest rows that rebuild a unit's relationships are stored;
consumers rebuild the rest (gnomepy `RelationshipGraph`). A new DERIVATION_VERSION re-derives every subject once (see
`rederive_on_new_version`), since an incremental pass would leave rows the old rules wrote.
"""
import dataclasses
import logging
from datetime import datetime, timezone
from typing import Protocol

from gnomepy.registry import RegistryClient

from classifier.db import ClassifierDB
from classifier.propositions.derive import DERIVATION_VERSION, Derived, derivation_units, derive_unit, minimal_rows
from classifier.propositions.drift import check_drift, sample
from classifier.propositions.attempts import AttemptStore, record, waiting
from classifier.propositions.extract import extract_propositions
from classifier.propositions.inputs import ExtractionInput, is_primary_side
from classifier.propositions.llm import PropositionExtractor
from classifier.propositions.subjects import (
    CombinedSubjectIndex,
    Embed,
    InMemorySubjectIndex,
    StoredSubjectIndex,
    embed_subjects,
)
from classifier.propositions.types import EXTRACTOR_VERSION, Method
from classifier.propositions.venue import READERS_VERSION
from classifier.utils import bulk_create_chunked

logger = logging.getLogger(__name__)

DERIVED_METHOD = "derived"
DEFAULT_DERIVE_LIMIT = 200
# Rows per reconcile request: the registry writes each row in turn inside API Gateway's 29 s, and the body has to stay
# under Lambda's 6 MB.
RECONCILE_CHUNK_ROWS = 2000
DRIFT_POOL = 500


@dataclasses.dataclass
class PassResult:
    read: int = 0
    stored: int = 0
    pending: int = 0
    subjects_derived: int = 0
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    backing_off: int = 0
    parked: int = 0
    gave_up: int = 0

    def add(self, other: "PassResult") -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def _plan_reads(
    db: ClassifierDB, security_ids: list[int], attempts: AttemptStore | None, llm_enabled: bool, result: PassResult,
) -> tuple[list[ExtractionInput], dict]:
    """The inputs to read this pass, and stored first sides for second sides read without theirs. A security is read
    when its source or the extractor changed and it isn't waiting out an earlier attempt; a changed first side brings
    its second side (a negation, no model call)."""
    inputs = {i.security_id: i for i in db.get_extraction_inputs(security_ids)}
    stored = db.get_proposition_versions(security_ids)
    due = {sid for sid, item in inputs.items() if stored.get(sid) != (item.source_hash, EXTRACTOR_VERSION)}
    if attempts is not None and due:
        skip = waiting(attempts, {sid: inputs[sid].source_hash for sid in due}, EXTRACTOR_VERSION, llm_enabled,
                       READERS_VERSION)
        result.backing_off += len(skip)
        due -= skip
    if not due:
        return [], {}

    events = db.get_event_ids_for_securities(sorted(due))
    siblings = {i.security_id: i for i in db.get_extraction_inputs(sorted(db.get_security_ids_for_events(list(events))))}
    by_event: dict[int, list[ExtractionInput]] = {}
    for item in siblings.values():
        by_event.setdefault(item.event_id, []).append(item)
    read = set(due)
    first_of: dict[int, int] = {}
    for sid in due:
        item = siblings.get(sid) or inputs[sid]
        others = [o for o in by_event.get(item.event_id, []) if o.security_id != sid]
        if is_primary_side(item):
            read.update(o.security_id for o in others if not is_primary_side(o))
        else:
            first = next((o for o in others if is_primary_side(o)), None)
            if first is not None:
                first_of[sid] = first.security_id
    stored_firsts = db.get_stored_propositions(sorted(set(first_of.values()) - read))
    partners = {}
    for second, first in first_of.items():
        if first in read:
            continue
        held = stored_firsts.get(first)
        if held is not None and held[2] == siblings[first].source_hash:
            partners[second] = (held[0], Method(held[1]))
        elif attempts is None or not waiting(attempts, {first: siblings[first].source_hash}, EXTRACTOR_VERSION,
                                             llm_enabled, READERS_VERSION):
            read.add(first)
    return [siblings.get(sid) or inputs[sid] for sid in sorted(read)], partners


def read_propositions(
    security_ids: list[int],
    *,
    db: ClassifierDB,
    registry: RegistryClient,
    extractor: PropositionExtractor | None,
    embed: Embed,
    attempts: AttemptStore | None = None,
) -> PassResult:
    result = PassResult()
    items, partners = _plan_reads(db, security_ids, attempts, extractor is not None, result)
    if not items:
        return result
    index = CombinedSubjectIndex(StoredSubjectIndex(db, embed), InMemorySubjectIndex(embed))
    extractions, report = extract_propositions(items, extractor, index=index, partners=partners)
    report.log()
    result.read, result.pending = len(extractions), len(report.failed)
    if attempts is not None:
        source_of = {i.security_id: i.source_hash for i in items}
        # A reader that fell back with the LLM off records nothing: the contract stays due (free to retry with code
        # readers, and the VenueReaderFallbacks alarm reports it) so a reader fix reads it on the next pass.
        unrecorded = set(report.unavailable) | (set(report.fell_back) if extractor is None else set())
        needs_llm = set(report.needs_llm)
        result.gave_up, result.parked = record(
            attempts,
            failed=[(sid, source_of[sid]) for sid in report.failed if sid not in unrecorded and sid not in needs_llm],
            needs_llm=[(sid, source_of[sid]) for sid in report.failed if sid in needs_llm],
            read=sorted(extractions),
            extractor_version=EXTRACTOR_VERSION,
            readers_version=READERS_VERSION,
        )

    subjects = {e.proposition.subject for e in extractions.values()}
    descriptions = {k: report.known[k].description if k in report.known else "" for k in subjects}
    subject_id_by_key: dict[str, int] = {}
    for _, chunk in bulk_create_chunked([{"key": k, "description": descriptions[k]} for k in sorted(subjects)],
                                        "proposition subjects"):
        for row in registry.create_proposition_subjects(chunk):
            subject_id_by_key[row["key"]] = row["subject_id"]

    rows = [
        dict(security_id=sid, extracted_subject_id=subject_id_by_key[e.proposition.subject],
             fields=e.proposition.to_fields(), method=e.method.value, extractor_version=e.extractor_version,
             source_hash=e.source_hash)
        for sid, e in sorted(extractions.items())
    ]
    for _, chunk in bulk_create_chunked(rows, "contract propositions"):
        result.stored += sum(1 for r in registry.upsert_contract_propositions(chunk) if r.get("changed"))
    embedded = embed_subjects(db, embed)
    if embedded:
        logger.info("Embedded %d proposition subjects", embedded)
    return result


def read_backlog(
    *,
    db: ClassifierDB,
    registry: RegistryClient,
    extractor: PropositionExtractor | None,
    embed: Embed,
    limit: int,
    after: int = 0,
    attempts: AttemptStore | None = None,
) -> tuple[PassResult, int]:
    """Reads up to `limit` open securities after `after` that still lack a current proposition (see
    `securities_needing_propositions`). Returns the pass and where the next one starts: past this page, or back at the
    start once the backlog is walked. Walking it, rather than always taking the first page, keeps securities nothing
    could read this time (the LLM is off, or failed on them) from blocking the rest."""
    due = db.securities_needing_propositions(EXTRACTOR_VERSION, limit, after)
    following = due[-1] if len(due) == limit else 0
    if not due:
        return PassResult(), following
    return read_propositions(due, db=db, registry=registry, extractor=extractor, embed=embed, attempts=attempts), following


def _chunks(security_ids: list[int], rows: list[Derived]) -> list[tuple[list[int], list[Derived]]]:
    """Splits a unit into reconcile requests of about RECONCILE_CHUNK_ROWS rows, each carrying every row that touches
    its securities, so each request replaces exactly its securities' rows (a row between two chunks goes in both)."""
    touching: dict[int, list[int]] = {sid: [] for sid in security_ids}
    for i, row in enumerate(rows):
        touching[row.security_id_a].append(i)
        touching[row.security_id_b].append(i)
    out: list[tuple[list[int], list[Derived]]] = []
    ids: list[int] = []
    picked: set[int] = set()
    for sid in security_ids:
        if ids and len(picked | set(touching[sid])) > RECONCILE_CHUNK_ROWS:
            out.append((ids, [rows[i] for i in sorted(picked)]))
            ids, picked = [], set()
        ids.append(sid)
        picked.update(touching[sid])
    if ids:
        out.append((ids, [rows[i] for i in sorted(picked)]))
    return out


def _derive_subject(db: ClassifierDB, registry: RegistryClient, subject_id: int, result: PassResult):
    contracts, changed, read_at = db.get_subject_contracts(subject_id)
    units = derivation_units(contracts)
    requests: list[tuple[list[int], list[Derived]]] = []
    for unit in units:
        ids = [c.security_id for c in unit.contracts]
        if not changed.intersection(ids):
            continue
        rows, reduced = minimal_rows(unit, derive_unit(unit))
        if not reduced:
            # Counted by the <Name>BatchFailures alarm: the unit is stored unreduced, correct but larger.
            logger.error("relationship reduction mismatch: subject=%d unit=%d", subject_id, ids[0])
        requests.extend(_chunks(ids, rows))
    # A changed contract no rule can use (a field left unknown) still loses the relationships it had.
    in_units = {c.security_id for unit in units for c in unit.contracts}
    unusable = sorted(changed - in_units)
    for start in range(0, len(unusable), RECONCILE_CHUNK_ROWS):
        requests.append((unusable[start:start + RECONCILE_CHUNK_ROWS], []))
    requests = requests or [([], [])]
    for i, (ids, rows) in enumerate(requests):
        # Only the last request marks the subject derived, so a failure part way leaves it due and the next pass
        # repeats the whole set.
        last = i == len(requests) - 1
        counts = registry.reconcile_contract_relationships(
            subject_id=subject_id if last else None,
            security_ids=ids,
            relationships=[
                dict(security_id_a=d.security_id_a, security_id_b=d.security_id_b, relationship_type=d.relationship_type,
                     flags=list(d.flags), evidence=d.evidence)
                for d in rows
            ],
            method=DERIVED_METHOD,
            derivation_version=DERIVATION_VERSION,
            derived_at=read_at.astimezone(timezone.utc).isoformat() if last else None,
        )
        result.inserted += counts.get("inserted", 0)
        result.updated += counts.get("updated", 0)
        result.deleted += counts.get("deleted", 0)
    result.subjects_derived += 1


def derive_due_subjects(
    *,
    db: ClassifierDB,
    registry: RegistryClient,
    limit: int = DEFAULT_DERIVE_LIMIT,
    result: PassResult | None = None,
) -> PassResult:
    """Derives and reconciles up to `limit` subjects due a derivation."""
    result = result or PassResult()
    for subject_id in db.dirty_subjects(limit):
        try:
            _derive_subject(db, registry, subject_id, result)
        except Exception:
            # One subject that can't be derived stays due and is retried, without holding up the rest. The
            # <Name>BatchFailures alarm counts this line.
            logger.exception("subject derivation failed: subject=%d", subject_id)
    return result


class VersionMarker(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> object: ...


DERIVATION_VERSION_KEY = "derive:version"


def rederive_on_new_version(db: ClassifierDB, marker: VersionMarker) -> int:
    """Marks every subject due a full derivation the first time this DERIVATION_VERSION runs; returns how many. The
    marker is one Redis key, kept without an expiry (one key can't crowd the node); losing it only costs one extra full
    derivation."""
    if marker.get(DERIVATION_VERSION_KEY) == DERIVATION_VERSION:
        return 0
    count = db.reset_derivations()
    marker.set(DERIVATION_VERSION_KEY, DERIVATION_VERSION)
    logger.info("Derivation version %s: %d subjects due a full derivation", DERIVATION_VERSION, count)
    return count


def run_drift_check(*, db: ClassifierDB, extractor: PropositionExtractor, size: int, today: datetime | None = None) -> int:
    """Today's drift sample; returns how many fields disagreed (each is logged for the alarm)."""
    pool = db.sample_extraction_inputs(DRIFT_POOL)
    drifts = check_drift(sample(pool, size, (today or datetime.now(timezone.utc)).date()), extractor)
    logger.info("Drift check: %d disagreements", len(drifts))
    return len(drifts)
