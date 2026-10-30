"""The relationships worker's pass: read the propositions of new or changed securities, store them with their
subjects, then derive and reconcile the relationships of every subject that is due (its propositions changed, or a
person merged or unmerged it).

Propositions are only re-read when a security's venue source or the extractor changed (source hash, extractor
version), so a redelivered message costs nothing. A due subject re-derives only the derivation units holding a
contract that changed since its last derivation, and each unit's relationships are replaced as one set, which makes
redelivery and re-derivation idempotent. Only the fewest rows that rebuild a unit's relationships are stored;
consumers rebuild the rest (gnomepy `RelationshipGraph`). Bumping DERIVATION_VERSION therefore needs every subject's derived_at
cleared, so that all units are re-derived.
"""
import dataclasses
import logging
from datetime import datetime, timezone

from gnomepy.registry import RegistryClient

from classifier.db import ClassifierDB
from classifier.propositions.derive import DERIVATION_VERSION, Derived, derivation_units, derive_unit, minimal_rows
from classifier.propositions.drift import check_drift, sample
from classifier.propositions.extract import extract_propositions
from classifier.propositions.llm import PropositionExtractor
from classifier.propositions.subjects import (
    CombinedSubjectIndex,
    Embed,
    InMemorySubjectIndex,
    StoredSubjectIndex,
    embed_subjects,
)
from classifier.propositions.types import EXTRACTOR_VERSION
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


def _needs_reading(db: ClassifierDB, security_ids: list[int]) -> list[int]:
    stored = db.get_proposition_versions(security_ids)
    inputs = {i.security_id: i for i in db.get_extraction_inputs(security_ids)}
    return [sid for sid, item in inputs.items() if stored.get(sid) != (item.source_hash, EXTRACTOR_VERSION)]


def read_propositions(
    security_ids: list[int],
    *,
    db: ClassifierDB,
    registry: RegistryClient,
    extractor: PropositionExtractor | None,
    embed: Embed,
) -> PassResult:
    result = PassResult()
    changed = _needs_reading(db, security_ids)
    if not changed:
        return result
    # A binary's second side is read from its first, so whole events are read together.
    events = db.get_event_ids_for_securities(changed)
    items = db.get_extraction_inputs(sorted(db.get_security_ids_for_events(list(events))))
    index = CombinedSubjectIndex(StoredSubjectIndex(db, embed), InMemorySubjectIndex(embed))
    extractions, report = extract_propositions(items, extractor, index=index)
    report.log()
    result.read, result.pending = len(extractions), len(report.failed)

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
) -> tuple[PassResult, int]:
    """Reads up to `limit` open securities after `after` that still lack a current proposition (see
    `securities_needing_propositions`). Returns the pass and where the next one starts: past this page, or back at the
    start once the backlog is walked. Walking it, rather than always taking the first page, keeps securities nothing
    could read this time (the LLM is off, or failed on them) from blocking the rest."""
    due = db.securities_needing_propositions(EXTRACTOR_VERSION, limit, after)
    following = due[-1] if len(due) == limit else 0
    if not due:
        return PassResult(), following
    return read_propositions(due, db=db, registry=registry, extractor=extractor, embed=embed), following


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


def run_drift_check(*, db: ClassifierDB, extractor: PropositionExtractor, size: int, today: datetime | None = None) -> int:
    """Today's drift sample; returns how many fields disagreed (each is logged for the alarm)."""
    pool = db.sample_extraction_inputs(DRIFT_POOL)
    drifts = check_drift(sample(pool, size, (today or datetime.now(timezone.utc)).date()), extractor)
    logger.info("Drift check: %d disagreements", len(drifts))
    return len(drifts)
