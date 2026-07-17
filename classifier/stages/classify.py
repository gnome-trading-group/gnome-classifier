import logging

from classifier.cache import ClassifierCache
from classifier.constants import MIN_CONFIDENCE
from classifier.db import ClassifierDB
from classifier.utils import bulk_create_chunked
from classifier.relationships.rule_based import find_hedgeable_pairs
from classifier.relationships.semantic import (
    build_judgment_requests,
    derive_semantic_relationships,
    find_semantic_candidates,
    parse_judgment_responses,
)
from classifier.relationships.structural import find_complement_pairs, find_mutually_exclusive_pairs
from classifier.types import Confidence, JudgedRelationship, RelationshipMatch, RelationshipType, SecurityId, EventId
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)


def classify_structural(
    registry: RegistryClient,
    new_security_ids: list[SecurityId],
    *,
    db: ClassifierDB,
    min_confidence: Confidence = MIN_CONFIDENCE,
) -> dict:
    """Run structural + rule-based classification (complement, ME, hedgeable).

    Writes relationships to the registry and returns summary counts.
    """
    new_sids = set(new_security_ids)

    if new_sids:
        new_event_ids = db.get_event_ids_for_securities(list(new_sids))
        new_ecs = db.get_event_contracts_for_events(list(new_event_ids))
        new_events = db.get_events_for_ids(list(new_event_ids))
    else:
        new_ecs = db.get_all_event_contracts()
        new_events = db.get_unresolved_events()

    hedge_keywords = db.get_hedge_keywords()

    existing_relationships = db.get_contract_relationships_for_securities(new_security_ids)
    existing_pairs: set[tuple[SecurityId, SecurityId]] = set()
    for rel in existing_relationships:
        if rel.method != "manual":
            existing_pairs.add((rel.security_id_a, rel.security_id_b))
            existing_pairs.add((rel.security_id_b, rel.security_id_a))

    pending: list[RelationshipMatch] = []
    pending.extend(find_complement_pairs(new_ecs))
    pending.extend(find_mutually_exclusive_pairs(new_ecs))
    pending.extend(find_hedgeable_pairs(new_ecs, new_events, hedge_keywords))

    logger.debug("structural pending: %d, new_sids: %d", len(pending), len(new_sids))
    best: dict[tuple[SecurityId, SecurityId], tuple[str, Confidence, str]] = {}
    for sid_a, sid_b, rel_type, conf, method in pending:
        if new_sids and sid_a not in new_sids and sid_b not in new_sids:
            continue
        pair = (sid_a, sid_b)
        if pair in existing_pairs:
            continue
        if pair not in best or conf > best[pair][1]:
            best[pair] = (rel_type, conf, method)

    written = 0
    skipped_low_confidence = 0
    pending_rels: list[dict] = []

    for (sid_a, sid_b), (rel_type, conf, method) in best.items():
        if conf < min_confidence:
            skipped_low_confidence += 1
            continue
        pending_rels.append(dict(
            security_id_a=sid_a,
            security_id_b=sid_b,
            relationship_type=rel_type,
            confidence=conf,
            method=method,
        ))

    for _, chunk in bulk_create_chunked(pending_rels, "structural relationships", batch_size=1000):
        created_list = registry.bulk_create_contract_relationships(chunk)
        written += len(created_list)

    return {
        "relationships_written": written,
        "relationships_skipped_low_confidence": skipped_low_confidence,
        "relationship_errors": 0,
    }


def prepare_semantic_batch(
    new_security_ids: list[SecurityId],
    *,
    cache: ClassifierCache | None = None,
    db: ClassifierDB,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Find candidate event pairs, check cache, build Claude API requests.

    Returns (api_requests, pending_context, cached_results) where:
    - api_requests: list of request dicts ready for batch_client.create_messages or submit_batch
    - pending_context: JSON-serializable context for parse_judgment_responses
    - cached_results: already-resolved JudgedRelationship-like dicts from cache

    Does NOT submit the batch — caller chooses sync (create_messages) or async (submit_batch).
    """
    new_sids = set(new_security_ids)

    if new_sids:
        new_event_ids: set[EventId] | None = db.get_event_ids_for_securities(list(new_sids))
        event_contracts = db.get_event_contracts_for_events(list(new_event_ids))
        events = db.get_events_for_ids(list(new_event_ids))
    else:
        new_event_ids = None
        event_contracts = db.get_all_event_contracts()
        events = db.get_unresolved_events()

    embeddings = db.get_embeddings(list(new_event_ids)) if new_event_ids else {}
    logger.info("Loaded %d embeddings for new events", len(embeddings))

    pending_pairs, cached_judged = find_semantic_candidates(
        events, event_contracts, embeddings,
        db=db, new_event_ids=new_event_ids, cache=cache,
    )
    logger.info("Semantic candidates: %d pending, %d cached", len(pending_pairs), len(cached_judged))

    api_requests, pending_context = build_judgment_requests(pending_pairs)

    cached_results = [
        {
            "security_id_a": r.security_id_a,
            "security_id_b": r.security_id_b,
            "relationship_type": r.relationship_type,
            "confidence": r.confidence,
        }
        for r in cached_judged
    ]

    return api_requests, pending_context, cached_results


def process_semantic_results(
    registry: RegistryClient,
    responses: dict[str, object],
    pending_context: list[dict],
    cached_results: list[dict],
    new_security_ids: list[SecurityId],
    *,
    cache: ClassifierCache | None = None,
    db: ClassifierDB,
    min_confidence: Confidence = MIN_CONFIDENCE,
) -> dict:
    """Parse Claude responses, combine with cached results, dedup, and write relationships.

    Returns summary counts of relationships written.
    """
    event_contracts = db.get_all_event_contracts()

    judged = parse_judgment_responses(responses, pending_context, cache)

    cached_judged = [
        JudgedRelationship(
            security_id_a=r["security_id_a"],
            security_id_b=r["security_id_b"],
            relationship_type=RelationshipType(r["relationship_type"]),
            confidence=float(r["confidence"]),
        )
        for r in cached_results
    ]

    all_matches = derive_semantic_relationships(judged + cached_judged, event_contracts)

    existing_relationships = db.get_contract_relationships_for_securities(new_security_ids)
    existing_pairs: set[tuple[SecurityId, SecurityId]] = set()
    for rel in existing_relationships:
        if rel.method != "manual":
            existing_pairs.add((rel.security_id_a, rel.security_id_b))
            existing_pairs.add((rel.security_id_b, rel.security_id_a))

    new_sids = set(new_security_ids)
    best: dict[tuple[SecurityId, SecurityId], tuple[str, Confidence, str]] = {}
    for match in all_matches:
        sid_a, sid_b = match.security_id_a, match.security_id_b
        rel_type, conf, method = match.relationship_type, match.confidence, match.method
        if new_sids and sid_a not in new_sids and sid_b not in new_sids:
            continue
        pair = (sid_a, sid_b)
        if pair in existing_pairs:
            continue
        if pair not in best or conf > best[pair][1]:
            best[pair] = (rel_type, conf, method)

    written = 0
    skipped_low_confidence = 0
    pending_rels: list[dict] = []

    for (sid_a, sid_b), (rel_type, conf, method) in best.items():
        if conf < min_confidence:
            skipped_low_confidence += 1
            continue
        pending_rels.append(dict(
            security_id_a=sid_a,
            security_id_b=sid_b,
            relationship_type=rel_type,
            confidence=conf,
            method=method,
        ))

    for _, chunk in bulk_create_chunked(pending_rels, "semantic relationships", batch_size=1000):
        created_list = registry.bulk_create_contract_relationships(chunk)
        written += len(created_list)

    return {
        "relationships_written": written,
        "relationships_skipped_low_confidence": skipped_low_confidence,
        "relationship_errors": 0,
    }
