import logging

import voyageai

from classifier.cache import ClassifierCache
from classifier.client import BatchAnthropicClient
from classifier.constants import MIN_CONFIDENCE
from classifier.db import ClassifierDB
from classifier.utils import bulk_create_chunked
from classifier.relationships.rule_based import find_hedgeable_pairs
from classifier.relationships.semantic import embed_events_voyage, find_semantic_matches
from classifier.relationships.structural import find_complement_pairs, find_mutually_exclusive_pairs
from classifier.types import Confidence, RelationshipMatch, SecurityId, EventId
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)


def classify_relationships(
    registry: RegistryClient,
    batch_client: BatchAnthropicClient,
    voyage_client: voyageai.Client,
    new_security_ids: list[SecurityId],
    *,
    skip_judgment: bool = False,
    skip_semantic: bool = False,
    cache: ClassifierCache | None = None,
    db: ClassifierDB,
    min_confidence: Confidence = MIN_CONFIDENCE,
) -> dict[str, int]:
    # ── Load data ────────────────────────────────────────────────────────────
    events = db.get_unresolved_events()
    event_contracts = db.get_all_event_contracts()
    securities = db.get_all_securities()
    currencies = db.get_all_currencies()

    new_sids = set(new_security_ids)
    new_event_ids: set[EventId] | None = None
    if new_sids:
        new_event_ids = {ec.event_id for ec in event_contracts if ec.security_id in new_sids}

    existing_relationships = db.get_contract_relationships_for_securities(new_security_ids)
    existing_pairs: set[tuple[SecurityId, SecurityId]] = set()
    for rel in existing_relationships:
        if rel.method != "manual":
            existing_pairs.add((rel.security_id_a, rel.security_id_b))
            existing_pairs.add((rel.security_id_b, rel.security_id_a))

    # ── Structural + rule-based relationships ────────────────────────────────
    # Scoped to new events only — prior runs already wrote relationships for existing events.
    # Semantic matching below still receives full event_contracts for neighbor event details.
    new_ecs = [ec for ec in event_contracts if ec.event_id in new_event_ids] if new_event_ids else event_contracts
    new_events = [e for e in events if e.event_id in new_event_ids] if new_event_ids else events

    pending: list[RelationshipMatch] = []
    pending.extend(find_complement_pairs(new_ecs))
    pending.extend(find_mutually_exclusive_pairs(new_ecs))
    pending.extend(find_hedgeable_pairs(new_ecs, new_events, securities, currencies))

    # ── Semantic relationships ───────────────────────────────────────────────
    if not skip_semantic:
        stored_embeddings = db.get_embeddings([ev.event_id for ev in events])
        logger.info("Loaded %d event embeddings from DB", len(stored_embeddings))
        embeddings = embed_events_voyage(voyage_client, events, cached_embeddings=stored_embeddings)
        new_embeddings = {eid: emb for eid, emb in embeddings.items() if eid not in stored_embeddings}
        if new_embeddings:
            logger.info("Writing %d new event embeddings to DB", len(new_embeddings))
            db.put_embeddings(new_embeddings)
        semantic = find_semantic_matches(
            batch_client, events, event_contracts, embeddings,
            new_event_ids=new_event_ids, skip_judgment=skip_judgment, cache=cache,
            db=db,
        )
        logger.debug("semantic matches: %d", len(semantic))
        pending.extend(semantic)

    # ── Dedup + filter to new securities ────────────────────────────────────
    logger.debug("pending total: %d, new_sids: %d", len(pending), len(new_sids))
    best: dict[tuple[SecurityId, SecurityId], tuple[str, Confidence, str]] = {}
    for sid_a, sid_b, rel_type, conf, method in pending:
        if new_sids and sid_a not in new_sids and sid_b not in new_sids:
            continue
        pair = (sid_a, sid_b)
        if pair in existing_pairs:
            continue
        if pair not in best or conf > best[pair][1]:
            best[pair] = (rel_type, conf, method)
    logger.debug("best after dedup: %d", len(best))

    # ── Write to registry ────────────────────────────────────────────────────
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

    for _, chunk in bulk_create_chunked(pending_rels, "relationships"):
        created_list = registry.bulk_create_contract_relationships(chunk)
        written += len(created_list)

    return {
        "relationships_written": written,
        "relationships_skipped_low_confidence": skipped_low_confidence,
        "relationship_errors": 0,
    }
