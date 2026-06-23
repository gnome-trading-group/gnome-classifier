import logging

import anthropic
import voyageai

from classifier.cache import ClassifierCache
from classifier.constants import MIN_CONFIDENCE
from classifier.relationships.rule_based import find_hedgeable_pairs
from classifier.relationships.semantic import embed_events_voyage, find_semantic_matches
from classifier.relationships.structural import find_complement_pairs, find_mutually_exclusive_pairs
from classifier.types import Confidence, Embedding, RelationshipMatch, SecurityId
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)


def classify_relationships(
    registry: RegistryClient,
    anthropic_client: anthropic.Anthropic,
    voyage_client: voyageai.Client,
    new_security_ids: list[SecurityId],
    skip_judgment: bool = False,
    skip_semantic: bool = False,
    cache: ClassifierCache | None = None,
    min_confidence: Confidence = MIN_CONFIDENCE,
) -> dict[str, int]:
    events = registry.get_event()
    event_contracts = registry.get_event_contracts()
    securities = registry.get_security()
    currencies = registry.get_currency()
    existing_relationships = registry.get_contract_relationships()

    existing_pairs: set[tuple[SecurityId, SecurityId]] = {
        (rel.security_id_a, rel.security_id_b)
        for rel in existing_relationships
        if rel.method != "manual"
    }

    new_sids = set(new_security_ids)
    new_event_ids: set[EventId] | None = None
    if new_sids:
        new_event_ids = {ec.event_id for ec in event_contracts if ec.security_id in new_sids}

    pending: list[RelationshipMatch] = []

    pending.extend(find_complement_pairs(event_contracts))
    pending.extend(find_mutually_exclusive_pairs(event_contracts))
    pending.extend(find_hedgeable_pairs(event_contracts, events, securities, currencies))

    if not skip_semantic:
        embeddings = embed_events_voyage(voyage_client, events)
        semantic = find_semantic_matches(
            anthropic_client, events, event_contracts, embeddings,
            new_event_ids=new_event_ids, skip_judgment=skip_judgment, cache=cache,
        )
        logger.debug("semantic matches: %d", len(semantic))
        pending.extend(semantic)
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

    if pending_rels:
        created_list = registry.bulk_create_contract_relationships(pending_rels)
        written += len(created_list)

    return {
        "relationships_written": written,
        "relationships_skipped_low_confidence": skipped_low_confidence,
        "relationship_errors": 0,
    }
