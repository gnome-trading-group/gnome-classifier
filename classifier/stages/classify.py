import dataclasses
import logging

from classifier.cache import ClassifierCache
from classifier.client import BatchAnthropicClient
from classifier.constants import (
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    DEFAULT_HEDGEABLE_WITH_CONFIDENCE,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_NEIGHBOR_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_JUDGMENT_MODEL,
    DEFAULT_STRUCTURAL_CONFIDENCE,
    DEFAULT_SYNC_THRESHOLD,
)
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


def _load_event_data(
    db: ClassifierDB,
    new_security_ids: list[SecurityId],
) -> tuple[set[SecurityId], set[EventId] | None, list, list]:
    new_sids = set(new_security_ids)
    if new_sids:
        new_event_ids = db.get_event_ids_for_securities(list(new_sids))
        event_contracts = db.get_event_contracts_for_events(list(new_event_ids))
        events = db.get_events_for_ids(list(new_event_ids))
    else:
        new_event_ids = None
        event_contracts = db.get_all_event_contracts()
        events = db.get_unresolved_events()
    return new_sids, new_event_ids, event_contracts, events


def _dedup_and_write_relationships(
    registry: RegistryClient,
    matches: list[RelationshipMatch],
    new_security_ids: list[SecurityId],
    *,
    db: ClassifierDB,
    min_confidence: Confidence = DEFAULT_MIN_CONFIDENCE,
    label: str = "relationships",
) -> tuple[dict, list[dict]]:
    existing_relationships = db.get_contract_relationships_for_securities(new_security_ids)
    existing_pairs: set[tuple[SecurityId, SecurityId]] = set()
    for rel in existing_relationships:
        if rel.method != "manual":
            existing_pairs.add((rel.security_id_a, rel.security_id_b))
            existing_pairs.add((rel.security_id_b, rel.security_id_a))

    new_sids = set(new_security_ids)
    best: dict[tuple[SecurityId, SecurityId], tuple[str, Confidence, str]] = {}
    for match in matches:
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
    written_rels: list[dict] = []

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

    for _, chunk in bulk_create_chunked(pending_rels, label, batch_size=1000):
        created_list = registry.bulk_create_contract_relationships(chunk)
        written += len(created_list)
        written_rels.extend(created_list)

    return {
        "relationships_written": written,
        "relationships_skipped_low_confidence": skipped_low_confidence,
        "relationship_errors": 0,
    }, written_rels


def classify_structural(
    registry: RegistryClient,
    new_security_ids: list[SecurityId],
    *,
    db: ClassifierDB,
    min_confidence: Confidence = DEFAULT_MIN_CONFIDENCE,
    structural_confidence: float = DEFAULT_STRUCTURAL_CONFIDENCE,
    hedgeable_with_confidence: float = DEFAULT_HEDGEABLE_WITH_CONFIDENCE,
) -> tuple[dict, list[dict]]:
    """Run structural + rule-based classification (complement, ME, hedgeable).

    Writes relationships to the registry and returns (summary_counts, written_relationships).
    """
    new_sids, _, event_contracts, events = _load_event_data(db, new_security_ids)
    hedge_keywords = db.get_hedge_keywords()

    pending: list[RelationshipMatch] = []
    pending.extend(find_complement_pairs(event_contracts, confidence=structural_confidence))
    pending.extend(find_mutually_exclusive_pairs(event_contracts, confidence=structural_confidence))
    pending.extend(find_hedgeable_pairs(event_contracts, events, hedge_keywords, confidence=hedgeable_with_confidence))

    logger.debug("structural pending: %d, new_sids: %d", len(pending), len(new_sids))
    return _dedup_and_write_relationships(
        registry, pending, new_security_ids,
        db=db, min_confidence=min_confidence, label="structural relationships",
    )


def prepare_semantic_batch(
    new_security_ids: list[SecurityId],
    *,
    cache: ClassifierCache | None = None,
    db: ClassifierDB,
    threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    neighbor_limit: int = DEFAULT_NEIGHBOR_SEARCH_LIMIT,
    allowed_categories: set[str] | None = None,
    model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Find candidate event pairs, check cache, build Claude API requests.

    Returns (api_requests, pending_context, cached_results) where:
    - api_requests: list of request dicts ready for batch_client.create_messages
    - pending_context: JSON-serializable context for parse_judgment_responses
    - cached_results: already-resolved JudgedRelationship-like dicts from cache

    Does NOT submit the batch — caller passes these to batch_client.create_messages.
    """
    _, new_event_ids, event_contracts, events = _load_event_data(db, new_security_ids)

    embeddings = db.get_embeddings(list(new_event_ids)) if new_event_ids else {}
    logger.info("Loaded %d embeddings for new events", len(embeddings))

    pending_pairs, cached_judged = find_semantic_candidates(
        events, event_contracts, embeddings,
        db=db, new_event_ids=new_event_ids, cache=cache,
        threshold=threshold, neighbor_limit=neighbor_limit,
        allowed_categories=allowed_categories, model=model,
    )
    logger.info("Semantic candidates: %d pending, %d cached", len(pending_pairs), len(cached_judged))

    api_requests, pending_context = build_judgment_requests(pending_pairs, model=model)

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
    min_confidence: Confidence = DEFAULT_MIN_CONFIDENCE,
    model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL,
) -> tuple[dict, list[dict]]:
    """Parse Claude responses, combine with cached results, dedup, and write relationships.

    Returns (summary_counts, written_relationships).
    """
    event_contracts = db.get_event_contracts_for_securities(new_security_ids)

    judged = parse_judgment_responses(responses, pending_context, cache, model=model)

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
    return _dedup_and_write_relationships(
        registry, all_matches, new_security_ids,
        db=db, min_confidence=min_confidence, label="semantic relationships",
    )


@dataclasses.dataclass
class ClassificationResult:
    structural: dict
    semantic: dict
    written_relationships: list[dict] = dataclasses.field(default_factory=list)

    @property
    def relationships_written(self) -> int:
        return self.structural.get("relationships_written", 0) + self.semantic.get("relationships_written", 0)

    @property
    def relationships_skipped_low_confidence(self) -> int:
        return (
            self.structural.get("relationships_skipped_low_confidence", 0)
            + self.semantic.get("relationships_skipped_low_confidence", 0)
        )


def classify_semantic_sync(
    registry: RegistryClient,
    batch_client: BatchAnthropicClient,
    new_security_ids: list[SecurityId],
    *,
    cache,
    db,
    threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    neighbor_limit: int = DEFAULT_NEIGHBOR_SEARCH_LIMIT,
    allowed_categories: set[str] | None = None,
    model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL,
    min_confidence: Confidence = DEFAULT_MIN_CONFIDENCE,
    sync_threshold: int = DEFAULT_SYNC_THRESHOLD,
) -> tuple[dict, list[dict]]:
    api_requests, pending_context, cached_results = prepare_semantic_batch(
        new_security_ids, cache=cache, db=db,
        threshold=threshold, neighbor_limit=neighbor_limit,
        allowed_categories=allowed_categories, model=model,
    )
    responses = batch_client.create_messages(api_requests, sync_threshold=sync_threshold) if api_requests else {}
    return process_semantic_results(
        registry, responses, pending_context, cached_results, new_security_ids,
        cache=cache, db=db, min_confidence=min_confidence, model=model,
    )


def run_classification_sync(
    registry: RegistryClient,
    batch_client: BatchAnthropicClient,
    new_security_ids: list[SecurityId],
    *,
    cache,
    db,
    skip_semantic: bool = False,
    min_confidence: Confidence = DEFAULT_MIN_CONFIDENCE,
    structural_confidence: float = DEFAULT_STRUCTURAL_CONFIDENCE,
    hedgeable_with_confidence: float = DEFAULT_HEDGEABLE_WITH_CONFIDENCE,
    threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    neighbor_limit: int = DEFAULT_NEIGHBOR_SEARCH_LIMIT,
    allowed_categories: set[str] | None = None,
    model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL,
    sync_threshold: int = DEFAULT_SYNC_THRESHOLD,
) -> ClassificationResult:
    structural_counts, structural_rels = classify_structural(
        registry, new_security_ids, db=db,
        min_confidence=min_confidence,
        structural_confidence=structural_confidence,
        hedgeable_with_confidence=hedgeable_with_confidence,
    )
    semantic_counts: dict = {}
    semantic_rels: list[dict] = []
    if not skip_semantic:
        semantic_counts, semantic_rels = classify_semantic_sync(
            registry, batch_client, new_security_ids, cache=cache, db=db,
            threshold=threshold, neighbor_limit=neighbor_limit,
            allowed_categories=allowed_categories, model=model,
            min_confidence=min_confidence, sync_threshold=sync_threshold,
        )
    return ClassificationResult(
        structural=structural_counts,
        semantic=semantic_counts,
        written_relationships=structural_rels + semantic_rels,
    )
