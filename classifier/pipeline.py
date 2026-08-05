import dataclasses
import logging

from classifier.constants import (
    DEFAULT_CANONICALIZE_BATCH_SIZE,
    DEFAULT_CANONICALIZE_MODEL,
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    DEFAULT_HEDGEABLE_WITH_CONFIDENCE,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_NEIGHBOR_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_JUDGMENT_MODEL,
    DEFAULT_STRUCTURAL_CONFIDENCE,
    DEFAULT_SYNC_THRESHOLD,
    DEFAULT_VOYAGE_EMBED_CHUNK_SIZE,
    DEFAULT_VOYAGE_EMBEDDING_MODEL,
)
from classifier.client import BatchVoyageClient
from classifier.stages.classify import ClassificationResult, run_classification_sync
from classifier.stages.embed import embed_and_update
from classifier.stages.entities import create_entities
from classifier.stages.fetch import fetch_exchanges
from classifier.types import Confidence, EntityResult

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PipelineResult:
    entity_result: EntityResult
    classification: ClassificationResult | None


def create_entities_and_embed(
    registry,
    batch_client,
    contracts,
    *,
    voyage_client: BatchVoyageClient,
    cache=None,
    db,
    canonicalize_model: str = DEFAULT_CANONICALIZE_MODEL,
    canonicalize_batch_size: int = DEFAULT_CANONICALIZE_BATCH_SIZE,
    sync_threshold: int = DEFAULT_SYNC_THRESHOLD,
    voyage_model: str = DEFAULT_VOYAGE_EMBEDDING_MODEL,
    voyage_chunk_size: int = DEFAULT_VOYAGE_EMBED_CHUNK_SIZE,
    debug: bool = False,
) -> EntityResult:
    entity_result = create_entities(
        registry, batch_client, contracts, cache=cache, db=db,
        canonicalize_model=canonicalize_model,
        canonicalize_batch_size=canonicalize_batch_size,
        sync_threshold=sync_threshold, debug=debug,
    )
    return embed_and_update(voyage_client, entity_result, db, voyage_model=voyage_model, voyage_chunk_size=voyage_chunk_size, debug=debug)


def run_full_pipeline_sync(
    registry,
    batch_client,
    contracts,
    *,
    voyage_client: BatchVoyageClient,
    cache,
    db,
    skip_classify: bool = False,
    skip_semantic: bool = False,
    min_confidence: Confidence = DEFAULT_MIN_CONFIDENCE,
    structural_confidence: float = DEFAULT_STRUCTURAL_CONFIDENCE,
    hedgeable_with_confidence: float = DEFAULT_HEDGEABLE_WITH_CONFIDENCE,
    threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    neighbor_limit: int = DEFAULT_NEIGHBOR_SEARCH_LIMIT,
    allowed_categories: set[str] | None = None,
    model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL,
    sync_threshold: int = DEFAULT_SYNC_THRESHOLD,
    canonicalize_model: str = DEFAULT_CANONICALIZE_MODEL,
    canonicalize_batch_size: int = DEFAULT_CANONICALIZE_BATCH_SIZE,
    voyage_model: str = DEFAULT_VOYAGE_EMBEDDING_MODEL,
    voyage_chunk_size: int = DEFAULT_VOYAGE_EMBED_CHUNK_SIZE,
    debug: bool = False,
) -> PipelineResult:
    entity_result = create_entities_and_embed(
        registry, batch_client, contracts,
        voyage_client=voyage_client, cache=cache, db=db,
        canonicalize_model=canonicalize_model,
        canonicalize_batch_size=canonicalize_batch_size,
        sync_threshold=sync_threshold,
        voyage_model=voyage_model,
        voyage_chunk_size=voyage_chunk_size,
        debug=debug,
    )
    if skip_classify or not entity_result.has_new_entities:
        return PipelineResult(entity_result=entity_result, classification=None)
    classification = run_classification_sync(
        registry, batch_client, entity_result.new_security_ids,
        cache=cache, db=db, skip_semantic=skip_semantic,
        min_confidence=min_confidence,
        structural_confidence=structural_confidence,
        hedgeable_with_confidence=hedgeable_with_confidence,
        threshold=threshold, neighbor_limit=neighbor_limit,
        allowed_categories=allowed_categories, model=model,
        sync_threshold=sync_threshold,
        debug=debug,
    )
    return PipelineResult(entity_result=entity_result, classification=classification)
