import dataclasses
import logging

from classifier.client import BatchAnthropicClient, BatchVoyageClient
from classifier.stages.classify import classify_structural, prepare_semantic_batch, process_semantic_results
from classifier.stages.entities import create_entities
from classifier.types import EntityResult, SecurityId
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import Exchange

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ClassificationResult:
    structural: dict
    semantic: dict

    @property
    def relationships_written(self) -> int:
        return self.structural.get("relationships_written", 0) + self.semantic.get("relationships_written", 0)

    @property
    def relationships_skipped_low_confidence(self) -> int:
        return (
            self.structural.get("relationships_skipped_low_confidence", 0)
            + self.semantic.get("relationships_skipped_low_confidence", 0)
        )


@dataclasses.dataclass
class PipelineResult:
    entity_result: EntityResult
    classification: ClassificationResult | None


def fetch_exchanges(
    registry: RegistryClient,
    adapter_name: str | None = None,
) -> dict[str, Exchange]:
    exchanges = registry.get_exchange()
    exchange_by_name = {e.exchange_name.lower(): e for e in exchanges}
    if adapter_name:
        key = adapter_name.lower()
        if key not in exchange_by_name:
            raise ValueError(f"Unknown adapter '{adapter_name}'. Choices: {list(exchange_by_name)}")
        return {key: exchange_by_name[key]}
    return exchange_by_name


def embed_and_update(
    voyage_client: BatchVoyageClient,
    entity_result: EntityResult,
    db,
) -> EntityResult:
    """Embed events without embeddings and widen new_security_ids to include their securities."""
    events_to_embed = db.get_events_without_embeddings()
    if not events_to_embed:
        return entity_result
    all_embedded_event_ids: list[int] = []
    for chunk_embs in voyage_client.embed_events(events_to_embed):
        db.put_embeddings(chunk_embs)
        all_embedded_event_ids.extend(chunk_embs.keys())
    if all_embedded_event_ids:
        extra = db.get_security_ids_for_events(all_embedded_event_ids)
        updated_ids = list(set(entity_result.new_security_ids) | extra)
        entity_result = dataclasses.replace(entity_result, new_security_ids=updated_ids)
        logger.info("Embedded %d events, %d securities to classify", len(all_embedded_event_ids), len(updated_ids))
    return entity_result


def create_entities_and_embed(
    registry,
    batch_client,
    contracts,
    *,
    voyage_client: BatchVoyageClient,
    cache=None,
    db,
) -> EntityResult:
    entity_result = create_entities(registry, batch_client, contracts, cache=cache, db=db)
    return embed_and_update(voyage_client, entity_result, db)


def classify_semantic_sync(
    registry: RegistryClient,
    batch_client: BatchAnthropicClient,
    new_security_ids: list[SecurityId],
    *,
    cache,
    db,
) -> dict:
    """Run the full semantic classification sequence synchronously."""
    api_requests, pending_context, cached_results = prepare_semantic_batch(
        new_security_ids, cache=cache, db=db,
    )
    responses = batch_client.create_messages(api_requests) if api_requests else {}
    return process_semantic_results(
        registry, responses, pending_context, cached_results, new_security_ids,
        cache=cache, db=db,
    )


def run_classification_sync(
    registry: RegistryClient,
    batch_client: BatchAnthropicClient,
    new_security_ids: list[SecurityId],
    *,
    cache,
    db,
    skip_semantic: bool = False,
) -> ClassificationResult:
    """Run structural classification and optionally semantic classification."""
    structural = classify_structural(registry, new_security_ids, db=db)
    semantic: dict = {}
    if not skip_semantic:
        semantic = classify_semantic_sync(registry, batch_client, new_security_ids, cache=cache, db=db)
    return ClassificationResult(structural=structural, semantic=semantic)


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
) -> PipelineResult:
    """Full blocking pipeline: entity creation + embedding + structural + semantic classification."""
    entity_result = create_entities_and_embed(
        registry, batch_client, contracts,
        voyage_client=voyage_client, cache=cache, db=db,
    )
    if skip_classify or not entity_result.has_new_entities:
        return PipelineResult(entity_result=entity_result, classification=None)
    classification = run_classification_sync(
        registry, batch_client, entity_result.new_security_ids,
        cache=cache, db=db, skip_semantic=skip_semantic,
    )
    return PipelineResult(entity_result=entity_result, classification=classification)
