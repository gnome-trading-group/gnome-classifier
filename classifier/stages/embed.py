import dataclasses
import logging

from classifier.client import BatchVoyageClient
from classifier.constants import DEFAULT_VOYAGE_EMBED_CHUNK_SIZE, DEFAULT_VOYAGE_EMBEDDING_MODEL
from classifier.types import EntityResult

logger = logging.getLogger(__name__)


def embed_and_update(
    voyage_client: BatchVoyageClient,
    entity_result: EntityResult,
    db,
    voyage_model: str = DEFAULT_VOYAGE_EMBEDDING_MODEL,
    voyage_chunk_size: int = DEFAULT_VOYAGE_EMBED_CHUNK_SIZE,
    debug: bool = False,
) -> EntityResult:
    events_to_embed = db.get_events_without_embeddings()
    if not events_to_embed:
        return entity_result
    if debug:
        logger.info("[DEBUG] embed: %d events need embeddings", len(events_to_embed))
        for ev in events_to_embed[:50]:
            logger.info("[DEBUG] embed:   event_id=%d %r", ev.event_id, ev.title[:80])
        if len(events_to_embed) > 50:
            logger.info("[DEBUG] embed:   ... and %d more", len(events_to_embed) - 50)
    all_embedded_event_ids: list[int] = []
    for chunk_embs in voyage_client.embed_events(events_to_embed, chunk_size=voyage_chunk_size, model=voyage_model):
        db.put_embeddings(chunk_embs)
        all_embedded_event_ids.extend(chunk_embs.keys())
    if all_embedded_event_ids:
        extra = db.get_security_ids_for_events(all_embedded_event_ids)
        updated_ids = list(set(entity_result.new_security_ids) | extra)
        entity_result = dataclasses.replace(entity_result, new_security_ids=updated_ids)
        logger.info("Embedded %d events, %d securities to classify", len(all_embedded_event_ids), len(updated_ids))
    return entity_result
