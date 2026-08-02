import logging

from classifier.stages.embed import embed_and_update
from classifier.types import EntityResult
from classifier.workers.base import BaseWorker, parse_security_messages
from classifier.workers.config import WorkerConfig, init_db, init_runtime_config, init_voyage

logger = logging.getLogger(__name__)


class EmbedWorker(BaseWorker):
    _worker_messages_attr = "embed_max_messages"
    _worker_wait_attr = "embed_max_wait_seconds"

    def __init__(self, config: WorkerConfig):
        super().__init__(
            input_queue_url=config.entities_queue_url,
            output_queue_url=config.embeddings_queue_url,
        )
        self._config = config
        self._voyage_client = None
        self._db = None

    def _setup(self):
        self._voyage_client = init_voyage()
        self._db = init_db()
        self._runtime_config = init_runtime_config()

    def process_batch(self, messages: list[dict]) -> list[dict]:
        security_ids, symbol_by_id = parse_security_messages(messages)

        if not security_ids:
            return []

        cfg = self._runtime_config.config
        entity_result = EntityResult(
            events_created=0, securities_created=0, listings_created=0,
            event_contracts_created=0, listing_specs_created=0,
            new_security_ids=security_ids, new_security_symbols=[symbol_by_id[sid] for sid in security_ids],
            created_event_ids=[], created_event_names=[],
        )
        updated_result = embed_and_update(
            self._voyage_client, entity_result, self._db,
            voyage_model=cfg.models.voyage_embedding_model,
            voyage_chunk_size=cfg.processing.voyage_embed_chunk_size,
        )

        return [
            {"type": "security", "security_id": sid, "security_symbol": symbol_by_id.get(sid, "")}
            for sid in updated_result.new_security_ids
        ]
