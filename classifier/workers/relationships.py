import json
import logging

import boto3

from classifier.stages.classify import run_classification_sync
from classifier.workers.base import BaseWorker, parse_security_messages
from classifier.workers.config import WorkerConfig, init_anthropic, init_cache, init_db, init_registry, init_runtime_config

logger = logging.getLogger(__name__)


class RelationshipsWorker(BaseWorker):
    _worker_messages_attr = "relationships_max_messages"
    _worker_wait_attr = "relationships_max_wait_seconds"

    def __init__(self, config: WorkerConfig):
        super().__init__(
            input_queue_url=config.embeddings_queue_url,
            output_queue_url=None,
        )
        self._config = config
        self._sns_topic_arn = config.notifications_topic_arn
        self._registry = None
        self._batch_client = None
        self._cache = None
        self._db = None
        self._sns = None

    def _setup(self):
        self._registry = init_registry()
        self._batch_client = init_anthropic()
        self._cache = init_cache()
        self._db = init_db()
        self._sns = boto3.client("sns")
        self._runtime_config = init_runtime_config()

    def process_batch(self, messages: list[dict]) -> list[dict]:
        security_ids, symbol_by_id = parse_security_messages(messages)

        if not security_ids:
            return []

        cfg = self._runtime_config.config
        classification = run_classification_sync(
            self._registry, self._batch_client, security_ids,
            cache=self._cache, db=self._db,
            skip_semantic=not cfg.feature_flags.semantic_judgements_enabled,
            min_confidence=cfg.thresholds.min_confidence,
            structural_confidence=cfg.thresholds.structural_confidence,
            hedgeable_with_confidence=cfg.thresholds.hedgeable_with_confidence,
            threshold=cfg.thresholds.embedding_similarity_threshold,
            neighbor_limit=cfg.processing.neighbor_search_limit,
            allowed_categories=set(cfg.category_filter.allowed_categories) if cfg.category_filter.enabled else None,
            model=cfg.models.semantic_judgment_model,
            sync_threshold=cfg.processing.anthropic_sync_threshold,
        )

        for rel in classification.written_relationships:
            sid_a = rel.get("security_id_a")
            sid_b = rel.get("security_id_b")
            try:
                self._sns.publish(
                    TopicArn=self._sns_topic_arn,
                    Message=json.dumps({
                        "type": "relationship",
                        "security_id_a": sid_a,
                        "security_id_b": sid_b,
                        "security_symbol_a": symbol_by_id.get(sid_a, ""),
                        "security_symbol_b": symbol_by_id.get(sid_b, ""),
                        "relationship_type": rel.get("relationship_type"),
                        "confidence": rel.get("confidence"),
                        "method": rel.get("method"),
                    }),
                )
            except Exception:
                logger.warning("SNS publish failed, notification lost", exc_info=True)

        if classification.written_relationships:
            logger.info(
                "Relationships: %d written for %d securities",
                len(classification.written_relationships), len(security_ids),
            )

        return []
