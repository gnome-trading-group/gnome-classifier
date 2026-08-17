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
        self._runtime_config = init_runtime_config()
        self._sns = boto3.client("sns")

    def process_batch(self, messages: list[dict]) -> list[dict]:
        security_ids, symbol_by_id = parse_security_messages(messages)

        if not security_ids:
            return []

        created_events: dict[int, str] = {}
        for msg in messages:
            body = json.loads(msg["Body"])
            if "created_event_id" in body:
                created_events[body["created_event_id"]] = body["created_event_name"]

        cfg = self._runtime_config.config
        classification = run_classification_sync(
            self._registry, self._batch_client, security_ids,
            cache=self._cache, db=self._db,
            skip_semantic=not cfg.feature_flags.semantic_judgements_enabled,
            min_confidence=cfg.thresholds.min_confidence,
            hedgeable_with_confidence=cfg.thresholds.hedgeable_with_confidence,
            threshold=cfg.thresholds.embedding_similarity_threshold,
            neighbor_limit=cfg.processing.neighbor_search_limit,
            allowed_categories=set(cfg.category_filter.allowed_categories) if cfg.category_filter.enabled else None,
            model=cfg.models.semantic_judgment_model,
            sync_threshold=cfg.processing.anthropic_sync_threshold,
            debug=cfg.feature_flags.debug,
        )

        if classification.written_relationships:
            logger.info(
                "Relationships: %d written for %d securities",
                len(classification.written_relationships), len(security_ids),
            )

        if created_events:
            self._publish_to_sns({
                "type": "new_events",
                "created_event_ids": list(created_events.keys()),
                "created_event_names": list(created_events.values()),
            })

        return []

    def _publish_to_sns(self, payload: dict):
        try:
            self._sns.publish(TopicArn=self._sns_topic_arn, Message=json.dumps(payload))
        except Exception:
            logger.warning("SNS publish failed, notification lost", exc_info=True)
