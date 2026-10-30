import json
import logging
import os
import time

import boto3

from classifier.constants import DEFAULT_VOYAGE_EMBEDDING_MODEL
from classifier.propositions.attempts import RedisAttemptStore
from classifier.propositions.llm import PropositionExtractor
from classifier.stages.propositions import derive_due_subjects, read_backlog, read_propositions, run_drift_check
from classifier.workers.base import BaseWorker, message_body, parse_security_messages
from classifier.workers.config import WorkerConfig, claude_client, init_db, init_registry, init_runtime_config, init_voyage

logger = logging.getLogger(__name__)

# Idle passes read the backlog of securities without a proposition and derive subjects that came due without a message
# (a person merging two subjects); polls return every 20s, so this keeps them to about one a minute.
IDLE_DERIVE_INTERVAL_SECONDS = 60


class RelationshipsWorker(BaseWorker):
    _worker_messages_attr = "relationships_max_messages"
    _worker_wait_attr = "relationships_max_wait_seconds"

    def __init__(self, config: WorkerConfig):
        super().__init__(
            input_queue_url=config.entities_queue_url,
            output_queue_url=None,
        )
        self._config = config
        self._sns_topic_arn = config.notifications_topic_arn
        self._registry = None
        self._voyage = None
        self._db = None
        self._sns = None
        self._attempts = None
        self._client_for = claude_client
        self._last_idle_derive = 0.0
        self._last_drift_check = 0.0
        self._backlog_after = 0

    def _setup(self):
        self._registry = init_registry()
        self._voyage = init_voyage()
        self._db = init_db()
        self._runtime_config = init_runtime_config()
        self._sns = boto3.client("sns")
        self._attempts = RedisAttemptStore(os.environ["REDIS_ENDPOINT"])

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        model = self._runtime_config.config.models.voyage_embedding_model or DEFAULT_VOYAGE_EMBEDDING_MODEL
        return self._voyage.embed(texts, model=model, input_type=kind)

    def _extractor(self) -> PropositionExtractor | None:
        """Claude for the securities the code readers can't read, or None while `proposition_llm_enabled` is off."""
        cfg = self._runtime_config.config
        if not cfg.feature_flags.proposition_llm_enabled:
            return None
        return PropositionExtractor(self._client_for, cfg.models.proposition_providers(), cfg.models.proposition_models)

    def _read_backlog(self):
        result, self._backlog_after = read_backlog(
            db=self._db, registry=self._registry, extractor=self._extractor(), embed=self._embed,
            limit=self._runtime_config.config.processing.relationships_backlog_limit, after=self._backlog_after,
            attempts=self._attempts,
        )
        return result

    def process_batch(self, messages: list[dict]) -> list[dict]:
        security_ids, _ = parse_security_messages(messages)
        if not security_ids:
            return []

        created_events: dict[int, str] = {}
        for msg in messages:
            body = message_body(msg)
            if "created_event_id" in body:
                created_events[body["created_event_id"]] = body["created_event_name"]

        cfg = self._runtime_config.config
        if cfg.feature_flags.relationships_enabled:
            result = read_propositions(security_ids, db=self._db, registry=self._registry,
                                       extractor=self._extractor(), embed=self._embed, attempts=self._attempts)
            result.add(self._read_backlog())
            derive_due_subjects(db=self._db, registry=self._registry,
                                limit=cfg.processing.relationships_derive_limit, result=result)
            logger.info("Relationships pass: %s", result)
            self._maybe_check_drift()
        else:
            logger.info("Relationships pass: disabled")

        if created_events:
            self._publish_to_sns({
                "type": "new_events",
                "created_event_ids": list(created_events.keys()),
                "created_event_names": list(created_events.values()),
            })
        return []

    def on_idle(self) -> None:
        cfg = self._runtime_config.config
        now = time.monotonic()
        if now - self._last_idle_derive < IDLE_DERIVE_INTERVAL_SECONDS:
            return
        self._last_idle_derive = now
        # Logged every idle pass, even with nothing to do or relationships off: the RelationshipsSilent alarm counts
        # these and the batch passes, so a worker that hangs goes quiet. Keep the wording in step with its filter.
        if not cfg.feature_flags.relationships_enabled:
            logger.info("Relationships pass: disabled")
            return
        result = self._read_backlog()
        derive_due_subjects(db=self._db, registry=self._registry,
                            limit=cfg.processing.relationships_derive_limit, result=result)
        logger.info("Relationships pass: %s", result)
        self._maybe_check_drift()

    def _maybe_check_drift(self) -> None:
        cfg = self._runtime_config.config
        extractor = self._extractor()
        # The drift check re-reads code-read contracts with Claude, so it is paid too.
        if extractor is None:
            return
        now = time.monotonic()
        if self._last_drift_check and now - self._last_drift_check < cfg.worker_params.drift_interval_seconds:
            return
        self._last_drift_check = now
        run_drift_check(db=self._db, extractor=extractor, size=cfg.processing.drift_sample_size)

    def _publish_to_sns(self, payload: dict):
        try:
            self._sns.publish(TopicArn=self._sns_topic_arn, Message=json.dumps(payload))
        except Exception:
            logger.warning("SNS publish failed, notification lost", exc_info=True)
