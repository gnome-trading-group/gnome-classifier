import dataclasses
import json
import logging

import boto3

from classifier.adapters.types import AdapterContract
from classifier.stages.entities import create_entities
from classifier.stages.resolve import detect_resolved_events
from classifier.stages.stale import deactivate_stale_events
from classifier.workers.base import BaseWorker
from classifier.workers.config import WorkerConfig, init_anthropic, init_cache, init_db, init_registry, init_runtime_config

logger = logging.getLogger(__name__)


class NormalizeWorker(BaseWorker):
    _worker_messages_attr = "normalize_max_messages"
    _worker_wait_attr = "normalize_max_wait_seconds"

    def __init__(self, config: WorkerConfig):
        super().__init__(
            input_queue_url=config.contracts_queue_url,
            output_queue_url=config.entities_queue_url,
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
        self._runtime_config = init_runtime_config()
        self._runtime_config.refresh()
        self._batch_client = init_anthropic()
        self._cache = init_cache()
        self._db = init_db()
        self._sns = boto3.client("sns")

    def process_batch(self, messages: list[dict]) -> list[dict]:
        new_contracts: list[AdapterContract] = []
        resolved_by_exchange: dict[int, set[str]] = {}
        stale_events: list[tuple[int, str]] = []

        for msg in messages:
            body = json.loads(msg["Body"])
            if body["type"] == "new":
                for c in body["contracts"]:
                    new_contracts.append(AdapterContract(**c))
            elif body["type"] == "resolved":
                exchange_id = body["exchange_id"]
                resolved_by_exchange.setdefault(exchange_id, set()).add(body["native_id"])
            elif body["type"] == "stale":
                stale_events.append((body["exchange_id"], body["native_event_id"]))

        entities_queue_messages: list[dict] = []

        cfg = self._runtime_config.config

        if resolved_by_exchange:
            resolution_counts = detect_resolved_events(resolved_by_exchange, self._registry, self._db,
                                                       debug=cfg.feature_flags.debug)
            if any(v for v in resolution_counts.values()):
                self._publish_to_sns({"type": "resolution", **resolution_counts})
                logger.info("Resolution: %s", resolution_counts)

        if stale_events:
            stale_counts = deactivate_stale_events(stale_events, self._registry, self._db,
                                                   debug=cfg.feature_flags.debug)
            if any(v for v in stale_counts.values()):
                self._publish_to_sns({"type": "stale_cleanup", **stale_counts})
                logger.info("Stale cleanup: %s", stale_counts)

        if new_contracts:
            canonicalize_enabled = cfg.feature_flags.canonicalization_enabled
            entity_result = create_entities(
                self._registry, self._batch_client, new_contracts,
                cache=self._cache, db=self._db,
                canonicalize_enabled=canonicalize_enabled,
                canonicalize_model=cfg.models.canonicalize_model,
                canonicalize_batch_size=cfg.processing.canonicalize_batch_size,
                sync_threshold=cfg.processing.anthropic_sync_threshold,
                debug=cfg.feature_flags.debug,
            )
            if entity_result.has_new_entities:
                for sid, symbol in zip(entity_result.new_security_ids, entity_result.new_security_symbols):
                    entities_queue_messages.append({"type": "new_security", "security_id": sid, "security_symbol": symbol})
                self._publish_to_sns({
                    "type": "new_entity",
                    "new_symbols": entity_result.new_security_symbols,
                    "created_event_ids": entity_result.created_event_ids,
                    "created_event_names": entity_result.created_event_names,
                    **entity_result.counts,
                })
                logger.info(
                    "Entities: %d new securities from %d contracts",
                    len(entity_result.new_security_ids), len(new_contracts),
                )

        return entities_queue_messages

    def _publish_to_sns(self, payload: dict):
        try:
            self._sns.publish(TopicArn=self._sns_topic_arn, Message=json.dumps(payload))
        except Exception:
            logger.warning("SNS publish failed, notification lost", exc_info=True)
