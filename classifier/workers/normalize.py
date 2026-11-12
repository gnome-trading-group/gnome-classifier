import dataclasses
import json
import logging

import boto3

from classifier.adapters.types import AdapterContract
from classifier.stages.entities import create_entities
from classifier.stages.resolve import deactivate_closed_listings
from classifier.stages.stale import deactivate_stale_events
from classifier.workers.base import BaseWorker, message_body
from classifier.client import ClaudeAsker
from classifier.workers.config import WorkerConfig, claude_client, init_cache, init_db, init_registry, init_runtime_config

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
        self._client_for = claude_client
        self._cache = None
        self._db = None
        self._sns = None

    def _setup(self):
        self._registry = init_registry()
        self._runtime_config = init_runtime_config()
        self._runtime_config.refresh()
        self._cache = init_cache()
        self._db = init_db()
        self._sns = boto3.client("sns")

    def _canonicalizer(self) -> ClaudeAsker | None:
        """Claude for the categories and tags exchange labels leave open, or None while `canonicalization_enabled` is
        off."""
        cfg = self._runtime_config.config
        if not cfg.feature_flags.canonicalization_enabled:
            return None
        return ClaudeAsker(self._client_for, cfg.models.providers(), cfg.models.canonicalize_models)

    def process_batch(self, messages: list[dict]) -> list[dict]:
        new_contracts: list[AdapterContract] = []
        closed_by_exchange: dict[int, set[str]] = {}
        stale_events: list[tuple[int, str]] = []

        for msg in messages:
            body = message_body(msg)
            if body["type"] == "new":
                for c in body["contracts"]:
                    new_contracts.append(AdapterContract(**c))
            elif body["type"] == "closed":
                closed_by_exchange.setdefault(body["exchange_id"], set()).add(body["native_id"])
            elif body["type"] == "stale":
                stale_events.append((body["exchange_id"], body["native_event_id"]))

        entities_queue_messages: list[dict] = []

        cfg = self._runtime_config.config

        # Closing only deactivates listings, so there is nothing to announce: events resolve on their settlement values,
        # in the fetch worker.
        if closed_by_exchange:
            deactivate_closed_listings(closed_by_exchange, self._registry, self._db, debug=cfg.feature_flags.debug)

        if stale_events:
            stale_counts = deactivate_stale_events(stale_events, self._registry, self._db,
                                                   debug=cfg.feature_flags.debug)
            if any(v for v in stale_counts.values()):
                self._publish_to_sns({"type": "resolved", **stale_counts})
                logger.info("Stale cleanup: %s", stale_counts)

        if new_contracts:
            entity_result = create_entities(
                self._registry, self._canonicalizer(), new_contracts,
                cache=self._cache, db=self._db,
                canonicalize_batch_size=cfg.processing.canonicalize_batch_size,
                debug=cfg.feature_flags.debug,
            )
            if entity_result.has_new_entities:
                created_event_sids: dict[int, tuple[int, str]] = {}
                if entity_result.created_event_ids:
                    name_by_eid = dict(zip(entity_result.created_event_ids, entity_result.created_event_names))
                    ecs = self._db.get_event_contracts_for_events(entity_result.created_event_ids)
                    for ec in ecs:
                        if ec.event_id in name_by_eid:
                            created_event_sids[ec.security_id] = (ec.event_id, name_by_eid[ec.event_id])

                for sid, symbol in zip(entity_result.new_security_ids, entity_result.new_security_symbols):
                    msg: dict = {"type": "new_security", "security_id": sid, "security_symbol": symbol}
                    if sid in created_event_sids:
                        msg["created_event_id"], msg["created_event_name"] = created_event_sids[sid]
                    entities_queue_messages.append(msg)

                logger.info(
                    "Entities: %d new securities from %d contracts",
                    len(entity_result.new_security_ids), len(new_contracts),
                )
            # The registry dropped these securities' relationships with their old source; the relationships worker
            # restores them as soon as it reads the new one, rather than when its backlog walk reaches them.
            sent = {m["security_id"] for m in entities_queue_messages}
            changed = [sid for sid in entity_result.changed_source_security_ids if sid not in sent]
            entities_queue_messages.extend({"type": "source_changed", "security_id": sid} for sid in changed)
            if changed:
                logger.info("Entities: %d venue sources changed", len(changed))
            if entity_result.event_contracts_relabelled or entity_result.securities_renamed \
                    or entity_result.listings_renamed:
                logger.info(
                    "Entities: renamed by the venue: %d outcome labels, %d securities, %d listings",
                    entity_result.event_contracts_relabelled, entity_result.securities_renamed,
                    entity_result.listings_renamed,
                )

        return entities_queue_messages

    def _publish_to_sns(self, payload: dict):
        try:
            self._sns.publish(TopicArn=self._sns_topic_arn, Message=json.dumps(payload))
        except Exception:
            logger.warning("SNS publish failed, notification lost", exc_info=True)
