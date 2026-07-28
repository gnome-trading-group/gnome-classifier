import json
import logging

from classifier.notifications import format_notification_blocks, send_slack_notification
from classifier.workers.base import BaseWorker
from classifier.workers.config import WorkerConfig, fetch_secret, init_runtime_config

logger = logging.getLogger(__name__)


class NotifyWorker(BaseWorker):
    _worker_messages_attr = "notify_max_messages"
    _worker_wait_attr = "notify_max_wait_seconds"

    def __init__(self, config: WorkerConfig):
        super().__init__(
            input_queue_url=config.slack_queue_url,
            output_queue_url=None,
        )
        self._config = config
        self._slack_token: str | None = None

    def _setup(self):
        self._runtime_config = init_runtime_config()
        if self._config.slack_bot_token_secret:
            self._slack_token = fetch_secret(self._config.slack_bot_token_secret)

    def process_batch(self, messages: list[dict]) -> list[dict]:
        if not self._slack_token or not self._config.slack_channel:
            logger.info("Slack not configured, skipping notification")
            return []

        new_symbols: list[str] = []
        entity_counts: dict = {}
        resolution_counts: dict = {}
        relationships_written = 0

        for msg in messages:
            try:
                outer = json.loads(msg["Body"])
                # SNS wraps the message in an envelope; unwrap if present
                if outer.get("Type") == "Notification":
                    payload = json.loads(outer["Message"])
                else:
                    payload = outer
            except (json.JSONDecodeError, KeyError):
                logger.warning("Malformed notification message, skipping: %s", msg.get("MessageId", "?"))
                continue

            msg_type = payload.get("type")
            if msg_type == "new_entity":
                new_symbols.append(payload.get("security_symbol", ""))
                for k in ("events_created", "securities_created", "listings_created"):
                    entity_counts[k] = entity_counts.get(k, 0) + payload.get(k, 0)
            elif msg_type == "resolution":
                for k in ("events_resolved", "securities_deactivated", "listings_deactivated"):
                    resolution_counts[k] = resolution_counts.get(k, 0) + payload.get(k, 0)
            elif msg_type == "relationship":
                relationships_written += 1

        blocks = format_notification_blocks(new_symbols, entity_counts, resolution_counts, relationships_written)
        send_slack_notification(self._slack_token, self._config.slack_channel, blocks)
        return []
