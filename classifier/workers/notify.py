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

        created_events: list[tuple[int, str]] = []
        resolved_events: list[tuple[int, str]] = []

        for msg in messages:
            try:
                outer = json.loads(msg["Body"])
                if outer.get("Type") == "Notification":
                    payload = json.loads(outer["Message"])
                else:
                    payload = outer
            except (json.JSONDecodeError, KeyError):
                logger.warning("Malformed notification message, skipping: %s", msg.get("MessageId", "?"))
                continue

            msg_type = payload.get("type")
            if msg_type == "new_events":
                ids = payload.get("created_event_ids", [])
                names = payload.get("created_event_names", [])
                created_events.extend(zip(ids, names))
            elif msg_type == "resolved":
                ids = payload.get("resolved_event_ids", [])
                names = payload.get("resolved_event_names", [])
                resolved_events.extend(zip(ids, names))

        if not created_events and not resolved_events:
            return []

        blocks = format_notification_blocks(created_events, resolved_events)
        send_slack_notification(self._slack_token, self._config.slack_channel, blocks)
        return []
