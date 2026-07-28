import json
import logging
import signal
import time
from abc import ABC, abstractmethod

import boto3

from classifier.constants import DEFAULT_MAX_MESSAGES, DEFAULT_MAX_WAIT_SECONDS, DEFAULT_VISIBILITY_TIMEOUT
from classifier.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

_SEND_BATCH_SIZE = 10


def sqs_send_batch(sqs_client, queue_url: str, messages: list[dict]):
    for i in range(0, len(messages), _SEND_BATCH_SIZE):
        chunk = messages[i:i + _SEND_BATCH_SIZE]
        entries = [{"Id": str(j), "MessageBody": json.dumps(msg)} for j, msg in enumerate(chunk)]
        resp = sqs_client.send_message_batch(QueueUrl=queue_url, Entries=entries)
        failed = resp.get("Failed", [])
        if failed:
            ids = [f["Id"] for f in failed]
            raise RuntimeError(f"SQS send_message_batch partial failure: {len(failed)} of {len(chunk)} failed: {ids}")


def parse_security_messages(messages: list[dict]) -> tuple[list[int], dict[int, str]]:
    security_ids: list[int] = []
    symbol_by_id: dict[int, str] = {}
    for msg in messages:
        body = json.loads(msg["Body"])
        sid = body["security_id"]
        security_ids.append(sid)
        symbol_by_id[sid] = body.get("security_symbol", "")
    return security_ids, symbol_by_id


class BaseWorker(ABC):
    _worker_messages_attr: str | None = None
    _worker_wait_attr: str | None = None

    def __init__(
        self,
        input_queue_url: str | None,
        output_queue_url: str | None,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS,
        visibility_timeout: int = DEFAULT_VISIBILITY_TIMEOUT,
        poll_wait_seconds: int = 20,
    ):
        self._input_queue_url = input_queue_url
        self._output_queue_url = output_queue_url
        self._max_messages = max_messages
        self._max_wait_seconds = max_wait_seconds
        self._visibility_timeout = visibility_timeout
        self._poll_wait_seconds = poll_wait_seconds
        self._running = False
        self._sqs = boto3.client("sqs")
        self._runtime_config: RuntimeConfig | None = None

    def run(self):
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self._running = True
        self._setup()
        logger.info("%s started", type(self).__name__)
        while self._running:
            batch = self._collect_batch()
            if not batch:
                continue
            try:
                output_messages = self.process_batch(batch)
            except Exception:
                logger.exception("process_batch failed; messages will redeliver after visibility timeout")
                continue
            if output_messages and self._output_queue_url:
                self._send_results(output_messages)
            self._delete_messages(batch)
        logger.info("%s stopped", type(self).__name__)

    def _collect_batch(self) -> list[dict]:
        messages: list[dict] = []
        deadline: float | None = None
        max_messages = self._max_messages
        max_wait_seconds = self._max_wait_seconds
        while self._running:
            self._runtime_config.refresh()
            wp = self._runtime_config.config.worker_params
            max_messages = self._resolve_worker_max_messages(wp) or self._max_messages
            max_wait_seconds = self._resolve_worker_max_wait_seconds(wp) or self._max_wait_seconds
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait = min(self._poll_wait_seconds, max(1, int(remaining)))
            else:
                wait = self._poll_wait_seconds

            want = min(_SEND_BATCH_SIZE, max_messages - len(messages))
            response = self._sqs.receive_message(
                QueueUrl=self._input_queue_url,
                MaxNumberOfMessages=want,
                WaitTimeSeconds=wait,
                VisibilityTimeout=self._visibility_timeout,
                AttributeNames=["All"],
            )
            new_msgs = response.get("Messages", [])
            if new_msgs:
                if deadline is None:
                    deadline = time.monotonic() + max_wait_seconds
                messages.extend(new_msgs)
            if len(messages) >= max_messages:
                break

        return messages

    def _resolve_worker_max_messages(self, wp) -> int | None:
        return getattr(wp, self._worker_messages_attr, None) if self._worker_messages_attr else None

    def _resolve_worker_max_wait_seconds(self, wp) -> int | None:
        return getattr(wp, self._worker_wait_attr, None) if self._worker_wait_attr else None

    def _send_results(self, messages: list[dict]):
        sqs_send_batch(self._sqs, self._output_queue_url, messages)

    def _delete_messages(self, messages: list[dict]):
        for i in range(0, len(messages), _SEND_BATCH_SIZE):
            chunk = messages[i:i + _SEND_BATCH_SIZE]
            entries = [
                {"Id": str(j), "ReceiptHandle": msg["ReceiptHandle"]}
                for j, msg in enumerate(chunk)
            ]
            resp = self._sqs.delete_message_batch(QueueUrl=self._input_queue_url, Entries=entries)
            failed = resp.get("Failed", [])
            if failed:
                logger.warning("delete_message_batch partial failure: %d of %d failed (will redeliver)", len(failed), len(chunk))

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received, draining current batch")
        self._running = False

    @abstractmethod
    def process_batch(self, messages: list[dict]) -> list[dict]:
        """Process a batch of SQS messages. Return output messages for the output queue."""

    @abstractmethod
    def _setup(self):
        """Initialize clients and resources before the main loop."""
