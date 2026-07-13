import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

BATCH_TIME_THRESHOLD_MINUTES = 2.0


@dataclass
class ModelRateLimit:
    input_tpm: int
    output_tpm: int


class BatchAnthropicClient:
    """Wraps anthropic.Anthropic and transparently routes requests to either parallel sync
    calls or the Anthropic Message Batches API based on estimated token usage vs rate limits."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        rate_limits: dict[str, ModelRateLimit] | None = None,
        poll_interval: int = 30,
        max_batch_wait: int = 480,
    ):
        self._client = client
        self._rate_limits = rate_limits or {}
        self._poll_interval = poll_interval
        self._max_batch_wait = max_batch_wait

    def create_messages(self, requests: list[dict]) -> dict[str, anthropic.types.Message]:
        """Submit N requests, return {custom_id: Message}.

        Each request must have "custom_id" (str) and "params" (dict matching messages.create kwargs).
        Missing results (errored/expired) are omitted from the return dict — callers should
        treat a missing custom_id as a graceful failure and skip that item.
        """
        if not requests:
            return {}

        model = requests[0]["params"].get("model", "")

        if self._should_batch(requests, model):
            logger.info("Using batch API for %d requests (model=%s)", len(requests), model)
            try:
                return self._batch_create(requests)
            except Exception as e:
                logger.warning("Batch API failed, falling back to sync: %s", e)

        logger.info("Using sync API for %d requests (model=%s)", len(requests), model)
        return self._sync_create(requests)

    def _should_batch(self, requests: list[dict], model: str) -> bool:
        limit = self._rate_limits.get(model)
        if limit is None or len(requests) <= 1:
            return False

        est_input = sum(self._estimate_input_tokens(r) for r in requests)
        est_output = sum(r["params"].get("max_tokens", 1000) for r in requests)

        minutes_input = est_input / limit.input_tpm
        minutes_output = est_output / limit.output_tpm
        estimated_minutes = max(minutes_input, minutes_output)

        logger.debug(
            "Rate limit estimate: model=%s requests=%d est_input=%d est_output=%d est_minutes=%.2f",
            model, len(requests), est_input, est_output, estimated_minutes,
        )
        return estimated_minutes > BATCH_TIME_THRESHOLD_MINUTES

    def _estimate_input_tokens(self, request: dict) -> int:
        params = request["params"]
        text = ""
        system = params.get("system", "")
        if isinstance(system, str):
            text += system
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict):
                    text += block.get("text", "")
        for msg in params.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                text += content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text += block.get("text", "")
        return max(1, len(text) // 4)

    def _sync_create(self, requests: list[dict]) -> dict[str, anthropic.types.Message]:
        results: dict[str, anthropic.types.Message] = {}
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(self._client.messages.create, **req["params"]): req["custom_id"]
                for req in requests
            }
            for future in as_completed(futures):
                custom_id = futures[future]
                try:
                    results[custom_id] = future.result()
                except Exception as e:
                    logger.warning("Sync request %s failed: %s", custom_id, e)
        return results

    def _batch_create(self, requests: list[dict]) -> dict[str, anthropic.types.Message]:
        batch = self._client.messages.batches.create(
            requests=[{"custom_id": r["custom_id"], "params": r["params"]} for r in requests]
        )
        logger.info("Submitted batch %s (%d requests)", batch.id, len(requests))

        deadline = time.monotonic() + self._max_batch_wait
        while time.monotonic() < deadline:
            status = self._client.messages.batches.retrieve(batch.id)
            if status.processing_status == "ended":
                break
            logger.debug("Batch %s still processing...", batch.id)
            time.sleep(self._poll_interval)
        else:
            raise TimeoutError(f"Batch {batch.id} did not complete within {self._max_batch_wait}s")

        results: dict[str, anthropic.types.Message] = {}
        for item in self._client.messages.batches.results(batch.id):
            match item.result.type:
                case "succeeded":
                    results[item.custom_id] = item.result.message
                case "errored":
                    logger.warning("Batch request %s errored: %s", item.custom_id, item.result.error)
                case "expired":
                    logger.warning("Batch request %s expired", item.custom_id)
                case "canceled":
                    logger.warning("Batch request %s canceled", item.custom_id)

        logger.info("Batch %s complete: %d/%d succeeded", batch.id, len(results), len(requests))
        return results
