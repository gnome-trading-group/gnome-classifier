import dataclasses
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

logger = logging.getLogger(__name__)

SYNC_THRESHOLD = 10


@dataclasses.dataclass
class SubmitResult:
    batch_id: str | None = None
    responses: dict[str, anthropic.types.Message] | None = None

    @property
    def is_complete(self) -> bool:
        return self.responses is not None


class BatchAnthropicClient:
    def __init__(
        self,
        client: anthropic.Anthropic,
        poll_interval: int = 30,
        max_batch_wait: int = 14400,
    ):
        self._client = client
        self._poll_interval = poll_interval
        self._max_batch_wait = max_batch_wait

    def create_messages(self, requests: list[dict]) -> dict[str, anthropic.types.Message]:
        """Submit N requests, return {custom_id: Message}. Blocks until complete."""
        if not requests:
            return {}
        result = self.submit_batch(requests)
        if result.is_complete:
            return result.responses
        batch_id = result.batch_id
        deadline = time.monotonic() + self._max_batch_wait
        while time.monotonic() < deadline:
            status = self.check_batch(batch_id)
            if status == "ended":
                break
            logger.debug("Batch %s still processing...", batch_id)
            time.sleep(self._poll_interval)
        else:
            raise TimeoutError(f"Batch {batch_id} did not complete within {self._max_batch_wait}s")
        results = self.collect_batch_results(batch_id)
        logger.info("Batch %s complete: %d/%d succeeded", batch_id, len(results), len(requests))
        return results

    def submit_batch(self, requests: list[dict]) -> SubmitResult:
        """Submit requests. Returns SubmitResult — sync (≤10 requests) or with batch_id (>10)."""
        if not requests:
            return SubmitResult(responses={})
        model = requests[0]["params"].get("model", "")
        if len(requests) <= SYNC_THRESHOLD:
            logger.info("Sync processing %d requests (model=%s)", len(requests), model)
            return SubmitResult(responses=self._sync_create(requests))
        logger.info("Submitting batch of %d requests (model=%s)", len(requests), model)
        batch = self._client.messages.batches.create(
            requests=[{"custom_id": r["custom_id"], "params": r["params"]} for r in requests]
        )
        logger.info("Submitted batch %s", batch.id)
        return SubmitResult(batch_id=batch.id)

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

    def check_batch(self, batch_id: str) -> str:
        """Return the current processing_status of a batch."""
        status = self._client.messages.batches.retrieve(batch_id)
        return status.processing_status

    def collect_batch_results(self, batch_id: str) -> dict[str, anthropic.types.Message]:
        """Retrieve all results for a completed batch. Returns {custom_id: Message}."""
        results: dict[str, anthropic.types.Message] = {}
        for item in self._client.messages.batches.results(batch_id):
            match item.result.type:
                case "succeeded":
                    results[item.custom_id] = item.result.message
                case "errored":
                    logger.warning("Batch request %s errored: %s", item.custom_id, item.result.error)
                case "expired":
                    logger.warning("Batch request %s expired", item.custom_id)
                case "canceled":
                    logger.warning("Batch request %s canceled", item.custom_id)
        logger.info("Batch %s collected: %d results", batch_id, len(results))
        return results
