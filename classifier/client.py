import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

logger = logging.getLogger(__name__)


class BatchAnthropicClient:
    def __init__(
        self,
        client: anthropic.Anthropic,
        poll_interval: int = 30,
        max_batch_wait: int = 900,
    ):
        self._client = client
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
        logger.info("Using batch API for %d requests (model=%s)", len(requests), model)
        try:
            return self._batch_create(requests)
        except Exception as e:
            logger.warning("Batch API failed, falling back to sync: %s", e)
            return self._sync_create(requests)

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
