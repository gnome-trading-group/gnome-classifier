import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import voyageai
from gnomepy.registry.types import Event

logger = logging.getLogger(__name__)


class BatchVoyageClient:
    def __init__(self, client: voyageai.Client, max_workers: int = 5, max_retries: int = 3):
        self._client = client
        self._max_workers = max_workers
        self._max_retries = max_retries

    def embed_events(self, events: list[Event], chunk_size: int = 2000) -> Iterator[dict[int, list[float]]]:
        for i in range(0, len(events), chunk_size):
            chunk = events[i:i + chunk_size]
            texts = []
            for event in chunk:
                text = event.title
                if event.description:
                    text += ". " + event.description[:200]
                texts.append(text)
            result_embeddings = self.embed(texts)
            yield {event.event_id: emb for event, emb in zip(chunk, result_embeddings)}

    def embed(self, texts: list[str], model: str = "voyage-3", input_type: str = "document") -> list[list[float]]:
        if not texts:
            return []
        batches = [(i, texts[i:i + 128]) for i in range(0, len(texts), 128)]
        total = len(batches)
        results: dict[int, list[list[float]]] = {}

        logger.info("Embedding %d texts in %d batches (workers=%d)", len(texts), total, self._max_workers)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._embed_with_retry, batch_texts, model, input_type): idx
                for idx, batch_texts in batches
            }
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if completed % 10 == 0 or completed == total:
                    logger.info("Embedded %d/%d batches", completed, total)

        all_embeddings: list[list[float]] = []
        for idx, _ in batches:
            all_embeddings.extend(results[idx])
        return all_embeddings

    def _embed_with_retry(self, texts: list[str], model: str, input_type: str) -> list[list[float]]:
        for attempt in range(self._max_retries):
            try:
                result = self._client.embed(texts, model=model, input_type=input_type)
                return result.embeddings
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
                logger.warning("Voyage embed retry %d/%d: %s", attempt + 1, self._max_retries, e)
