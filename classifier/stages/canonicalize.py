import json
import logging
from typing import Any

from classifier.cache import ClassifierCache
from classifier.client import BatchAnthropicClient
from classifier.constants import CANONICALIZE_BATCH_SIZE, STANDARDIZED_CATEGORIES
from classifier.types import CanonicalizeInput, NativeKey

logger = logging.getLogger(__name__)

CANONICALIZE_MODEL = "claude-haiku-4-5-20251001"

_CATEGORIES_STR = ", ".join(sorted(STANDARDIZED_CATEGORIES))


def _parse_response(response: Any) -> Any:
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _parse_canonical_result(item: dict, raw_title: str) -> dict:
    category = item.get("category", "OTHER")
    if category not in STANDARDIZED_CATEGORIES:
        category = "OTHER"
    tags = item.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    else:
        tags = [t for t in tags if isinstance(t, str)][:8]
    return {"title": item.get("title", raw_title), "category": category, "tags": tags}


def _build_chunk_prompt(batch: list[CanonicalizeInput]) -> str:
    event_lines = "\n".join(
        f"[{j + 1}] Title: {ev.raw_title} | Description: {(ev.description or '')[:200]} | Category: {ev.category or ''}"
        for j, ev in enumerate(batch)
    )
    return f"""You are standardizing prediction market events for a cross-exchange registry.

For each event below, generate:
1. title: Clean, exchange-neutral, concise title for this prediction market question. Preserve all dates exactly as stated.
2. category: One of {_CATEGORIES_STR}
3. tags: 3-8 lowercase keyword tags

Events:
{event_lines}

Respond with a JSON array, one object per event, echoing the input number as "id":
[{{"id": 1, "title": "...", "category": "...", "tags": ["..."]}}, ...]"""


def canonicalize_events(
    batch_client: BatchAnthropicClient,
    events: list[CanonicalizeInput],
    cache: ClassifierCache | None = None,
) -> dict[NativeKey, dict[str, Any]]:
    """Canonicalize a list of CanonicalizeInput records.
    Returns a mapping from (exchange_id, native_id) to {"title", "category", "tags"}."""
    results: dict[NativeKey, dict] = {}
    if not events:
        return results

    uncached: list[CanonicalizeInput] = []
    if cache is not None:
        cached_bulk = cache.get_canonicalization_bulk(
            CANONICALIZE_MODEL, [(ev.exchange_id, ev.native_id) for ev in events]
        )
        for ev in events:
            cached = cached_bulk.get((ev.exchange_id, ev.native_id))
            if cached is not None:
                results[(ev.exchange_id, ev.native_id)] = cached
            else:
                uncached.append(ev)
        logger.info("canonicalize: %d cache hits, %d to call Claude", len(cached_bulk), len(uncached))
    else:
        uncached = list(events)
        logger.info("canonicalize: no cache, %d to call Claude", len(uncached))

    chunks: list[list[CanonicalizeInput]] = [
        uncached[i:i + CANONICALIZE_BATCH_SIZE]
        for i in range(0, len(uncached), CANONICALIZE_BATCH_SIZE)
    ]

    requests = [
        {
            "custom_id": f"canon_{i}",
            "params": {
                "model": CANONICALIZE_MODEL,
                "max_tokens": 300 * len(chunk),
                "messages": [{"role": "user", "content": _build_chunk_prompt(chunk)}],
            },
        }
        for i, chunk in enumerate(chunks)
    ]

    responses = batch_client.create_messages(requests)

    for i, chunk in enumerate(chunks):
        custom_id = f"canon_{i}"
        response = responses.get(custom_id)
        by_id: dict[int, dict] = {}

        if response is not None:
            try:
                parsed = _parse_response(response)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and isinstance(item.get("id"), int):
                            by_id[item["id"]] = item
            except Exception as e:
                raw = response.content[0].text if response.content else "<empty>"
                logger.warning("Chunk %d parse failed: %s raw=%r", i, e, raw[:500])

        missed: list[CanonicalizeInput] = []
        for j, ev in enumerate(chunk):
            item = by_id.get(j + 1)
            if item is not None:
                result = _parse_canonical_result(item, ev.raw_title)
                results[(ev.exchange_id, ev.native_id)] = result
                if cache is not None:
                    cache.put_canonicalization(CANONICALIZE_MODEL, ev.exchange_id, ev.native_id, result)
            else:
                missed.append(ev)

        if missed:
            logger.warning("Chunk %d: %d/%d matched, retrying %d individually",
                           i, len(chunk) - len(missed), len(chunk), len(missed))
            for ev in missed:
                result = _canonicalize_single(batch_client._client, ev.raw_title, ev.description, ev.category)
                if result is None:
                    continue
                results[(ev.exchange_id, ev.native_id)] = result
                if cache is not None:
                    cache.put_canonicalization(CANONICALIZE_MODEL, ev.exchange_id, ev.native_id, result)

    failed = len(events) - len(results)
    if failed > 0:
        logger.warning("canonicalize: %d events failed canonicalization, will retry next run", failed)

    return results


def _canonicalize_single(
    client,
    raw_title: str,
    description: str | None,
    exchange_category: str | None,
) -> dict | None:
    prompt = f"""You are standardizing a prediction market event for a cross-exchange registry.

Exchange-provided title: {raw_title}
Description: {description or ''}
Exchange category: {exchange_category or ''}

Generate:
1. title: Clean, exchange-neutral, concise title for this prediction market question. Preserve all dates exactly as stated.
2. category: One of {_CATEGORIES_STR}
3. tags: 3-8 lowercase keyword tags

Respond with JSON only: {{"title": "...", "category": "...", "tags": ["..."]}}"""

    response = None
    try:
        response = client.messages.create(
            model=CANONICALIZE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _parse_response(response)
        return _parse_canonical_result(result, raw_title)
    except Exception as e:
        raw = response.content[0].text if response is not None else "<no response>"
        logger.warning("Canonicalization failed for '%s': %s stop_reason=%s raw=%r", raw_title, e,
                       response.stop_reason if response is not None else None, raw[:500])
        return None
