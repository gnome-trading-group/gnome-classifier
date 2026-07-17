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


def prepare_canon_batch(
    events: list[CanonicalizeInput],
    cache: ClassifierCache | None = None,
) -> tuple[list[dict], list[dict], dict[NativeKey, dict]]:
    """Separate cached events and build API requests for uncached ones.

    Returns:
      api_requests: request dicts for batch_client.submit_batch / create_messages
      canon_context: JSON-serializable chunk metadata list (for parse_canon_results)
      cached_results: dict[NativeKey, dict] for cache hits
    """
    if not events:
        return [], [], {}

    cached_results: dict[NativeKey, dict] = {}
    uncached: list[CanonicalizeInput] = []

    if cache is not None:
        cached_bulk = cache.get_canonicalization_bulk(
            CANONICALIZE_MODEL, [(ev.exchange_id, ev.native_id) for ev in events]
        )
        for ev in events:
            cached = cached_bulk.get((ev.exchange_id, ev.native_id))
            if cached is not None:
                cached_results[(ev.exchange_id, ev.native_id)] = cached
            else:
                uncached.append(ev)
        logger.info("prepare_canon_batch: %d cache hits, %d to call Claude", len(cached_results), len(uncached))
    else:
        uncached = list(events)
        logger.info("prepare_canon_batch: no cache, %d to call Claude", len(uncached))

    chunks = [uncached[i:i + CANONICALIZE_BATCH_SIZE] for i in range(0, len(uncached), CANONICALIZE_BATCH_SIZE)]

    api_requests = [
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

    canon_context = [
        {
            "custom_id": f"canon_{i}",
            "events": [
                {
                    "raw_title": ev.raw_title,
                    "description": ev.description,
                    "category": ev.category,
                    "exchange_id": ev.exchange_id,
                    "native_id": ev.native_id,
                }
                for ev in chunk
            ],
        }
        for i, chunk in enumerate(chunks)
    ]

    return api_requests, canon_context, cached_results


def parse_canon_results(
    responses: dict,
    canon_context: list[dict],
    cache: ClassifierCache | None,
    anthropic_client: Any,
) -> dict[NativeKey, dict]:
    """Parse batch API responses using canon_context. Retries missed items individually.

    Returns dict[NativeKey, dict] for uncached events only — caller merges cached_results.
    """
    results: dict[NativeKey, dict] = {}

    for chunk_info in canon_context:
        custom_id = chunk_info["custom_id"]
        chunk_events = chunk_info["events"]
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
                logger.warning("Chunk %s parse failed: %s raw=%r", custom_id, e, raw[:500])

        missed: list[dict] = []
        for j, ev_info in enumerate(chunk_events):
            nk: NativeKey = (ev_info["exchange_id"], ev_info["native_id"])
            item = by_id.get(j + 1)
            if item is not None:
                result = _parse_canonical_result(item, ev_info["raw_title"])
                results[nk] = result
                if cache is not None:
                    cache.put_canonicalization(CANONICALIZE_MODEL, nk[0], nk[1], result)
            else:
                missed.append(ev_info)

        if missed:
            logger.warning("Chunk %s: %d/%d matched, retrying %d individually",
                           custom_id, len(chunk_events) - len(missed), len(chunk_events), len(missed))
            for ev_info in missed:
                nk = (ev_info["exchange_id"], ev_info["native_id"])
                result = _canonicalize_single(
                    anthropic_client, ev_info["raw_title"],
                    ev_info.get("description"), ev_info.get("category"),
                )
                if result is None:
                    continue
                results[nk] = result
                if cache is not None:
                    cache.put_canonicalization(CANONICALIZE_MODEL, nk[0], nk[1], result)

    uncached_total = sum(len(c["events"]) for c in canon_context)
    failed = uncached_total - len(results)
    if failed > 0:
        logger.warning("parse_canon_results: %d events failed, will retry next run", failed)

    return results


def canonicalize_events(
    batch_client: BatchAnthropicClient,
    events: list[CanonicalizeInput],
    cache: ClassifierCache | None = None,
) -> dict[NativeKey, dict[str, Any]]:
    """Canonicalize a list of CanonicalizeInput records.
    Returns a mapping from (exchange_id, native_id) to {"title", "category", "tags"}."""
    if not events:
        return {}
    api_requests, canon_context, cached_results = prepare_canon_batch(events, cache)
    responses = batch_client.create_messages(api_requests)
    results = parse_canon_results(responses, canon_context, cache, batch_client._client)
    results.update(cached_results)
    return results


def _canonicalize_single(
    client: Any,
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
