import json
import logging
from typing import Any

import anthropic

from classifier.cache import ClassifierCache
from classifier.constants import CANONICALIZE_BATCH_SIZE, STANDARDIZED_CATEGORIES
from classifier.types import CanonicalizeInput

logger = logging.getLogger(__name__)

CANONICALIZE_MODEL = "claude-haiku-4-5-20251001"

_CATEGORIES_STR = ", ".join(sorted(STANDARDIZED_CATEGORIES))


def _parse_canonical_result(item: dict, raw_title: str) -> dict:
    category = item.get("category", "OTHER")
    if category not in STANDARDIZED_CATEGORIES:
        category = "OTHER"
    tags = item.get("tags", [])
    if not isinstance(tags, list) or not (3 <= len(tags) <= 8):
        tags = []
    return {"title": item.get("title", raw_title), "category": category, "tags": tags}


def canonicalize_events(
    client: anthropic.Anthropic,
    events: list[CanonicalizeInput],
    cache: ClassifierCache | None = None,
) -> dict[str, dict[str, Any]]:
    """Canonicalize a list of CanonicalizeInput records.
    Returns a mapping from raw_title to {"title", "category", "tags"}."""
    results: dict[str, dict] = {}

    uncached: list[CanonicalizeInput] = []
    if cache is not None:
        for ev in events:
            cached = cache.get_canonicalization(CANONICALIZE_MODEL, ev.exchange_id, ev.native_id)
            if cached is not None:
                results[ev.raw_title] = cached
            else:
                uncached.append(ev)
    else:
        uncached = list(events)

    for i in range(0, len(uncached), CANONICALIZE_BATCH_SIZE):
        batch = uncached[i:i + CANONICALIZE_BATCH_SIZE]
        event_lines = "\n".join(
            f"[{j + 1}] Title: {ev.raw_title} | Description: {(ev.description or '')[:200]} | Category: {ev.category or ''}"
            for j, ev in enumerate(batch)
        )
        prompt = f"""You are standardizing prediction market events for a cross-exchange registry.

For each event below, generate:
1. title: Clean, exchange-neutral, concise title for this prediction market question
2. category: One of {_CATEGORIES_STR}
3. tags: 3-8 lowercase keyword tags

Events:
{event_lines}

Respond with a JSON array in the same order, one object per event:
[{{"title": "...", "category": "...", "tags": ["..."]}}, ...]"""

        batch_results = None
        try:
            response = client.messages.create(
                model=CANONICALIZE_MODEL,
                max_tokens=250 * len(batch),
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == len(batch):
                batch_results = parsed
            else:
                logger.warning(
                    "Batch canonicalization returned wrong length (expected %d, got %d)",
                    len(batch), len(parsed) if isinstance(parsed, list) else -1,
                )
        except Exception as e:
            logger.warning("Batch canonicalization failed at offset %d: %s", i, e)

        if batch_results is not None:
            for j, ev in enumerate(batch):
                result = _parse_canonical_result(batch_results[j], ev.raw_title)
                results[ev.raw_title] = result
                if cache is not None:
                    cache.put_canonicalization(CANONICALIZE_MODEL, ev.exchange_id, ev.native_id, result)
        else:
            for ev in batch:
                result = _canonicalize_single(client, ev.raw_title, ev.description, ev.category)
                results[ev.raw_title] = result
                if cache is not None:
                    cache.put_canonicalization(CANONICALIZE_MODEL, ev.exchange_id, ev.native_id, result)

    return results


def _canonicalize_single(
    client: anthropic.Anthropic,
    raw_title: str,
    description: str | None,
    exchange_category: str | None,
) -> dict:
    prompt = f"""You are standardizing a prediction market event for a cross-exchange registry.

Exchange-provided title: {raw_title}
Description: {description or ''}
Exchange category: {exchange_category or ''}

Generate:
1. title: Clean, exchange-neutral, concise title for this prediction market question.
2. category: One of {_CATEGORIES_STR}
3. tags: 3-8 lowercase keyword tags

Respond with JSON only: {{"title": "...", "category": "...", "tags": ["..."]}}"""

    try:
        response = client.messages.create(
            model=CANONICALIZE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(response.content[0].text.strip())
        return _parse_canonical_result(result, raw_title)
    except Exception as e:
        logger.warning("Canonicalization failed for '%s': %s", raw_title, e)
        return {"title": raw_title, "category": "OTHER", "tags": []}
