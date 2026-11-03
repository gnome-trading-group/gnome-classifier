import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from classifier import taxonomy
from classifier.cache import ClassifierCache
from classifier.client.claude import ClaudeAsker
from classifier.constants import DEFAULT_CANONICALIZE_BATCH_SIZE, STANDARDIZED_CATEGORIES
from classifier.types import CanonicalizeInput, NativeKey

logger = logging.getLogger(__name__)

_MAX_TOKENS_PER_EVENT = 300
EFFORT = "low"
MAX_WORKERS = 8

# The boundaries between neighbouring categories are where models disagree, so each definition says what it excludes.
_CATEGORY_DEFINITIONS = """\
- SPORTS: games, players, teams, leagues, tournaments and athletes' awards; esports included
- POLITICS: elections, governments, officials, legislation, geopolitics and wars
- ECONOMICS: macro data (inflation, jobs, GDP), central banks and interest rates, trade and tariffs
- FINANCE: companies (IPOs, valuations, earnings, acquisitions, executives), stocks, indices and commodities; a tech or crypto company's IPO or valuation is FINANCE
- CRYPTO: cryptocurrencies, tokens and their prices, token launches, airdrops
- TECHNOLOGY: tech and AI products, launches, capabilities and outages
- SCIENCE: space, research, discoveries, mathematics
- HEALTH: diseases, outbreaks, medicine, drug approvals
- WEATHER: temperatures, storms, climate measurements
- ENTERTAINMENT: film, TV, music, celebrities, video games and streaming culture
- LEGAL: courts, trials, lawsuits, indictments and pardons
- OTHER: none of the above"""

# Exchanges label events with free-form categories ("Midterms", "FDV"); models echo those back as the category unless
# told not to. Tags come back as additions only, since the exchange tags are merged in ahead of them.
_INSTRUCTIONS = f"""You are categorizing prediction market events for a cross-exchange registry. The exchange's own labels are already applied; you fill in only what they left open.

Categories:
{_CATEGORY_DEFINITIONS}

For each event:
- category: when the event says "Category: <NAME> (fixed)", return that name unchanged. When it says "Category: choose", pick exactly one category above, uppercase as written. The exchange label is only a hint; never return it as the category.
- tags: tags to ADD to the exchange tags already listed, up to 6 in total. Add every person, team, organization, token, place or asset named in the title that the exchange tags don't already include, plus one short theme ("ipo", "acquisition", "pardon", "retirement") when the exchange tags don't already say what kind of event it is. Return [] only when the exchange tags already include all of them.
  Tags are lowercase, in their shortest common form ("unemployment", not "us unemployment rate"), 1-3 words each. No numbers, prices, thresholds, dates, years or times; no category names; nothing about how the market works or resolves ("total points", "spread", "over/under", "price", "winner").

Examples:
Title: Houston vs Jacksonville: Fantasy Points: Woody Marks | Exchange tags: football | Category: SPORTS (fixed)
-> {{"category": "SPORTS", "tags": ["nfl", "woody marks", "houston texans", "jacksonville jaguars"]}}
Title: Will Snapchat be acquired before 2027? | Exchange tags: business, big tech, tech, ai, acquisitions | Category: choose
-> {{"category": "FINANCE", "tags": ["snapchat"]}}
Title: Will there be at least 4750 measles cases in the U.S. in 2026? | Exchange tags: measles | Category: choose
-> {{"category": "HEALTH", "tags": ["us"]}}
Title: Who will Trump pardon?: Bob Menendez | Exchange tags: trump | Category: POLITICS (fixed)
-> {{"category": "POLITICS", "tags": ["bob menendez", "pardon"]}}
Title: Will GameStop acquire eBay? | Exchange tags: acquisitions | Category: choose
-> {{"category": "FINANCE", "tags": ["gamestop", "ebay"]}}"""


def _parse_canonical_result(item: dict) -> dict:
    category = item.get("category")
    # Models often title-case the label ("Politics"); treating that as unknown would file the event under OTHER.
    category = category.strip().upper() if isinstance(category, str) else "OTHER"
    if category not in STANDARDIZED_CATEGORIES:
        category = "OTHER"
    tags = item.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    # The same cleanup as exchange tags is the backstop for the prompt rules.
    return {"category": category, "tags": taxonomy.clean_tags(tags)}


def needs_llm(ev: CanonicalizeInput) -> bool:
    return ev.mapped_category is None or len(ev.exchange_tags) < taxonomy.MIN_STATIC_TAGS


def static_result(ev: CanonicalizeInput) -> dict:
    """Category and tags from the exchange's own labels alone."""
    return {"category": ev.mapped_category or "OTHER", "tags": list(ev.exchange_tags)}


def _merge(ev: CanonicalizeInput, llm: dict) -> dict:
    # The exchange's labels win; the model only fills the category they left open and tops up thin tags.
    return {
        "category": ev.mapped_category or llm["category"],
        "tags": taxonomy.merge_tags(ev.exchange_tags, llm["tags"]),
    }


def _one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def _fields(ev: CanonicalizeInput) -> str:
    category = f"{ev.mapped_category} (fixed)" if ev.mapped_category else "choose"
    # Some exchange titles carry newlines (Kalshi: " XRP price at ...?\n: $1.69991 or above"), which would split an
    # event across prompt lines.
    return (
        f"Title: {_one_line(ev.raw_title)} | Description: {_one_line(ev.description)[:200]}"
        f" | Exchange label: {_one_line(ev.category)}"
        f" | Exchange tags: {', '.join(ev.exchange_tags) or 'none'} | Category: {category}"
    )


def _title_key(raw_title: str) -> str:
    return hashlib.sha256(raw_title.encode()).hexdigest()[:6]


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "key": {"type": "string"},
                    "category": {"type": "string", "enum": sorted(STANDARDIZED_CATEGORIES)},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "key", "category", "tags"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["events"],
    "additionalProperties": False,
}


def _build_prompt(batch: list[CanonicalizeInput]) -> str:
    event_lines = "\n".join(f"[{j + 1}] (key: {_title_key(ev.raw_title)}) {_fields(ev)}" for j, ev in enumerate(batch))
    return f"""Events:
{event_lines}

Return one entry per event, echoing its number as "id" and its key as "key"."""


def _match_chunk(text: str | None, chunk: list[CanonicalizeInput], request_id: str) -> tuple[dict[NativeKey, dict], list[CanonicalizeInput]]:
    """Match an answer back to its events by id, validated by the echoed key. Returns (matched, missed)."""
    by_id: dict[int, dict] = {}
    if text is not None:
        try:
            for item in json.loads(text)["events"]:
                if isinstance(item, dict) and isinstance(item.get("id"), int):
                    by_id[item["id"]] = item
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Chunk %s parse failed: %s raw=%r", request_id, e, text[:500])

    matched: dict[NativeKey, dict] = {}
    missed: list[CanonicalizeInput] = []
    for j, ev in enumerate(chunk):
        item = by_id.get(j + 1)
        if item is None:
            missed.append(ev)
            continue
        expected_key = _title_key(ev.raw_title)
        echoed_key = item.get("key", "")
        if echoed_key != expected_key:
            logger.warning(
                "Canonicalization key mismatch: echoed=%r expected=%r (id=%d, title=%r)",
                echoed_key, expected_key, j + 1, ev.raw_title[:60],
            )
            missed.append(ev)
            continue
        matched[(ev.exchange_id, ev.native_id)] = _parse_canonical_result(item)
    return matched, missed


def canonicalize_events(
    asker: ClaudeAsker,
    events: list[CanonicalizeInput],
    cache: ClassifierCache | None = None,
    batch_size: int = DEFAULT_CANONICALIZE_BATCH_SIZE,
    debug: bool = False,
    max_workers: int = MAX_WORKERS,
) -> dict[NativeKey, dict[str, Any]]:
    """Categorize a list of CanonicalizeInput records.

    Events whose exchange labels give a category and enough tags skip the model; the rest go to it and keep whatever
    their labels did give. Returns a mapping from (exchange_id, native_id) to {"category", "tags"}."""
    if not events:
        return {}
    results = {(ev.exchange_id, ev.native_id): static_result(ev) for ev in events if not needs_llm(ev)}
    pending = [ev for ev in events if needs_llm(ev)]
    logger.info("canonicalize: %d events from exchange labels, %d need %s", len(results), len(pending), asker.model)
    llm_results = _llm_canonicalize(asker, pending, cache, batch_size, debug, max_workers)
    for ev in pending:
        results[(ev.exchange_id, ev.native_id)] = _merge(ev, llm_results[(ev.exchange_id, ev.native_id)])
    return results


def _ask_chunks(asker: ClaudeAsker, chunks: dict[str, list[CanonicalizeInput]], max_workers: int) -> tuple[dict[str, str], int, int]:
    """{request id: answer text} for the requests that got an answer, and the tokens they used. A failed request is
    logged and left out, so its events count as missed."""
    answers: dict[str, str] = {}
    input_tokens = output_tokens = 0
    if not chunks:
        return answers, 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(asker.ask_with_usage, _INSTRUCTIONS, _build_prompt(chunk), RESPONSE_SCHEMA,
                        _MAX_TOKENS_PER_EVENT * len(chunk), EFFORT): request_id
            for request_id, chunk in chunks.items()
        }
        for future in as_completed(futures):
            request_id = futures[future]
            try:
                answer = future.result()
            except Exception as e:
                logger.warning("Canonicalization request %s failed: %s", request_id, e)
                continue
            answers[request_id] = answer.text
            input_tokens += answer.input_tokens
            output_tokens += answer.output_tokens
    return answers, input_tokens, output_tokens


def _llm_canonicalize(
    asker: ClaudeAsker,
    events: list[CanonicalizeInput],
    cache: ClassifierCache | None,
    batch_size: int,
    debug: bool,
    max_workers: int,
) -> dict[NativeKey, dict[str, Any]]:
    """The model's category and tags per event, cached per model; chunk misses are retried one event at a time."""
    if not events:
        return {}
    model = asker.model

    results: dict[NativeKey, dict] = {}
    uncached: list[CanonicalizeInput] = []
    if cache is not None:
        cached_bulk = cache.get_canonicalization_bulk(model, [(ev.exchange_id, ev.native_id) for ev in events])
        for ev in events:
            cached = cached_bulk.get((ev.exchange_id, ev.native_id))
            if cached is not None:
                results[(ev.exchange_id, ev.native_id)] = cached
            else:
                uncached.append(ev)
    else:
        uncached = list(events)
    logger.info("canonicalize: %d cache hits, %d to call %s", len(results), len(uncached), model)
    if not uncached:
        return results

    chunks = {f"canon_{i}": uncached[i:i + batch_size] for i in range(0, len(uncached), batch_size)}
    answers, input_tokens, output_tokens = _ask_chunks(asker, chunks, max_workers)

    fresh: dict[NativeKey, dict] = {}
    missed: list[CanonicalizeInput] = []
    for request_id, chunk in chunks.items():
        matched, chunk_missed = _match_chunk(answers.get(request_id), chunk, request_id)
        fresh.update(matched)
        missed.extend(chunk_missed)
        if chunk_missed:
            logger.warning("Chunk %s: %d/%d matched, retrying %d individually",
                           request_id, len(matched), len(chunk), len(chunk_missed))

    singles = {f"single_{i}": [ev] for i, ev in enumerate(missed)}
    retry_answers, retry_in, retry_out = _ask_chunks(asker, singles, max_workers)
    input_tokens += retry_in
    output_tokens += retry_out
    for request_id, chunk in singles.items():
        matched, _ = _match_chunk(retry_answers.get(request_id), chunk, request_id)
        fresh.update(matched)

    logger.info(
        "canonicalize: %d events via %s (%d individual retries), %d input / %d output tokens",
        len(fresh), model, len(missed), input_tokens, output_tokens,
    )

    if cache is not None:
        for (exchange_id, native_id), result in fresh.items():
            cache.put_canonicalization(model, exchange_id, native_id, result)
    if debug:
        for ev in uncached:
            result = fresh.get((ev.exchange_id, ev.native_id))
            if result is not None:
                logger.info("[DEBUG] canonicalize: %r [%s] tags=%s",
                            ev.raw_title[:60], result["category"], result["tags"])

    failed_keys = [(ev.exchange_id, ev.native_id) for ev in uncached if (ev.exchange_id, ev.native_id) not in fresh]
    if failed_keys:
        logger.error("canonicalize: %d events permanently failed: %s", len(failed_keys), failed_keys)
        raise RuntimeError(f"{len(failed_keys)} events failed canonicalization after individual retry: {failed_keys}")

    results.update(fresh)
    return results
