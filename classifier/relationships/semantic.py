import json
import logging
from collections import defaultdict

from classifier.cache import ClassifierCache
from classifier.client import BatchAnthropicClient
from classifier.db import ClassifierDB
from classifier.types import (
    Embedding,
    EventId,
    JudgedRelationship,
    RelationshipMatch,
    RelationshipType,
    SecurityId,
)
from classifier.relationships.structural import build_complement_map, derive_complement_relationships, primary_contracts
from gnomepy.registry.types import Event, EventContract

logger = logging.getLogger(__name__)

EMBEDDING_SIMILARITY_THRESHOLD = 0.80

MODEL = "claude-sonnet-4-6"

_JUDGE_SYSTEM_PROMPT = """You are classifying relationships between specific prediction market contracts for trading purposes.

For each pair of contracts (one from A, one from B) that has a meaningful trading relationship, return an entry. Use these types:
- EQUIVALENT: Same question worded differently (direct arbitrage)
- IMPLIES: Contract A[i] being true logically implies contract B[j] must be true. Use "direction": "B_IMPLIES_A" if the reverse.
- CORRELATED: Same underlying asset/entity, outcomes tend to move together but neither strictly implies the other. Different assets are NEVER CORRELATED (BTC and ETH are NONE).
- MUTUALLY_EXCLUSIVE: Both contracts CANNOT BOTH RESOLVE YES — they are logically incompatible outcomes (e.g., "Candidate A wins" and "Candidate B wins" in the same race). Do NOT use this for contracts that merely seem like opposites but can both resolve as stated — e.g., "election called by June 30 — No" and "election called by December 31 — Yes" CAN both be true (election called in September), so they are NOT mutually exclusive.

- NONE / omit: No meaningful trading relationship

Most pairs are unrelated — only include pairs with genuine trading signal. Return [] if none.

Respond with a JSON array only:
[{"a": 1, "b": 1, "type": "EQUIVALENT", "confidence": 0.95}, ...]
For IMPLIES entries add "direction": "A_IMPLIES_B" or "B_IMPLIES_A".
Only output the JSON array, nothing else."""


def find_semantic_candidates(
    events: list[Event],
    event_contracts: list[EventContract],
    embeddings: dict[EventId, Embedding],
    *,
    db: ClassifierDB,
    new_event_ids: set[EventId] | None = None,
    cache: ClassifierCache | None = None,
) -> tuple[list[tuple[Event, Event, list[EventContract], list[EventContract], float]], list[JudgedRelationship]]:
    """Find candidate event pairs via embedding similarity, splitting into cache hits and pending.

    Returns (pending_pairs, cached_judged) where pending_pairs need Claude judgment and
    cached_judged are already-resolved relationships from the cache.
    """
    by_event: dict[EventId, list[EventContract]] = defaultdict(list)
    for ec in event_contracts:
        by_event[ec.event_id].append(ec)

    event_by_id = {e.event_id: e for e in events}

    candidate_pairs: dict[tuple[EventId, EventId], float] = {}
    for eid in (new_event_ids or set()):
        if eid not in embeddings or eid not in by_event:
            continue
        neighbors = db.find_neighbors(embeddings[eid], EMBEDDING_SIMILARITY_THRESHOLD)
        for neighbor_eid, sim in neighbors:
            if neighbor_eid == eid or neighbor_eid not in by_event:
                continue
            pair = (min(eid, neighbor_eid), max(eid, neighbor_eid))
            if pair not in candidate_pairs or sim > candidate_pairs[pair]:
                candidate_pairs[pair] = sim

    cached_judged: list[JudgedRelationship] = []
    pending: list[tuple[Event, Event, list[EventContract], list[EventContract], float]] = []

    for (eid_a, eid_b), similarity in candidate_pairs.items():
        try:
            ev_a, ev_b = event_by_id.get(eid_a), event_by_id.get(eid_b)
            if ev_a is None or ev_b is None:
                continue
            if ev_a.category and ev_b.category and ev_a.category != ev_b.category:
                continue

            contracts_a = by_event[eid_a]
            contracts_b = by_event[eid_b]
            primary_a = primary_contracts(contracts_a)
            primary_b = primary_contracts(contracts_b)
            if not primary_a or not primary_b:
                continue

            labels_a = [ec.outcome_label for ec in primary_a]
            labels_b = [ec.outcome_label for ec in primary_b]

            if cache is not None:
                cached = cache.get_judgment(MODEL, ev_a.title, labels_a, ev_b.title, labels_b)
                if cached is not None:
                    cached_items, a_is_first = cached
                    cached_judged.extend(_parse_cached_judgment(cached_items, primary_a, primary_b, a_is_first))
                    continue

            pending.append((ev_a, ev_b, contracts_a, contracts_b, similarity))
        except Exception as e:
            logger.error("Failed comparing events %d and %d: %s", eid_a, eid_b, e)

    return pending, cached_judged


def build_judgment_requests(
    pending: list[tuple[Event, Event, list[EventContract], list[EventContract], float]],
) -> tuple[list[dict], list[dict]]:
    """Build Claude API requests and a JSON-serializable context for later result processing.

    Returns (api_requests, pending_context). Each context entry stores the data needed to
    parse responses and write cache entries without re-querying ORM objects.
    """
    api_requests = []
    pending_context = []

    for idx, (ev_a, ev_b, contracts_a, contracts_b, similarity) in enumerate(pending):
        primary_a = primary_contracts(contracts_a)
        primary_b = primary_contracts(contracts_b)
        contracts_a_lines = "  ".join(f"[{i+1}] {ec.outcome_label}" for i, ec in enumerate(primary_a))
        contracts_b_lines = "  ".join(f"[{i+1}] {ec.outcome_label}" for i, ec in enumerate(primary_b))
        user_content = (
            f"Event A: {ev_a.title}\n"
            f"  Contracts: {contracts_a_lines}\n\n"
            f"Event B: {ev_b.title}\n"
            f"  Contracts: {contracts_b_lines}\n\n"
            f"Embedding similarity: {similarity:.3f}"
        )
        custom_id = f"j_{idx}"
        api_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 300,
                "system": [{"type": "text", "text": _JUDGE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user_content}],
            },
        })
        pending_context.append({
            "custom_id": custom_id,
            "event_a_id": ev_a.event_id,
            "event_b_id": ev_b.event_id,
            "event_a_title": ev_a.title,
            "event_b_title": ev_b.title,
            "primary_a_sids": [ec.security_id for ec in primary_a],
            "primary_b_sids": [ec.security_id for ec in primary_b],
            "labels_a": [ec.outcome_label for ec in primary_a],
            "labels_b": [ec.outcome_label for ec in primary_b],
            "similarity": similarity,
        })

    return api_requests, pending_context


def parse_judgment_responses(
    responses: dict[str, object],
    pending_context: list[dict],
    cache: ClassifierCache | None,
) -> list[JudgedRelationship]:
    """Parse Claude batch API responses using the saved pending_context.

    Writes cache entries for successful judgments.
    """
    results: list[JudgedRelationship] = []
    ctx_by_id = {entry["custom_id"]: entry for entry in pending_context}

    for custom_id, response in responses.items():
        ctx = ctx_by_id.get(custom_id)
        if ctx is None:
            continue
        idx_to_sid_a = {i + 1: sid for i, sid in enumerate(ctx["primary_a_sids"])}
        idx_to_sid_b = {i + 1: sid for i, sid in enumerate(ctx["primary_b_sids"])}
        idx_to_label_a = {i + 1: lbl for i, lbl in enumerate(ctx["labels_a"])}
        idx_to_label_b = {i + 1: lbl for i, lbl in enumerate(ctx["labels_b"])}

        judged, cache_items = _parse_response_text(
            response.content[0].text.strip(),
            idx_to_sid_a, idx_to_sid_b, idx_to_label_a, idx_to_label_b,
        )
        results.extend(judged)

        if cache is not None and cache_items:
            a_is_first = (ctx["event_a_title"], "|".join(ctx["labels_a"])) <= (ctx["event_b_title"], "|".join(ctx["labels_b"]))
            cache.put_judgment(
                MODEL,
                ctx["event_a_title"], ctx["labels_a"],
                ctx["event_b_title"], ctx["labels_b"],
                cache_items, a_is_first,
            )

    return results


def derive_semantic_relationships(
    judged: list[JudgedRelationship],
    event_contracts: list[EventContract],
) -> list[RelationshipMatch]:
    """Run complement derivation and convert to RelationshipMatch list."""
    complement_of = build_complement_map(event_contracts)
    derived = derive_complement_relationships(judged, complement_of)
    return [
        RelationshipMatch(r.security_id_a, r.security_id_b, r.relationship_type, r.confidence, "embedding")
        for r in judged + derived
    ]


def _parse_response_text(
    raw: str,
    idx_to_sid_a: dict[int, SecurityId],
    idx_to_sid_b: dict[int, SecurityId],
    idx_to_label_a: dict[int, str],
    idx_to_label_b: dict[int, str],
) -> tuple[list[JudgedRelationship], list[dict]]:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    logger.debug("judge_relationship response: %s", raw)

    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            logger.debug("response is not a list: %r", items)
            return [], []

        results: list[JudgedRelationship] = []
        cache_items: list[dict] = []

        for item in items:
            idx_a = item.get("a")
            idx_b = item.get("b")
            rel_type_str = item.get("type", "NONE")
            confidence = float(item.get("confidence", 0.70))
            if idx_a not in idx_to_sid_a or idx_b not in idx_to_sid_b:
                logger.debug("skipping item (index out of range): %s", item)
                continue
            if rel_type_str not in RelationshipType.__members__:
                logger.debug("skipping item (invalid type): %s", item)
                continue
            if confidence < 0.70:
                logger.debug("skipping item (low confidence): %s", item)
                continue

            sid_a = idx_to_sid_a[idx_a]
            sid_b = idx_to_sid_b[idx_b]
            rt = RelationshipType(rel_type_str)
            direction = item.get("direction", "A_IMPLIES_B")

            cache_item: dict = {
                "first_label": idx_to_label_a[idx_a],
                "second_label": idx_to_label_b[idx_b],
                "type": rel_type_str,
                "confidence": confidence,
            }
            if rt == RelationshipType.IMPLIES:
                cache_item["direction"] = direction
            cache_items.append(cache_item)

            if rt == RelationshipType.IMPLIES and direction == "B_IMPLIES_A":
                results.append(JudgedRelationship(sid_b, sid_a, rt, confidence))
            elif rt == RelationshipType.IMPLIES:
                results.append(JudgedRelationship(sid_a, sid_b, rt, confidence))
            else:
                results.append(JudgedRelationship(sid_a, sid_b, rt, confidence))
                results.append(JudgedRelationship(sid_b, sid_a, rt, confidence))

        return results, cache_items
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.debug("judge_relationship parse error: %s — raw: %r", e, raw)
        return [], []


def _parse_cached_judgment(
    cached_items: list[dict],
    yes_a: list[EventContract],
    yes_b: list[EventContract],
    a_is_first: bool,
) -> list[JudgedRelationship]:
    if a_is_first:
        label_to_sid_first = {ec.outcome_label: ec.security_id for ec in yes_a}
        label_to_sid_second = {ec.outcome_label: ec.security_id for ec in yes_b}
    else:
        label_to_sid_first = {ec.outcome_label: ec.security_id for ec in yes_b}
        label_to_sid_second = {ec.outcome_label: ec.security_id for ec in yes_a}

    results: list[JudgedRelationship] = []
    for item in cached_items:
        sid_first = label_to_sid_first.get(item["first_label"])
        sid_second = label_to_sid_second.get(item["second_label"])
        if sid_first is None or sid_second is None:
            continue
        rel_type_str = item["type"]
        if rel_type_str not in RelationshipType.__members__:
            continue
        rt = RelationshipType(rel_type_str)
        confidence = float(item["confidence"])
        direction = item.get("direction", "A_IMPLIES_B")

        if a_is_first:
            sid_a, sid_b = sid_first, sid_second
        else:
            sid_a, sid_b = sid_second, sid_first

        if rt == RelationshipType.IMPLIES and direction == "B_IMPLIES_A":
            results.append(JudgedRelationship(sid_b, sid_a, rt, confidence))
        elif rt == RelationshipType.IMPLIES:
            results.append(JudgedRelationship(sid_a, sid_b, rt, confidence))
        else:
            results.append(JudgedRelationship(sid_a, sid_b, rt, confidence))
            results.append(JudgedRelationship(sid_b, sid_a, rt, confidence))

    return results
