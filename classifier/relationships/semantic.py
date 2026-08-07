import json
import logging
from collections import defaultdict

from classifier.cache import ClassifierCache
from classifier.client import BatchAnthropicClient
from classifier.constants import DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD, DEFAULT_MIN_CONFIDENCE, DEFAULT_SEMANTIC_JUDGMENT_MODEL
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
from classifier.utils import strip_code_fences
from gnomepy.registry.types import Event, EventContract

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are classifying relationships between specific prediction market contracts for trading purposes.

For each pair of contracts (one from A, one from B) that has a meaningful trading relationship, return an entry. Use these types:
- EQUIVALENT: Same question worded differently (direct arbitrage). Requires identical numeric thresholds/targets AND identical dates/deadlines — "above 7,750" vs "above 7,795" is NOT equivalent, and "through August 2" vs "through August 4" is NOT equivalent.
- IMPLIES: Contract A[i] being true logically implies contract B[j] must be true. Use "direction": "B_IMPLIES_A" if the reverse.
- MUTUALLY_EXCLUSIVE: Both contracts CANNOT BOTH RESOLVE YES — they are logically incompatible outcomes (e.g., "Candidate A wins" and "Candidate B wins" in the same race). Do NOT use this for contracts that merely seem like opposites but can both resolve as stated — e.g., "election called by June 30 — No" and "election called by December 31 — Yes" CAN both be true (election called in September), so they are NOT mutually exclusive.
- NONE / omit: No meaningful trading relationship

Most pairs are unrelated — only include pairs with genuine trading signal. Return [] if none.

Respond with a JSON array only:
[{"a": 1, "b": 1, "type": "EQUIVALENT", "confidence": 0.95}, ...]
For IMPLIES entries add "direction": "A_IMPLIES_B" or "B_IMPLIES_A".
Only output the JSON array, nothing else.
Each contract pair (a, b) must appear at most once — pick the single most accurate relationship type.

---

## Examples

### EQUIVALENT — same question, different phrasing across exchanges

Event A: Will Bitcoin price exceed $100,000 by end of 2025?
  Description:
  Contracts: [1] Yes  [2] No
Event B: Bitcoin above $100k on December 31, 2025?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.943
Output: [{"a": 1, "b": 1, "type": "EQUIVALENT", "confidence": 0.96}, {"a": 2, "b": 2, "type": "EQUIVALENT", "confidence": 0.96}]

### EQUIVALENT — same election, both outcomes map

Event A: 2024 US Presidential Election winner
  Description:
  Contracts: [1] Trump  [2] Harris
Event B: Will Donald Trump win the 2024 US Presidential Election?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.921
Output: [{"a": 1, "b": 1, "type": "EQUIVALENT", "confidence": 0.97}, {"a": 2, "b": 2, "type": "EQUIVALENT", "confidence": 0.89}]

### IMPLIES — higher threshold implies lower (A→B)

Event A: Will Bitcoin exceed $200,000 at any point in 2026?
  Description:
  Contracts: [1] Yes  [2] No
Event B: Will Bitcoin exceed $100,000 at any point in 2026?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.891
Output: [{"a": 1, "b": 1, "type": "IMPLIES", "confidence": 0.98, "direction": "A_IMPLIES_B"}]

### IMPLIES — superset implies subset, direction reversed (B→A)

Event A: Will the Fed cut interest rates at least once in 2026?
  Description:
  Contracts: [1] Yes  [2] No
Event B: Will the Fed cut interest rates at least three times in 2026?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.874
Output: [{"a": 1, "b": 1, "type": "IMPLIES", "confidence": 0.97, "direction": "B_IMPLIES_A"}]

### NOT EQUIVALENT — different price thresholds are distinct questions

Event A: S&P 500 price on Aug 5, 2026 at 10am EDT: 7,795 or above
  Description:
  Contracts: [1] Yes
Event B: S&P 500 price on Aug 5, 2026 at 10am EDT: 7,750 or above
  Description:
  Contracts: [1] Yes
Embedding similarity: 0.961
Output: [{"a": 1, "b": 1, "type": "IMPLIES", "confidence": 0.97, "direction": "A_IMPLIES_B"}]

### NOT EQUIVALENT — different deadlines are distinct questions

Event A: Israel Iran Ceasefire Continues Through August 4, 2026
  Description:
  Contracts: [1] Yes
Event B: Israel Iran Ceasefire Continues Through August 2, 2026
  Description:
  Contracts: [1] Yes
Embedding similarity: 0.957
Output: [{"a": 1, "b": 1, "type": "IMPLIES", "confidence": 0.97, "direction": "A_IMPLIES_B"}]

### MUTUALLY_EXCLUSIVE — only one outcome can resolve YES

Event A: 2028 US Presidential Election winner
  Description:
  Contracts: [1] Democratic candidate  [2] Republican candidate
Event B: 2028 US Presidential Election: Will Republicans win?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.903
Output: [{"a": 1, "b": 2, "type": "MUTUALLY_EXCLUSIVE", "confidence": 0.94}, {"a": 2, "b": 1, "type": "EQUIVALENT", "confidence": 0.95}]

### NOT mutually exclusive — time-bound contracts are often IMPLIES, not ME

Event A: Will the US enter a recession by June 30, 2026?
  Description:
  Contracts: [1] Yes  [2] No
Event B: Will the US enter a recession by December 31, 2026?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.908
Output: [{"a": 1, "b": 1, "type": "IMPLIES", "confidence": 0.97, "direction": "A_IMPLIES_B"}]

### NONE — different underlying assets, no trading relationship

Event A: Will Ethereum price exceed $5,000 by end of 2026?
  Description:
  Contracts: [1] Yes  [2] No
Event B: Will Bitcoin price exceed $200,000 by end of 2026?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.813
Output: []

### NONE — same domain, no structural relationship between outcomes

Event A: Who wins the 2026 FIFA World Cup?
  Description:
  Contracts: [1] Brazil  [2] France  [3] Germany  [4] Field
Event B: Will the 2026 FIFA World Cup final have more than 2 goals?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.801
Output: []

### NONE — superficially related topic but no trading relationship

Event A: Will inflation in the US exceed 3% in 2026?
  Description:
  Contracts: [1] Yes  [2] No
Event B: Will the Fed raise interest rates in 2026?
  Description:
  Contracts: [1] Yes  [2] No
Embedding similarity: 0.821
Output: []

### NONE — esports full match vs individual map are different scopes

Event A: Team Jenz vs Power Rangers - EPL Masters 2026 Dota 2 Match
  Description: Match winner: Team Jenz vs Power Rangers
  Contracts: [1] Team Jenz  [2] Power Rangers
Event B: Dota 2: Team Jenz vs Power Rangers Map 1
  Description: Map 1 winner
  Contracts: [1] Team Jenz  [2] Power Rangers
Embedding similarity: 0.948
Output: []"""


def find_semantic_candidates(
    events: list[Event],
    event_contracts: list[EventContract],
    *,
    db: ClassifierDB,
    new_event_ids: set[EventId] | None = None,
    cache: ClassifierCache | None = None,
    threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    neighbor_limit: int = 50,
    allowed_categories: set[str] | None = None,
    model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL,
    debug: bool = False,
) -> tuple[list[tuple[Event, Event, list[EventContract], list[EventContract], float]], list[JudgedRelationship]]:
    """Find candidate event pairs via embedding similarity, splitting into cache hits and pending.

    Returns (pending_pairs, cached_judged) where pending_pairs need Claude judgment and
    cached_judged are already-resolved relationships from the cache.
    """
    by_event: dict[EventId, list[EventContract]] = defaultdict(list)
    for ec in event_contracts:
        by_event[ec.event_id].append(ec)

    event_by_id = {e.event_id: e for e in events}

    if new_event_ids is None:
        db_event_ids = None
    else:
        db_event_ids = [
            eid for eid in new_event_ids
            if eid in by_event
            and not (allowed_categories and (ev := event_by_id.get(eid)) and ev.category and ev.category not in allowed_categories)
        ]
    all_pairs = db.find_all_neighbor_pairs(threshold, event_ids=db_event_ids, neighbor_limit=neighbor_limit)

    neighbor_event_ids = set()
    for eid_a, eid_b, _ in all_pairs:
        if eid_a not in by_event:
            neighbor_event_ids.add(eid_a)
        if eid_b not in by_event:
            neighbor_event_ids.add(eid_b)
    if neighbor_event_ids:
        neighbor_ids_list = list(neighbor_event_ids)
        for ec in db.get_event_contracts_for_events(neighbor_ids_list):
            by_event[ec.event_id].append(ec)
        for ev in db.get_events_for_ids(neighbor_ids_list):
            event_by_id[ev.event_id] = ev

    candidate_pairs: dict[tuple[EventId, EventId], float] = {}
    for eid_a, eid_b, sim in all_pairs:
        if eid_a not in by_event or eid_b not in by_event:
            continue
        if allowed_categories and new_event_ids is None:
            ev_a, ev_b = event_by_id.get(eid_a), event_by_id.get(eid_b)
            a_ok = not (ev_a and ev_a.category and ev_a.category not in allowed_categories)
            b_ok = not (ev_b and ev_b.category and ev_b.category not in allowed_categories)
            if not a_ok and not b_ok:
                continue
        pair = (eid_a, eid_b)
        if pair not in candidate_pairs or sim > candidate_pairs[pair]:
            candidate_pairs[pair] = sim

    # First pass: resolve events/contracts for all valid pairs, build cache key list
    prepared: list[tuple[Event, Event, list[EventContract], list[EventContract], list[str], list[str], float]] = []
    cache_keys: list[tuple[str, list[str], str, list[str]]] = []

    for (eid_a, eid_b), similarity in candidate_pairs.items():
        try:
            ev_a, ev_b = event_by_id.get(eid_a), event_by_id.get(eid_b)
            if ev_a is None or ev_b is None:
                continue

            contracts_a = by_event[eid_a]
            contracts_b = by_event[eid_b]
            primary_a = primary_contracts(contracts_a)
            primary_b = primary_contracts(contracts_b)
            if not primary_a or not primary_b:
                continue

            labels_a = [ec.outcome_label for ec in primary_a]
            labels_b = [ec.outcome_label for ec in primary_b]
            prepared.append((ev_a, ev_b, contracts_a, contracts_b, labels_a, labels_b, similarity))
            cache_keys.append((ev_a.title, labels_a, ev_b.title, labels_b))
        except Exception as e:
            logger.error("Failed comparing events %d and %d: %s", eid_a, eid_b, e)

    # Bulk cache lookup — single pipelined round trip
    cache_hits = cache.get_judgment_bulk(model, cache_keys) if cache is not None else {}

    # Second pass: partition into cached_judged and pending
    cached_judged: list[JudgedRelationship] = []
    pending: list[tuple[Event, Event, list[EventContract], list[EventContract], float]] = []

    for i, (ev_a, ev_b, contracts_a, contracts_b, labels_a, labels_b, similarity) in enumerate(prepared):
        if i in cache_hits:
            cached_items, a_is_first = cache_hits[i]
            primary_a = primary_contracts(contracts_a)
            primary_b = primary_contracts(contracts_b)
            cached_judged.extend(_parse_cached_judgment(cached_items, primary_a, primary_b, a_is_first))
        else:
            pending.append((ev_a, ev_b, contracts_a, contracts_b, similarity))

    if debug:
        logger.info("[DEBUG] semantic: %d candidate pairs, %d cache hits, %d pending judgment",
                    len(prepared), len(cache_hits), len(pending))
        for ev_a, ev_b, _, _, similarity in pending[:50]:
            logger.info("[DEBUG] semantic: pair: %r vs %r sim=%.3f",
                        ev_a.title[:60], ev_b.title[:60], similarity)
        if len(pending) > 50:
            logger.info("[DEBUG] semantic: ... and %d more pending pairs", len(pending) - 50)

    return pending, cached_judged


def build_judgment_requests(
    pending: list[tuple[Event, Event, list[EventContract], list[EventContract], float]],
    model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL,
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
            f"  Description: {(ev_a.description or '')[:200]}\n"
            f"  Contracts: {contracts_a_lines}\n\n"
            f"Event B: {ev_b.title}\n"
            f"  Description: {(ev_b.description or '')[:200]}\n"
            f"  Contracts: {contracts_b_lines}\n\n"
            f"Embedding similarity: {similarity:.3f}"
        )
        custom_id = f"j_{idx}"
        api_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": model,
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
    model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL,
    debug: bool = False,
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

        if debug:
            title_a = ctx.get("event_a_title", "?")[:60]
            title_b = ctx.get("event_b_title", "?")[:60]
            if judged:
                types = list({j.relationship_type for j in judged})
                conf = max(j.confidence for j in judged)
                logger.info("[DEBUG] semantic: judgment: %r vs %r -> %s conf=%.2f",
                            title_a, title_b, "/".join(str(t) for t in types), conf)
            else:
                logger.info("[DEBUG] semantic: judgment: %r vs %r -> NONE", title_a, title_b)

        if cache is not None and cache_items is not None:
            a_is_first = (ctx["event_a_title"], "|".join(ctx["labels_a"])) <= (ctx["event_b_title"], "|".join(ctx["labels_b"]))
            cache.put_judgment(
                model,
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
) -> tuple[list[JudgedRelationship], list[dict] | None]:
    raw = strip_code_fences(raw)
    logger.debug("judge_relationship response: %s", raw)

    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            logger.debug("response is not a list: %r", items)
            return [], None

        results: list[JudgedRelationship] = []
        cache_items: list[dict] = []

        best_per_pair: dict[tuple[int, int], dict] = {}
        for item in items:
            idx_a = item.get("a")
            idx_b = item.get("b")
            rel_type_str = item.get("type", "NONE")
            confidence = float(item.get("confidence", DEFAULT_MIN_CONFIDENCE))
            if idx_a not in idx_to_sid_a or idx_b not in idx_to_sid_b:
                logger.debug("skipping item (index out of range): %s", item)
                continue
            if rel_type_str not in RelationshipType.__members__:
                logger.debug("skipping item (invalid type): %s", item)
                continue
            if confidence < DEFAULT_MIN_CONFIDENCE:
                logger.debug("skipping item (low confidence): %s", item)
                continue
            pair_key = (idx_a, idx_b)
            if pair_key not in best_per_pair or confidence > best_per_pair[pair_key].get("confidence", 0):
                best_per_pair[pair_key] = item

        for item in best_per_pair.values():
            idx_a = item.get("a")
            idx_b = item.get("b")
            rel_type_str = item.get("type", "NONE")
            confidence = float(item.get("confidence", DEFAULT_MIN_CONFIDENCE))
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
        return [], None


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
