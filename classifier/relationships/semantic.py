import json
import logging
from collections import defaultdict

import anthropic
import voyageai

from classifier.cache import ClassifierCache
from classifier.types import (
    Embedding,
    EventId,
    JudgedRelationship,
    RelationshipMatch,
    RelationshipType,
    SecurityId,
)
from classifier.relationships.structural import build_complement_map
from classifier.utils import cosine_similarity
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


def embed_events_voyage(
    voyage_client: voyageai.Client,
    events: list[Event],
) -> dict[EventId, Embedding]:
    embeddings: dict[EventId, Embedding] = {}
    texts = []
    ids_to_embed = []
    for event in events:
        if event.embedding:
            embeddings[event.event_id] = event.embedding
            continue
        text = event.title
        if event.description:
            text += ". " + event.description[:200]
        texts.append(text)
        ids_to_embed.append(event.event_id)

    for i in range(0, len(texts), 128):
        batch_texts = texts[i:i + 128]
        batch_ids = ids_to_embed[i:i + 128]
        result = voyage_client.embed(batch_texts, model="voyage-3", input_type="document")  # type: ignore[attr-defined]
        for eid, emb in zip(batch_ids, result.embeddings):
            embeddings[eid] = emb

    return embeddings


def _make_union_find() -> tuple[dict[EventId, EventId], dict[EventId, set[EventId]]]:
    parent: dict[EventId, EventId] = {}
    members: dict[EventId, set[EventId]] = {}
    return parent, members


def _uf_find(parent: dict[EventId, EventId], x: EventId) -> EventId:
    while parent.get(x, x) != x:
        parent[x] = parent.get(parent[x], parent[x])
        x = parent[x]
    return x


def _uf_union(parent: dict[EventId, EventId], members: dict[EventId, set[EventId]], a: EventId, b: EventId) -> None:
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra == rb:
        return
    parent[ra] = rb
    combined = members.get(ra, {ra}) | members.get(rb, {rb})
    members[rb] = combined
    members[ra] = combined


def _try_clone_from_equivalent(
    eid_a: EventId,
    eid_b: EventId,
    equiv_parent: dict[EventId, EventId],
    equiv_members: dict[EventId, set[EventId]],
    judged_results: dict[tuple[EventId, EventId], list[JudgedRelationship]],
    by_event: dict[EventId, list[EventContract]],
) -> list[JudgedRelationship] | None:
    for orig, other in [(eid_a, eid_b), (eid_b, eid_a)]:
        peers = equiv_members.get(_uf_find(equiv_parent, orig), set()) - {orig}
        for peer in peers:
            key = (min(peer, other), max(peer, other))
            if key not in judged_results:
                continue
            label_to_peer = {ec.outcome_label: ec.security_id for ec in by_event[peer]}
            label_to_orig = {ec.outcome_label: ec.security_id for ec in by_event[orig]}
            sid_map = {label_to_peer[l]: label_to_orig[l] for l in label_to_peer if l in label_to_orig}
            cloned: list[JudgedRelationship] = []
            for r in judged_results[key]:
                new_s1 = sid_map.get(r.security_id_a, r.security_id_a)
                new_s2 = sid_map.get(r.security_id_b, r.security_id_b)
                if r.relationship_type == RelationshipType.IMPLIES:
                    cloned.append(JudgedRelationship(new_s1, new_s2, r.relationship_type, r.confidence))
                else:
                    cloned.append(JudgedRelationship(new_s1, new_s2, r.relationship_type, r.confidence))
                    cloned.append(JudgedRelationship(new_s2, new_s1, r.relationship_type, r.confidence))
            return cloned
    return None


def find_semantic_matches(
    anthropic_client: anthropic.Anthropic,
    events: list[Event],
    event_contracts: list[EventContract],
    embeddings: dict[EventId, Embedding],
    new_event_ids: set[EventId] | None = None,
    skip_judgment: bool = False,
    cache: ClassifierCache | None = None,
) -> list[RelationshipMatch]:
    by_event: dict[EventId, list[EventContract]] = defaultdict(list)
    for ec in event_contracts:
        by_event[ec.event_id].append(ec)

    complement_of = build_complement_map(event_contracts)

    event_by_id = {e.event_id: e for e in events}
    event_ids = [e.event_id for e in events if e.event_id in embeddings and e.event_id in by_event]
    matches: list[RelationshipMatch] = []
    would_judge_count = 0

    me_parent, me_members = _make_union_find()
    equiv_parent, equiv_members = _make_union_find()
    judged_results: dict[tuple[EventId, EventId], list[JudgedRelationship]] = {}

    for i in range(len(event_ids)):
        for j in range(i + 1, len(event_ids)):
            eid_a, eid_b = event_ids[i], event_ids[j]

            if new_event_ids is not None and eid_a not in new_event_ids and eid_b not in new_event_ids:
                continue

            try:
                ev_a, ev_b = event_by_id[eid_a], event_by_id[eid_b]
                if ev_a.category and ev_b.category and ev_a.category != ev_b.category:
                    continue

                similarity = cosine_similarity(embeddings[eid_a], embeddings[eid_b])

                if similarity < EMBEDDING_SIMILARITY_THRESHOLD:
                    continue

                contracts_a = by_event[eid_a]
                contracts_b = by_event[eid_b]

                if (eid_a in me_parent or eid_b in me_parent) and _uf_find(me_parent, eid_a) == _uf_find(me_parent, eid_b):
                    yes_a = [ec for ec in contracts_a if ec.outcome_label.lower() != "no"]
                    yes_b = [ec for ec in contracts_b if ec.outcome_label.lower() != "no"]
                    for ec_a in yes_a:
                        for ec_b in yes_b:
                            matches.append(RelationshipMatch(ec_a.security_id, ec_b.security_id, RelationshipType.MUTUALLY_EXCLUSIVE, 0.95, "embedding"))
                            matches.append(RelationshipMatch(ec_b.security_id, ec_a.security_id, RelationshipType.MUTUALLY_EXCLUSIVE, 0.95, "embedding"))
                    logger.debug("Propagated ME: '%s' vs '%s'", ev_a.title, ev_b.title)
                    continue

                cloned = _try_clone_from_equivalent(
                    eid_a, eid_b, equiv_parent, equiv_members, judged_results, by_event
                )
                if cloned is not None:
                    matches.extend(
                        RelationshipMatch(r.security_id_a, r.security_id_b, r.relationship_type, r.confidence, "embedding")
                        for r in cloned
                    )
                    logger.debug("Cloned from equivalent: '%s' vs '%s'", ev_a.title, ev_b.title)
                    continue

                if skip_judgment:
                    would_judge_count += 1
                    logger.info(
                        "Would judge: '%s' vs '%s' (similarity=%.3f, %d×%d contracts)",
                        ev_a.title, ev_b.title, similarity, len(contracts_a), len(contracts_b),
                    )
                    continue

                contract_matches = _judge_relationship(
                    anthropic_client, ev_a, ev_b, contracts_a, contracts_b, similarity, cache=cache
                )
                judged_results[(min(eid_a, eid_b), max(eid_a, eid_b))] = contract_matches

                if any(r.relationship_type == RelationshipType.MUTUALLY_EXCLUSIVE for r in contract_matches):
                    _uf_union(me_parent, me_members, eid_a, eid_b)
                if any(r.relationship_type == RelationshipType.EQUIVALENT for r in contract_matches):
                    _uf_union(equiv_parent, equiv_members, eid_a, eid_b)

                matches.extend(
                    RelationshipMatch(r.security_id_a, r.security_id_b, r.relationship_type, r.confidence, "embedding")
                    for r in contract_matches
                )
            except Exception as e:
                logger.error("Failed comparing events %d and %d: %s", eid_a, eid_b, e)

    if skip_judgment:
        logger.info("skip_judgment: would have called Claude %d times", would_judge_count)

    derived_implies: list[RelationshipMatch] = []
    for m in matches:
        if m.relationship_type != RelationshipType.MUTUALLY_EXCLUSIVE:
            continue
        comp_a = complement_of.get(m.security_id_a)
        comp_b = complement_of.get(m.security_id_b)
        if comp_b is not None:
            derived_implies.append(RelationshipMatch(m.security_id_a, comp_b, RelationshipType.IMPLIES, m.confidence, "embedding"))
        if comp_a is not None:
            derived_implies.append(RelationshipMatch(m.security_id_b, comp_a, RelationshipType.IMPLIES, m.confidence, "embedding"))
    matches.extend(derived_implies)

    return matches


def _judge_relationship(
    client: anthropic.Anthropic,
    event_a: Event,
    event_b: Event,
    contracts_a: list[EventContract],
    contracts_b: list[EventContract],
    similarity: float,
    cache: ClassifierCache | None = None,
) -> list[JudgedRelationship]:
    """Returns judged contract pairs. For IMPLIES, security_id_a is antecedent and security_id_b is consequent.
    Symmetric types emit both directions."""
    yes_a = [ec for ec in contracts_a if ec.outcome_label.lower() != "no"]
    yes_b = [ec for ec in contracts_b if ec.outcome_label.lower() != "no"]
    if not yes_a or not yes_b:
        return []

    complement_of = build_complement_map(contracts_a + contracts_b)

    labels_a = [ec.outcome_label for ec in yes_a]
    labels_b = [ec.outcome_label for ec in yes_b]

    if cache is not None:
        cached = cache.get_judgment(MODEL, event_a.title, labels_a, event_b.title, labels_b)
        if cached is not None:
            cached_items, a_is_first = cached
            return _parse_cached_judgment(cached_items, yes_a, yes_b, complement_of, a_is_first)

    contracts_a_lines = "  ".join(f"[{i+1}] {ec.outcome_label}" for i, ec in enumerate(yes_a))
    contracts_b_lines = "  ".join(f"[{i+1}] {ec.outcome_label}" for i, ec in enumerate(yes_b))

    user_content = (
        f"Event A: {event_a.title}\n"
        f"  Contracts: {contracts_a_lines}\n\n"
        f"Event B: {event_b.title}\n"
        f"  Contracts: {contracts_b_lines}\n\n"
        f"Embedding similarity: {similarity:.3f}"
    )

    logger.debug("judge_relationship user message:\n%s", user_content)

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=[{"type": "text", "text": _JUDGE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    logger.debug("judge_relationship response: %s", raw)

    idx_to_sid_a = {i + 1: ec.security_id for i, ec in enumerate(yes_a)}
    idx_to_sid_b = {i + 1: ec.security_id for i, ec in enumerate(yes_b)}
    idx_to_label_a = {i + 1: ec.outcome_label for i, ec in enumerate(yes_a)}
    idx_to_label_b = {i + 1: ec.outcome_label for i, ec in enumerate(yes_b)}
    logger.debug("idx_to_sid_a: %s  idx_to_sid_b: %s", idx_to_sid_a, idx_to_sid_b)

    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            logger.debug("response is not a list: %r", items)
            return []
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

        if cache is not None and cache_items:
            a_is_first = (event_a.title, "|".join(labels_a)) <= (event_b.title, "|".join(labels_b))
            cache.put_judgment(MODEL, event_a.title, labels_a, event_b.title, labels_b, cache_items, a_is_first)

        derived: list[JudgedRelationship] = []
        for r in results:
            comp_a = complement_of.get(r.security_id_a)
            comp_b = complement_of.get(r.security_id_b)
            if comp_a is None or comp_b is None:
                continue
            if r.relationship_type == RelationshipType.IMPLIES:
                derived.append(JudgedRelationship(comp_b, comp_a, r.relationship_type, r.confidence))
            elif r.relationship_type in (RelationshipType.EQUIVALENT, RelationshipType.CORRELATED):
                derived.append(JudgedRelationship(comp_a, comp_b, r.relationship_type, r.confidence))
                derived.append(JudgedRelationship(comp_b, comp_a, r.relationship_type, r.confidence))
        results.extend(derived)

        logger.debug("judge_relationship returning %d results (%d derived)", len(results), len(derived))
        return results
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.debug("judge_relationship parse error: %s — raw: %r", e, raw)
        return []


def _parse_cached_judgment(
    cached_items: list[dict],
    yes_a: list[EventContract],
    yes_b: list[EventContract],
    complement_of: dict[SecurityId, SecurityId],
    a_is_first: bool,
) -> list[JudgedRelationship]:
    """Reconstruct sid-based results from label-based cache entries."""
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

    derived: list[JudgedRelationship] = []
    for r in results:
        comp_a = complement_of.get(r.security_id_a)
        comp_b = complement_of.get(r.security_id_b)
        if comp_a is None or comp_b is None:
            continue
        if r.relationship_type == RelationshipType.IMPLIES:
            derived.append(JudgedRelationship(comp_b, comp_a, r.relationship_type, r.confidence))
        elif r.relationship_type in (RelationshipType.EQUIVALENT, RelationshipType.CORRELATED):
            derived.append(JudgedRelationship(comp_a, comp_b, r.relationship_type, r.confidence))
            derived.append(JudgedRelationship(comp_b, comp_a, r.relationship_type, r.confidence))
    results.extend(derived)

    return results
