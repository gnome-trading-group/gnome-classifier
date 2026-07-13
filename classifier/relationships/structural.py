from collections import defaultdict

from gnomepy.registry.types import EventContract

from classifier.types import EventId, JudgedRelationship, RelationshipMatch, RelationshipType, SecurityId

STRUCTURAL_CONFIDENCE = 1.0


def primary_contracts(contracts: list[EventContract]) -> list[EventContract]:
    """For a single event's contracts, return the subset to send to the LLM for judgment.
    Binary events (exactly 2 contracts): return one representative, chosen by lowest event_contract_id.
    All other events: return all contracts unchanged."""
    if len(contracts) == 2:
        return [min(contracts, key=lambda ec: ec.event_contract_id)]
    return contracts


def build_complement_map(
    event_contracts: list[EventContract],
) -> dict[SecurityId, SecurityId]:
    by_event: dict[EventId, list[EventContract]] = defaultdict(list)
    for ec in event_contracts:
        by_event[ec.event_id].append(ec)
    complement_of: dict[SecurityId, SecurityId] = {}
    for ecs in by_event.values():
        if len(ecs) == 2:
            complement_of[ecs[0].security_id] = ecs[1].security_id
            complement_of[ecs[1].security_id] = ecs[0].security_id
    return complement_of


def find_complement_pairs(
    event_contracts: list[EventContract],
) -> list[RelationshipMatch]:
    complement_of = build_complement_map(event_contracts)
    return [
        RelationshipMatch(a, b, RelationshipType.COMPLEMENT, STRUCTURAL_CONFIDENCE, "structural")
        for a, b in complement_of.items()
    ]


def derive_complement_relationships(
    matches: list[JudgedRelationship],
    complement_of: dict[SecurityId, SecurityId],
) -> list[JudgedRelationship]:
    """Derive additional relationships via complement mapping.

    - IMPLIES(A→B)              → IMPLIES(comp(B)→comp(A))
    - EQUIVALENT/CORRELATED(A,B) → EQUIVALENT/CORRELATED(comp(A),comp(B))
    - MUTUALLY_EXCLUSIVE(A,B)   → IMPLIES(A→comp(B))  and  IMPLIES(B→comp(A))
    """
    derived: list[JudgedRelationship] = []
    for r in matches:
        comp_a = complement_of.get(r.security_id_a)
        comp_b = complement_of.get(r.security_id_b)
        if r.relationship_type == RelationshipType.IMPLIES:
            if comp_a is not None and comp_b is not None:
                derived.append(JudgedRelationship(comp_b, comp_a, r.relationship_type, r.confidence))
        elif r.relationship_type in (RelationshipType.EQUIVALENT, RelationshipType.CORRELATED):
            if comp_a is not None and comp_b is not None:
                derived.append(JudgedRelationship(comp_a, comp_b, r.relationship_type, r.confidence))
                derived.append(JudgedRelationship(comp_b, comp_a, r.relationship_type, r.confidence))
        elif r.relationship_type == RelationshipType.MUTUALLY_EXCLUSIVE:
            if comp_b is not None:
                derived.append(JudgedRelationship(r.security_id_a, comp_b, RelationshipType.IMPLIES, r.confidence))
            if comp_a is not None:
                derived.append(JudgedRelationship(r.security_id_b, comp_a, RelationshipType.IMPLIES, r.confidence))
    return derived


def find_mutually_exclusive_pairs(
    event_contracts: list[EventContract],
) -> list[RelationshipMatch]:
    by_event: dict[EventId, list[SecurityId]] = defaultdict(list)
    for ec in event_contracts:
        by_event[ec.event_id].append(ec.security_id)

    pairs: list[RelationshipMatch] = []
    for ids in by_event.values():
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j:
                    pairs.append(RelationshipMatch(
                        ids[i], ids[j],
                        RelationshipType.MUTUALLY_EXCLUSIVE,
                        STRUCTURAL_CONFIDENCE,
                        "structural",
                    ))
    return pairs
