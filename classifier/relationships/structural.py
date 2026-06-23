from collections import defaultdict

from gnomepy.registry.types import EventContract

from classifier.types import EventId, RelationshipMatch, RelationshipType, SecurityId

STRUCTURAL_CONFIDENCE = 1.0


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
