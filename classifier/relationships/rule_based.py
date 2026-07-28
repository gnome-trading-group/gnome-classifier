import re

from classifier.constants import DEFAULT_HEDGEABLE_WITH_CONFIDENCE
from classifier.types import RelationshipMatch, RelationshipType, SecurityId
from gnomepy.registry.types import Event, EventContract


def find_hedgeable_pairs(
    event_contracts: list[EventContract],
    events: list[Event],
    hedge_keywords: list[tuple[int, str]],
    confidence: float = DEFAULT_HEDGEABLE_WITH_CONFIDENCE,
) -> list[RelationshipMatch]:
    if not hedge_keywords:
        return []

    event_by_id = {e.event_id: e for e in events}

    keyword_to_security_ids: dict[str, list[SecurityId]] = {}
    for security_id, keyword in hedge_keywords:
        keyword_to_security_ids.setdefault(keyword.lower(), []).append(security_id)

    matches: list[RelationshipMatch] = []
    for ec in event_contracts:
        event = event_by_id.get(ec.event_id)
        if event is None:
            continue
        text = (event.title + " " + ec.outcome_label).lower()
        for kw, security_ids in keyword_to_security_ids.items():
            if re.search(rf'\b{re.escape(kw)}\b', text):
                for tradeable_sid in security_ids:
                    matches.append(RelationshipMatch(
                        ec.security_id, tradeable_sid,
                        RelationshipType.HEDGEABLE_WITH,
                        confidence,
                        "rule",
                    ))

    return matches
