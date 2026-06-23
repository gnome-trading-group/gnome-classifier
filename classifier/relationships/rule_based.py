import re

from classifier.types import CurrencyId, RelationshipMatch, RelationshipType, SecurityId
from gnomepy.registry.types import Currency, Event, EventContract, Security, SecurityType

HEDGEABLE_WITH_CONFIDENCE = 0.90


def find_hedgeable_pairs(
    event_contracts: list[EventContract],
    events: list[Event],
    existing_securities: list[Security],
    currencies: list[Currency],
) -> list[RelationshipMatch]:
    event_by_id = {e.event_id: e for e in events}

    currency_by_id = {c.currency_id: c for c in currencies}
    keyword_to_currency_id: dict[str, CurrencyId] = {}
    for c in currencies:
        keyword_to_currency_id[c.symbol.lower()] = c.currency_id
        if c.name:
            keyword_to_currency_id[c.name.lower()] = c.currency_id

    securities_by_keyword: dict[str, list[Security]] = {}
    for sec in existing_securities:
        if sec.type == SecurityType.EVENT_CONTRACT or sec.base_currency_id is None:
            continue
        currency = currency_by_id.get(sec.base_currency_id)
        if currency is None:
            continue
        keywords = {currency.symbol.lower()}
        if currency.name:
            keywords.add(currency.name.lower())
        for kw in keywords:
            securities_by_keyword.setdefault(kw, []).append(sec)

    all_keywords = set(securities_by_keyword)
    matches: list[RelationshipMatch] = []
    for ec in event_contracts:
        event = event_by_id.get(ec.event_id)
        if event is None:
            continue
        text = (event.title + " " + ec.outcome_label).lower()
        for kw in all_keywords:
            if re.search(rf'\b{re.escape(kw)}\b', text):
                for tradeable_sec in securities_by_keyword[kw]:
                    matches.append(RelationshipMatch(
                        ec.security_id, tradeable_sec.security_id,
                        RelationshipType.HEDGEABLE_WITH,
                        HEDGEABLE_WITH_CONFIDENCE,
                        "rule",
                    ))

    return matches
