import dataclasses
import logging
from datetime import timedelta

import anthropic
import voyageai

from classifier.adapters.types import AdapterContract
from classifier.cache import ClassifierCache
from classifier.types import CanonicalizeInput, Embedding, EventId, ListingId, SecurityId
from classifier.constants import (
    DEDUP_COSINE_THRESHOLD,
    DEDUP_EXPIRY_TOLERANCE_HOURS,
    EMBED_BATCH_SIZE,
)
from classifier.stages.canonicalize import canonicalize_events
from classifier.utils import cosine_similarity, expiry_close, from_dict, generate_security_symbol
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import Currency, EventContract, Listing, SecurityType

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _EmbedCandidate:
    raw_titles: list[str]
    canonical_title: str
    category: str
    tags: list[str]
    expiry: str | None
    description: str | None
    text: str
    embedding: Embedding | None = None


def create_entities(
    registry: RegistryClient,
    voyage: voyageai.Client,
    anthropic_client: anthropic.Anthropic,
    contracts: list[AdapterContract],
    exchange_by_name: dict,
    cache: ClassifierCache | None = None,
) -> dict:
    if not contracts:
        return {
            "events_created": 0, "securities_created": 0, "listings_created": 0,
            "event_contracts_created": 0, "listing_specs_created": 0,
            "new_security_ids": [], "new_security_symbols": [], "embeddings": {},
        }

    existing_events = registry.get_event()
    existing_securities = registry.get_security()
    existing_security_id_set = {s.security_id for s in existing_securities}
    existing_listings = registry.get_listing()
    existing_event_contracts = registry.get_event_contracts()
    currencies = registry.get_currency()
    existing_listing_specs = registry.get_listing_spec()

    currency_by_symbol = {c.symbol: c for c in currencies}
    exchange_name_by_id = {e.exchange_id: name for name, e in exchange_by_name.items()}
    listing_by_key = {f"{l.exchange_id}:{l.exchange_security_id}": l for l in existing_listings}
    event_contract_by_key = {f"{ec.event_id}:{ec.security_id}": ec for ec in existing_event_contracts}
    spec_by_listing_id: dict[ListingId, object] = {s.listing_id: s for s in existing_listing_specs}

    contracts_by_raw_title: dict[str, list[AdapterContract]] = {}
    for c in contracts:
        contracts_by_raw_title.setdefault(c.event_title, []).append(c)

    existing_exchange_events = registry.get_exchange_events()
    exchange_event_by_key: dict[str, object] = {
        f"{ee.exchange_id}:{ee.native_event_id}": ee for ee in existing_exchange_events
    }

    event_id_by_raw: dict[str, EventId] = {}
    events_to_canonicalize: list[CanonicalizeInput] = []
    for raw_title, group in contracts_by_raw_title.items():
        c = group[0]
        key = f"{c.exchange_id}:{c.exchange_event_native_id}"
        ee = exchange_event_by_key.get(key)
        if ee:
            event_id_by_raw[raw_title] = ee.event_id
            continue
        events_to_canonicalize.append(CanonicalizeInput(raw_title, c.event_description, c.event_category, c.exchange_id, c.exchange_event_native_id))

    canonical_by_raw = canonicalize_events(anthropic_client, events_to_canonicalize, cache=cache)

    event_by_id = {ev.event_id: ev for ev in existing_events}
    event_info_by_raw: dict[str, dict] = dict(canonical_by_raw)
    for raw_title, event_id in event_id_by_raw.items():
        if raw_title not in event_info_by_raw:
            ev = event_by_id.get(event_id)
            if ev:
                event_info_by_raw[raw_title] = {
                    "title": ev.title,
                    "category": ev.category or "OTHER",
                    "tags": ev.tags or [],
                }

    existing_embeddings: dict[EventId, Embedding] = {
        ev.event_id: ev.embedding for ev in existing_events if ev.embedding
    }

    events_created = 0
    securities_created = 0
    listings_created = 0
    event_contracts_created = 0
    listing_specs_created = 0

    created_event_records: list[tuple[str, str | None, int]] = [
        (ev.title, ev.expiry, ev.event_id) for ev in existing_events
    ]

    embed_candidates: list[_EmbedCandidate] = []
    for raw_title, canonical_info in event_info_by_raw.items():
        canonical_title = canonical_info["title"]
        category = canonical_info["category"]
        tags = canonical_info["tags"]
        group = contracts_by_raw_title[raw_title]
        expiry = group[0].event_expiry
        description = group[0].event_description

        existing_match_id = next(
            (eid for t, exp, eid in created_event_records
             if t == canonical_title and expiry_close(expiry, exp, timedelta(hours=DEDUP_EXPIRY_TOLERANCE_HOURS))),
            None,
        )
        if existing_match_id is not None:
            event_id_by_raw[raw_title] = existing_match_id
            continue

        candidate_match_idx = next(
            (idx for idx, cand in enumerate(embed_candidates)
             if cand.canonical_title == canonical_title and expiry_close(expiry, cand.expiry, timedelta(hours=DEDUP_EXPIRY_TOLERANCE_HOURS))),
            None,
        )
        if candidate_match_idx is not None:
            embed_candidates[candidate_match_idx].raw_titles.append(raw_title)
            continue

        text = canonical_title
        if description:
            text += ". " + description[:200]
        embed_candidates.append(_EmbedCandidate(
            raw_titles=[raw_title],
            canonical_title=canonical_title,
            category=category,
            tags=tags,
            expiry=expiry,
            description=description,
            text=text,
        ))

    texts = [cand.text for cand in embed_candidates]
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        result = voyage.embed(batch, model="voyage-3", input_type="document")
        for j, emb in enumerate(result.embeddings):
            embed_candidates[i + j].embedding = emb

    pending_events: list[dict] = []
    pending_raw_to_event_idx: dict[str, int] = {}

    for cand in embed_candidates:
        matched_event_id = None

        best_sim = 0.0
        best_eid = None
        for eid, emb in existing_embeddings.items():
            sim = cosine_similarity(cand.embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_eid = eid
        if best_sim >= DEDUP_COSINE_THRESHOLD and best_eid is not None:
            matched_event_id = best_eid

        if matched_event_id is not None:
            for raw_title in cand.raw_titles:
                event_id_by_raw[raw_title] = matched_event_id
            created_event_records.append((cand.canonical_title, cand.expiry, matched_event_id))
            continue

        event_idx = len(pending_events)
        for raw_title in cand.raw_titles:
            pending_raw_to_event_idx[raw_title] = event_idx
        pending_events.append(dict(
            title=cand.canonical_title,
            description=cand.description,
            category=cand.category,
            tags=cand.tags,
            embedding=cand.embedding,
            expiry=cand.expiry,
        ))

    created_event_ids: list[int | None] = [None] * len(pending_events)
    if pending_events:
        created_list = registry.bulk_create_events(pending_events)
        events_created += len(created_list)
        for idx, created in enumerate(created_list):
            new_eid = created["event_id"]
            created_event_ids[idx] = new_eid
            ev = pending_events[idx]
            created_event_records.append((ev["title"], ev.get("expiry"), new_eid))
            if ev.get("embedding"):
                existing_embeddings[new_eid] = ev["embedding"]

    for raw_title, event_idx in pending_raw_to_event_idx.items():
        eid = created_event_ids[event_idx] if event_idx < len(created_event_ids) else None
        if eid is not None:
            event_id_by_raw[raw_title] = eid

    pending_exchange_events: list[dict] = []
    for raw_title, group in contracts_by_raw_title.items():
        c = group[0]
        key = f"{c.exchange_id}:{c.exchange_event_native_id}"
        if key in exchange_event_by_key:
            continue
        eid = event_id_by_raw.get(raw_title)
        if eid is None:
            continue
        pending_exchange_events.append(dict(
            exchange_id=c.exchange_id,
            event_id=eid,
            native_event_id=c.exchange_event_native_id,
            raw_title=raw_title,
        ))
        exchange_event_by_key[key] = True

    if pending_exchange_events:
        registry.bulk_create_exchange_events(pending_exchange_events)

    all_currency_symbols = (
        {c.base_currency for c in contracts}
        | {c.quote_currency for c in contracts}
        | {c.settle_currency for c in contracts}
    )
    for sym in all_currency_symbols:
        if sym not in currency_by_symbol:
            try:
                created_curr = registry._post("/currencies", {"symbol": sym})
                currency_by_symbol[sym] = from_dict(Currency, created_curr)
            except Exception as e:
                logger.error("Failed to create currency '%s': %s", sym, e)

    security_id_by_symbol: dict[str, SecurityId] = {s.symbol: s.security_id for s in existing_securities}
    security_id_by_outcome: dict[tuple[str, str], SecurityId] = {}

    seen_symbols: dict[str, AdapterContract] = {}
    for c in contracts:
        canonical_info = event_info_by_raw.get(c.event_title)
        if canonical_info is None:
            continue
        symbol = generate_security_symbol(canonical_info["title"], c.outcome_label)
        if symbol not in seen_symbols:
            seen_symbols[symbol] = c

    pending_securities: list[dict] = []
    pending_security_symbols: list[str] = []
    pending_security_outcomes: list[tuple[str, str]] = []

    for symbol, c in seen_symbols.items():
        canonical_info = event_info_by_raw[c.event_title]
        if symbol in security_id_by_symbol:
            security_id_by_outcome[(c.event_title, c.outcome_label)] = security_id_by_symbol[symbol]
            continue

        base_ccy = currency_by_symbol.get(c.base_currency)
        quote_ccy = currency_by_symbol.get(c.quote_currency)
        settle_ccy = currency_by_symbol.get(c.settle_currency)

        pending_securities.append(dict(
            symbol=symbol,
            type=SecurityType.EVENT_CONTRACT,
            contract_type=c.contract_type,
            asset_class=c.asset_class,
            base_currency_id=base_ccy.currency_id if base_ccy else None,
            quote_currency_id=quote_ccy.currency_id if quote_ccy else None,
            settle_currency_id=settle_ccy.currency_id if settle_ccy else None,
            inverse=c.inverse,
            quanto=c.is_quanto,
            expiry=c.event_expiry,
            active=True,
        ))
        pending_security_symbols.append(symbol)
        pending_security_outcomes.append((c.event_title, c.outcome_label))

    if pending_securities:
        created_list = registry.bulk_create_securities(pending_securities)
        securities_created += len(created_list)
        for idx, created in enumerate(created_list):
            new_sid = created["security_id"]
            sym = pending_security_symbols[idx]
            security_id_by_symbol[sym] = new_sid
            security_id_by_outcome[pending_security_outcomes[idx]] = new_sid

    pending_listings: list[dict] = []
    pending_listing_keys: list[str] = []

    for c in contracts:
        key = f"{c.exchange_id}:{c.exchange_security_id}"
        if key in listing_by_key:
            continue
        sid = security_id_by_outcome.get((c.event_title, c.outcome_label))
        if sid is None:
            continue

        listing_by_key[key] = None  # type: ignore[assignment]
        pending_listings.append(dict(
            exchange_id=c.exchange_id,
            security_id=sid,
            exchange_security_id=c.exchange_security_id,
            exchange_security_symbol=c.exchange_security_symbol,
        ))
        pending_listing_keys.append(key)

    if pending_listings:
        created_list = registry.bulk_create_listings(pending_listings)
        listings_created += len(created_list)
        for idx, created in enumerate(created_list):
            listing_by_key[pending_listing_keys[idx]] = from_dict(Listing, created)

    pending_ecs: list[dict] = []
    pending_ec_keys: list[str] = []

    for c in contracts:
        event_id = event_id_by_raw.get(c.event_title)
        sid = security_id_by_outcome.get((c.event_title, c.outcome_label))
        if event_id is None or sid is None:
            continue
        ec_key = f"{event_id}:{sid}"
        if ec_key in event_contract_by_key:
            continue

        event_contract_by_key[ec_key] = None  # type: ignore[assignment]
        pending_ecs.append(dict(
            event_id=event_id,
            security_id=sid,
            outcome_label=c.outcome_label,
        ))
        pending_ec_keys.append(ec_key)

    if pending_ecs:
        created_list = registry.bulk_create_event_contracts(pending_ecs)
        event_contracts_created += len(created_list)
        for idx, created in enumerate(created_list):
            event_contract_by_key[pending_ec_keys[idx]] = from_dict(EventContract, created)

    pending_specs: list[dict] = []

    for c in contracts:
        key = f"{c.exchange_id}:{c.exchange_security_id}"
        listing = listing_by_key.get(key)
        if listing is None:
            continue
        listing_id = listing.listing_id if hasattr(listing, "listing_id") else listing["listing_id"]
        if listing_id in spec_by_listing_id:
            continue

        spec_by_listing_id[listing_id] = True  # type: ignore[assignment]
        pending_specs.append(dict(
            listing_id=listing_id,
            tick_size=c.tick_size,
            lot_size=c.lot_size,
            min_notional=c.min_notional,
            contract_multiplier=c.contract_multiplier,
        ))

    if pending_specs:
        try:
            created_list = registry.bulk_create_listing_specs(pending_specs)
            listing_specs_created += len(created_list)
        except Exception as e:
            logger.error("Bulk listing_spec creation failed: %s", e)

    all_securities_after = registry.get_security()
    new_security_ids = [s.security_id for s in all_securities_after if s.security_id not in existing_security_id_set]
    new_security_symbols = [s.symbol for s in all_securities_after if s.security_id not in existing_security_id_set]

    return {
        "events_created": events_created,
        "securities_created": securities_created,
        "listings_created": listings_created,
        "event_contracts_created": event_contracts_created,
        "listing_specs_created": listing_specs_created,
        "new_security_ids": new_security_ids,
        "new_security_symbols": new_security_symbols,
    }


