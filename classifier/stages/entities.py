import logging
from collections import defaultdict
from datetime import timedelta

from classifier.adapters.types import AdapterContract
from classifier.cache import ClassifierCache
from classifier.client import BatchAnthropicClient
from classifier.db import ClassifierDB
from classifier.types import CanonicalizeInput, EventId, NativeKey, SecurityId
from classifier.constants import DEDUP_EXPIRY_TOLERANCE_HOURS
from classifier.stages.canonicalize import canonicalize_events
from classifier.utils import bulk_create_chunked, expiry_close, from_dict, generate_security_symbol
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import EventContract, Listing, SecurityType

logger = logging.getLogger(__name__)


class _CurrencyProxy:
    # Wraps a currency_id int so downstream helpers can call .currency_id
    # without needing a full ORM object.
    def __init__(self, currency_id: int):
        self.currency_id = currency_id


class _ListingProxy:
    # Wraps a listing_id int so downstream helpers can call .listing_id
    # without needing a full ORM object.
    def __init__(self, listing_id: int):
        self.listing_id = listing_id


def _native_key(c: AdapterContract) -> NativeKey:
    return (c.exchange_id, c.exchange_event_native_id)


def create_entities(
    registry: RegistryClient,
    batch_client: BatchAnthropicClient,
    contracts: list[AdapterContract],
    *,
    cache: ClassifierCache | None = None,
    db: ClassifierDB,
) -> dict:
    if not contracts:
        return {
            "events_created": 0, "securities_created": 0, "listings_created": 0,
            "event_contracts_created": 0, "listing_specs_created": 0,
            "new_security_ids": [], "new_security_symbols": [],
        }

    contracts_by_native: dict[NativeKey, list[AdapterContract]] = {}
    for c in contracts:
        contracts_by_native.setdefault(_native_key(c), []).append(c)

    # ── Skip already-registered contracts ────────────────────────────
    # Most runs (99%) find nothing new. Checking exchange_events first
    # lets us skip canonicalization/embedding entirely for known contracts.
    # Cache is checked before DB to avoid a query per contract group.
    event_id_by_native: dict[NativeKey, EventId] = {}
    events_to_canonicalize: list[CanonicalizeInput] = []
    seen_exchange_events: set[NativeKey] = set()

    all_native_keys = list(contracts_by_native.keys())

    cached: dict[NativeKey, int] = {}
    if cache is not None:
        cached = cache.get_exchange_event_bulk(all_native_keys)

    cache_miss_keys = [nk for nk in all_native_keys if nk not in cached]
    if cache_miss_keys:
        all_exchange_events = db.get_all_exchange_events()
        db_results = {nk: all_exchange_events[nk] for nk in cache_miss_keys if nk in all_exchange_events}
    else:
        db_results = {}

    if cache is not None and db_results:
        cache.put_exchange_event_bulk(db_results)

    for nk, group in contracts_by_native.items():
        event_id = cached.get(nk) or db_results.get(nk)
        if event_id is not None:
            event_id_by_native[nk] = event_id
            seen_exchange_events.add(nk)
        else:
            c = group[0]
            exchange_id, native_id = nk
            events_to_canonicalize.append(
                CanonicalizeInput(
                    c.event_title, c.event_description, c.event_category,
                    exchange_id, native_id,
                )
            )

    # ── Canonicalize titles ──────────────────────────────────────────
    # Different exchanges use different titles for the same event
    # ("BTC $100k?" vs "Will Bitcoin reach $100,000?"). Claude normalizes
    # these so the title+expiry dedup downstream can match them.
    canonical_by_native = canonicalize_events(batch_client, events_to_canonicalize, cache=cache)

    # Build event_info_by_native: for known events, load from DB; for new, use canonical result.
    mapped_event_ids = list(set(event_id_by_native.values()))
    mapped_event_info = db.get_events(mapped_event_ids) if mapped_event_ids else {}
    event_info_by_native: dict[NativeKey, dict] = {}
    for nk, event_id in event_id_by_native.items():
        info = mapped_event_info.get(event_id)
        if info:
            event_info_by_native[nk] = info
    for nk, info in canonical_by_native.items():
        event_info_by_native[nk] = info

    # ── Title + expiry dedup ─────────────────────────────────────────
    # Match on Claude's canonical title — deterministic, not probabilistic.
    # Semantic equivalence (different titles, same meaning) is handled by
    # the relationship classifier with LLM verification, not here.
    created_event_records: list[tuple[str, str | None, int]] = list(db.get_events_for_dedup())
    pending_events, pending_native_to_event_idx, event_id_by_native = _title_expiry_dedup(
        event_info_by_native, contracts_by_native, created_event_records, event_id_by_native,
    )

    # ── Create events in registry ────────────────────────────────────
    # Only truly new events reach this point. Registry API handles writes;
    # we store the returned event_ids for future dedup runs.
    events_created, created_event_ids = _create_events(registry, pending_events, created_event_records)

    for nk, event_idx in pending_native_to_event_idx.items():
        eid = created_event_ids[event_idx] if event_idx < len(created_event_ids) else None
        if eid is not None:
            event_id_by_native[nk] = eid

    # ── Map exchange events ──────────────────────────────────────────
    # Links each exchange's native contract ID to our canonical event_id.
    # seen_exchange_events is pre-populated with already-existing mappings
    # so we only create entries for genuinely new contracts.
    _create_exchange_events(registry, contracts_by_native, event_id_by_native, seen_exchange_events)

    # ── Resolve reference data (currencies, securities, listings) ────
    # Each entity type is queried from DB before creation to avoid duplicates.
    # _CurrencyProxy and _ListingProxy wrap DB integer IDs so the shared
    # creation helpers below can access .currency_id / .listing_id without
    # needing full ORM objects from the registry API.
    seen_symbols = _collect_seen_symbols(contracts, event_info_by_native)
    all_symbols_needed = list(seen_symbols.keys())

    existing_secs = db.get_existing_securities(all_symbols_needed)
    pre_existing_security_ids = set(existing_secs.values())

    currency_by_symbol_ids = db.get_currencies()
    currency_by_symbol: dict[str, object] = {}
    all_currency_symbols = (
        {c.base_currency for c in contracts}
        | {c.quote_currency for c in contracts}
        | {c.settle_currency for c in contracts}
    )
    for sym in all_currency_symbols:
        if sym in currency_by_symbol_ids:
            currency_by_symbol[sym] = _CurrencyProxy(currency_by_symbol_ids[sym])
        else:
            try:
                created_curr = registry.create_currency(symbol=sym)
                cid = created_curr["currency_id"]
                currency_by_symbol[sym] = _CurrencyProxy(cid)
                currency_by_symbol_ids[sym] = cid
            except Exception as e:
                logger.error("Failed to create currency '%s': %s", sym, e)

    security_id_by_symbol: dict[str, SecurityId] = dict(existing_secs)
    security_id_by_outcome: dict[tuple[NativeKey, str], SecurityId] = {}

    securities_created, security_id_by_symbol, security_id_by_outcome = _create_securities(
        registry, seen_symbols, event_info_by_native, contracts_by_native,
        security_id_by_symbol, security_id_by_outcome, currency_by_symbol,
    )

    # Back-fill outcome map for contracts whose NativeKey wasn't the "first seen" for their symbol.
    # Happens when two exchanges share the same canonical title — seen_symbols records one NativeKey
    # per symbol, but all exchanges' contracts must resolve to the same security.
    for c in contracts:
        nk = _native_key(c)
        if (nk, c.outcome_label) in security_id_by_outcome:
            continue
        info = event_info_by_native.get(nk)
        if info is None:
            continue
        symbol = generate_security_symbol(info["title"], c.outcome_label)
        sid = security_id_by_symbol.get(symbol)
        if sid is not None:
            security_id_by_outcome[(nk, c.outcome_label)] = sid

    listing_keys_needed = [
        (c.exchange_id, c.exchange_security_id)
        for c in contracts
        if security_id_by_outcome.get((_native_key(c), c.outcome_label)) is not None
    ]
    existing_listing_map = db.get_existing_listings(list(set(listing_keys_needed)))
    listing_by_key: dict[str, object] = {
        f"{eid}:{esid}": _ListingProxy(lid)
        for (eid, esid), lid in existing_listing_map.items()
    }

    listings_created = _create_listings(registry, contracts, security_id_by_outcome, listing_by_key)

    ec_keys_needed = [
        (event_id_by_native[_native_key(c)], security_id_by_outcome[(_native_key(c), c.outcome_label)])
        for c in contracts
        if _native_key(c) in event_id_by_native
        and (_native_key(c), c.outcome_label) in security_id_by_outcome
    ]
    existing_ecs = db.get_existing_event_contracts(list(set(ec_keys_needed)))
    event_contract_by_key: dict[str, object] = {f"{eid}:{sid}": True for eid, sid in existing_ecs}

    event_contracts_created = _create_event_contracts(
        registry, contracts, event_id_by_native, security_id_by_outcome, event_contract_by_key,
    )

    listing_ids_needed = [
        listing_by_key[f"{c.exchange_id}:{c.exchange_security_id}"].listing_id
        for c in contracts
        if f"{c.exchange_id}:{c.exchange_security_id}" in listing_by_key
        and listing_by_key[f"{c.exchange_id}:{c.exchange_security_id}"] is not None
    ]
    existing_specs = db.get_existing_listing_specs(list(set(listing_ids_needed)))
    spec_by_listing_id: dict[int, object] = {lid: True for lid in existing_specs}

    listing_specs_created = _create_listing_specs(registry, contracts, listing_by_key, spec_by_listing_id)

    # ── Compute new security IDs ─────────────────────────────────────
    # Returned to the Step Function so the classify stage knows which
    # securities need relationship classification.
    new_security_ids = [
        sid for sid in security_id_by_outcome.values()
        if sid not in pre_existing_security_ids
    ]
    all_new_symbols = {v: k for k, v in security_id_by_symbol.items() if k not in existing_secs}
    new_security_symbols = [all_new_symbols.get(sid, "") for sid in new_security_ids]

    return {
        "events_created": events_created,
        "securities_created": securities_created,
        "listings_created": listings_created,
        "event_contracts_created": event_contracts_created,
        "listing_specs_created": listing_specs_created,
        "new_security_ids": new_security_ids,
        "new_security_symbols": new_security_symbols,
    }


def _title_expiry_dedup(
    event_info_by_native: dict[NativeKey, dict],
    contracts_by_native: dict[NativeKey, list[AdapterContract]],
    created_event_records: list[tuple[str, str | None, int]],
    event_id_by_native: dict[NativeKey, EventId],
) -> tuple[list[dict], dict[NativeKey, int], dict[NativeKey, EventId]]:
    event_id_by_native = dict(event_id_by_native)
    records_by_title: dict[str, list[tuple[str | None, int]]] = defaultdict(list)
    for title, expiry, eid in created_event_records:
        records_by_title[title].append((expiry, eid))

    pending_events: list[dict] = []
    pending_native_to_event_idx: dict[NativeKey, int] = {}
    pending_by_title: list[tuple[str, str | None, int]] = []

    for nk, canonical_info in event_info_by_native.items():
        if nk in event_id_by_native:
            continue
        canonical_title = canonical_info["title"]
        group = contracts_by_native[nk]
        expiry = group[0].event_expiry

        existing_match_id = next(
            (eid for exp, eid in records_by_title.get(canonical_title, [])
             if expiry_close(expiry, exp, timedelta(hours=DEDUP_EXPIRY_TOLERANCE_HOURS))),
            None,
        )
        if existing_match_id is not None:
            event_id_by_native[nk] = existing_match_id
            continue

        pending_match_idx = next(
            (idx for idx, (title, exp, _) in enumerate(pending_by_title)
             if title == canonical_title
             and expiry_close(expiry, exp, timedelta(hours=DEDUP_EXPIRY_TOLERANCE_HOURS))),
            None,
        )
        if pending_match_idx is not None:
            pending_native_to_event_idx[nk] = pending_match_idx
            continue

        event_idx = len(pending_events)
        pending_native_to_event_idx[nk] = event_idx
        pending_events.append(dict(
            title=canonical_title,
            description=group[0].event_description,
            category=canonical_info["category"],
            tags=canonical_info["tags"],
            expiry=expiry,
        ))
        pending_by_title.append((canonical_title, expiry, event_idx))

    return pending_events, pending_native_to_event_idx, event_id_by_native


def _create_events(
    registry: RegistryClient,
    pending_events: list[dict],
    created_event_records: list[tuple[str, str | None, int]],
) -> tuple[int, list[int | None]]:
    events_created = 0
    created_event_ids: list[int | None] = [None] * len(pending_events)
    for chunk_start, chunk in bulk_create_chunked(pending_events, "events"):
        created_list = registry.bulk_create_events(chunk)
        events_created += len(created_list)
        for chunk_idx, created in enumerate(created_list):
            idx = chunk_start + chunk_idx
            new_eid = created["event_id"]
            created_event_ids[idx] = new_eid
            ev = pending_events[idx]
            created_event_records.append((ev["title"], ev.get("expiry"), new_eid))
    return events_created, created_event_ids


def _create_exchange_events(
    registry: RegistryClient,
    contracts_by_native: dict[NativeKey, list[AdapterContract]],
    event_id_by_native: dict[NativeKey, EventId],
    seen_exchange_events: set[NativeKey],
) -> int:
    pending: list[dict] = []
    for nk, group in contracts_by_native.items():
        if nk in seen_exchange_events:
            continue
        exchange_id, native_id = nk
        eid = event_id_by_native.get(nk)
        if eid is None:
            continue
        pending.append(dict(
            exchange_id=exchange_id,
            event_id=eid,
            native_event_id=native_id,
            raw_title=group[0].event_title,
        ))
        seen_exchange_events.add(nk)
    for _, chunk in bulk_create_chunked(pending, "exchange events"):
        registry.bulk_create_exchange_events(chunk)
    return len(pending)


def _collect_seen_symbols(
    contracts: list[AdapterContract],
    event_info_by_native: dict[NativeKey, dict],
) -> dict[str, AdapterContract]:
    seen: dict[str, AdapterContract] = {}
    for c in contracts:
        info = event_info_by_native.get(_native_key(c))
        if info is None:
            continue
        symbol = generate_security_symbol(info["title"], c.outcome_label)
        if symbol not in seen:
            seen[symbol] = c
    return seen


def _create_securities(
    registry: RegistryClient,
    seen_symbols: dict[str, AdapterContract],
    event_info_by_native: dict[NativeKey, dict],
    contracts_by_native: dict[NativeKey, list[AdapterContract]],
    security_id_by_symbol: dict[str, SecurityId],
    security_id_by_outcome: dict[tuple[NativeKey, str], SecurityId],
    currency_by_symbol: dict,
) -> tuple[int, dict[str, SecurityId], dict[tuple[NativeKey, str], SecurityId]]:
    pending_securities: list[dict] = []
    pending_symbols: list[str] = []
    pending_outcomes: list[tuple[NativeKey, str]] = []

    for symbol, c in seen_symbols.items():
        nk = _native_key(c)
        canonical_info = event_info_by_native[nk]
        if symbol in security_id_by_symbol:
            security_id_by_outcome[(nk, c.outcome_label)] = security_id_by_symbol[symbol]
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
        pending_symbols.append(symbol)
        pending_outcomes.append((nk, c.outcome_label))

    securities_created = 0
    for chunk_start, chunk in bulk_create_chunked(pending_securities, "securities"):
        created_list = registry.bulk_create_securities(chunk)
        securities_created += len(created_list)
        for chunk_idx, created in enumerate(created_list):
            idx = chunk_start + chunk_idx
            new_sid = created["security_id"]
            sym = pending_symbols[idx]
            security_id_by_symbol[sym] = new_sid
            security_id_by_outcome[pending_outcomes[idx]] = new_sid

    return securities_created, security_id_by_symbol, security_id_by_outcome


def _create_listings(
    registry: RegistryClient,
    contracts: list[AdapterContract],
    security_id_by_outcome: dict[tuple[NativeKey, str], SecurityId],
    listing_by_key: dict[str, object],
) -> int:
    pending_listings: list[dict] = []
    pending_listing_keys: list[str] = []

    for c in contracts:
        key = f"{c.exchange_id}:{c.exchange_security_id}"
        if key in listing_by_key:
            continue
        sid = security_id_by_outcome.get((_native_key(c), c.outcome_label))
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

    listings_created = 0
    for chunk_start, chunk in bulk_create_chunked(pending_listings, "listings"):
        try:
            created_list = registry.bulk_create_listings(chunk)
            listings_created += len(created_list)
            for chunk_idx, created in enumerate(created_list):
                idx = chunk_start + chunk_idx
                listing_by_key[pending_listing_keys[idx]] = from_dict(Listing, created)
        except Exception as e:
            logger.error(
                "Bulk listing creation failed (chunk starting at %d): %s — first item: %s",
                chunk_start, e, chunk[0] if chunk else None,
            )
    return listings_created


def _create_event_contracts(
    registry: RegistryClient,
    contracts: list[AdapterContract],
    event_id_by_native: dict[NativeKey, EventId],
    security_id_by_outcome: dict[tuple[NativeKey, str], SecurityId],
    event_contract_by_key: dict[str, object],
) -> int:
    pending_ecs: list[dict] = []
    pending_ec_keys: list[str] = []

    for c in contracts:
        nk = _native_key(c)
        event_id = event_id_by_native.get(nk)
        sid = security_id_by_outcome.get((nk, c.outcome_label))
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

    event_contracts_created = 0
    for chunk_start, chunk in bulk_create_chunked(pending_ecs, "event contracts"):
        created_list = registry.bulk_create_event_contracts(chunk)
        event_contracts_created += len(created_list)
        for chunk_idx, created in enumerate(created_list):
            idx = chunk_start + chunk_idx
            event_contract_by_key[pending_ec_keys[idx]] = from_dict(EventContract, created)
    return event_contracts_created


def _create_listing_specs(
    registry: RegistryClient,
    contracts: list[AdapterContract],
    listing_by_key: dict[str, object],
    spec_by_listing_id: dict[int, object],
) -> int:
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

    listing_specs_created = 0
    for _, chunk in bulk_create_chunked(pending_specs, "listing specs"):
        try:
            created_list = registry.bulk_create_listing_specs(chunk)
            listing_specs_created += len(created_list)
        except Exception as e:
            logger.error("Bulk listing_spec creation failed: %s", e)
    return listing_specs_created
