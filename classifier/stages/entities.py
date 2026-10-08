import dataclasses
import logging

from classifier.adapters.types import AdapterContract
from classifier.cache import ClassifierCache
from classifier.client import BatchAnthropicClient
from classifier.db import ClassifierDB
from classifier.types import CanonicalizeInput, EntityResult, EventId, NativeKey, SecurityId
from classifier.constants import DEFAULT_CANONICALIZE_BATCH_SIZE, DEFAULT_CANONICALIZE_MODEL
from classifier.stages.canonicalize import canonicalize_events
from classifier.utils import bulk_create_chunked
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import SecurityType

logger = logging.getLogger(__name__)

ListingKey = tuple[int, str]


def _native_key(c: AdapterContract) -> NativeKey:
    return (c.exchange_id, c.exchange_event_native_id)


def _listing_key(c: AdapterContract) -> ListingKey:
    return (c.exchange_id, c.exchange_security_id)


def _empty_result() -> EntityResult:
    return EntityResult(
        events_created=0, securities_created=0, listings_created=0,
        event_contracts_created=0, listing_specs_created=0, listing_specs_updated=0,
        new_security_ids=[], new_security_symbols=[],
        created_event_ids=[], created_event_names=[],
    )


@dataclasses.dataclass
class EntityContext:
    contracts_by_native: dict[NativeKey, list[AdapterContract]]
    event_id_by_native: dict[NativeKey, EventId]


def prepare_canonicalization_inputs(
    contracts: list[AdapterContract],
    cache: ClassifierCache | None,
    db: ClassifierDB,
) -> tuple[list[CanonicalizeInput], EntityContext]:
    """Determine which native events are new and need categorization.

    Returns (events_to_canonicalize, entity_context).
    """
    contracts_by_native: dict[NativeKey, list[AdapterContract]] = {}
    for c in contracts:
        contracts_by_native.setdefault(_native_key(c), []).append(c)

    all_native_keys = list(contracts_by_native.keys())
    cached: dict[NativeKey, int] = cache.get_exchange_event_bulk(all_native_keys) if cache is not None else {}
    cache_miss_keys = [nk for nk in all_native_keys if nk not in cached]
    db_results = db.get_exchange_events(cache_miss_keys) if cache_miss_keys else {}
    if cache is not None and db_results:
        cache.put_exchange_event_bulk(db_results)

    event_id_by_native: dict[NativeKey, EventId] = {}
    events_to_canonicalize: list[CanonicalizeInput] = []
    for nk, group in contracts_by_native.items():
        event_id = cached.get(nk) or db_results.get(nk)
        if event_id is not None:
            event_id_by_native[nk] = event_id
        else:
            c = group[0]
            events_to_canonicalize.append(
                CanonicalizeInput(c.event_title, c.event_description, c.event_category, nk[0], nk[1])
            )

    return events_to_canonicalize, EntityContext(
        contracts_by_native=contracts_by_native,
        event_id_by_native=event_id_by_native,
    )


def create_entities_from_canonical(
    registry: RegistryClient,
    canonical_by_native: dict[NativeKey, dict],
    entity_ctx: EntityContext,
    contracts: list[AdapterContract],
    *,
    cache: ClassifierCache | None = None,
    db: ClassifierDB,
    debug: bool = False,
) -> EntityResult:
    """Create events, securities, listings and event contracts for a batch of contracts.

    Events are identified by their native exchange event and securities by their listing
    (exchange_id, exchange_security_id); titles and symbols never decide identity.
    """
    if not contracts:
        return _empty_result()

    contracts_by_native = entity_ctx.contracts_by_native
    event_id_by_native = dict(entity_ctx.event_id_by_native)

    created_event_ids, created_event_names = _create_events(
        registry, contracts_by_native, canonical_by_native, event_id_by_native,
    )
    if cache is not None and created_event_ids:
        created = set(created_event_ids)
        cache.put_exchange_event_bulk({nk: eid for nk, eid in event_id_by_native.items() if eid in created})

    unique_contracts = list({_listing_key(c): c for c in contracts}.values())
    existing_listings = db.get_existing_listings([_listing_key(c) for c in unique_contracts])
    listing_id_by_key: dict[ListingKey, int] = {k: lid for k, (lid, _) in existing_listings.items()}
    security_id_by_key: dict[ListingKey, SecurityId] = {k: sid for k, (_, sid) in existing_listings.items()}

    unlisted_contracts = [c for c in unique_contracts if _listing_key(c) not in listing_id_by_key]
    currency_ids = _resolve_currencies(registry, db, unlisted_contracts)
    securities_created, new_security_ids, new_security_symbols = _create_securities(
        registry, db, unlisted_contracts, currency_ids, security_id_by_key,
    )
    listings_created = _create_listings(registry, unlisted_contracts, security_id_by_key, listing_id_by_key)
    event_contracts_created = _create_event_contracts(
        registry, db, unique_contracts, event_id_by_native, security_id_by_key,
    )
    spec_by_listing_id = db.get_existing_listing_specs(list(set(listing_id_by_key.values())))
    listing_specs_created, listing_specs_updated = _sync_listing_specs(
        registry, unique_contracts, listing_id_by_key, spec_by_listing_id,
    )

    _reconcile_stale_entities(registry, contracts, entity_ctx.event_id_by_native, db)

    if debug and (created_event_ids or securities_created):
        logger.info("[DEBUG] entities: %d events created, %d securities created, %d listings created",
                    len(created_event_ids), securities_created, listings_created)
        for eid, name in zip(created_event_ids[:50], created_event_names[:50]):
            logger.info("[DEBUG] entities:   event id=%d %r", eid, name[:80])
        if len(created_event_ids) > 50:
            logger.info("[DEBUG] entities:   ... and %d more events", len(created_event_ids) - 50)
        for sid, sym in zip(new_security_ids[:50], new_security_symbols[:50]):
            logger.info("[DEBUG] entities:   security id=%d %s", sid, sym)
        if len(new_security_ids) > 50:
            logger.info("[DEBUG] entities:   ... and %d more securities", len(new_security_ids) - 50)

    return EntityResult(
        events_created=len(created_event_ids),
        securities_created=securities_created,
        listings_created=listings_created,
        event_contracts_created=event_contracts_created,
        listing_specs_created=listing_specs_created,
        listing_specs_updated=listing_specs_updated,
        new_security_ids=new_security_ids,
        new_security_symbols=new_security_symbols,
        created_event_ids=created_event_ids,
        created_event_names=created_event_names,
    )


def create_entities(
    registry: RegistryClient,
    batch_client: BatchAnthropicClient,
    contracts: list[AdapterContract],
    *,
    cache: ClassifierCache | None = None,
    db: ClassifierDB,
    canonicalize_enabled: bool = True,
    canonicalize_model: str = DEFAULT_CANONICALIZE_MODEL,
    canonicalize_batch_size: int = DEFAULT_CANONICALIZE_BATCH_SIZE,
    sync_threshold: int = 10,
    debug: bool = False,
) -> EntityResult:
    if not contracts:
        return _empty_result()
    events_to_canon, entity_ctx = prepare_canonicalization_inputs(contracts, cache, db)
    if canonicalize_enabled:
        canonical = canonicalize_events(
            batch_client, events_to_canon, cache=cache,
            model=canonicalize_model, batch_size=canonicalize_batch_size,
            sync_threshold=sync_threshold, debug=debug,
        )
    else:
        canonical = {
            (ev.exchange_id, ev.native_id): {"category": ev.category or "OTHER", "tags": []}
            for ev in events_to_canon
        }
    return create_entities_from_canonical(registry, canonical, entity_ctx, contracts, cache=cache, db=db, debug=debug)


def _create_events(
    registry: RegistryClient,
    contracts_by_native: dict[NativeKey, list[AdapterContract]],
    canonical_by_native: dict[NativeKey, dict],
    event_id_by_native: dict[NativeKey, EventId],
) -> tuple[list[EventId], list[str]]:
    pending_keys: list[NativeKey] = []
    pending_events: list[dict] = []
    for nk, group in contracts_by_native.items():
        if nk in event_id_by_native:
            continue
        c = group[0]
        info = canonical_by_native.get(nk, {})
        pending_keys.append(nk)
        pending_events.append(dict(
            title=c.event_title,
            description=c.event_description,
            category=info.get("category", "OTHER"),
            tags=info.get("tags", []),
            expiry=c.event_expiry,
            exchange_id=c.exchange_id,
            native_event_id=c.exchange_event_native_id,
            native_url=c.exchange_event_native_url,
        ))

    created_ids: list[EventId] = []
    created_names: list[str] = []
    for chunk_start, chunk in bulk_create_chunked(pending_events, "events"):
        for chunk_idx, created in enumerate(registry.bulk_create_events(chunk)):
            idx = chunk_start + chunk_idx
            event_id_by_native[pending_keys[idx]] = created["event_id"]
            created_ids.append(created["event_id"])
            created_names.append(pending_events[idx]["title"])
    return created_ids, created_names


def _resolve_currencies(
    registry: RegistryClient,
    db: ClassifierDB,
    contracts: list[AdapterContract],
) -> dict[str, int]:
    currency_ids = db.get_currencies()
    needed = {c.base_currency for c in contracts} | {c.quote_currency for c in contracts} | {c.settle_currency for c in contracts}
    for sym in needed - currency_ids.keys():
        currency_ids[sym] = registry.create_currency(symbol=sym)["currency_id"]
    return currency_ids


def _create_securities(
    registry: RegistryClient,
    db: ClassifierDB,
    contracts: list[AdapterContract],
    currency_ids: dict[str, int],
    security_id_by_key: dict[ListingKey, SecurityId],
) -> tuple[int, list[SecurityId], list[str]]:
    """Create one security per unlisted contract.

    A security left without a listing by an earlier failed attempt is reused, since nothing
    else can own it. Any other symbol clash fails the insert on sm.security's unique symbol,
    so two contracts can never silently share a security.
    """
    orphan_by_symbol = db.get_unlisted_securities([c.security_symbol for c in contracts])

    new_security_ids: list[SecurityId] = []
    new_symbols: list[str] = []
    pending: list[dict] = []
    pending_keys: list[ListingKey] = []
    for c in contracts:
        orphan_sid = orphan_by_symbol.pop(c.security_symbol, None)
        if orphan_sid is not None:
            security_id_by_key[_listing_key(c)] = orphan_sid
            new_security_ids.append(orphan_sid)
            new_symbols.append(c.security_symbol)
            continue
        pending_keys.append(_listing_key(c))
        pending.append(dict(
            symbol=c.security_symbol,
            type=SecurityType.EVENT_CONTRACT,
            contract_type=c.contract_type,
            asset_class=c.asset_class,
            base_currency_id=currency_ids.get(c.base_currency),
            quote_currency_id=currency_ids.get(c.quote_currency),
            settle_currency_id=currency_ids.get(c.settle_currency),
            inverse=c.inverse,
            quanto=c.is_quanto,
            expiry=c.event_expiry,
            active=True,
        ))

    created_count = 0
    for chunk_start, chunk in bulk_create_chunked(pending, "securities"):
        created_list = registry.bulk_create_securities(chunk)
        created_count += len(created_list)
        for chunk_idx, created in enumerate(created_list):
            idx = chunk_start + chunk_idx
            security_id_by_key[pending_keys[idx]] = created["security_id"]
            new_security_ids.append(created["security_id"])
            new_symbols.append(pending[idx]["symbol"])
    return created_count, new_security_ids, new_symbols


def _create_listings(
    registry: RegistryClient,
    contracts: list[AdapterContract],
    security_id_by_key: dict[ListingKey, SecurityId],
    listing_id_by_key: dict[ListingKey, int],
) -> int:
    pending = [
        dict(
            exchange_id=c.exchange_id,
            security_id=security_id_by_key[_listing_key(c)],
            exchange_security_id=c.exchange_security_id,
            exchange_security_symbol=c.exchange_security_symbol,
        )
        for c in contracts
    ]
    created_count = 0
    for chunk_start, chunk in bulk_create_chunked(pending, "listings"):
        created_list = registry.bulk_create_listings(chunk)
        created_count += len(created_list)
        for chunk_idx, created in enumerate(created_list):
            c = contracts[chunk_start + chunk_idx]
            listing_id_by_key[_listing_key(c)] = created["listing_id"]
    return created_count


def _create_event_contracts(
    registry: RegistryClient,
    db: ClassifierDB,
    contracts: list[AdapterContract],
    event_id_by_native: dict[NativeKey, EventId],
    security_id_by_key: dict[ListingKey, SecurityId],
) -> int:
    wanted: dict[tuple[EventId, SecurityId], str] = {}
    for c in contracts:
        event_id = event_id_by_native.get(_native_key(c))
        security_id = security_id_by_key.get(_listing_key(c))
        if event_id is None or security_id is None:
            continue
        wanted.setdefault((event_id, security_id), c.outcome_label)

    # A security belongs to exactly one event. One already linked elsewhere means an adapter changed the event it
    # maps a market to, which would split the market across two events; skip it rather than link it twice.
    linked = db.get_event_ids_by_security([sid for _, sid in wanted])
    pending = []
    for (eid, sid), label in wanted.items():
        if sid not in linked:
            pending.append(dict(event_id=eid, security_id=sid, outcome_label=label))
        elif linked[sid] != eid:
            # Counted by the IdentityRegressions metric filter (cdk/lib/stacks/classifier-stack.ts); keep the wording.
            logger.error("Security %d is already in event %d; not linking it to event %d", sid, linked[sid], eid)
    created_count = 0
    for _, chunk in bulk_create_chunked(pending, "event contracts"):
        created_count += len(registry.bulk_create_event_contracts(chunk))
    return created_count


def _sync_listing_specs(
    registry: RegistryClient,
    contracts: list[AdapterContract],
    listing_id_by_key: dict[ListingKey, int],
    spec_by_listing_id: dict[int, tuple[int, int, int, int, int]],
) -> tuple[int, int]:
    pending_specs: list[dict] = []
    updates = 0
    for c in contracts:
        listing_id = listing_id_by_key.get(_listing_key(c))
        if listing_id is None:
            continue
        new_vals = (int(c.tick_size), int(c.lot_size), int(c.min_notional), int(c.contract_multiplier), int(c.min_size))
        existing = spec_by_listing_id.get(listing_id)
        if existing == new_vals:
            continue
        if existing is not None:
            updates += 1
        pending_specs.append(dict(
            listing_id=listing_id,
            tick_size=c.tick_size,
            lot_size=c.lot_size,
            min_notional=c.min_notional,
            contract_multiplier=c.contract_multiplier,
            min_size=c.min_size,
        ))

    posted = 0
    for _, chunk in bulk_create_chunked(pending_specs, "listing specs"):
        try:
            posted += len(registry.bulk_create_listing_specs(chunk))
        except Exception as e:
            logger.error("Bulk listing_spec creation failed: %s", e)
    return posted - updates, updates


def _reconcile_stale_entities(
    registry: RegistryClient,
    contracts: list[AdapterContract],
    preexisting_event_id_by_native: dict[NativeKey, EventId],
    db: ClassifierDB,
) -> None:
    """Deactivate listings of pre-existing events that the exchange no longer lists.

    Every message carries the full contract group of its native event, so any active
    listing of that event missing from the batch has been removed by the exchange.
    """
    if not preexisting_event_id_by_native:
        return
    old_security_ids = db.get_security_ids_for_events(list(set(preexisting_event_id_by_native.values())))
    if not old_security_ids:
        return
    existing_listings = db.get_active_listings_for_securities(list(old_security_ids))
    if not existing_listings:
        return

    batch_exchange_ids = {nk[0] for nk in preexisting_event_id_by_native}
    current_keys = {_listing_key(c) for c in contracts}
    stale_listing_ids: list[int] = []
    stale_security_ids: set[int] = set()
    for lid, sid, lex_id, lex_esid in existing_listings:
        # Only judge listings on exchanges this batch speaks for.
        if lex_id not in batch_exchange_ids:
            continue
        if (lex_id, lex_esid) not in current_keys:
            stale_listing_ids.append(lid)
            stale_security_ids.add(sid)

    if stale_listing_ids:
        registry.bulk_patch_listings([{"listing_id": lid, "active": False} for lid in stale_listing_ids])
        logger.info("Deactivated %d stale listings", len(stale_listing_ids))

    if stale_security_ids:
        still_active = db.get_securities_with_active_listings(list(stale_security_ids))
        sids_to_deactivate = list(stale_security_ids - still_active)
        if sids_to_deactivate:
            registry.bulk_patch_securities([{"security_id": sid, "active": False} for sid in sids_to_deactivate])
            logger.info("Deactivated %d stale securities", len(sids_to_deactivate))
