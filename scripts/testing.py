import dataclasses
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


from classifier.cache.base import ClassifierCache
from classifier.client import BedrockConverseClient, ConverseResult
from classifier.propositions.attempts import Attempt
from classifier.propositions.derive import Contract
from classifier.propositions.inputs import ExtractionInput
from classifier.propositions.subjects import KnownSubject
from classifier.propositions.types import Proposition
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import (
    ContractType,
    Currency,
    Event,
    EventContract,
    Exchange,
    Listing,
    ListingSpec,
    Security,
)


class MemoryClassifierCache(ClassifierCache):
    """In-memory cache for use in tests and scripts without Redis."""

    def __init__(self):
        self._canon: dict[str, dict] = {}

    def get_canonicalization(self, model: str, exchange_id: int, native_id: str) -> dict | None:
        return self._canon.get(self._canon_hash(model, exchange_id, native_id))

    def get_canonicalization_bulk(self, model: str, pairs: list[tuple[int, str]]) -> dict[tuple[int, str], dict]:
        return {
            p: self._canon[self._canon_hash(model, *p)]
            for p in pairs
            if self._canon_hash(model, *p) in self._canon
        }

    def put_canonicalization(self, model: str, exchange_id: int, native_id: str, result: dict) -> None:
        self._canon[self._canon_hash(model, exchange_id, native_id)] = result


class FakeConverseClient(BedrockConverseClient):
    """Answers canonicalization prompts with a fixed category and tags, echoing each event's id and key."""

    def __init__(self, category: str = "OTHER", tags: list[str] | None = None):
        super().__init__(client=None)
        self._category = category
        self._tags = tags or []
        self.calls = 0

    def converse(self, model: str, prompt: str, max_tokens: int) -> ConverseResult:
        self.calls += 1
        items = []
        single = None
        for line in prompt.splitlines():
            if line.startswith("[") and "(key: " in line:
                items.append({
                    "id": int(line[1:line.index("]")]), "key": line.split("(key: ", 1)[1].split(")", 1)[0],
                    "category": self._category, "tags": self._tags,
                })
            elif line.startswith("Exchange-provided title:"):
                single = {"category": self._category, "tags": self._tags}
        return ConverseResult(text=json.dumps(single if single is not None else items), input_tokens=0, output_tokens=0)


class StubRegistry(RegistryClient):
    """In-memory registry that simulates empty DB state. All writes are stored locally."""

    def __init__(self):
        self._next_id = 1
        self._events: list[Event] = []
        self._securities: list[Security] = []
        self._listings: list[Listing] = []
        self._listing_specs: list[ListingSpec] = []
        self._min_size_by_spec_id: dict[int, int] = {}
        # Fee state lives beside the gnomepy dataclasses, which may predate these fields.
        self._fee_keys_by_listing: dict[int, tuple[str | None, str | None]] = {}
        self._fee_schedules: list[dict] = []
        self._event_contracts: list[EventContract] = []
        self._contract_sources: dict[int, dict] = {}
        # Proposition state as the registry keeps it; `_clock` stands in for now() so "changed since derived" is
        # ordered without real time.
        self._clock = 0
        self._subjects: dict[str, dict] = {}
        self._propositions: dict[int, dict] = {}
        # When each proposition last changed, standing in for sm.contract_proposition.date_modified.
        self._proposition_modified: dict[int, int] = {}
        self._relationships: dict[tuple[int, int], dict] = {}
        self._currencies: list[Currency] = []
        self._settlement_by_event_contract: dict[int, int] = {}
        # When each listing was last written, standing in for sm.listing.date_modified; unset reads as just now.
        self._listing_modified_at: dict[int, datetime] = {}

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def get_exchange(self) -> list[Exchange]:
        return [
            Exchange(exchange_id=1, exchange_code="POLYMARKET_INTL", exchange_name="Polymarket (International)", region="", date_modified="", date_created=""),
            Exchange(exchange_id=2, exchange_code="KALSHI", exchange_name="Kalshi", region="", date_modified="", date_created=""),
            Exchange(exchange_id=3, exchange_code="HYPERLIQUID", exchange_name="Hyperliquid", region="", date_modified="", date_created=""),
        ]

    def get_currency(self) -> list[Currency]:
        return list(self._currencies)

    def get_security(self) -> list[Security]:
        return list(self._securities)

    def get_listing(self) -> list[Listing]:
        return list(self._listings)

    def get_listing_spec(self) -> list[ListingSpec]:
        return list(self._listing_specs)

    def get_event(self, resolved: bool | None = None) -> list[Event]:
        if resolved is None:
            return list(self._events)
        return [e for e in self._events if e.resolved == resolved]

    def get_event_contracts(self) -> list[EventContract]:
        return list(self._event_contracts)

    def bulk_create_events(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            event_id = self._alloc_id()
            d = {
                "event_id": event_id,
                "title": item.get("title", ""),
                "description": item.get("description"),
                "category": item.get("category"),
                "tags": item.get("tags"),
                "resolved": False,
                "resolved_at": None,
                "expiry": item.get("expiry"),
                "date_modified": "",
                "date_created": "",
                "exchange_id": item["exchange_id"],
                "native_event_id": item["native_event_id"],
                "native_url": item.get("native_url"),
            }
            if any(
                e.exchange_id == d["exchange_id"] and e.native_event_id == d["native_event_id"]
                for e in self._events
            ):
                raise ValueError(f"duplicate native event {d['exchange_id']}:{d['native_event_id']}")
            self._events.append(Event(**d))
            results.append(d)
        return results

    def bulk_create_securities(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            if any(s.symbol == item.get("symbol") for s in self._securities):
                raise ValueError(f"duplicate security symbol {item.get('symbol')}")
            security_id = self._alloc_id()
            d = {"security_id": security_id, **item}
            self._securities.append(Security(
                security_id=security_id,
                symbol=item.get("symbol", ""),
                type=item.get("type", 0),
                contract_type=item.get("contract_type", 0),
                asset_class=item.get("asset_class", 0),
                base_currency_id=item.get("base_currency_id"),
                quote_currency_id=item.get("quote_currency_id"),
                settle_currency_id=item.get("settle_currency_id"),
                inverse=item.get("inverse", False),
                is_quanto=item.get("quanto", False),
                expiry=item.get("expiry"),
                strike_price=None,
                active=item.get("active", True),
                underlying_security_id=None,
                description=None,
                date_modified="",
                date_created="",
            ))
            results.append(d)
        return results

    def bulk_create_listings(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            listing_id = self._alloc_id()
            d = {
                "listing_id": listing_id,
                "security_id": item["security_id"],
                "exchange_id": item["exchange_id"],
                "exchange_security_id": item.get("exchange_security_id"),
                "exchange_security_symbol": item.get("exchange_security_symbol"),
                "active": item.get("active", True),
                "date_modified": "",
                "date_created": "",
            }
            self._listings.append(Listing(**d))
            self._fee_keys_by_listing[listing_id] = (item.get("fee_group"), item.get("fee_event"))
            results.append(d)
        return results

    def bulk_create_event_contracts(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            ec_id = self._alloc_id()
            d = {
                "event_contract_id": ec_id,
                "event_id": item["event_id"],
                "security_id": item["security_id"],
                "outcome_label": item["outcome_label"],
                "date_created": "",
            }
            self._event_contracts.append(EventContract(**d))
            results.append(d)
        return results

    def bulk_create_listing_specs(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            spec_id = self._alloc_id()
            d = {
                "id": spec_id,
                "listing_id": item["listing_id"],
                "tick_size": item["tick_size"],
                "lot_size": item["lot_size"],
                "min_notional": item["min_notional"],
                "contract_multiplier": item["contract_multiplier"],
                "recorded_at": "",
            }
            self._listing_specs.append(ListingSpec(**d))
            self._min_size_by_spec_id[spec_id] = item.get("min_size", 0)
            results.append(d)
        return results

    def bulk_create_fee_schedules(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            row = {"fee_schedule_id": self._alloc_id(), "recorded_at": len(self._fee_schedules), **item}
            self._fee_schedules.append(row)
            results.append(row)
        return results

    def patch_event_contract(self, event_contract_id: int, **kwargs) -> dict:
        for i, ec in enumerate(self._event_contracts):
            if ec.event_contract_id == event_contract_id:
                updated = dataclasses.replace(ec, **kwargs)
                self._event_contracts[i] = updated
                return dataclasses.asdict(updated)
        return {}

    def patch_event(self, event_id: int, **kwargs) -> dict:
        for i, ev in enumerate(self._events):
            if ev.event_id == event_id:
                updated = dataclasses.replace(ev, **kwargs)
                self._events[i] = updated
                return dataclasses.asdict(updated)
        return {}

    def patch_security(self, security_id: int, **kwargs) -> dict:
        for i, s in enumerate(self._securities):
            if s.security_id == security_id:
                updated = dataclasses.replace(s, **kwargs)
                self._securities[i] = updated
                return dataclasses.asdict(updated)
        return {}

    def patch_listing(self, listing_id: int, **kwargs) -> dict:
        if "fee_group" in kwargs or "fee_event" in kwargs:
            group, event = self._fee_keys_by_listing.get(listing_id, (None, None))
            self._fee_keys_by_listing[listing_id] = (kwargs.pop("fee_group", group), kwargs.pop("fee_event", event))
        for i, l in enumerate(self._listings):
            if l.listing_id == listing_id:
                updated = dataclasses.replace(l, **kwargs)
                self._listings[i] = updated
                self._listing_modified_at[listing_id] = datetime.now(timezone.utc)
                return dataclasses.asdict(updated)
        return {}

    def bulk_patch_events(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            event_id = item["event_id"]
            kwargs = {k: v for k, v in item.items() if k != "event_id"}
            results.append(self.patch_event(event_id, **kwargs))
        return results

    def bulk_patch_securities(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            security_id = item["security_id"]
            kwargs = {k: v for k, v in item.items() if k != "security_id"}
            results.append(self.patch_security(security_id, **kwargs))
        return results

    def bulk_patch_event_contracts(self, items: list[dict]) -> list[dict]:
        # Kept beside the EventContract rows so the stub doesn't depend on gnomepy carrying settlement fields; the
        # first value recorded wins, as in the registry.
        for item in items:
            self._settlement_by_event_contract.setdefault(item["event_contract_id"], item["settlement_price"])
        return [dict(item) for item in items]

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _subject_by_id(self, subject_id: int) -> dict:
        return next(s for s in self._subjects.values() if s["subject_id"] == subject_id)

    def create_proposition_subjects(self, items: list[dict]) -> list[dict]:
        rows = []
        for item in items:
            subject = self._subjects.get(item["key"])
            if subject is None:
                subject = self._subjects[item["key"]] = {
                    "subject_id": self._alloc_id(), "key": item["key"], "description": item.get("description") or "",
                    "date_modified": self._tick(), "derived_at": None, "embedding": None,
                }
            elif item.get("description"):
                subject["description"] = item["description"]
            rows.append({k: subject[k] for k in ("subject_id", "key", "description")})
        return rows

    def upsert_contract_propositions(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            before = self._propositions.get(item["security_id"])
            row = {**item, "subject_id": item["extracted_subject_id"]}
            changed = before != row
            if changed:
                self._propositions[item["security_id"]] = row
                self._proposition_modified[item["security_id"]] = self._tick()
                touched = {row["subject_id"]} | ({before["subject_id"]} if before else set())
                for subject_id in touched:
                    self._subject_by_id(subject_id)["date_modified"] = self._tick()
            results.append({"security_id": item["security_id"], "subject_id": row["subject_id"], "changed": changed})
        return results

    def reconcile_contract_relationships(self, *, security_ids, relationships, method, derivation_version,
                                         subject_id=None, derived_at=None) -> dict:
        ids = set(security_ids)
        if any(r["security_id_a"] not in ids and r["security_id_b"] not in ids for r in relationships):
            raise ValueError("A relationship touches no security in securityIds")
        desired = {(r["security_id_a"], r["security_id_b"]): r for r in relationships}
        stale = [k for k, v in self._relationships.items()
                 if v["method"] != "manual" and (k[0] in ids or k[1] in ids) and k not in desired]
        for k in stale:
            del self._relationships[k]
        inserted = sum(1 for k in desired if k not in self._relationships)
        for k, r in desired.items():
            self._relationships[k] = {**r, "method": method, "derivation_version": derivation_version}
        if subject_id is not None:
            self._subject_by_id(subject_id)["derived_at"] = self._tick()
        return {"inserted": inserted, "updated": 0, "deleted": len(stale), "unchanged": len(desired) - inserted}

    def _drop_derived_relationships(self, security_id: int):
        for k in [k for k, v in self._relationships.items() if v["method"] != "manual" and security_id in k]:
            del self._relationships[k]

    def upsert_contract_sources(self, items: list[dict]) -> list[dict]:
        if len({item["security_id"] for item in items}) != len(items):
            raise ValueError("Each security may appear once")
        results = []
        for item in items:
            sid = item["security_id"]
            if item["source"] is None:
                proposition = self._propositions.pop(sid, None)
                source = self._contract_sources.pop(sid, None)
                self._drop_derived_relationships(sid)
                if proposition:
                    self._subject_by_id(proposition["subject_id"])["date_modified"] = self._tick()
                results.append({"security_id": sid, "changed": bool(proposition or source)})
                continue
            before = self._contract_sources.get(sid)
            changed = (before or {}).get("source_hash") != item["source_hash"]
            self._contract_sources[sid] = dict(item)
            if changed and before:
                self._drop_derived_relationships(sid)
            results.append({"security_id": sid, "changed": changed})
        return results

    def bulk_patch_listings(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            listing_id = item["listing_id"]
            kwargs = {k: v for k, v in item.items() if k != "listing_id"}
            results.append(self.patch_listing(listing_id, **kwargs))
        return results

    def _post(self, path: str, body: dict) -> dict:
        if path == "/currencies":
            currency_id = self._alloc_id()
            d = {
                "currency_id": currency_id,
                "symbol": body.get("symbol", ""),
                "name": None,
                "decimals": 6,
                "date_modified": "",
                "date_created": "",
            }
            self._currencies.append(Currency(**d))
            return d
        return {}

    def _post_bulk(self, path: str, items: list[dict]) -> list[dict]:
        return []

    def _patch(self, path: str, params: dict, body: dict) -> dict:
        return {}

    def get_dry_run_data(self) -> dict:
        return {
            "events": [dataclasses.asdict(e) for e in self._events],
            "securities": [dataclasses.asdict(s) for s in self._securities],
            "listings": [dataclasses.asdict(l) for l in self._listings],
            "event_contracts": [dataclasses.asdict(ec) for ec in self._event_contracts],
        }


class InMemoryAttemptStore:
    """The attempt store with expiry against a clock the test moves."""

    def __init__(self):
        self.clock = 0.0
        self._items: dict[int, tuple[Attempt, float]] = {}

    def now(self) -> float:
        return self.clock

    def get_many(self, security_ids: list[int]) -> dict[int, Attempt]:
        return {sid: held[0] for sid in security_ids if (held := self._items.get(sid)) and held[1] > self.clock}

    def put(self, security_id: int, attempt: Attempt, ttl_seconds: int) -> None:
        self._items[security_id] = (attempt, self.clock + ttl_seconds)

    def clear(self, security_ids: list[int]) -> None:
        for sid in security_ids:
            self._items.pop(sid, None)


class StubDB:
    """In-memory ClassifierDB that reads from a StubRegistry's internal state.

    Shares state with StubRegistry live — writes made via StubRegistry are
    immediately visible here, so tests can use both together without sync issues.
    """

    def __init__(self, registry: StubRegistry):
        self._r = registry

    def get_exchange_event(self, exchange_id: int, native_id: str) -> int | None:
        for ev in self._r._events:
            if ev.exchange_id == exchange_id and ev.native_event_id == native_id:
                return ev.event_id
        return None

    def get_exchange_events(self, keys: list[tuple[int, str]]) -> dict[tuple[int, str], int]:
        key_set = set(keys)
        return {
            (ev.exchange_id, ev.native_event_id): ev.event_id
            for ev in self._r._events
            if (ev.exchange_id, ev.native_event_id) in key_set
        }

    def get_events(self, event_ids: list[int]) -> dict[int, dict]:
        return {
            ev.event_id: {"title": ev.title, "category": ev.category or "OTHER", "tags": ev.tags or []}
            for ev in self._r._events
            if ev.event_id in event_ids
        }

    def get_currencies(self) -> dict[str, int]:
        return {c.symbol: c.currency_id for c in self._r._currencies}

    def get_unlisted_securities(self, symbols: list[str]) -> dict[str, int]:
        sym_set = set(symbols)
        listed = {l.security_id for l in self._r._listings}
        return {
            s.symbol: s.security_id for s in self._r._securities
            if s.symbol in sym_set and s.security_id not in listed
        }

    def get_all_security_ids(self) -> set[int]:
        return {s.security_id for s in self._r._securities if s.active}

    def get_existing_listings(self, keys: list[tuple[int, str]]) -> dict[tuple[int, str], tuple[int, int]]:
        key_set = set(keys)
        return {
            (l.exchange_id, l.exchange_security_id): (l.listing_id, l.security_id)
            for l in self._r._listings
            if (l.exchange_id, l.exchange_security_id) in key_set
        }

    def securities_listed_more_than_once(self, security_ids: list[int]) -> set[int]:
        counts = Counter(l.security_id for l in self._r._listings if l.security_id in set(security_ids))
        return {sid for sid, n in counts.items() if n > 1}

    def _modified_within(self, listing: Listing, max_age: timedelta) -> bool:
        modified_at = self._r._listing_modified_at.get(listing.listing_id)
        return modified_at is None or datetime.now(timezone.utc) - modified_at < max_age

    def get_unsettled_contracts(
        self, exchange_id: int, max_age: timedelta, after_id: int, limit: int, include_active: bool = False,
    ) -> list[tuple[int, str]]:
        eligible = {l.security_id: l.exchange_security_id for l in self._r._listings
                    if l.exchange_id == exchange_id
                    and ((not l.active and self._modified_within(l, max_age)) or (l.active and include_active))}
        return sorted(
            (ec.event_contract_id, eligible[ec.security_id]) for ec in self._r._event_contracts
            if ec.security_id in eligible and ec.event_contract_id not in self._r._settlement_by_event_contract
            and ec.event_contract_id > after_id
        )[:limit]

    def get_unsettled_event_contracts(
        self, exchange_id: int, exchange_security_ids: list[str],
    ) -> dict[str, tuple[int, int]]:
        wanted = set(exchange_security_ids)
        id_by_security = {l.security_id: l.exchange_security_id for l in self._r._listings
                          if l.exchange_id == exchange_id and l.exchange_security_id in wanted}
        return {
            id_by_security[ec.security_id]: (ec.event_contract_id, ec.event_id)
            for ec in self._r._event_contracts
            if ec.security_id in id_by_security
            and ec.event_contract_id not in self._r._settlement_by_event_contract
        }

    def get_unresolved_event_settlements(self, event_ids: list[int]) -> dict[int, dict[int, bool]]:
        unresolved = {ev.event_id for ev in self._r._events if not ev.resolved and ev.event_id in event_ids}
        result: dict[int, dict[int, bool]] = {}
        for ec in self._r._event_contracts:
            if ec.event_id in unresolved:
                result.setdefault(ec.event_id, {})[ec.event_contract_id] = (
                    ec.event_contract_id in self._r._settlement_by_event_contract)
        return result

    def get_retired_unsettled_listings(self, exchange_id: int, max_age: timedelta) -> list[tuple[int, int, str]]:
        unsettled = {ec.security_id for ec in self._r._event_contracts
                     if ec.event_contract_id not in self._r._settlement_by_event_contract}
        return [
            (l.listing_id, l.security_id, l.exchange_security_id) for l in self._r._listings
            if l.exchange_id == exchange_id and not l.active and l.security_id in unsettled
            and self._modified_within(l, max_age)
        ]

    def get_resolved_event_ids_for_securities(self, security_ids: list[int]) -> set[int]:
        resolved = {ev.event_id for ev in self._r._events if ev.resolved}
        wanted = set(security_ids)
        return {ec.event_id for ec in self._r._event_contracts if ec.security_id in wanted and ec.event_id in resolved}

    def get_event_ids_by_security(self, security_ids: list[int]) -> dict[int, int]:
        id_set = set(security_ids)
        return {ec.security_id: ec.event_id for ec in self._r._event_contracts if ec.security_id in id_set}

    def get_existing_listing_specs(self, listing_ids: list[int]) -> dict[int, tuple[int, int, int, int, int]]:
        id_set = set(listing_ids)
        seen: dict[int, tuple[int, int, int, int, int]] = {}
        for s in self._r._listing_specs:
            if s.listing_id in id_set:
                min_size = self._r._min_size_by_spec_id.get(s.id, 0)
                seen[s.listing_id] = (s.tick_size, s.lot_size, s.min_notional, s.contract_multiplier, min_size)
        return seen

    def get_listing_fee_keys(self, listing_ids: list[int]) -> dict[int, tuple[str | None, str | None]]:
        return {lid: self._r._fee_keys_by_listing.get(lid, (None, None))
                for lid in listing_ids if any(l.listing_id == lid for l in self._r._listings)}

    def get_fee_source_refs(self, exchange_id: int) -> set[str]:
        return {row["source_ref"] for row in self._r._fee_schedules
                if row.get("exchange_id") == exchange_id and row.get("source_ref")}

    def get_unresolved_events(self) -> list[Event]:
        return self._r.get_event(resolved=False)

    def get_all_event_contracts(self) -> list[EventContract]:
        resolved_ids = {ev.event_id for ev in self._r._events if ev.resolved}
        return [ec for ec in self._r._event_contracts if ec.event_id not in resolved_ids]

    def get_all_securities(self) -> list[Security]:
        return [s for s in self._r._securities if s.active]

    def get_securities_with_active_listings(self, security_ids: list[int]) -> set[int]:
        sid_set = set(security_ids)
        return {
            l.security_id for l in self._r._listings
            if l.security_id in sid_set and l.active
        }

    def get_active_listings_for_securities(
        self, security_ids: list[int],
    ) -> list[tuple[int, int, int, str]]:
        sid_set = set(security_ids)
        return [
            (l.listing_id, l.security_id, l.exchange_id, l.exchange_security_id)
            for l in self._r._listings
            if l.security_id in sid_set and l.active
        ]

    def get_all_active_listings(self) -> list:
        return [l for l in self._r._listings if l.active]

    def get_event_ids_for_securities(self, security_ids: list[int]) -> set[int]:
        sid_set = set(security_ids)
        resolved_ids = {ev.event_id for ev in self._r._events if ev.resolved}
        return {
            ec.event_id for ec in self._r._event_contracts
            if ec.security_id in sid_set and ec.event_id not in resolved_ids
        }

    def get_event_contracts_for_events(self, event_ids: list[int]) -> list:
        id_set = set(event_ids)
        resolved_ids = {ev.event_id for ev in self._r._events if ev.resolved}
        return [
            ec for ec in self._r._event_contracts
            if ec.event_id in id_set and ec.event_id not in resolved_ids
        ]

    def get_active_listings_by_exchange_security(
        self, exchange_id: int, exchange_security_ids: list[str],
    ) -> list[tuple[int, int, str]]:
        id_set = set(exchange_security_ids)
        return [
            (l.listing_id, l.security_id, l.exchange_security_id)
            for l in self._r._listings
            if l.exchange_id == exchange_id
            and l.exchange_security_id in id_set
            and l.active
        ]

    def get_event_ids_for_security(self, security_id: int) -> list[int]:
        return [ec.event_id for ec in self._r._event_contracts if ec.security_id == security_id]

    def get_active_security_count_for_event(self, event_id: int) -> int:
        security_ids = {
            ec.security_id for ec in self._r._event_contracts if ec.event_id == event_id
        }
        return sum(1 for s in self._r._securities if s.security_id in security_ids and s.active)

    def get_all_currencies(self) -> list[Currency]:
        return self._r.get_currency()

    def get_security_ids_for_events(self, event_ids: list[int]) -> set[int]:
        id_set = set(event_ids)
        return {ec.security_id for ec in self._r._event_contracts if ec.event_id in id_set}

    def _exchange_code(self, exchange_id: int) -> str:
        return next(e.exchange_code for e in self._r.get_exchange() if e.exchange_id == exchange_id)

    def get_extraction_inputs(self, security_ids: list[int]) -> list[ExtractionInput]:
        out = []
        for sid in security_ids:
            stored = self._r._contract_sources.get(sid)
            ec = next((c for c in self._r._event_contracts if c.security_id == sid), None)
            event = next((e for e in self._r._events if ec and e.event_id == ec.event_id), None)
            security = next((x for x in self._r._securities if x.security_id == sid), None)
            if stored is None or event is None or event.resolved or security is None:
                continue
            out.append(ExtractionInput(
                security_id=sid, exchange=self._exchange_code(stored["exchange_id"]), event_id=event.event_id,
                contract_type=ContractType(security.contract_type), event_title=event.title,
                outcome_label=ec.outcome_label, description=event.description, expiry=event.expiry,
                source=stored["source"], source_hash=stored["source_hash"],
            ))
        return out

    def sample_extraction_inputs(self, limit: int) -> list[ExtractionInput]:
        return self.get_extraction_inputs(sorted(self._r._contract_sources))[:limit]

    def securities_needing_propositions(self, extractor_version: str, limit: int, after: int = 0) -> list[int]:
        open_sids = {i.security_id for i in self.get_extraction_inputs(sorted(self._r._contract_sources))}
        due = [sid for sid in sorted(open_sids)
               if sid > after and ((p := self._r._propositions.get(sid)) is None
               or (p["source_hash"], p["extractor_version"]) != (self._r._contract_sources[sid]["source_hash"], extractor_version))]
        return due[:limit]

    def get_proposition_versions(self, security_ids: list[int]) -> dict[int, tuple[str, str]]:
        return {sid: (p["source_hash"], p["extractor_version"]) for sid, p in self._r._propositions.items()
                if sid in set(security_ids)}

    def get_stored_propositions(self, security_ids: list[int]) -> dict[int, tuple[Proposition, str, str]]:
        out = {}
        for sid in security_ids:
            p = self._r._propositions.get(sid)
            if p is not None:
                key = self._r._subject_by_id(p["extracted_subject_id"])["key"]
                out[sid] = (Proposition.from_fields(key, p["fields"]), p["method"], p["source_hash"])
        return out

    def dirty_subjects(self, limit: int) -> list[int]:
        due = [s for s in self._r._subjects.values() if s["derived_at"] is None or s["date_modified"] > s["derived_at"]]
        return [s["subject_id"] for s in sorted(due, key=lambda s: s["date_modified"])][:limit]

    def _complement(self, security_id: int) -> int | None:
        ec = next((c for c in self._r._event_contracts if c.security_id == security_id), None)
        if ec is None:
            return None
        siblings = [c.security_id for c in self._r._event_contracts if c.event_id == ec.event_id]
        binary = {x.security_id for x in self._r._securities if x.contract_type == ContractType.BINARY}
        if len(siblings) != 2 or not set(siblings) <= binary:
            return None
        return next(sid for sid in siblings if sid != security_id)

    def get_subject_contracts(self, subject_id: int) -> tuple[list[Contract], set[int], datetime]:
        subject = self._r._subject_by_id(subject_id)
        open_sids = {i.security_id for i in self.get_extraction_inputs(sorted(self._r._contract_sources))}
        contracts, changed = [], set()
        for sid, p in sorted(self._r._propositions.items()):
            source = self._r._contract_sources.get(sid)
            if p["subject_id"] != subject_id or sid not in open_sids or source["source_hash"] != p["source_hash"]:
                continue
            contracts.append(Contract(sid, self._exchange_code(source["exchange_id"]),
                                      Proposition.from_fields(subject["key"], p["fields"]), self._complement(sid)))
            if subject["derived_at"] is None or self._r._proposition_modified[sid] > subject["derived_at"]:
                changed.add(sid)
        return contracts, changed, datetime(2026, 1, 1, tzinfo=timezone.utc)

    def nearest_subjects(self, embedding: list[float], limit: int) -> list[KnownSubject]:
        def cos(v):
            dot = sum(a * b for a, b in zip(v, embedding))
            norm = math.sqrt(sum(a * a for a in v)) * math.sqrt(sum(b * b for b in embedding))
            return dot / norm if norm else 0.0
        stored = [s for s in self._r._subjects.values() if s["embedding"] is not None]
        ranked = sorted(stored, key=lambda s: -cos(s["embedding"]))[:limit]
        return [KnownSubject(s["key"], s["description"]) for s in ranked]

    def subjects_without_embeddings(self, limit: int = 1000) -> list[KnownSubject]:
        return [KnownSubject(s["key"], s["description"]) for s in self._r._subjects.values() if s["embedding"] is None][:limit]

    def reset_derivations(self) -> int:
        derived = [s for s in self._r._subjects.values() if s["derived_at"] is not None]
        for subject in derived:
            subject["derived_at"] = None
        return len(derived)

    def put_subject_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        for key, vector in embeddings.items():
            self._r._subjects[key]["embedding"] = vector

    def get_active_exchange_native_ids(self) -> dict[int, set[str]]:
        result: dict[int, set[str]] = {}
        for ev in self._r._events:
            if not ev.resolved and ev.native_event_id is not None:
                result.setdefault(ev.exchange_id, set()).add(ev.native_event_id)
        return result
