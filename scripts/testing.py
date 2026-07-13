import dataclasses
import json
import math
from unittest.mock import MagicMock

import anthropic

from gnomepy.registry import RegistryClient
from gnomepy.registry.types import (
    ContractRelationship,
    Currency,
    Event,
    EventContract,
    Exchange,
    ExchangeEvent,
    Listing,
    ListingSpec,
    Security,
)


def no_op_anthropic_client() -> anthropic.Anthropic:
    """Mock Anthropic client that echoes raw event titles with category=OTHER, tags=[]."""
    def _fake_create(*args, **kwargs):
        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        content = messages[0].get("content", "") if messages else ""
        titles = []
        for line in content.splitlines():
            if line.startswith("[") and "] Title: " in line:
                title = line.split("] Title: ", 1)[1].split(" | ")[0].strip()
                titles.append({"title": title, "category": "OTHER", "tags": []})
            elif line.startswith("Exchange-provided title:"):
                title = line.split(":", 1)[1].strip()
                titles.append({"title": title, "category": "OTHER", "tags": []})
        if not titles:
            text = json.dumps([])
        elif len(titles) == 1:
            text = json.dumps(titles[0])
        else:
            text = json.dumps(titles)
        mock_content = MagicMock()
        mock_content.text = text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = _fake_create
    return client


def no_op_voyage_client():
    """Mock Voyage client that returns zero embeddings."""
    client = MagicMock()
    result = MagicMock()
    result.embeddings = []
    client.embed.return_value = result
    return client


class StubRegistry(RegistryClient):
    """In-memory registry that simulates empty DB state. All writes are stored locally."""

    def __init__(self):
        self._next_id = 1
        self._events: list[Event] = []
        self._securities: list[Security] = []
        self._listings: list[Listing] = []
        self._listing_specs: list[ListingSpec] = []
        self._event_contracts: list[EventContract] = []
        self._exchange_events: list[ExchangeEvent] = []
        self._contract_relationships: list[ContractRelationship] = []
        self._currencies: list[Currency] = []

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def get_exchange(self) -> list[Exchange]:
        return [
            Exchange(exchange_id=1, exchange_name="polymarket", region="", schema_type="", date_modified="", date_created=""),
            Exchange(exchange_id=2, exchange_name="kalshi", region="", schema_type="", date_modified="", date_created=""),
            Exchange(exchange_id=3, exchange_name="hyperliquid", region="", schema_type="", date_modified="", date_created=""),
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

    def get_contract_relationships(self) -> list[ContractRelationship]:
        return list(self._contract_relationships)

    def get_exchange_events(self) -> list[ExchangeEvent]:
        return list(self._exchange_events)

    def bulk_create_events(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            event_id = self._alloc_id()
            d = {
                "event_id": event_id,
                "title": item.get("title", ""),
                "description": item.get("description"),
                "category": item.get("category"),
                "resolution_source": None,
                "tags": item.get("tags"),
                "resolved": False,
                "resolved_at": None,
                "expiry": item.get("expiry"),
                "date_modified": "",
                "date_created": "",
            }
            self._events.append(Event(**d))
            results.append(d)
        return results

    def bulk_create_securities(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
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
                "date_modified": "",
                "date_created": "",
            }
            self._listings.append(Listing(**d))
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
            results.append(d)
        return results

    def bulk_create_exchange_events(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            ee_id = self._alloc_id()
            d = {
                "exchange_event_id": ee_id,
                "exchange_id": item["exchange_id"],
                "event_id": item["event_id"],
                "native_event_id": item["native_event_id"],
                "raw_title": item.get("raw_title", ""),
                "date_created": "",
            }
            self._exchange_events.append(ExchangeEvent(**d))
            results.append(d)
        return results

    def bulk_create_contract_relationships(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            rel_id = self._alloc_id()
            d = {
                "relationship_id": rel_id,
                "security_id_a": item["security_id_a"],
                "security_id_b": item["security_id_b"],
                "relationship_type": item["relationship_type"],
                "confidence": item["confidence"],
                "method": item["method"],
                "reviewed": False,
                "reviewed_at": None,
                "date_created": "",
            }
            self._contract_relationships.append(ContractRelationship(**d))
            results.append(d)
        return results

    def patch_event_contract(self, event_contract_id: int, **kwargs) -> dict:
        for i, ec in enumerate(self._event_contracts):
            if ec.event_contract_id == event_contract_id:
                updated = dataclasses.replace(ec, **kwargs)
                self._event_contracts[i] = updated
                return dataclasses.asdict(updated)
        return {}

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
            "relationships": [dataclasses.asdict(r) for r in self._contract_relationships],
        }


class StubDB:
    """In-memory ClassifierDB that reads from a StubRegistry's internal state.

    Shares state with StubRegistry live — writes made via StubRegistry are
    immediately visible here, so tests can use both together without sync issues.
    """

    def __init__(self, registry: StubRegistry):
        self._r = registry
        self._embeddings: dict[int, list[float]] = {}

    def get_exchange_event(self, exchange_id: int, native_id: str) -> int | None:
        for ee in self._r._exchange_events:
            if ee.exchange_id == exchange_id and ee.native_event_id == native_id:
                return ee.event_id
        return None

    def get_all_exchange_events(self) -> dict[tuple[int, str], int]:
        return {(ee.exchange_id, ee.native_event_id): ee.event_id for ee in self._r._exchange_events}

    def get_events(self, event_ids: list[int]) -> dict[int, dict]:
        return {
            ev.event_id: {"title": ev.title, "category": ev.category or "OTHER", "tags": ev.tags or []}
            for ev in self._r._events
            if ev.event_id in event_ids
        }

    def get_events_for_dedup(self) -> list[tuple[str, str | None, int]]:
        return [(ev.title, ev.expiry, ev.event_id) for ev in self._r._events]

    def get_currencies(self) -> dict[str, int]:
        return {c.symbol: c.currency_id for c in self._r._currencies}

    def get_existing_securities(self, symbols: list[str]) -> dict[str, int]:
        sym_set = set(symbols)
        return {s.symbol: s.security_id for s in self._r._securities if s.symbol in sym_set}

    def get_all_security_ids(self) -> set[int]:
        return {s.security_id for s in self._r._securities}

    def get_existing_listings(self, keys: list[tuple[int, str]]) -> dict[tuple[int, str], int]:
        key_set = set(keys)
        return {
            (l.exchange_id, l.exchange_security_id): l.listing_id
            for l in self._r._listings
            if (l.exchange_id, l.exchange_security_id) in key_set
        }

    def get_existing_event_contracts(self, keys: list[tuple[int, int]]) -> set[tuple[int, int]]:
        key_set = set(keys)
        return {
            (ec.event_id, ec.security_id) for ec in self._r._event_contracts
            if (ec.event_id, ec.security_id) in key_set
        }

    def get_existing_listing_specs(self, listing_ids: list[int]) -> set[int]:
        id_set = set(listing_ids)
        return {s.listing_id for s in self._r._listing_specs if s.listing_id in id_set}

    def get_unresolved_events(self) -> list[Event]:
        return self._r.get_event(resolved=False)

    def get_all_event_contracts(self) -> list[EventContract]:
        return self._r.get_event_contracts()

    def get_all_securities(self) -> list[Security]:
        return self._r.get_security()

    def get_all_currencies(self) -> list[Currency]:
        return self._r.get_currency()

    def get_contract_relationships_for_securities(self, security_ids: list[int]) -> list[ContractRelationship]:
        sid_set = set(security_ids)
        return [
            r for r in self._r._contract_relationships
            if r.security_id_a in sid_set or r.security_id_b in sid_set
        ]

    def find_neighbors(
        self, embedding: list[float], threshold: float, limit: int = 50
    ) -> list[tuple[int, float]]:
        results = []
        for eid, emb in self._embeddings.items():
            dot = sum(x * y for x, y in zip(embedding, emb))
            denom = math.sqrt(sum(x * x for x in embedding)) * math.sqrt(sum(x * x for x in emb))
            sim = dot / denom if denom else 0.0
            if sim >= threshold:
                results.append((eid, sim))
        return sorted(results, key=lambda x: -x[1])[:limit]

    def get_embeddings(self, event_ids: list[int]) -> dict[int, list[float]]:
        return {eid: self._embeddings[eid] for eid in event_ids if eid in self._embeddings}

    def put_embeddings(self, embeddings: dict[int, list[float]]) -> None:
        self._embeddings.update(embeddings)
