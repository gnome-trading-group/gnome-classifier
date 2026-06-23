import dataclasses

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

    def get_event(self) -> list[Event]:
        return list(self._events)

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
                "embedding": item.get("embedding"),
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
