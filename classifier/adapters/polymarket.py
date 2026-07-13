import json
import logging

import requests

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.adapters.types import AdapterContract
from classifier.types import ExchangeId

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"
PAGE_SIZE = 500

CONTRACT_MULTIPLIER = 1e9
TICK_SIZE = 10_000_000
LOT_SIZE = 1_000_000


class PolymarketAdapter:
    exchange_name = "polymarket"

    def fetch(self, exchange_id: ExchangeId) -> list[AdapterContract]:
        events = self._fetch_all_events()
        contracts: list[AdapterContract] = []
        for event in events:
            contracts.extend(self._map_event(exchange_id, event))
        return contracts

    def _fetch_all_events(self) -> list[dict]:
        events: list[dict] = []
        after_cursor: str | None = None
        while True:
            params: dict = {
                "active": "true",
                "closed": "false",
                "limit": PAGE_SIZE,
            }
            if after_cursor is not None:
                params["after_cursor"] = after_cursor

            try:
                res = requests.get(f"{GAMMA_API_URL}/events/keyset", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except Exception as e:
                logger.error("Polymarket API error at cursor=%s: %s", after_cursor, e)
                break

            events.extend(data.get("events", []))
            after_cursor = data.get("next_cursor")
            if after_cursor is None:
                break

        return events

    def _map_event(self, exchange_id: ExchangeId, event: dict) -> list[AdapterContract]:
        markets = [m for m in event.get("markets", []) if not m.get("closed", False)]
        if not markets:
            return []

        event_description = event.get("description") or None

        contracts: list[AdapterContract] = []
        for market in markets:
            question = market.get("question", "")
            condition_id = market.get("conditionId", "")
            if not condition_id:
                continue
            expiry = market.get("endDate")

            raw_outcomes = market.get("outcomes", "[]")
            raw_token_ids = market.get("clobTokenIds", "[]")
            try:
                outcomes = raw_outcomes if isinstance(raw_outcomes, list) else json.loads(raw_outcomes)
                token_ids = raw_token_ids if isinstance(raw_token_ids, list) else json.loads(raw_token_ids)
            except Exception:
                continue

            if not outcomes or not token_ids:
                continue

            is_binary = len(outcomes) == 2
            contract_type = ContractType.BINARY if is_binary else ContractType.MULTI_OUTCOME

            for outcome, token_id in zip(outcomes, token_ids):
                contracts.append(AdapterContract(
                    exchange_id=exchange_id,
                    exchange_security_id=f"{condition_id}:{token_id}",
                    exchange_security_symbol=f"{question[:60]} -- {outcome}",
                    base_currency="USDC",
                    quote_currency="USDC",
                    settle_currency="USDC",
                    security_type=SecurityType.EVENT_CONTRACT,
                    contract_type=contract_type,
                    asset_class=AssetClass.PREDICTION,
                    inverse=False,
                    is_quanto=False,
                    tick_size=TICK_SIZE,
                    lot_size=LOT_SIZE,
                    min_notional=0.0,
                    contract_multiplier=CONTRACT_MULTIPLIER,
                    event_title=question,
                    outcome_label=outcome,
                    event_description=event_description,
                    event_category=None,
                    event_expiry=expiry,
                    exchange_event_native_id=condition_id,
                ))

        return contracts
