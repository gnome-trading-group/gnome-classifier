import json
import logging
from datetime import datetime, timezone, timedelta

import requests.exceptions

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.adapters.types import AdapterContract
from classifier.client.http import RateLimitedSession
from classifier.types import ExchangeId

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"
PAGE_SIZE = 500

CONTRACT_MULTIPLIER = 1e9
TICK_SIZE = 10_000_000
LOT_SIZE = 1_000_000


class PolymarketAdapter:
    exchange_name = "polymarket"

    def __init__(self, session: RateLimitedSession | None = None):
        self._session = session or RateLimitedSession(min_request_interval=0.1)

    def fetch(self, exchange_id: ExchangeId) -> list[AdapterContract]:
        events = self._fetch_all_events()
        contracts: list[AdapterContract] = []
        for event in events:
            contracts.extend(self._map_event(exchange_id, event))
        return contracts

    def fetch_resolved(self, exchange_id: ExchangeId, lookback_days: int) -> set[str]:
        resolved: set[str] = set()

        for event in self._fetch_closed_events(lookback_days):
            for market in event.get("markets", []):
                condition_id = market.get("conditionId", "")
                if not condition_id:
                    continue
                raw_token_ids = market.get("clobTokenIds", "[]")
                try:
                    token_ids = raw_token_ids if isinstance(raw_token_ids, list) else json.loads(raw_token_ids)
                except Exception:
                    continue
                for token_id in token_ids:
                    resolved.add(f"{condition_id}:{token_id}")

        for event in self._fetch_all_events():
            for market in event.get("markets", []):
                if not market.get("closed", False):
                    continue
                condition_id = market.get("conditionId", "")
                if not condition_id:
                    continue
                raw_token_ids = market.get("clobTokenIds", "[]")
                try:
                    token_ids = raw_token_ids if isinstance(raw_token_ids, list) else json.loads(raw_token_ids)
                except Exception:
                    continue
                for token_id in token_ids:
                    resolved.add(f"{condition_id}:{token_id}")

        return resolved

    def _fetch_closed_events(self, lookback_days: int) -> list[dict]:
        since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        events: list[dict] = []
        after_cursor: str | None = None
        while True:
            params: dict = {
                "active": "false",
                "closed": "true",
                "end_date_min": since,
                "limit": PAGE_SIZE,
            }
            if after_cursor is not None:
                params["after_cursor"] = after_cursor
            try:
                res = self._session.get(f"{GAMMA_API_URL}/events/keyset", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.RetryError as e:
                logger.error("Polymarket closed events retries exhausted at cursor=%s: %s", after_cursor, e)
                break
            except requests.exceptions.RequestException as e:
                logger.error("Polymarket closed events API error at cursor=%s: %s", after_cursor, e)
                break
            events.extend(data.get("events", []))
            after_cursor = data.get("next_cursor")
            if after_cursor is None:
                break
        return events

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
                res = self._session.get(f"{GAMMA_API_URL}/events/keyset", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.RetryError as e:
                logger.error("Polymarket API retries exhausted at cursor=%s: %s", after_cursor, e)
                break
            except requests.exceptions.RequestException as e:
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
        event_title = event.get("title", "")
        event_slug = event.get("slug", "")

        is_neg_risk_group = len(markets) > 1 and all(m.get("negRisk") for m in markets)

        if is_neg_risk_group:
            event_volume: float = sum(m.get("volumeNum") or 0.0 for m in markets)
            return self._map_neg_risk_group(
                exchange_id, markets, event_title, event_slug, event_description, event_volume,
            )

        contracts: list[AdapterContract] = []
        for market in markets:
            market_volume: float = market.get("volumeNum") or 0.0
            contracts.extend(self._map_binary_market(exchange_id, market, event_description, event_slug, market_volume))
        return contracts

    def _map_neg_risk_group(
        self,
        exchange_id: ExchangeId,
        markets: list[dict],
        event_title: str,
        event_slug: str,
        event_description: str | None,
        event_volume: float = 0.0,
    ) -> list[AdapterContract]:
        symbol_base = f"{event_title[:60]} -- "
        contracts: list[AdapterContract] = []
        for market in markets:
            condition_id = market.get("conditionId", "")
            if not condition_id:
                continue
            raw_token_ids = market.get("clobTokenIds", "[]")
            try:
                token_ids = raw_token_ids if isinstance(raw_token_ids, list) else json.loads(raw_token_ids)
            except Exception:
                continue
            if not token_ids:
                continue
            yes_token_id = token_ids[0]
            outcome_label = market.get("groupItemTitle") or market.get("question", "")
            contracts.append(AdapterContract(
                exchange_id=exchange_id,
                exchange_security_id=f"{condition_id}:{yes_token_id}",
                exchange_security_symbol=f"{symbol_base}{outcome_label}"[:100],
                base_currency="USDC",
                quote_currency="USDC",
                settle_currency="USDC",
                security_type=SecurityType.EVENT_CONTRACT,
                contract_type=ContractType.MULTI_OUTCOME,
                asset_class=AssetClass.PREDICTION,
                inverse=False,
                is_quanto=False,
                tick_size=TICK_SIZE,
                lot_size=LOT_SIZE,
                min_notional=0.0,
                contract_multiplier=CONTRACT_MULTIPLIER,
                event_title=event_title,
                outcome_label=outcome_label,
                event_description=event_description,
                event_category=None,
                event_expiry=market.get("endDate"),
                exchange_event_native_id=event_slug,
                exchange_event_native_url=f"https://polymarket.com/event/{event_slug}",
                event_volume=event_volume,
            ))
        return contracts

    def _map_binary_market(
        self,
        exchange_id: ExchangeId,
        market: dict,
        event_description: str | None,
        event_slug: str = "",
        event_volume: float = 0.0,
    ) -> list[AdapterContract]:
        question = market.get("question", "")
        condition_id = market.get("conditionId", "")
        if not condition_id:
            return []
        expiry = market.get("endDate")

        raw_outcomes = market.get("outcomes", "[]")
        raw_token_ids = market.get("clobTokenIds", "[]")
        try:
            outcomes = raw_outcomes if isinstance(raw_outcomes, list) else json.loads(raw_outcomes)
            token_ids = raw_token_ids if isinstance(raw_token_ids, list) else json.loads(raw_token_ids)
        except Exception:
            return []

        if not outcomes or not token_ids:
            return []

        is_binary = len(outcomes) == 2
        contract_type = ContractType.BINARY if is_binary else ContractType.MULTI_OUTCOME

        native_url = f"https://polymarket.com/event/{event_slug}" if event_slug else None
        contracts = []
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
                exchange_event_native_url=native_url,
                event_volume=event_volume,
            ))
        return contracts
