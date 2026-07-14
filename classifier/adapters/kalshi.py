import logging
from datetime import datetime, timedelta, timezone

import requests

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.adapters.types import AdapterContract
from classifier.types import ExchangeId

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
PAGE_SIZE = 200

CONTRACT_MULTIPLIER = 1_000_000_000
TICK_SIZE = 1_000_000
LOT_SIZE = 1_000_000


class KalshiAdapter:
    exchange_name = "kalshi"

    def fetch(self, exchange_id: ExchangeId) -> list[AdapterContract]:
        events = self._fetch_active_events()
        contracts: list[AdapterContract] = []
        for event in events:
            contracts.extend(self._map_event(exchange_id, event))
        return contracts

    def fetch_resolved(self, exchange_id: ExchangeId, lookback_days: int) -> set[str]:
        events = self._fetch_settled_events(lookback_days)
        resolved: set[str] = set()
        for event in events:
            markets = event.get("markets", [])
            is_multi = event.get("mutually_exclusive", False) and len(markets) > 1
            for market in markets:
                ticker = market.get("ticker", "")
                if not ticker:
                    continue
                if is_multi:
                    resolved.add(ticker)
                else:
                    resolved.add(f"{ticker}:yes")
                    resolved.add(f"{ticker}:no")
        return resolved

    def _fetch_settled_events(self, lookback_days: int) -> list[dict]:
        min_ts = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
        events: list[dict] = []
        cursor = ""
        while True:
            params: dict = {
                "with_nested_markets": "true",
                "status": "settled",
                "min_close_ts": min_ts,
                "limit": PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                res = requests.get(f"{BASE_URL}/events", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except Exception as e:
                logger.error("Kalshi settled API error: %s", e)
                break
            page = data.get("events", [])
            events.extend(page)
            cursor = data.get("cursor", "")
            if not cursor:
                break
        return events

    def _fetch_active_events(self) -> list[dict]:
        events: list[dict] = []
        cursor = ""
        while True:
            params: dict = {
                "with_nested_markets": "true",
                "status": "open",
                "limit": PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                res = requests.get(f"{BASE_URL}/events", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except Exception as e:
                logger.error("Kalshi API error: %s", e)
                break

            page = data.get("events", [])
            events.extend(page)

            cursor = data.get("cursor", "")
            if not cursor:
                break

        return events

    def _map_event(self, exchange_id: ExchangeId, event: dict) -> list[AdapterContract]:
        markets = event.get("markets", [])
        if not markets:
            return []

        event_title = event.get("title", "")
        event_description = event.get("sub_title") or None
        event_category = event.get("category") or None
        event_ticker = event.get("event_ticker", "")
        if not event_ticker:
            return []
        is_multi = event.get("mutually_exclusive", False) and len(markets) > 1

        has_sub_markets = not is_multi and len(markets) > 1

        contracts: list[AdapterContract] = []
        for market in markets:
            ticker = market.get("ticker", "")
            if not ticker:
                continue

            expiry = market.get("close_time") or market.get("expiration_time")

            if is_multi:
                outcome = market.get("yes_sub_title") or ticker
                exchange_security_symbol_base = f"{event_title[:60]} -- "
                contracts.append(AdapterContract(
                    exchange_id=exchange_id,
                    exchange_security_id=ticker,
                    exchange_security_symbol=f"{exchange_security_symbol_base}{outcome}"[:100],
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
                    outcome_label=outcome,
                    event_description=event_description,
                    event_category=event_category,
                    event_expiry=expiry,
                    exchange_event_native_id=event_ticker,
                ))
            else:
                if has_sub_markets:
                    sub_title = market.get("yes_sub_title") or ticker
                    market_event_title = f"{event_title}: {sub_title}"
                    native_id = ticker
                else:
                    market_event_title = event_title
                    native_id = event_ticker
                exchange_security_symbol_base = f"{market_event_title[:60]} -- "
                for side in ("Yes", "No"):
                    contracts.append(AdapterContract(
                        exchange_id=exchange_id,
                        exchange_security_id=f"{ticker}:{side.lower()}",
                        exchange_security_symbol=f"{exchange_security_symbol_base}{side}"[:100],
                        base_currency="USDC",
                        quote_currency="USDC",
                        settle_currency="USDC",
                        security_type=SecurityType.EVENT_CONTRACT,
                        contract_type=ContractType.BINARY,
                        asset_class=AssetClass.PREDICTION,
                        inverse=False,
                        is_quanto=False,
                        tick_size=TICK_SIZE,
                        lot_size=LOT_SIZE,
                        min_notional=0.0,
                        contract_multiplier=CONTRACT_MULTIPLIER,
                        event_title=market_event_title,
                        outcome_label=side,
                        event_description=event_description,
                        event_category=event_category,
                        event_expiry=expiry,
                        exchange_event_native_id=native_id,
                    ))

        return contracts
