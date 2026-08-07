import logging
import re
from datetime import datetime, timedelta, timezone

import requests.exceptions

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.adapters.types import AdapterContract
from classifier.client.http import RateLimitedSession
from classifier.types import ExchangeId

logger = logging.getLogger(__name__)

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
PAGE_SIZE = 200

CONTRACT_MULTIPLIER = 1_000_000_000
TICK_SIZE = 1_000_000
LOT_SIZE = 1_000_000


def _slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    slug = re.sub(r'-+', '-', slug)
    return slug


class KalshiAdapter:
    exchange_name = "kalshi"

    def __init__(self, session: RateLimitedSession | None = None):
        self._session = session or RateLimitedSession(min_request_interval=0.15)

    def fetch(self, exchange_id: ExchangeId) -> list[AdapterContract]:
        events = self._fetch_active_events()
        contracts: list[AdapterContract] = []
        for event in events:
            contracts.extend(self._map_event(exchange_id, event))
        return contracts

    def fetch_resolved(self, exchange_id: ExchangeId, lookback_days: int) -> set[str]:
        resolved: set[str] = set()

        for event in self._fetch_settled_events(lookback_days):
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

        for event in self._fetch_active_events():
            markets = event.get("markets", [])
            is_multi = event.get("mutually_exclusive", False) and len(markets) > 1
            for market in markets:
                if market.get("status", "active") == "active":
                    continue
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
                res = self._session.get(f"{BASE_URL}/events", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.RetryError as e:
                logger.error("Kalshi settled API retries exhausted: %s", e)
                break
            except requests.exceptions.RequestException as e:
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
                res = self._session.get(f"{BASE_URL}/events", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.RetryError as e:
                logger.error("Kalshi API retries exhausted: %s", e)
                break
            except requests.exceptions.RequestException as e:
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

        if is_multi:
            event_description = markets[0].get("rules_secondary") or markets[0].get("rules_primary") or event_description

        has_sub_markets = not is_multi and len(markets) > 1

        series_ticker = event.get("series_ticker", "")
        sub_title = event.get("sub_title", "")
        if series_ticker:
            slug = _slugify(f"{event_title} {sub_title}".strip())
            native_url: str | None = f"https://kalshi.com/markets/{series_ticker.lower()}/{slug}"
        else:
            native_url = None

        def _parse_volume(market: dict) -> float | None:
            raw = market.get("volume_fp")
            if raw is None:
                return None
            try:
                return float(raw)
            except (ValueError, TypeError):
                return None

        event_volume: float | None = None
        if not has_sub_markets:
            vols = [v for m in markets if (v := _parse_volume(m)) is not None]
            if vols:
                event_volume = sum(vols)

        contracts: list[AdapterContract] = []
        for market in markets:
            ticker = market.get("ticker", "")
            if not ticker:
                continue

            if market.get("status", "active") != "active":
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
                    exchange_event_native_url=native_url,
                    event_volume=event_volume,
                ))
            else:
                if has_sub_markets:
                    sub_title_market = market.get("yes_sub_title") or ticker
                    market_event_title = f"{event_title}: {sub_title_market}"
                    native_id = ticker
                    market_volume = _parse_volume(market)
                    market_description = market.get("rules_primary") or event_description
                else:
                    sub_title_single = market.get("yes_sub_title", "")
                    if sub_title_single and sub_title_single.lower() not in event_title.lower():
                        market_event_title = f"{event_title}: {sub_title_single}"
                    else:
                        market_event_title = event_title
                    native_id = event_ticker
                    market_volume = event_volume
                    market_description = market.get("rules_primary") or event_description
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
                        event_description=market_description,
                        event_category=event_category,
                        event_expiry=expiry,
                        exchange_event_native_id=native_id,
                        exchange_event_native_url=native_url,
                        event_volume=market_volume,
                    ))

        return contracts
