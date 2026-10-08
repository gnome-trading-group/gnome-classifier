import itertools
import logging
import re
from datetime import datetime, timedelta, timezone

import requests.exceptions

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.adapters.settlement import PRICE_SCALE, chunked, to_price
from classifier.adapters.types import AdapterContract
from classifier.client.http import RateLimitedSession
from classifier.types import ExchangeId
from classifier.utils import format_security_symbol

logger = logging.getLogger(__name__)

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
PAGE_SIZE = 200
# Tickers per /markets lookup when collecting settlements.
SETTLEMENT_BATCH_SIZE = 100
# Kalshi marks a market determined, then may amend or dispute it; only finalized values are paid out.
FINAL_STATUS = "finalized"

CONTRACT_MULTIPLIER = 1_000_000_000
SIZE_SCALE = 1_000_000
TICK_SIZE = 10_000_000
LOT_SIZE = 10_000


def _market_tick_size(market: dict) -> int:
    ranges = market.get("price_ranges")
    if not ranges:
        return TICK_SIZE
    try:
        steps = [float(r["step"]) for r in ranges if "step" in r]
        if not steps:
            return TICK_SIZE
        # Tapered markets tick finer at the edges than in the middle; the listing spec holds one tick,
        # so use the coarsest step, which every band accepts, rather than one the middle band rejects.
        return round(max(steps) * CONTRACT_MULTIPLIER)
    except (ValueError, TypeError):
        return TICK_SIZE


def _dollar_volume_24h(market: dict) -> float:
    # Kalshi reports volume in contracts; contracts x last price approximates the one-sided dollar
    # notional that Polymarket and Hyperliquid report, so one volume threshold fits all three.
    try:
        return float(market.get("volume_24h_fp") or 0) * float(market.get("last_price_dollars") or 0)
    except (TypeError, ValueError):
        return 0.0


def _is_multi_outcome(event: dict) -> bool:
    # Only the flag decides: deciding on the market count too made a market change type and identity whenever
    # Kalshi added or removed a sibling. A mutually exclusive event with one market is a multi-outcome event with a
    # single outcome listed so far.
    return bool(event.get("mutually_exclusive", False))


def _slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    slug = re.sub(r'-+', '-', slug)
    return slug


class KalshiAdapter:
    exchange_code = "KALSHI"
    symbol_prefix = "KX"

    def __init__(self, session: RateLimitedSession | None = None):
        self._session = session or RateLimitedSession(min_request_interval=0.15)

    def fetch(self, exchange_id: ExchangeId):
        cursor = ""
        while True:
            params: dict = {"with_nested_markets": "true", "status": "open", "limit": PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            try:
                res = self._session.get(f"{BASE_URL}/events", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.RetryError as e:
                logger.error("Kalshi API retries exhausted: %s", e)
                raise
            except requests.exceptions.RequestException as e:
                logger.error("Kalshi API error: %s", e)
                raise
            page = [c for event in data.get("events", []) for c in self._map_event(exchange_id, event)]
            if page:
                yield page
            cursor = data.get("cursor", "")
            if not cursor:
                return

    def fetch_resolved(self, exchange_id: ExchangeId, lookback_days: int) -> set[str]:
        resolved: set[str] = set()

        # Kalshi lists events as settled while some of their markets are still trading, so
        # only each market's own status decides whether it resolved.
        for event in itertools.chain(self._fetch_settled_events(lookback_days), self._fetch_active_events()):
            markets = event.get("markets", [])
            is_multi = _is_multi_outcome(event)
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

    def fetch_settlements(self, exchange_security_ids: set[str]) -> dict[str, int]:
        ids_by_ticker: dict[str, list[str]] = {}
        for security_id in exchange_security_ids:
            ids_by_ticker.setdefault(security_id.partition(":")[0], []).append(security_id)

        settlements: dict[str, int] = {}
        for tickers in chunked(sorted(ids_by_ticker), SETTLEMENT_BATCH_SIZE):
            for market in self._fetch_markets(tickers):
                if market.get("status") != FINAL_STATUS:
                    continue
                yes_price = to_price(market.get("settlement_value_dollars"))
                if yes_price is None:
                    logger.warning("Kalshi %s finalized without a usable settlement value: %r",
                                   market.get("ticker"), market.get("settlement_value_dollars"))
                    continue
                for security_id in ids_by_ticker.get(market.get("ticker", ""), []):
                    # The NO listing trades in NO terms (the gateway inverts its orders), so it pays 1 - YES.
                    settlements[security_id] = PRICE_SCALE - yes_price if security_id.endswith(":no") else yes_price
        return settlements

    def _fetch_markets(self, tickers: list[str]) -> list[dict]:
        params = {"tickers": ",".join(tickers), "limit": len(tickers)}
        try:
            res = self._session.get(f"{BASE_URL}/markets", params=params, timeout=30)
            res.raise_for_status()
            return res.json().get("markets", [])
        except requests.exceptions.RetryError as e:
            logger.error("Kalshi markets retries exhausted: %s", e)
            raise
        except requests.exceptions.RequestException as e:
            logger.error("Kalshi markets API error: %s", e)
            raise

    def _fetch_settled_events(self, lookback_days: int):
        min_ts = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
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
                raise
            except requests.exceptions.RequestException as e:
                logger.error("Kalshi settled API error: %s", e)
                raise
            yield from data.get("events", [])
            cursor = data.get("cursor", "")
            if not cursor:
                return

    def _fetch_active_events(self):
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
                raise
            except requests.exceptions.RequestException as e:
                logger.error("Kalshi API error: %s", e)
                raise
            yield from data.get("events", [])
            cursor = data.get("cursor", "")
            if not cursor:
                return

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
        is_multi = _is_multi_outcome(event)

        if is_multi:
            event_description = markets[0].get("rules_secondary") or markets[0].get("rules_primary") or event_description

        series_ticker = event.get("series_ticker", "")
        sub_title = event.get("sub_title", "")
        if series_ticker:
            slug = _slugify(f"{event_title} {sub_title}".strip())
            native_url: str | None = f"https://kalshi.com/markets/{series_ticker.lower()}/{slug}"
        else:
            native_url = None

        event_volume = sum(_dollar_volume_24h(m) for m in markets)

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
                    tick_size=_market_tick_size(market),
                    lot_size=LOT_SIZE,
                    min_notional=0.0,
                    contract_multiplier=CONTRACT_MULTIPLIER,
                    event_title=event_title,
                    outcome_label=outcome,
                    event_description=event_description,
                    event_category=event_category,
                    event_expiry=expiry,
                    exchange_event_native_id=event_ticker,
                    security_symbol=format_security_symbol(self.symbol_prefix, ticker),
                    exchange_event_native_url=native_url,
                    event_volume=event_volume,
                ))
            else:
                # Each market of an event that isn't mutually exclusive is its own yes/no question, so it is its
                # own event, keyed by its market ticker however many siblings Kalshi lists alongside it.
                # Titles are only written when an event is created and never decide identity, so they may still
                # look at siblings: a lone market skips a sub-title its event title already names.
                market_sub_title = market.get("yes_sub_title", "")
                if len(markets) > 1:
                    market_event_title = f"{event_title}: {market_sub_title or ticker}"
                elif market_sub_title and market_sub_title.lower() not in event_title.lower():
                    market_event_title = f"{event_title}: {market_sub_title}"
                else:
                    market_event_title = event_title
                native_id = ticker
                market_volume = _dollar_volume_24h(market)
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
                        tick_size=_market_tick_size(market),
                        lot_size=LOT_SIZE,
                        min_notional=0.0,
                        contract_multiplier=CONTRACT_MULTIPLIER,
                        event_title=market_event_title,
                        outcome_label=side,
                        event_description=market_description,
                        event_category=event_category,
                        event_expiry=expiry,
                        exchange_event_native_id=native_id,
                        security_symbol=format_security_symbol(self.symbol_prefix, ticker, side),
                        exchange_event_native_url=native_url,
                        event_volume=market_volume,
                    ))

        return contracts
