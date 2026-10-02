import logging
import re
from datetime import datetime, timedelta, timezone

import requests.exceptions

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.adapters.types import AdapterContract
from classifier.client.http import RateLimitedSession
from classifier.types import ExchangeId
from classifier.utils import format_security_symbol

logger = logging.getLogger(__name__)

GATEWAY_URL = "https://gateway.polymarket.us/v1"
# The gateway silently caps pages at 500.
PAGE_SIZE = 500

CONTRACT_MULTIPLIER = 1_000_000_000
SIZE_SCALE = 1_000_000
TICK_SIZE = 10_000_000
LOT_SIZE = 10_000

MARKET_STATUS_OPEN = "MARKET_STATUS_OPEN"

# The gateway documents volume fields but never returns them; it does honour volumeNumMin/Max
# filters (shares). Querying disjoint ranges buckets each market by volume without the values.
VOLUME_BUCKETS_SHARES = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 1_000_000)
# Shares x an assumed mid price of 0.5 approximates the one-sided dollar volume the other
# adapters report, so the shared min_event_volume threshold applies across exchanges.
ASSUMED_PRICE = 0.5


def _market_tick_size(market: dict) -> int:
    raw = market.get("orderPriceMinTickSize")
    if raw is None:
        return TICK_SIZE
    try:
        return round(float(raw) * CONTRACT_MULTIPLIER)
    except (ValueError, TypeError):
        return TICK_SIZE


def _market_lot_size(market: dict) -> int:
    raw = market.get("minimumTradeQty")
    if raw is None:
        return LOT_SIZE
    try:
        return round(float(raw) * SIZE_SCALE)
    except (ValueError, TypeError):
        return LOT_SIZE


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _market_event_title(event_title: str, market: dict) -> str:
    # Moneyline markets repeat the event title with different punctuation ("A vs B" / "A vs. B").
    market_title = market.get("title") or ""
    if not market_title or _normalize(market_title) in _normalize(event_title):
        return event_title
    return f"{event_title}: {market_title}"


def _is_open(market: dict) -> bool:
    return (
        market.get("active", False)
        and not market.get("closed", False)
        and market.get("status") == MARKET_STATUS_OPEN
    )


class PolymarketUsAdapter:
    exchange_code = "POLYMARKET_US"
    symbol_prefix = "PM_US"

    def __init__(self, session: RateLimitedSession | None = None):
        # Public gateway limit is 20 req/s per IP.
        self._session = session or RateLimitedSession(min_request_interval=0.1)

    def fetch(self, exchange_id: ExchangeId):
        volume_by_slug = self._fetch_volume_buckets()
        for events in self._paginate("events", {"active": "true", "closed": "false"}):
            page = [c for event in events for c in self._map_event(exchange_id, event, volume_by_slug)]
            if page:
                yield page

    def fetch_resolved(self, exchange_id: ExchangeId, lookback_days: int) -> set[str]:
        resolved: set[str] = set()

        since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for markets in self._paginate("markets", {"closed": "true", "endDateMin": since}):
            for market in markets:
                resolved.update(self._security_ids(market))

        # Markets resolve early (eliminated teams) while their endDate is still in the future.
        for events in self._paginate("events", {"active": "true", "closed": "false"}):
            for event in events:
                for market in event.get("markets") or []:
                    if not _is_open(market):
                        resolved.update(self._security_ids(market))

        return resolved

    def _fetch_volume_buckets(self) -> dict[str, float]:
        volume_by_slug: dict[str, float] = {}
        bounds = (*VOLUME_BUCKETS_SHARES, None)
        for low, high in zip(bounds, bounds[1:]):
            params = {"active": "true", "closed": "false", "volumeNumMin": str(low)}
            if high is not None:
                params["volumeNumMax"] = str(high)
            for markets in self._paginate("markets", params):
                for market in markets:
                    slug = market.get("slug")
                    if slug:
                        volume_by_slug[slug] = low * ASSUMED_PRICE
        return volume_by_slug

    def _paginate(self, resource: str, params: dict):
        offset = 0
        while True:
            page_params = {**params, "limit": PAGE_SIZE, "offset": offset}
            try:
                res = self._session.get(f"{GATEWAY_URL}/{resource}", params=page_params, timeout=30)
                res.raise_for_status()
                items = res.json().get(resource, [])
            except requests.exceptions.RetryError as e:
                logger.error("Polymarket US %s retries exhausted at offset=%d: %s", resource, offset, e)
                raise
            except requests.exceptions.RequestException as e:
                logger.error("Polymarket US %s API error at offset=%d: %s", resource, offset, e)
                raise
            if items:
                yield items
            if len(items) < PAGE_SIZE:
                return
            offset += PAGE_SIZE

    @staticmethod
    def _security_ids(market: dict) -> set[str]:
        slug = market.get("slug")
        if not slug:
            return set()
        return {f"{slug}:long", f"{slug}:short"}

    def _map_event(self, exchange_id: ExchangeId, event: dict, volume_by_slug: dict[str, float]) -> list[AdapterContract]:
        event_title = event.get("title", "")
        event_slug = event.get("slug", "")
        if not event_slug:
            return []
        native_url = f"https://polymarket.us/event/{event_slug}"
        event_category = event.get("category") or None

        contracts: list[AdapterContract] = []
        for market in event.get("markets") or []:
            if not _is_open(market):
                continue
            slug = market.get("slug", "")
            sides = market.get("marketSides") or []
            if not slug or sorted(bool(s.get("long")) for s in sides) != [False, True]:
                continue

            market_event_title = _market_event_title(event_title, market)
            symbol_base = f"{market_event_title[:60]} -- "
            for side in sides:
                # Each market is one book priced in long terms; the short side trades it at 1 - p.
                leg = "long" if side.get("long") else "short"
                outcome = side.get("description") or leg
                contracts.append(AdapterContract(
                    exchange_id=exchange_id,
                    exchange_security_id=f"{slug}:{leg}",
                    exchange_security_symbol=f"{symbol_base}{outcome}"[:100],
                    base_currency="USD",
                    quote_currency="USD",
                    settle_currency="USD",
                    security_type=SecurityType.EVENT_CONTRACT,
                    contract_type=ContractType.BINARY,
                    asset_class=AssetClass.PREDICTION,
                    inverse=False,
                    is_quanto=False,
                    tick_size=_market_tick_size(market),
                    lot_size=_market_lot_size(market),
                    min_notional=0.0,
                    contract_multiplier=CONTRACT_MULTIPLIER,
                    event_title=market_event_title,
                    outcome_label=outcome,
                    event_description=market.get("description") or event.get("description") or None,
                    event_category=event_category,
                    event_expiry=market.get("endDate") or event.get("endDate"),
                    exchange_event_native_id=slug,
                    security_symbol=format_security_symbol(self.symbol_prefix, slug, leg),
                    exchange_event_native_url=native_url,
                    event_volume=volume_by_slug.get(slug, 0.0),
                ))
        return contracts
