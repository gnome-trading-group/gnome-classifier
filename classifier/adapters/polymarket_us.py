import logging
import re
import time
from datetime import datetime, timedelta, timezone

import requests.exceptions

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier import taxonomy
from classifier.adapters.settlement import CLOSED, OPEN, MarketState, chunked, settled, to_price
from classifier.adapters.types import AdapterContract, pick_source_fields
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
# Closed and resolving markets have no side prices yet; only a resolved market's are what each side paid.
MARKET_STATUS_RESOLVED = "MARKET_STATUS_RESOLVED"
# Slugs per /markets lookup when asking about specific markets.
LOOKUP_BATCH_SIZE = 50

# The gateway documents volume fields but never returns them; it does honour volumeNumMin/Max
# filters (shares). Querying disjoint ranges buckets each market by volume without the values.
VOLUME_BUCKETS_SHARES = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 1_000_000)
# Since 2026-10-15 event lists carry only each event's primary markets (docs.polymarket.us/changelog); spreads, totals
# and props come from the event itself. Those full events are reused for this long, so a fetch cycle every minute
# doesn't refetch hundreds of them; a market added in between waits at most this long to be seen.
FULL_EVENT_TTL_SECONDS = 300
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


def _trimmed(event: dict) -> bool:
    """Whether the event list left some of the event's markets out."""
    count = (event.get("marketCounts") or {}).get("numMarkets")
    return isinstance(count, int) and count > len(event.get("markets") or [])


def _sport_slug(event: dict) -> str | None:
    """The event's sport (e.g. "table-tennis"), the fee group Polymarket US prices some sports by; the gateway marks
    the tag that names a sport with a `sport` object."""
    for tag in event.get("tags") or []:
        sport = tag.get("sport") if isinstance(tag, dict) else None
        if isinstance(sport, dict) and sport.get("slug"):
            return sport["slug"]
    return None



_SOURCE_EVENT_FIELDS = ("slug", "ticker", "title", "description", "category", "seriesSlug", "startDate", "endDate")
_SOURCE_MARKET_FIELDS = (
    "slug", "title", "subtitle", "question", "description", "outcomes", "marketType", "sportsMarketType",
    "sportsMarketTypeV2", "line", "startDate", "endDate", "gameStartTime", "resolutionSource",
)

_SOURCE_SIDE_FIELDS = ("description", "long", "identifier", "marketSideType")


def _with_sources(event: dict, contracts: list[AdapterContract]) -> list[AdapterContract]:
    event_source = pick_source_fields(event, _SOURCE_EVENT_FIELDS)
    market_by_slug = {m.get("slug"): m for m in event.get("markets") or []}
    for c in contracts:
        slug, _, leg = c.exchange_security_id.partition(":")
        c.proposition_source = {
            "venue": "POLYMARKET_US",
            "event": event_source,
            "market": pick_source_fields(market_by_slug.get(slug), _SOURCE_MARKET_FIELDS),
            # A side's price moves every fetch; only what it is stays in the source.
            "sides": [pick_source_fields(side, _SOURCE_SIDE_FIELDS)
                      for side in (market_by_slug.get(slug) or {}).get("marketSides") or []],
            "leg": leg,
        }
    return contracts


class PolymarketUsAdapter:
    exchange_code = "POLYMARKET_US"
    symbol_prefix = "PM_US"

    def __init__(self, session: RateLimitedSession | None = None):
        # Public gateway limit is 20 req/s per IP.
        self._session = session or RateLimitedSession(min_request_interval=0.1)
        # Full events by slug, with when they were fetched (FULL_EVENT_TTL_SECONDS).
        self._full_events: dict[str, tuple[float, dict]] = {}

    def fetch(self, exchange_id: ExchangeId):
        volume_by_slug = self._fetch_volume_buckets()
        for events in self._active_events():
            page = [c for event in events for c in _with_sources(event, self._map_event(exchange_id, event, volume_by_slug))]
            if page:
                yield page

    def fetch_resolved(self, exchange_id: ExchangeId, lookback_days: int) -> set[str]:
        resolved: set[str] = set()

        since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for markets in self._paginate("markets", {"closed": "true", "endDateMin": since}):
            for market in markets:
                resolved.update(self._security_ids(market))

        # Markets resolve early (eliminated teams) while their endDate is still in the future.
        for events in self._active_events():
            for event in events:
                for market in event.get("markets") or []:
                    if not _is_open(market):
                        resolved.update(self._security_ids(market))

        return resolved

    def fetch_market_states(self, exchange_security_ids: set[str]) -> dict[str, MarketState]:
        legs_by_slug: dict[str, list[str]] = {}
        for security_id in exchange_security_ids:
            legs_by_slug.setdefault(security_id.rpartition(":")[0], []).append(security_id)
        states: dict[str, MarketState] = {}
        for batch in chunked(sorted(legs_by_slug), LOOKUP_BATCH_SIZE):
            for market in self._fetch_markets(batch):
                legs = legs_by_slug.get(market.get("slug", ""), [])
                if _is_open(market):
                    states.update(dict.fromkeys(legs, OPEN))
                    continue
                states.update(dict.fromkeys(legs, CLOSED))
                if market.get("status") != MARKET_STATUS_RESOLVED:
                    continue
                for side in market.get("marketSides") or []:
                    # Each side reports its own payout, so the short side's is already 1 - long.
                    security_id = f"{market.get('slug')}:{'long' if side.get('long') else 'short'}"
                    if security_id not in states:
                        continue
                    price = to_price(side.get("price"))
                    if price is None:
                        logger.warning("Polymarket US %s resolved with an unusable price: %r", security_id, side.get("price"))
                        continue
                    states[security_id] = settled(price)
        return states

    def _fetch_markets(self, slugs: list[str]) -> list[dict]:
        params = {"slug": slugs, "limit": len(slugs)}
        try:
            res = self._session.get(f"{GATEWAY_URL}/markets", params=params, timeout=30)
            res.raise_for_status()
            return res.json().get("markets", [])
        except requests.exceptions.RetryError as e:
            logger.error("Polymarket US markets retries exhausted: %s", e)
            raise
        except requests.exceptions.RequestException as e:
            logger.error("Polymarket US markets API error: %s", e)
            raise

    def _active_events(self):
        """Pages of active events, each with all of its markets: a trimmed event is replaced by the event itself."""
        now = time.monotonic()
        seen: set[str] = set()
        for events in self._paginate("events", {"active": "true", "closed": "false"}):
            page = []
            for event in events:
                slug = event.get("slug")
                if slug and _trimmed(event):
                    seen.add(slug)
                    event = self._full_event(slug, now)
                page.append(event)
            yield page
        self._full_events = {slug: entry for slug, entry in self._full_events.items() if slug in seen}

    def _full_event(self, slug: str, now: float) -> dict:
        cached = self._full_events.get(slug)
        if cached is not None and now - cached[0] < FULL_EVENT_TTL_SECONDS:
            return cached[1]
        try:
            res = self._session.get(f"{GATEWAY_URL}/events/slug/{slug}", timeout=30)
            res.raise_for_status()
            event = res.json()["event"]
        except requests.exceptions.RetryError as e:
            logger.error("Polymarket US event %s retries exhausted: %s", slug, e)
            raise
        except requests.exceptions.RequestException as e:
            logger.error("Polymarket US event %s API error: %s", slug, e)
            raise
        self._full_events[slug] = (now, event)
        return event

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
        tag_labels = [t.get("label") if isinstance(t, dict) else t for t in event.get("tags") or []]
        mapped_category = taxonomy.polymarket_us_category(event_category)
        event_tags = taxonomy.polymarket_tags(tag_labels, event.get("seriesSlug"), event_slug)
        fee_group = _sport_slug(event)

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
                    mapped_category=mapped_category,
                    event_tags=list(event_tags),
                    event_expiry=market.get("endDate") or event.get("endDate"),
                    exchange_event_native_id=slug,
                    security_symbol=format_security_symbol(self.symbol_prefix, slug, leg),
                    exchange_event_native_url=native_url,
                    event_volume=volume_by_slug.get(slug, 0.0),
                    fee_group=fee_group,
                ))
        return contracts
