"""Top of book for a few securities, read from each venue's public market data, for checking a subject merge against
prices before making it. The same reads as the registry's mark poller (`lambda/sync/venues/books.ts`): Kalshi and
Polymarket US keep one book per market, so a NO or short listing is priced as its complement; each Polymarket
International outcome token has its own book. Hyperliquid isn't read: its securities go unchecked. Hyperliquid's
perp mids are read separately (`hyperliquid_mids`), for the touch reader.

A side a venue shows empty is None. A market that can't be read is logged and left out, so it never stops the rest.
"""
import dataclasses
import logging

from classifier.adapters.hyperliquid import BASE_URL as HYPERLIQUID_INFO_URL
from classifier.adapters.kalshi import BASE_URL as KALSHI_API
from classifier.adapters.polymarket_intl import CLOB_API_URL
from classifier.adapters.polymarket_us import GATEWAY_URL as POLYMARKET_US_GATEWAY
from classifier.client.http import RateLimitedSession

logger = logging.getLogger(__name__)

KALSHI_BATCH = 100
POLYMARKET_BATCH = 50
REQUEST_TIMEOUT = 20


@dataclasses.dataclass(frozen=True)
class Quote:
    bid: float | None
    ask: float | None

    @property
    def mid(self) -> float | None:
        return (self.bid + self.ask) / 2 if self.bid is not None and self.ask is not None else None


@dataclasses.dataclass(frozen=True)
class ListingRef:
    security_id: int
    exchange_code: str
    exchange_security_id: str


def _price(value) -> float | None:
    """A binary contract only trades strictly between 0 and 1; venues show an empty side at either end."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    return p if 0 < p < 1 else None


def _complement(q: Quote) -> Quote:
    return Quote(None if q.ask is None else 1 - q.ask, None if q.bid is None else 1 - q.bid)


def _split(exchange_security_id: str) -> tuple[str, str]:
    market, _, side = exchange_security_id.partition(":")
    return market, side.lower()


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def kalshi_quote(market: dict, side: str) -> Quote | None:
    if market.get("status") != "active":
        return None

    def level(price, size) -> float | None:
        try:
            if size is not None and float(size) <= 0:
                return None
        except (TypeError, ValueError):
            return None
        return _price(price)

    q = Quote(level(market.get("yes_bid_dollars"), market.get("yes_bid_size_fp")),
              level(market.get("yes_ask_dollars"), market.get("yes_ask_size_fp")))
    return _complement(q) if side == "no" else q


def polymarket_intl_quote(book: dict) -> Quote:
    """The CLOB lists levels worst first, so the best is searched for."""
    def levels(side) -> list[float]:
        out = []
        for level in side or []:
            try:
                if float(level.get("size") or 0) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            if (p := _price(level.get("price"))) is not None:
                out.append(p)
        return out

    bids, asks = levels(book.get("bids")), levels(book.get("asks"))
    return Quote(max(bids) if bids else None, min(asks) if asks else None)


def polymarket_us_quote(market: dict, side: str) -> Quote | None:
    if market.get("closed") is True or market.get("status") != "MARKET_STATUS_OPEN":
        return None
    q = Quote(_price((market.get("bestBidQuote") or {}).get("value")),
              _price((market.get("bestAskQuote") or {}).get("value")))
    return _complement(q) if side == "short" else q


def _kalshi(session: RateLimitedSession, refs: list[ListingRef]) -> dict[int, Quote]:
    markets: dict[str, dict] = {}
    for batch in _chunks(sorted({_split(r.exchange_security_id)[0] for r in refs}), KALSHI_BATCH):
        res = session.get(f"{KALSHI_API}/markets", params={"limit": KALSHI_BATCH, "tickers": ",".join(batch)},
                          timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        markets.update({m["ticker"]: m for m in res.json().get("markets") or []})
    out = {}
    for r in refs:
        market, side = _split(r.exchange_security_id)
        if market in markets and (q := kalshi_quote(markets[market], side)) is not None:
            out[r.security_id] = q
    return out


def _polymarket_intl(session: RateLimitedSession, refs: list[ListingRef]) -> dict[int, Quote]:
    def token(r: ListingRef) -> str:
        return r.exchange_security_id.partition(":")[2] or r.exchange_security_id

    books: dict[str, dict] = {}
    for batch in _chunks(sorted({token(r) for r in refs}), POLYMARKET_BATCH):
        res = session.post(f"{CLOB_API_URL}/books", json=[{"token_id": t} for t in batch], timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        books.update({str(b.get("asset_id")): b for b in res.json()})
    return {r.security_id: polymarket_intl_quote(books[token(r)]) for r in refs if token(r) in books}


def _polymarket_us(session: RateLimitedSession, refs: list[ListingRef]) -> dict[int, Quote]:
    markets: dict[str, dict] = {}
    for batch in _chunks(sorted({_split(r.exchange_security_id)[0] for r in refs}), POLYMARKET_BATCH):
        res = session.get(f"{POLYMARKET_US_GATEWAY}/markets", params=[("slug", s) for s in batch] + [("limit", len(batch))],
                          timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        markets.update({m["slug"]: m for m in res.json().get("markets") or []})
    out = {}
    for r in refs:
        market, side = _split(r.exchange_security_id)
        if market in markets and (q := polymarket_us_quote(markets[market], side)) is not None:
            out[r.security_id] = q
    return out


_READERS = {"KALSHI": _kalshi, "POLYMARKET_INTL": _polymarket_intl, "POLYMARKET_US": _polymarket_us}


def read_quotes(refs: list[ListingRef], session: RateLimitedSession | None = None) -> dict[int, Quote]:
    """{security_id: quote} for the listings a venue could price."""
    session = session or RateLimitedSession(min_request_interval=0.1)
    by_venue: dict[str, list[ListingRef]] = {}
    for r in refs:
        if r.exchange_code in _READERS and r.exchange_security_id:
            by_venue.setdefault(r.exchange_code, []).append(r)
    out: dict[int, Quote] = {}
    for venue, venue_refs in by_venue.items():
        try:
            out.update(_READERS[venue](session, venue_refs))
        except Exception as e:
            logger.warning("Reading %s quotes for %d listings failed: %s", venue, len(venue_refs), e)
    return out


def hyperliquid_mids(session: RateLimitedSession | None = None) -> dict[str, float]:
    """{perp coin: mid} for every Hyperliquid perp, including those on builder-deployed exchanges (`xyz:CL`), which
    only their own exchange's read lists. Raises on a failed read, so a pass reads nothing with no prices rather than
    leaving every touch's direction unknown."""
    session = session or RateLimitedSession(min_request_interval=0.1)

    def post(body: dict):
        res = session.post(HYPERLIQUID_INFO_URL, json=body, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        return res.json()

    dexes = [d["name"] for d in post({"type": "perpDexs"}) if d and d.get("name")]
    mids: dict[str, float] = {}
    for body in [{"type": "allMids"}] + [{"type": "allMids", "dex": dex} for dex in dexes]:
        mids.update({coin: float(raw) for coin, raw in post(body).items() if not coin.startswith("@") and float(raw) > 0})
    return mids
