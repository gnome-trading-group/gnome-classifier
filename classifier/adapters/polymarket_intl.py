import json
import logging
import time
from datetime import datetime, timezone, timedelta

import requests.exceptions

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier import taxonomy
from classifier.adapters.settlement import CLOSED, OPEN, MarketState, chunked, settled, to_price
from classifier.adapters.types import AdapterContract
from classifier.client.http import RateLimitedSession
from classifier.stages.fees import polymarket_intl_definition, polymarket_intl_schedules
from classifier.types import ExchangeId
from classifier.utils import format_security_symbol

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
PAGE_SIZE = 500
# Condition ids per /markets lookup when asking about specific markets; each is 66 characters of query string.
LOOKUP_BATCH_SIZE = 50
# Gamma stops answering for many markets some time after they close, though the CLOB still has them (with their
# winner). Those are looked up on the CLOB one request each, at most this many an hour, so a backlog drains over a few
# hours instead of stalling one cycle.
CLOB_LOOKUPS_PER_HOUR = 2_000
# A market closes before its result is final: UMA proposes one, which can be disputed, before it resolves.
FINAL_UMA_STATUS = "resolved"

CONTRACT_MULTIPLIER = 1e9
SIZE_SCALE = 1_000_000
TICK_SIZE = 10_000_000
LOT_SIZE = 10_000
MIN_SIZE = 5 * SIZE_SCALE


def _market_tick_size(market: dict) -> int:
    raw = market.get("orderPriceMinTickSize")
    if raw is None:
        return TICK_SIZE
    try:
        return round(float(raw) * CONTRACT_MULTIPLIER)
    except (ValueError, TypeError):
        return TICK_SIZE


# The CLOB also rejects marketable orders under $1, but the OMS cannot tell a marketable order from a resting
# one, so min_notional stays 0 rather than rejecting valid 5-share resting orders priced under $0.20.
def _market_min_size(market: dict) -> int:
    raw = market.get("orderMinSize")
    if raw is None:
        return MIN_SIZE
    try:
        return round(float(raw) * SIZE_SCALE)
    except (ValueError, TypeError):
        return MIN_SIZE


def _build_sports_event_title(event_title: str, market: dict) -> str | None:
    question = market.get("question", "")
    if question and question != event_title:
        return None  # question already differentiates this market

    market_type = market.get("sportsMarketType", "")
    line = market.get("line")

    if market_type == "moneyline" or not market_type:
        return None

    if market_type == "spreads":
        return f"{event_title}: Spread {line}" if line is not None else f"{event_title}: Spread"

    if market_type == "totals":
        return f"{event_title}: Total {line}" if line is not None else f"{event_title}: Total"

    # Player props: points, rebounds, assists, etc.
    player = market.get("groupItemTitle", "")
    stat = market_type.title()
    if player and line is not None:
        return f"{event_title}: {player} {stat} {line}"
    elif player:
        return f"{event_title}: {player} {stat}"
    elif line is not None:
        return f"{event_title}: {stat} {line}"
    return f"{event_title}: {stat}"


def _market_symbol_base(market: dict, condition_id: str) -> str:
    # Market slugs are unique per market, unlike event slugs, which several binary markets share.
    return market.get("slug") or condition_id


def _is_neg_risk(event: dict) -> bool:
    # The event's own flag, not whichever of its markets are still open: deciding on those would regroup an event
    # (new identity, new contract type) as its markets closed.
    flag = event.get("negRisk")
    if flag is not None:
        return bool(flag)
    markets = event.get("markets") or []
    return bool(markets) and all(m.get("negRisk") for m in markets)


def _json_list(value) -> list:
    # Gamma encodes its array fields as JSON strings.
    return value if isinstance(value, list) else json.loads(value)


class PolymarketIntlAdapter:
    exchange_code = "POLYMARKET_INTL"
    symbol_prefix = "PM_I"

    def __init__(self, session: RateLimitedSession | None = None):
        self._session = session or RateLimitedSession(min_request_interval=0.1)
        self._fee_definitions: dict[str, dict] = {}
        self._clob_window_start = -float("inf")
        self._clob_lookups = 0

    def fetch_fee_schedules(self, exchange_id: ExchangeId) -> list[dict]:
        return polymarket_intl_schedules(exchange_id, dict(self._fee_definitions), datetime.now(timezone.utc))

    def _fee_group(self, market: dict) -> str | None:
        """The market's fee type, remembering the schedule it carries for fetch_fee_schedules. Markets without fees
        have none, and pay the exchange's default (nothing)."""
        fee_type, definition = market.get("feeType"), polymarket_intl_definition(market)
        if not isinstance(fee_type, str) or definition is None:
            return None
        if self._fee_definitions.get(fee_type, definition) != definition:
            logger.warning("Polymarket International fee type %s changed from %s to %s", fee_type,
                           self._fee_definitions[fee_type]["components"], definition["components"])
        self._fee_definitions[fee_type] = definition
        return fee_type

    def fetch(self, exchange_id: ExchangeId):
        after_cursor: str | None = None
        while True:
            params: dict = {"active": "true", "closed": "false", "limit": PAGE_SIZE}
            if after_cursor is not None:
                params["after_cursor"] = after_cursor
            try:
                res = self._session.get(f"{GAMMA_API_URL}/events/keyset", params=params, timeout=30)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.RetryError as e:
                logger.error("Polymarket API retries exhausted at cursor=%s: %s", after_cursor, e)
                raise
            except requests.exceptions.RequestException as e:
                logger.error("Polymarket API error at cursor=%s: %s", after_cursor, e)
                raise
            page = [c for event in data.get("events", []) for c in self._map_event(exchange_id, event)]
            if page:
                yield page
            after_cursor = data.get("next_cursor")
            if after_cursor is None:
                return

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

    def fetch_market_states(self, exchange_security_ids: set[str]) -> dict[str, MarketState]:
        ids_by_condition: dict[str, list[str]] = {}
        for security_id in exchange_security_ids:
            ids_by_condition.setdefault(security_id.partition(":")[0], []).append(security_id)

        states: dict[str, MarketState] = {}
        found: set[str] = set()
        # Gamma returns only open markets unless asked for closed ones, and never both at once.
        for closed in (True, False):
            for batch in chunked(sorted(set(ids_by_condition) - found), LOOKUP_BATCH_SIZE):
                for market in self._fetch_markets(batch, closed):
                    condition_id = market.get("conditionId", "")
                    if condition_id in ids_by_condition:
                        found.add(condition_id)
                        states.update(self._gamma_states(market, ids_by_condition[condition_id]))
        for condition_id in sorted(set(ids_by_condition) - found):
            if not self._take_clob_lookup():
                break
            market = self._fetch_clob_market(condition_id)
            if market is not None:
                states.update(self._clob_states(market, ids_by_condition[condition_id]))
        return states

    @staticmethod
    def _gamma_states(market: dict, security_ids: list[str]) -> dict[str, MarketState]:
        if not market.get("closed"):
            return dict.fromkeys(security_ids, OPEN)
        states = dict.fromkeys(security_ids, CLOSED)
        if market.get("umaResolutionStatus") != FINAL_UMA_STATUS:
            return states
        try:
            token_ids = _json_list(market.get("clobTokenIds"))
            prices = _json_list(market.get("outcomePrices"))
        except (TypeError, ValueError):
            logger.warning("Polymarket %s resolved with unreadable tokens or prices", market.get("conditionId"))
            return states
        for token_id, raw_price in zip(token_ids, prices):
            security_id = f"{market.get('conditionId')}:{token_id}"
            if security_id not in states:
                continue
            price = to_price(raw_price)
            if price is None:
                logger.warning("Polymarket %s resolved with an unusable price: %r", security_id, raw_price)
                continue
            states[security_id] = settled(price)
        return states

    @staticmethod
    def _clob_states(market: dict, security_ids: list[str]) -> dict[str, MarketState]:
        # Placeholder options (e.g. "Person F") have no tokens yet, so nothing about them can be read.
        tokens = [t for t in market.get("tokens") or [] if t.get("token_id")]
        if not tokens:
            return {}
        if not market.get("closed"):
            return dict.fromkeys(security_ids, OPEN)
        states = dict.fromkeys(security_ids, CLOSED)
        # The CLOB names a winner only once the result is final.
        if not any(t.get("winner") for t in tokens):
            return states
        for token in tokens:
            security_id = f"{market.get('condition_id')}:{token['token_id']}"
            if security_id not in states:
                continue
            price = to_price(token.get("price"))
            if price is None:
                logger.warning("Polymarket %s resolved with an unusable CLOB price: %r", security_id, token.get("price"))
                continue
            states[security_id] = settled(price)
        return states

    def _take_clob_lookup(self) -> bool:
        now = time.monotonic()
        if now - self._clob_window_start >= 3600:
            self._clob_window_start, self._clob_lookups = now, 0
        if self._clob_lookups >= CLOB_LOOKUPS_PER_HOUR:
            return False
        self._clob_lookups += 1
        return True

    def _fetch_markets(self, condition_ids: list[str], closed: bool) -> list[dict]:
        params = {"condition_ids": condition_ids, "closed": "true" if closed else "false", "limit": len(condition_ids)}
        try:
            res = self._session.get(f"{GAMMA_API_URL}/markets", params=params, timeout=30)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RetryError as e:
            logger.error("Polymarket markets retries exhausted: %s", e)
            raise
        except requests.exceptions.RequestException as e:
            logger.error("Polymarket markets API error: %s", e)
            raise

    def _fetch_clob_market(self, condition_id: str) -> dict | None:
        try:
            res = self._session.get(f"{CLOB_API_URL}/markets/{condition_id}", timeout=30)
            if res.status_code == 404:
                return None
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RetryError as e:
            logger.error("Polymarket CLOB market retries exhausted: %s", e)
            raise
        except requests.exceptions.RequestException as e:
            logger.error("Polymarket CLOB market API error: %s", e)
            raise

    def _fetch_closed_events(self, lookback_days: int):
        since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
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
                raise
            except requests.exceptions.RequestException as e:
                logger.error("Polymarket closed events API error at cursor=%s: %s", after_cursor, e)
                raise
            yield from data.get("events", [])
            after_cursor = data.get("next_cursor")
            if after_cursor is None:
                return

    def _fetch_all_events(self):
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
                raise
            except requests.exceptions.RequestException as e:
                logger.error("Polymarket API error at cursor=%s: %s", after_cursor, e)
                raise
            yield from data.get("events", [])
            after_cursor = data.get("next_cursor")
            if after_cursor is None:
                return

    def _map_event(self, exchange_id: ExchangeId, event: dict) -> list[AdapterContract]:
        markets = [m for m in event.get("markets", []) if not m.get("closed", False)]
        if not markets:
            return []

        event_description = event.get("description") or None
        event_title = event.get("title", "")
        event_slug = event.get("slug", "")
        tag_labels = [t.get("label") for t in event.get("tags") or [] if isinstance(t, dict)]
        event_category = tag_labels[0] if tag_labels else None
        mapped_category = taxonomy.polymarket_intl_category(tag_labels)
        event_tags = taxonomy.polymarket_tags(tag_labels, event.get("seriesSlug"), event_slug)

        event_end_date = event.get("endDate")
        is_neg_risk_group = _is_neg_risk(event)

        if is_neg_risk_group:
            event_volume: float = sum(m.get("volume24hr") or 0.0 for m in markets)
            return self._map_neg_risk_group(
                exchange_id, markets, event_title, event_slug, event_description, event_volume,
                event_category=event_category,
                mapped_category=mapped_category,
                event_tags=event_tags,
                event_end_date=event_end_date,
            )

        contracts: list[AdapterContract] = []
        for market in markets:
            market_volume: float = market.get("volume24hr") or 0.0
            contracts.extend(self._map_binary_market(
                exchange_id, market, event_description, event_slug, market_volume,
                event_title_override=_build_sports_event_title(event_title, market),
                event_category=event_category,
                mapped_category=mapped_category,
                event_tags=event_tags,
                event_end_date=event_end_date,
            ))
        return contracts

    def _map_neg_risk_group(
        self,
        exchange_id: ExchangeId,
        markets: list[dict],
        event_title: str,
        event_slug: str,
        event_description: str | None,
        event_volume: float = 0.0,
        event_category: str | None = None,
        mapped_category: str | None = None,
        event_tags: list[str] | None = None,
        event_end_date: str | None = None,
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
                tick_size=_market_tick_size(market),
                lot_size=LOT_SIZE,
                min_notional=0.0,
                min_size=_market_min_size(market),
                contract_multiplier=CONTRACT_MULTIPLIER,
                event_title=event_title,
                outcome_label=outcome_label,
                event_description=event_description,
                event_category=event_category,
                mapped_category=mapped_category,
                event_tags=list(event_tags or []),
                event_expiry=market.get("endDate") or event_end_date,
                exchange_event_native_id=event_slug,
                security_symbol=format_security_symbol(self.symbol_prefix, _market_symbol_base(market, condition_id)),
                exchange_event_native_url=f"https://polymarket.com/event/{event_slug}",
                event_volume=event_volume,
                fee_group=self._fee_group(market),
            ))
        return contracts

    def _map_binary_market(
        self,
        exchange_id: ExchangeId,
        market: dict,
        event_description: str | None,
        event_slug: str = "",
        event_volume: float = 0.0,
        event_title_override: str | None = None,
        event_category: str | None = None,
        mapped_category: str | None = None,
        event_tags: list[str] | None = None,
        event_end_date: str | None = None,
    ) -> list[AdapterContract]:
        question = market.get("question", "")
        condition_id = market.get("conditionId", "")
        if not condition_id:
            return []
        expiry = market.get("endDate") or event_end_date

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
                tick_size=_market_tick_size(market),
                lot_size=LOT_SIZE,
                min_notional=0.0,
                min_size=_market_min_size(market),
                contract_multiplier=CONTRACT_MULTIPLIER,
                event_title=event_title_override or question,
                outcome_label=outcome,
                event_description=market.get("description") or event_description,
                event_category=event_category,
                mapped_category=mapped_category,
                event_tags=list(event_tags or []),
                event_expiry=expiry,
                exchange_event_native_id=condition_id,
                security_symbol=format_security_symbol(self.symbol_prefix, _market_symbol_base(market, condition_id), outcome),
                exchange_event_native_url=native_url,
                event_volume=event_volume,
                fee_group=self._fee_group(market),
            ))
        return contracts
