import logging
import re

import requests

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.adapters.types import AdapterContract
from classifier.types import ExchangeId

logger = logging.getLogger(__name__)

BASE_URL = "https://api.hyperliquid.xyz/info"

CONTRACT_MULTIPLIER = 1_000_000_000
TICK_SIZE = 1_000_000
LOT_SIZE = 1_000_000

_CATEGORY_RE = re.compile(r'category:([^\|]+)')
_EXPIRY_RE = re.compile(r'expiry:(\d{8})-(\d{4})')
_STRUCTURED_DESC_RE = re.compile(r'^(?:metadata=|class:)')
_KV_RE = re.compile(r'(\w+):([^\|]+)')
_INDEX_RE = re.compile(r'^index:(\d+)$')


def _parse_meta(description: str) -> dict[str, str]:
    return dict(_KV_RE.findall(description))


def _fmt_expiry(expiry_iso: str | None) -> str:
    if not expiry_iso:
        return "?"
    return f"{expiry_iso[:10]} {expiry_iso[11:16]} UTC"


def _fmt_price(value: str) -> str:
    try:
        n = float(value)
        if n == int(n):
            return f"${int(n):,}"
        return f"${n:,}"
    except ValueError:
        return f"${value}"


class HyperliquidAdapter:
    exchange_name = "hyperliquid"

    def fetch(self, exchange_id: ExchangeId) -> list[AdapterContract]:
        data = self._fetch_outcome_meta()
        outcomes = {o["outcome"]: o for o in data.get("outcomes", [])}
        questions = data.get("questions", [])
        return self._map_all(exchange_id, outcomes, questions)

    def fetch_resolved(self, exchange_id: ExchangeId, lookback_days: int) -> set[str]:
        data = self._fetch_outcome_meta()
        resolved: set[str] = set()
        for question in data.get("questions", []):
            for oid in question.get("settledNamedOutcomes", []):
                resolved.add(f"@{oid}")
                resolved.add(f"@{oid}:0")
                resolved.add(f"@{oid}:1")
        return resolved

    def _fetch_outcome_meta(self) -> dict:
        try:
            res = requests.post(BASE_URL, json={"type": "outcomeMeta"}, timeout=30)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            logger.error("Hyperliquid API error: %s", e)
            return {}

    def _map_all(self, exchange_id: ExchangeId, outcomes: dict, questions: list[dict]) -> list[AdapterContract]:
        contracts: list[AdapterContract] = []
        questioned_outcome_ids: set[int] = set()

        for question in questions:
            named = question.get("namedOutcomes", [])
            questioned_outcome_ids.update(named)
            fallback = question.get("fallbackOutcome")
            if fallback is not None:
                questioned_outcome_ids.add(fallback)
            settled = set(question.get("settledNamedOutcomes", []))
            active = [oid for oid in named if oid != fallback and oid not in settled]

            if len(active) > 1:
                active_outcomes = [outcomes[oid] for oid in active if oid in outcomes]
                native_id = f"q:{active[0]}"
                contracts.extend(self._map_multi_outcome(exchange_id, question, active_outcomes, native_id))
            elif len(active) == 1:
                outcome = outcomes.get(active[0])
                if outcome:
                    q_desc = question.get("description", "")
                    contracts.extend(self._map_binary(
                        exchange_id=exchange_id,
                        event_title=question.get("name", ""),
                        event_desc=self._human_desc(q_desc),
                        event_category=self._parse_category(q_desc),
                        event_expiry=self._parse_expiry(q_desc),
                        outcome=outcome,
                        exchange_event_native_id=f"o:{outcome['outcome']}",
                    ))

        for outcome in outcomes.values():
            if outcome["outcome"] not in questioned_outcome_ids:
                o_desc = outcome.get("description", "")
                meta = _parse_meta(o_desc)
                cls = meta.get("class", "")

                if cls == "priceBinary":
                    underlying = meta.get("underlying", "?")
                    target = meta.get("targetPrice", "?")
                    event_expiry = self._parse_expiry(o_desc)
                    event_title = f"Will {underlying} be above {_fmt_price(target)}? ({_fmt_expiry(event_expiry)})"
                    event_category = "CRYPTO"
                    event_desc = None
                else:
                    event_title = outcome.get("name", "")
                    event_desc = self._human_desc(o_desc)
                    event_category = self._parse_category(o_desc)
                    event_expiry = self._parse_expiry(o_desc)

                contracts.extend(self._map_binary(
                    exchange_id=exchange_id,
                    event_title=event_title,
                    event_desc=event_desc,
                    event_category=event_category,
                    event_expiry=event_expiry,
                    outcome=outcome,
                    exchange_event_native_id=f"o:{outcome['outcome']}",
                ))

        return contracts

    def _map_multi_outcome(self, exchange_id: ExchangeId, question: dict, active_outcomes: list[dict], exchange_event_native_id: str) -> list[AdapterContract]:
        q_desc = question.get("description", "")
        meta = _parse_meta(q_desc)
        cls = meta.get("class", "")

        if cls == "priceBucket":
            underlying = meta.get("underlying", "?")
            expiry_iso = self._parse_expiry(q_desc)
            event_title = f"{underlying} price range on {_fmt_expiry(expiry_iso)}"
            event_category = "CRYPTO"
            event_desc = None
            thresholds = [t.strip() for t in meta.get("priceThresholds", "").split(",") if t.strip()]

            def _bucket_label(outcome: dict) -> str:
                m = _INDEX_RE.match(outcome.get("description", ""))
                if not m or not thresholds:
                    return outcome.get("name", str(outcome["outcome"]))
                idx = int(m.group(1))
                if idx == 0:
                    return f"< {_fmt_price(thresholds[0])}"
                if idx >= len(thresholds):
                    return f"> {_fmt_price(thresholds[-1])}"
                return f"{_fmt_price(thresholds[idx - 1])} - {_fmt_price(thresholds[idx])}"
        else:
            event_title = question.get("name", "")
            event_desc = self._human_desc(q_desc)
            event_category = self._parse_category(q_desc)
            expiry_iso = self._parse_expiry(q_desc)
            thresholds = []

            def _bucket_label(outcome: dict) -> str:
                return outcome.get("name", str(outcome["outcome"]))

        symbol_base = f"{event_title[:60]} -- "
        contracts: list[AdapterContract] = []
        for outcome in active_outcomes:
            outcome_id = outcome["outcome"]
            outcome_label = _bucket_label(outcome)
            contracts.append(AdapterContract(
                exchange_id=exchange_id,
                exchange_security_id=f"@{outcome_id}",
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
                event_description=event_desc,
                event_category=event_category,
                event_expiry=expiry_iso,
                exchange_event_native_id=exchange_event_native_id,
            ))
        return contracts

    def _map_binary(
        self,
        exchange_id: ExchangeId,
        event_title: str,
        event_desc: str | None,
        event_category: str | None,
        event_expiry: str | None,
        outcome: dict,
        exchange_event_native_id: str,
    ) -> list[AdapterContract]:
        outcome_id = outcome["outcome"]
        side_specs = outcome.get("sideSpecs", [])
        symbol_base = f"{event_title[:60]} -- "

        contracts: list[AdapterContract] = []
        for i, side in enumerate(side_specs[:2]):
            side_name = side.get("name", "Yes" if i == 0 else "No")
            contracts.append(AdapterContract(
                exchange_id=exchange_id,
                exchange_security_id=f"@{outcome_id}:{i}",
                exchange_security_symbol=f"{symbol_base}{side_name}"[:100],
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
                event_title=event_title,
                outcome_label=side_name,
                event_description=event_desc,
                event_category=event_category,
                event_expiry=event_expiry,
                exchange_event_native_id=exchange_event_native_id,
            ))
        return contracts

    def _parse_category(self, description: str) -> str | None:
        m = _CATEGORY_RE.search(description)
        return m.group(1).strip() if m else None

    def _parse_expiry(self, description: str) -> str | None:
        m = _EXPIRY_RE.search(description)
        if not m:
            return None
        date_str, time_str = m.group(1), m.group(2)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{time_str[:2]}:{time_str[2:]}:00Z"

    def _human_desc(self, description: str) -> str | None:
        if not description or _STRUCTURED_DESC_RE.match(description):
            return None
        return description
