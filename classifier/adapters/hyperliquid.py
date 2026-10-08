import logging
from datetime import datetime

import requests.exceptions

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.adapters.settlement import PRICE_SCALE, to_price
from classifier.adapters.types import AdapterContract
from classifier.client.http import RateLimitedSession
from classifier.types import ExchangeId
from classifier.utils import format_security_symbol

logger = logging.getLogger(__name__)

BASE_URL = "https://api.hyperliquid.xyz/info"

CONTRACT_MULTIPLIER = 1_000_000_000
TICK_SIZE = 1_000_000
LOT_SIZE = 1_000_000

# settledOutcome is one request per outcome and info requests share a per-IP weight budget, so each settle cycle
# looks up at most this many; the rest wait for the next cycle.
MAX_SETTLEMENT_LOOKUPS = 40

_TEMPLATE_PREFIX = "template:"
_TIMESTAMP_FORMAT = "%Y%m%d-%H%M"


def _parse_values(description: str) -> dict[str, str]:
    """Parse Hyperliquid's `key:value|key:value` descriptions; free text yields no values."""
    values: dict[str, str] = {}
    for part in description.split("|"):
        key, sep, value = part.partition(":")
        if sep and key.isidentifier():
            values[key] = value
    return values


def _parse_timestamp(value: str | None) -> datetime | None:
    try:
        return datetime.strptime(value or "", _TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _iso(ts: datetime | None) -> str | None:
    return ts.strftime("%Y-%m-%dT%H:%M:00Z") if ts else None


def _display(ts: datetime) -> str:
    # Hyperliquid's own templates append " UTC" where they want it, so times render bare.
    return ts.strftime("%Y-%m-%d %H:%M")


class _KeepMissing(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _fill(pattern: str, values: dict[str, str]) -> str:
    return pattern.format_map(_KeepMissing(values))


def _coin(outcome_id: int, side: int) -> str:
    """Hyperliquid's tradeable coin name for one side of an outcome, e.g. `#62070`."""
    return f"#{outcome_id}{side}"


class OutcomeTemplate:
    """A Hyperliquid outcome template from the `outcomeTemplates` info request.

    Outcomes and questions reference a template by name (`template:<id>`) and carry its keyword
    values in their description (`key:value|key:value`); rendering fills the template's own name
    and description patterns with those values.
    """

    def __init__(self, raw: dict):
        self.id: str = raw["id"]
        self.name: str = raw.get("name", self.id)
        self.description: str | None = raw.get("description")
        self.keyword_types: dict[str, str] = dict(raw.get("keywords", []))

    def _formatted(self, values: dict[str, str]) -> dict[str, str]:
        formatted = dict(values)
        for key, kind in self.keyword_types.items():
            ts = _parse_timestamp(values.get(key)) if kind == "dateTime" else None
            if ts:
                formatted[key] = _display(ts)
        return formatted

    def render(self, pattern: str, values: dict[str, str]) -> str:
        return _fill(pattern, self._formatted(values))

    def expiry(self, values: dict[str, str]) -> datetime | None:
        # A template can carry several dateTime keywords (scheduled start, resolution deadline);
        # the latest is the furthest the market can run.
        stamps = [
            _parse_timestamp(values.get(key))
            for key, kind in self.keyword_types.items()
            if kind == "dateTime"
        ]
        return max((s for s in stamps if s), default=None)


class _Rendered:
    """Title, description, expiry and category for one Hyperliquid event."""

    def __init__(self, title: str, description: str | None, expiry: datetime | None, category: str | None = None):
        self.title = title
        self.description = description
        self.expiry = expiry
        self.category = category


class HyperliquidAdapter:
    exchange_code = "HYPERLIQUID"
    symbol_prefix = "HL"
    # Nothing retires a settled outcome's listing (it just leaves outcomeMeta, where resolve can't see it), so the
    # settle cycle looks at active outcomes too, a page at a time sized to the lookups it may make per cycle.
    settle_active_listings = True
    settle_page_size = MAX_SETTLEMENT_LOOKUPS

    def __init__(self, session: RateLimitedSession | None = None):
        self._session = session or RateLimitedSession(min_request_interval=0.1)

    def fetch(self, exchange_id: ExchangeId):
        meta = self._post_info("outcomeMeta")
        templates = {t.id: t for t in map(OutcomeTemplate, self._post_info("outcomeTemplates"))}
        _, asset_ctxs = self._post_info("spotMetaAndAssetCtxs")
        volume_by_coin = {ctx["coin"]: float(ctx.get("dayNtlVlm") or 0) for ctx in asset_ctxs}
        outcomes = {o["outcome"]: o for o in meta.get("outcomes", [])}
        page = self._map_all(exchange_id, outcomes, meta.get("questions", []), templates, volume_by_coin)
        if page:
            yield page

    def fetch_resolved(self, exchange_id: ExchangeId, lookback_days: int) -> set[str]:
        meta = self._post_info("outcomeMeta")
        return {
            _coin(oid, side)
            for question in meta.get("questions", [])
            for oid in question.get("settledNamedOutcomes", [])
            for side in (0, 1)
        }

    def fetch_settlements(self, exchange_security_ids: set[str]) -> dict[str, int]:
        sides_by_outcome: dict[int, list[tuple[str, int]]] = {}
        for security_id in exchange_security_ids:
            encoding = security_id.removeprefix("#")
            if not encoding.isdigit():
                continue
            outcome, side = divmod(int(encoding), 10)
            sides_by_outcome.setdefault(outcome, []).append((security_id, side))

        settlements: dict[str, int] = {}
        # The settle stage pages candidates at settle_page_size; this only guards a caller that doesn't.
        for outcome in sorted(sides_by_outcome)[:MAX_SETTLEMENT_LOOKUPS]:
            # Null until the outcome settles; settlement is automatic and final, with no dispute window.
            settled = self._post_info("settledOutcome", outcome=outcome)
            if not settled:
                continue
            yes_price = to_price(settled.get("settleFraction"))
            if yes_price is None:
                logger.warning("Hyperliquid outcome %d settled with an unusable fraction: %r",
                               outcome, settled.get("settleFraction"))
                continue
            for security_id, side in sides_by_outcome[outcome]:
                settlements[security_id] = yes_price if side == 0 else PRICE_SCALE - yes_price
        return settlements

    def _post_info(self, request_type: str, **params):
        try:
            res = self._session.post(BASE_URL, json={"type": request_type, **params}, timeout=30)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RetryError as e:
            logger.error("Hyperliquid API retries exhausted: %s", e)
            raise
        except requests.exceptions.RequestException as e:
            logger.error("Hyperliquid API error: %s", e)
            raise

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, name: str, description: str, templates: dict[str, OutcomeTemplate]) -> _Rendered:
        values = _parse_values(description)
        if name.startswith(_TEMPLATE_PREFIX):
            template = templates.get(name.removeprefix(_TEMPLATE_PREFIX))
            if template is None:
                logger.warning("Unknown Hyperliquid template %r; using its raw name as the title", name)
                return _Rendered(name.removeprefix(_TEMPLATE_PREFIX), None, None)
            rendered_desc = template.render(template.description, values) if template.description else None
            return _Rendered(template.render(template.name, values), rendered_desc, template.expiry(values))
        if values.get("class") == "priceBinary":
            return self._render_recurring_binary(values)
        if values.get("class") == "priceBucket":
            return self._render_recurring_buckets(values)
        return _Rendered(name, description if not values else None, _parse_timestamp(values.get("expiry")))

    # Hyperliquid's recurring daily markets predate templates and have no template definition,
    # so these two are the only renderings built here.
    def _render_recurring_binary(self, values: dict[str, str]) -> _Rendered:
        expiry = _parse_timestamp(values.get("expiry"))
        title = f"{values.get('underlying', '?')} above {values.get('targetPrice', '?')} at {_display(expiry) if expiry else '?'}?"
        return _Rendered(title, None, expiry, "CRYPTO")

    def _render_recurring_buckets(self, values: dict[str, str]) -> _Rendered:
        expiry = _parse_timestamp(values.get("expiry"))
        title = f"{values.get('underlying', '?')} price range at {_display(expiry) if expiry else '?'}"
        return _Rendered(title, None, expiry, "CRYPTO")

    def _outcome_label(self, outcome: dict, question_values: dict[str, str], templates: dict[str, OutcomeTemplate]) -> str:
        name = outcome.get("name", "")
        values = _parse_values(outcome.get("description", ""))
        if name.startswith(_TEMPLATE_PREFIX):
            return self._render(name, outcome.get("description", ""), templates).title
        thresholds = [t for t in question_values.get("priceThresholds", "").split(",") if t]
        if "index" in values and thresholds:
            idx = int(values["index"])
            if idx == 0:
                return f"< {thresholds[0]}"
            if idx >= len(thresholds):
                return f"> {thresholds[-1]}"
            return f"{thresholds[idx - 1]} - {thresholds[idx]}"
        return name or str(outcome["outcome"])

    # ── Mapping ───────────────────────────────────────────────────────────────

    def _map_all(
        self,
        exchange_id: ExchangeId,
        outcomes: dict,
        questions: list[dict],
        templates: dict[str, OutcomeTemplate],
        volume_by_coin: dict[str, float],
    ) -> list[AdapterContract]:
        contracts: list[AdapterContract] = []
        questioned_outcome_ids: set[int] = set()

        for question in questions:
            named = question.get("namedOutcomes", [])
            fallback = question.get("fallbackOutcome")
            questioned_outcome_ids.update(named)
            if fallback is not None:
                questioned_outcome_ids.add(fallback)
            settled = set(question.get("settledNamedOutcomes", []))
            active = [outcomes[oid] for oid in named if oid != fallback and oid not in settled and oid in outcomes]
            rendered = self._render(question.get("name", ""), question.get("description", ""), templates)
            q_values = _parse_values(question.get("description", ""))
            # The question id is stable while outcomes settle; keying on an outcome would start a
            # new event whenever that outcome resolved.
            native_id = f"q:{question['question']}"

            # A question's outcomes stay multi-outcome to the last one: mapping a lone remaining outcome as a binary
            # would give it a new NO listing and change its type as its siblings settle.
            if active:
                contracts.extend(self._map_question(
                    exchange_id, rendered, active, q_values, templates, volume_by_coin, native_id,
                ))

        for outcome in outcomes.values():
            if outcome["outcome"] in questioned_outcome_ids:
                continue
            rendered = self._render(outcome.get("name", ""), outcome.get("description", ""), templates)
            contracts.extend(self._map_binary(exchange_id, rendered, outcome, volume_by_coin, f"o:{outcome['outcome']}"))

        return contracts

    def _map_question(
        self,
        exchange_id: ExchangeId,
        rendered: _Rendered,
        active_outcomes: list[dict],
        question_values: dict[str, str],
        templates: dict[str, OutcomeTemplate],
        volume_by_coin: dict[str, float],
        native_id: str,
    ) -> list[AdapterContract]:
        volume = sum(volume_by_coin.get(_coin(o["outcome"], side), 0.0) for o in active_outcomes for side in (0, 1))
        contracts: list[AdapterContract] = []
        for outcome in active_outcomes:
            outcome_id = outcome["outcome"]
            label = self._outcome_label(outcome, question_values, templates)
            contracts.append(self._contract(
                exchange_id, rendered, outcome, label, ContractType.MULTI_OUTCOME, native_id, volume,
                exchange_security_id=_coin(outcome_id, 0),
                security_symbol=format_security_symbol(self.symbol_prefix, str(outcome_id)),
            ))
        return contracts

    def _map_binary(
        self,
        exchange_id: ExchangeId,
        rendered: _Rendered,
        outcome: dict,
        volume_by_coin: dict[str, float],
        native_id: str,
    ) -> list[AdapterContract]:
        outcome_id = outcome["outcome"]
        values = _parse_values(outcome.get("description", ""))
        side_specs = outcome.get("sideSpecs", [])[:2]
        volume = sum(volume_by_coin.get(_coin(outcome_id, side), 0.0) for side in range(len(side_specs)))
        contracts: list[AdapterContract] = []
        for side, spec in enumerate(side_specs):
            label = _fill((spec.get("name") or ("Yes" if side == 0 else "No")).removeprefix(_TEMPLATE_PREFIX), values)
            contracts.append(self._contract(
                exchange_id, rendered, outcome, label, ContractType.BINARY, native_id, volume,
                exchange_security_id=_coin(outcome_id, side),
                security_symbol=format_security_symbol(self.symbol_prefix, str(outcome_id), label),
            ))
        return contracts

    def _contract(
        self,
        exchange_id: ExchangeId,
        rendered: _Rendered,
        outcome: dict,
        label: str,
        contract_type: ContractType,
        native_id: str,
        volume: float,
        *,
        exchange_security_id: str,
        security_symbol: str,
    ) -> AdapterContract:
        quote = outcome.get("quoteToken", "USDC")
        return AdapterContract(
            exchange_id=exchange_id,
            exchange_security_id=exchange_security_id,
            exchange_security_symbol=f"{rendered.title[:60]} -- {label}"[:100],
            base_currency=quote,
            quote_currency=quote,
            settle_currency=quote,
            security_type=SecurityType.EVENT_CONTRACT,
            contract_type=contract_type,
            asset_class=AssetClass.PREDICTION,
            inverse=False,
            is_quanto=False,
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            min_notional=0.0,
            contract_multiplier=CONTRACT_MULTIPLIER,
            event_title=rendered.title,
            outcome_label=label,
            event_description=rendered.description,
            event_category=rendered.category,
            event_expiry=_iso(rendered.expiry),
            exchange_event_native_id=native_id,
            security_symbol=security_symbol,
            event_volume=volume,
        )
