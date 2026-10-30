import hashlib
import json
from dataclasses import dataclass, field

from gnomepy.registry.types import AssetClass, ContractType, SecurityType

from classifier.types import ExchangeId


@dataclass
class AdapterContract:
    exchange_id: ExchangeId
    exchange_security_id: str
    exchange_security_symbol: str
    base_currency: str
    quote_currency: str
    settle_currency: str
    security_type: SecurityType
    contract_type: ContractType
    asset_class: AssetClass
    inverse: bool
    is_quanto: bool
    tick_size: float
    lot_size: float
    min_notional: float
    contract_multiplier: float
    event_title: str
    outcome_label: str
    exchange_event_native_id: str
    security_symbol: str
    event_description: str | None = None
    # The exchange's own label, passed to the LLM as a hint.
    event_category: str | None = None
    # A registry category from the exchange's labels (classifier.taxonomy), or None when they leave it open.
    mapped_category: str | None = None
    event_tags: list[str] = field(default_factory=list)
    event_expiry: str | None = None
    exchange_event_native_url: str | None = None
    event_volume: float | None = None
    min_size: float = 0.0
    # The venue's keys that fee schedules target (sm.listing.fee_group / fee_event), e.g. a Kalshi series and event.
    # Set only where a schedule other than the exchange's default can apply.
    fee_group: str | None = None
    fee_event: str | None = None
    # The venue's own resolution-relevant fields for this contract (rules, strikes, resolution sources, times), stored
    # as sm.contract_source so propositions are read from everything the venue said. Static fields only: a price or
    # volume here would resend the contract every fetch.
    proposition_source: dict | None = None


def pick_source_fields(raw: dict | None, keys: tuple[str, ...]) -> dict:
    """The named fields a venue gave, leaving out absent and empty ones."""
    if not raw:
        return {}
    return {k: raw[k] for k in keys if raw.get(k) not in (None, "", [], {})}


def proposition_source_hash(source: dict | None) -> str:
    if not source:
        return ""
    return hashlib.sha256(json.dumps(source, sort_keys=True, default=str).encode()).hexdigest()[:16]
