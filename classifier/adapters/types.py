from dataclasses import dataclass

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
    event_description: str | None = None
    event_category: str | None = None
    event_expiry: str | None = None
