import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
    # The event's scheduled start as the venue lists it now (ISO-8601 UTC), and where the venue gives it. Like the
    # volume, a reading the registry is kept up to date with, not part of the contract (a reschedule resends nothing).
    event_start_time: str | None = None
    event_start_source: str | None = None
    # The game a sports market belongs to when the venue lists one game as several events, and whether it is one of
    # the game's core markets (who wins, by how much, how many in total, the final score): the volume gate lets a
    # game's core markets in together (`stages.fetch.GameCore`). Gate inputs only, not part of the contract.
    game_id: str | None = None
    game_core: bool = False
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


# A Polymarket game's events share the event slug up to the date (`ucl-aek1-rma-2026-11-04`,
# `...-2026-11-04-exact-score`, `...-more-markets`).
_POLYMARKET_GAME_SLUG = re.compile(r"^(?P<game>.+?-\d{4}-\d{2}-\d{2})(?:-|$)")


def polymarket_game(event_slug: str) -> str | None:
    """The game a Polymarket sports event slug belongs to, or None when it names no date."""
    m = _POLYMARKET_GAME_SLUG.match(event_slug or "")
    return m.group("game") if m else None


def utc_iso(text: str | None) -> str | None:
    """A venue's time as ISO-8601 UTC: "2026-11-04 17:45:00+00", "2026-11-07T17:00:00Z", or Hyperliquid's
    "20261104-2000" (UTC). None for anything else."""
    if not text:
        return None
    try:
        if len(text) == 13 and text[8] == "-":
            parsed = datetime.strptime(text, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()
