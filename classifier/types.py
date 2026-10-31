import dataclasses
from typing import NamedTuple

# ── ID type aliases ──────────────────────────────────────────────────
type EventId = int
type SecurityId = int
type ExchangeId = int
type ListingId = int
type CurrencyId = int
type EventContractId = int
type NativeKey = tuple[int, str]


# ── Structured tuple types ───────────────────────────────────────────

class CanonicalizeInput(NamedTuple):
    """Input record for canonicalize_events."""
    raw_title: str
    description: str | None
    category: str | None
    exchange_id: int
    native_id: str
    mapped_category: str | None = None
    exchange_tags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class EntityResult:
    """Result of the entity creation + embedding stage."""
    events_created: int
    securities_created: int
    listings_created: int
    event_contracts_created: int
    listing_specs_created: int
    listing_specs_updated: int
    event_contracts_relabelled: int
    securities_renamed: int
    listings_renamed: int
    new_security_ids: list[SecurityId]
    new_security_symbols: list[str]
    created_event_ids: list[EventId]
    created_event_names: list[str]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "events_created": self.events_created,
            "securities_created": self.securities_created,
            "listings_created": self.listings_created,
            "event_contracts_created": self.event_contracts_created,
            "listing_specs_created": self.listing_specs_created,
            "listing_specs_updated": self.listing_specs_updated,
            "event_contracts_relabelled": self.event_contracts_relabelled,
            "securities_renamed": self.securities_renamed,
            "listings_renamed": self.listings_renamed,
        }

    @property
    def has_new_entities(self) -> bool:
        return len(self.new_security_ids) > 0
