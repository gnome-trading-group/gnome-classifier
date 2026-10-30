"""What extraction reads for one security: its venue source plus the registry's view of the contract."""
import dataclasses
import json

from gnomepy.registry.types import ContractType


@dataclasses.dataclass(frozen=True)
class ExtractionInput:
    security_id: int
    exchange: str
    event_id: int
    contract_type: ContractType
    event_title: str
    outcome_label: str
    description: str | None
    expiry: str | None
    source: dict
    source_hash: str


def _outcomes(market: dict) -> list:
    raw = market.get("outcomes") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return raw if isinstance(raw, list) else []


_YES_NO = {"yes", "no", "long", "short", "over", "under"}


def _named_sides(names: list) -> bool:
    """Sides named for what they back (two teams) rather than Yes/No: each is read as its own proposition."""
    return bool(names) and not {str(n).strip().lower() for n in names} <= _YES_NO


def is_primary_side(item: ExtractionInput) -> bool:
    """For a Yes/No binary pair, the side whose proposition is read; the other side is its negation. Multi-outcome
    contracts and binaries whose sides are named (Clemson / North Carolina) are each read as their own proposition."""
    if item.contract_type != ContractType.BINARY or _named_sides([item.outcome_label]):
        return True
    source = item.source
    venue = source.get("venue")
    if venue == "KALSHI":
        return source.get("side", "yes") == "yes"
    if venue == "POLYMARKET_US":
        if _named_sides([s.get("description") for s in source.get("sides") or []]):
            return True
        return source.get("leg") == "long"
    if venue == "POLYMARKET_INTL":
        outcomes = _outcomes(source.get("market") or {})
        if _named_sides(outcomes):
            return True
        return not outcomes or source.get("outcome") == outcomes[0]
    if venue == "HYPERLIQUID":
        names = [s.get("name") for s in (source.get("outcome") or {}).get("sideSpecs") or []]
        if _named_sides([n for n in names if n]) and not any(str(n).startswith("{") for n in names):
            return True
        first = (names[0] if names else None) or "Yes"
        return source.get("label") in (first, "Yes")
    return True
