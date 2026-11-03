"""Propositions read by code from a contract's venue source, for the market families whose rules follow a fixed
template. Anything else, or a family member whose text doesn't parse, returns None and goes to the LLM extractor.

Every value comes from the rules text the venue settles on, or, where the venue's rules text lost a character (Kalshi's
Fed ">25bps"), from a subtitle that agrees with it; a structured field the source also has (a Kalshi strike) must
agree with it or the value is left unknown. One module per venue.

A venue module may also correct what the LLM read (`correct`) where the venue states a field structurally that its
rules text leaves out, and a reader may need the venues' current prices (`prices`, {coin: mid}) where only the price
says which way a market points.
"""
import hashlib
from collections.abc import Mapping
from pathlib import Path

from classifier.propositions.types import Proposition
from classifier.propositions.venue import hyperliquid, kalshi, polymarket_intl, polymarket_us

_READERS = {"KALSHI": kalshi, "HYPERLIQUID": hyperliquid, "POLYMARKET_INTL": polymarket_intl,
            "POLYMARKET_US": polymarket_us}

# Changes with any edit to the readers. A security parked because no reader could read it is looked at again once
# this changes, and the Relationships worker re-reads every open security the readers now read differently
# (`stages.propositions.recheck_readers`), so a new or fixed reader applies without anyone bumping a version.
READERS_VERSION = hashlib.sha256(
    b"".join(path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py")))
).hexdigest()[:12]


def extract_venue(source: dict, prices: Mapping[str, float] | None = None) -> Proposition | None:
    reader = _READERS.get(source.get("venue"))
    if reader is None:
        return None
    proposition = reader.read(source, prices or {})
    if proposition is not None and reader.is_no_side(source):
        proposition = proposition.negate()
    return proposition


def correct_venue(source: dict, proposition: Proposition) -> Proposition:
    """The LLM's reading with any field the venue states structurally put right."""
    fix = getattr(_READERS.get(source.get("venue")), "correct", None)
    return fix(source, proposition) if fix else proposition


def needs_prices(source: dict) -> bool:
    """Whether the reading depends on current prices, which the rules text alone doesn't give."""
    return source.get("venue") == "HYPERLIQUID" and hyperliquid.is_touch(source)


def claims(source: dict) -> bool:
    """Whether a source belongs to a family the readers handle, so a None from `extract_venue` means the venue changed
    its wording (or a field) rather than that the market is one the LLM is meant to read."""
    reader = _READERS.get(source.get("venue"))
    return reader is not None and reader.claims(source)
