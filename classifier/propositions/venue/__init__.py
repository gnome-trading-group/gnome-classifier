"""Propositions read by code from a contract's venue source, for the market families whose rules follow a fixed
template. Anything else, or a family member whose text doesn't parse, returns None and goes to the LLM extractor.

Every value comes from the rules text the venue settles on; a structured field the source also has (a Kalshi strike)
must agree with it or the value is left unknown. One module per venue.
"""
import hashlib
from pathlib import Path

from classifier.propositions.types import Proposition
from classifier.propositions.venue import hyperliquid, kalshi, polymarket_intl, polymarket_us

_READERS = {"KALSHI": kalshi, "HYPERLIQUID": hyperliquid, "POLYMARKET_INTL": polymarket_intl,
            "POLYMARKET_US": polymarket_us}

# Changes with any edit to the readers. A security parked because no reader could read it is looked at again once
# this changes, so a new or fixed reader picks up what it now reads without anyone bumping a version.
READERS_VERSION = hashlib.sha256(
    b"".join(path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py")))
).hexdigest()[:12]


def extract_venue(source: dict) -> Proposition | None:
    reader = _READERS.get(source.get("venue"))
    if reader is None:
        return None
    proposition = reader.read(source)
    if proposition is not None and reader.is_no_side(source):
        proposition = proposition.negate()
    return proposition


def claims(source: dict) -> bool:
    """Whether a source belongs to a family the readers handle, so a None from `extract_venue` means the venue changed
    its wording (or a field) rather than that the market is one the LLM is meant to read."""
    reader = _READERS.get(source.get("venue"))
    return reader is not None and reader.claims(source)
