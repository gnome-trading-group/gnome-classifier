from enum import StrEnum
from typing import NamedTuple

# ── ID type aliases ──────────────────────────────────────────────────
type EventId = int
type SecurityId = int
type ExchangeId = int
type ListingId = int
type CurrencyId = int
type EventContractId = int

# ── Semantic float aliases ───────────────────────────────────────────
type Confidence = float
type Similarity = float

# ── Embedding vector ─────────────────────────────────────────────────
type Embedding = list[float]


# ── Relationship type enum ───────────────────────────────────────────

class RelationshipType(StrEnum):
    COMPLEMENT = "COMPLEMENT"
    MUTUALLY_EXCLUSIVE = "MUTUALLY_EXCLUSIVE"
    EQUIVALENT = "EQUIVALENT"
    IMPLIES = "IMPLIES"
    CORRELATED = "CORRELATED"
    HEDGEABLE_WITH = "HEDGEABLE_WITH"


# ── Structured tuple types ───────────────────────────────────────────

class RelationshipMatch(NamedTuple):
    """A candidate relationship between two securities from any discovery method."""
    security_id_a: SecurityId
    security_id_b: SecurityId
    relationship_type: RelationshipType
    confidence: Confidence
    method: str


class JudgedRelationship(NamedTuple):
    """A relationship verdict from the LLM judge. Internal to semantic.py."""
    security_id_a: SecurityId
    security_id_b: SecurityId
    relationship_type: RelationshipType
    confidence: Confidence


class CanonicalizeInput(NamedTuple):
    """Input record for canonicalize_events."""
    raw_title: str
    description: str | None
    category: str | None
    exchange_id: int
    native_id: str
