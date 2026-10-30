"""What a security asks, as one typed record: the proposition relationships are derived from.

A proposition is read once per security from its venue source. Two propositions are only ever compared when they share
a subject; the rest of the fields decide whether one implies the other. A field the extractor could not determine is
listed in `unknown` rather than guessed, and blocks every rule that needs it.
"""
import dataclasses
import re
from datetime import datetime
from enum import StrEnum

EXTRACTOR_VERSION = "p1"

SUBJECT_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9:._/-]{0,199}$")


class Kind(StrEnum):
    THRESHOLD = "THRESHOLD"
    RANGE = "RANGE"
    OCCURS_BY = "OCCURS_BY"
    WINNER = "WINNER"
    OTHER = "OTHER"


class Comparator(StrEnum):
    GT = "GT"
    GTE = "GTE"
    LT = "LT"
    LTE = "LTE"


class Observation(StrEnum):
    # The value at one moment (a 3pm price, the score at a cut-off).
    AT_TIME = "AT_TIME"
    # Its extreme at any point in the window (touches, highest daily count in a month).
    ANY_TIME = "ANY_TIME"
    # A total accumulated over the window (game totals, season wins).
    CUMULATIVE = "CUMULATIVE"


class Stability(StrEnum):
    # Can still move around the observation time: prices, in-game numbers.
    LIVE = "LIVE"
    # Fixed by then: a review score after release, published official figures.
    SETTLED = "SETTLED"


class Method(StrEnum):
    VENUE = "VENUE"
    LLM = "LLM"


# The fields `unknown` may name.
FIELDS = (
    "subject", "measure", "kind", "comparator", "value", "low", "high", "unit", "observation", "window_start",
    "window_end", "entity", "scope", "stability", "resolution_source",
)


@dataclasses.dataclass(frozen=True)
class Proposition:
    subject: str
    kind: Kind
    measure: str | None = None
    comparator: Comparator | None = None
    value: float | None = None
    low: float | None = None
    high: float | None = None
    low_inclusive: bool = True
    high_inclusive: bool = True
    unit: str | None = None
    observation: Observation | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    entity: str | None = None
    scope: str | None = None
    stability: Stability = Stability.LIVE
    resolution_source: str | None = None
    negated: bool = False
    unknown: frozenset[str] = frozenset()

    def __post_init__(self):
        if not SUBJECT_KEY_PATTERN.match(self.subject):
            raise ValueError(f"subject {self.subject!r} is not a subject key")
        if self.kind == Kind.THRESHOLD and "comparator" not in self.unknown and self.comparator is None:
            raise ValueError("a threshold needs a comparator or comparator in unknown")
        if self.kind == Kind.THRESHOLD and "value" not in self.unknown and self.value is None:
            raise ValueError("a threshold needs a value or value in unknown")
        if self.kind == Kind.RANGE and not ({"low", "high"} & self.unknown) and (self.low is None or self.high is None):
            raise ValueError("a range needs low and high")
        if self.kind == Kind.WINNER and "entity" not in self.unknown and not self.entity:
            raise ValueError("a winner proposition needs an entity")
        bad = set(self.unknown) - set(FIELDS)
        if bad:
            raise ValueError(f"unknown names fields that do not exist: {sorted(bad)}")

    def negate(self) -> "Proposition":
        return dataclasses.replace(self, negated=not self.negated)

    def to_fields(self) -> dict:
        """The JSON stored as sm.contract_proposition.fields (the subject travels as the row's subject)."""
        out: dict = {}
        for f in dataclasses.fields(self):
            if f.name == "subject":
                continue
            v = getattr(self, f.name)
            if isinstance(v, datetime):
                v = v.isoformat()
            elif isinstance(v, frozenset):
                v = sorted(v)
            elif isinstance(v, StrEnum):
                v = v.value
            out[f.name] = v
        return out

    @classmethod
    def from_fields(cls, subject: str, fields: dict) -> "Proposition":
        def dt(v):
            return datetime.fromisoformat(v) if v else None

        def enum(e, v):
            return e(v) if v else None

        return cls(
            subject=subject,
            kind=Kind(fields["kind"]),
            measure=fields.get("measure"),
            comparator=enum(Comparator, fields.get("comparator")),
            value=fields.get("value"),
            low=fields.get("low"),
            high=fields.get("high"),
            low_inclusive=fields.get("low_inclusive", True),
            high_inclusive=fields.get("high_inclusive", True),
            unit=fields.get("unit"),
            observation=enum(Observation, fields.get("observation")),
            window_start=dt(fields.get("window_start")),
            window_end=dt(fields.get("window_end")),
            entity=fields.get("entity"),
            scope=fields.get("scope"),
            stability=Stability(fields.get("stability") or Stability.LIVE),
            resolution_source=fields.get("resolution_source"),
            negated=fields.get("negated", False),
            unknown=frozenset(fields.get("unknown") or ()),
        )


@dataclasses.dataclass(frozen=True)
class Extraction:
    """A proposition with how it was read and from what."""
    proposition: Proposition
    method: Method
    source_hash: str
    extractor_version: str = EXTRACTOR_VERSION


def subject_key(*parts: str) -> str:
    """Joins parts into a subject key: lowercase, words joined by '-', parts by ':'."""
    cleaned = [re.sub(r"[^a-z0-9.]+", "-", p.lower()).strip("-") for p in parts if p]
    key = ":".join(c for c in cleaned if c)
    if not SUBJECT_KEY_PATTERN.match(key):
        raise ValueError(f"cannot make a subject key from {parts!r}")
    return key
