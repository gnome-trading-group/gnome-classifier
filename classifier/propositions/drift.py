"""The daily drift check: re-reads a sample of code-read contracts with the LLM and compares the two readings field
by field. A disagreement means either the venue changed what its rules say while the wording still parses, or the
reader has a bug; each is logged for the metric filter and returned for the report.
"""
import dataclasses
import logging
import random
from datetime import date, timedelta

from classifier.propositions.extract import family, request_group
from classifier.propositions.inputs import ExtractionInput, is_primary_side
from classifier.propositions.llm import PropositionExtractor
from classifier.propositions.types import Proposition
from classifier.propositions.venue import extract_venue

logger = logging.getLogger(__name__)

COMPARED_FIELDS = ("kind", "comparator", "value", "low", "high", "observation", "negated")
WINDOW_TOLERANCE = timedelta(hours=12)
DEFAULT_SAMPLE = 40


@dataclasses.dataclass(frozen=True)
class Drift:
    security_id: int
    family: str
    field: str
    venue: object
    llm: object


def _differences(venue: Proposition, llm: Proposition) -> list[tuple[str, object, object]]:
    a, b = venue.to_fields(), llm.to_fields()
    out = [(f, a[f], b[f]) for f in COMPARED_FIELDS if a[f] != b[f]]
    if venue.window_end and llm.window_end and abs(venue.window_end - llm.window_end) > WINDOW_TOLERANCE:
        out.append(("window_end", a["window_end"], b["window_end"]))
    return out


def sample(items: list[ExtractionInput], size: int = DEFAULT_SAMPLE, day: date | None = None) -> list[ExtractionInput]:
    """A day's sample of code-readable first sides: the same on reruns within a day, different across days."""
    readable = sorted((i for i in items if is_primary_side(i) and extract_venue(i.source) is not None),
                      key=lambda i: i.security_id)
    return random.Random((day or date.today()).toordinal()).sample(readable, min(size, len(readable)))


def check_drift(items: list[ExtractionInput], extractor: PropositionExtractor) -> list[Drift]:
    groups: dict[str, tuple[str, list[ExtractionInput]]] = {}
    for item in items:
        groups.setdefault(request_group(item), (item.event_title, []))[1].append(item)
    answers = extractor.extract(groups, {})
    drifts = []
    for item in items:
        venue = extract_venue(item.source)
        answer = answers.get(item.security_id)
        if venue is None or answer is None:
            continue
        for field, venue_value, llm_value in _differences(venue, answer[0]):
            drift = Drift(item.security_id, family(item), field, venue_value, llm_value)
            # Log wording kept stable: a metric filter counts these.
            logger.warning("proposition drift: family=%s security=%d field=%s venue=%r llm=%r",
                           drift.family, drift.security_id, field, venue_value, llm_value)
            drifts.append(drift)
    return drifts
