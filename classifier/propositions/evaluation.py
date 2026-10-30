"""Scores extracted propositions against the labelled ones in `tests/fixtures/relationships/propositions.jsonl`.

Subjects are free-form keys, so they are scored by agreement: contracts labelled with the same subject group (the two
sides of a related golden pair, linked transitively) should be given the same subject, and likewise entity groups for
winners of the same contest. Other fields are compared exactly, except window_end, which may differ by up to 12 hours
(venues and the extractor disagree about a date's time zone).
"""
import dataclasses
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from gnomepy.registry.types import ContractType

from classifier.adapters.types import proposition_source_hash
from classifier.propositions.inputs import ExtractionInput
from classifier.propositions.types import Extraction

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "relationships" / "propositions.jsonl"
EXACT_FIELDS = ("kind", "measure", "comparator", "value", "low", "high", "observation", "scope", "negated")
WINDOW_TOLERANCE = timedelta(hours=12)


@dataclasses.dataclass(frozen=True)
class LabelledProposition:
    key: str
    item: ExtractionInput
    subject_group: str | None
    entity_group: str | None
    expected: dict | None
    review: bool


@dataclasses.dataclass
class PropositionReport:
    subject_pairs: int = 0
    subject_agreed: int = 0
    entity_pairs: int = 0
    entity_agreed: int = 0
    field_checked: Counter = dataclasses.field(default_factory=Counter)
    field_correct: Counter = dataclasses.field(default_factory=Counter)
    missing: list[str] = dataclasses.field(default_factory=list)
    disagreements: list[tuple[str, str, object, object]] = dataclasses.field(default_factory=list)
    split_groups: list[tuple[str, list[tuple[str, str]]]] = dataclasses.field(default_factory=list)

    @property
    def subject_agreement(self) -> float | None:
        return self.subject_agreed / self.subject_pairs if self.subject_pairs else None

    @property
    def entity_agreement(self) -> float | None:
        return self.entity_agreed / self.entity_pairs if self.entity_pairs else None

    def field_accuracy(self, field: str) -> float | None:
        n = self.field_checked[field]
        return self.field_correct[field] / n if n else None


INPUTS_PATH = GOLDEN_PATH.with_name("inputs.jsonl")


def _row_input(row: dict, security_id: int) -> ExtractionInput:
    return ExtractionInput(
        security_id=security_id, exchange=row["exchange"], event_id=_event_id(row),
        contract_type=ContractType[row["contract_type"]], event_title=row["event_title"],
        outcome_label=row["outcome_label"], description=row.get("description"), expiry=row.get("expiry"),
        source=row["source"], source_hash=proposition_source_hash(row["source"]),
    )


def load_inputs(path: Path = INPUTS_PATH) -> dict[str, ExtractionInput]:
    """{listing key: input} for every security in the golden pairs (and binary partners), sources rebuilt from what
    the registry stored."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["key"]: _row_input(row, i) for i, row in enumerate(rows, 1)}


def load_labelled(path: Path = GOLDEN_PATH) -> list[LabelledProposition]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(LabelledProposition(row["key"], _row_input(row, len(out) + 1), row.get("subject_group"),
                                       row.get("entity_group"), row.get("expected"), row.get("review", False)))
    return out


def _event_id(row: dict) -> int:
    # Binary sides share an event; the labels carry the venue's own event identity in the source.
    source = row["source"]
    market = source.get("market") or {}
    native = (market.get("ticker") or "").partition(":")[0] if source.get("venue") == "KALSHI" else (
        market.get("question") or market.get("title") or row["event_title"])
    return int(hashlib.sha256(f"{row['exchange']}|{native}".encode()).hexdigest()[:7], 16)


def _window_matches(expected: str | None, got: datetime | None) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    return abs(datetime.fromisoformat(expected) - got) <= WINDOW_TOLERANCE


def score(labelled: list[LabelledProposition], results: dict[int, Extraction]) -> PropositionReport:
    report = PropositionReport()
    subject_of = {}
    for lp in labelled:
        extraction = results.get(lp.item.security_id)
        if extraction is None:
            report.missing.append(lp.key)
            continue
        p = extraction.proposition
        subject_of[lp.key] = (p.subject, p.entity)
        if lp.expected is None:
            continue
        got = p.to_fields()
        for field in EXACT_FIELDS:
            report.field_checked[field] += 1
            if got[field] == lp.expected.get(field):
                report.field_correct[field] += 1
            else:
                report.disagreements.append((lp.key, field, lp.expected.get(field), got[field]))
        report.field_checked["window_end"] += 1
        if _window_matches(lp.expected.get("window_end"), p.window_end):
            report.field_correct["window_end"] += 1
        else:
            report.disagreements.append((lp.key, "window_end", lp.expected.get("window_end"), got["window_end"]))

    for attr, index in (("subject", 0), ("entity", 1)):
        groups: dict[str, list[str]] = defaultdict(list)
        for lp in labelled:
            group = lp.subject_group if attr == "subject" else lp.entity_group
            if group and lp.key in subject_of:
                groups[group].append(lp.key)
        for group, keys in groups.items():
            values = {k: subject_of[k][index] for k in keys}
            for a, b in itertools.combinations(keys, 2):
                setattr(report, f"{attr}_pairs", getattr(report, f"{attr}_pairs") + 1)
                if values[a] == values[b]:
                    setattr(report, f"{attr}_agreed", getattr(report, f"{attr}_agreed") + 1)
            if attr == "subject" and len(set(values.values())) > 1:
                report.split_groups.append((group, sorted(values.items())))
    return report
