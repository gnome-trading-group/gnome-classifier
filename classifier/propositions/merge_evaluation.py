"""Scores subject merge judgments against the hand-labelled pairs in `tests/fixtures/subjects/pairs.jsonl`.

Each golden pair holds both subjects as the judge sees them (`SubjectProfile.to_json`, read from prod) and the verdict:
`SAME`, `RELATED` or `DIFFERENT`. Only SAME merges, so SAME precision is what the gate checks: a wrong SAME makes
different contests' contracts comparable. RELATED and DIFFERENT are both "don't merge", so mixing them up costs nothing.
"""
import dataclasses
import json
from collections import Counter
from pathlib import Path

from classifier.propositions.merges import Judgment, SubjectProfile, Verdict

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "subjects" / "pairs.jsonl"
MISSING = "MISSING"


@dataclasses.dataclass(frozen=True)
class GoldenSubjectPair:
    id: str
    a: SubjectProfile
    b: SubjectProfile
    expected: Verdict
    note: str
    source: str

    @property
    def keys(self) -> tuple[str, str]:
        return self.a.key, self.b.key


@dataclasses.dataclass
class MergeReport:
    scored: list[tuple[GoldenSubjectPair, Judgment | None]]
    confusion: Counter

    @property
    def predicted_same(self) -> int:
        return sum(1 for _, j in self.scored if j and j.verdict == Verdict.SAME)

    @property
    def correct_same(self) -> int:
        return sum(1 for p, j in self.scored if j and j.verdict == Verdict.SAME and p.expected == Verdict.SAME)

    @property
    def expected_same(self) -> int:
        return sum(1 for p, _ in self.scored if p.expected == Verdict.SAME)

    @property
    def precision(self) -> float | None:
        return self.correct_same / self.predicted_same if self.predicted_same else None

    @property
    def recall(self) -> float | None:
        return self.correct_same / self.expected_same if self.expected_same else None

    def wrong_merges(self) -> list[tuple[GoldenSubjectPair, Judgment]]:
        return [(p, j) for p, j in self.scored if j and j.verdict == Verdict.SAME and p.expected != Verdict.SAME]

    def missed_merges(self) -> list[tuple[GoldenSubjectPair, Judgment | None]]:
        return [(p, j) for p, j in self.scored if p.expected == Verdict.SAME and (not j or j.verdict != Verdict.SAME)]

    def passes(self, min_precision: float) -> bool:
        return self.precision is None or self.precision >= min_precision


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenSubjectPair]:
    pairs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pairs.append(GoldenSubjectPair(
            id=row["id"], a=SubjectProfile.from_json(row["a"]), b=SubjectProfile.from_json(row["b"]),
            expected=Verdict(row["expected"]), note=row.get("note", ""), source=row.get("source", ""),
        ))
    return pairs


def score(golden: list[GoldenSubjectPair], judgments: dict[tuple[str, str], Judgment | None]) -> MergeReport:
    scored, confusion = [], Counter()
    for pair in golden:
        judgment = judgments.get(pair.keys)
        scored.append((pair, judgment))
        confusion[(pair.expected.value, judgment.verdict.value if judgment else MISSING)] += 1
    return MergeReport(scored, confusion)


def save_judgments(path: Path, judgments: dict[tuple[str, str], Judgment | None]) -> None:
    path.write_text(json.dumps([{"a": a, "b": b, "judgment": j.to_json() if j else None}
                                for (a, b), j in judgments.items()], indent=1))


def load_judgments(path: Path) -> dict[tuple[str, str], Judgment | None]:
    return {(r["a"], r["b"]): Judgment.from_json(r["judgment"]) if r["judgment"] else None
            for r in json.loads(path.read_text())}
