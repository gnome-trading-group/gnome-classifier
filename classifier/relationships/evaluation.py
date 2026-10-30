"""Scores predicted contract relationships against the hand-labelled golden pairs in
`tests/fixtures/relationships/pairs.jsonl`.

A golden pair names two securities by listing identity and the one relationship between them: `EQUIVALENT`,
`A_IMPLIES_B`, `B_IMPLIES_A` or `NONE`. Predictions are relationship rows (`EQUIVALENT` or `IMPLIES`, a -> b) between
any securities; only rows between the two sides of a golden pair are scored.

A predicted `IMPLIES` is correct when it is logically true, so one between an equivalent pair counts as correct (and
as a missed equivalence). A predicted `EQUIVALENT` is correct only on an equivalent pair. `IMPLIES` both ways reads as
`EQUIVALENT`.
"""
import dataclasses
import json
from collections import Counter
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "relationships"

EQUIVALENT = "EQUIVALENT"
A_IMPLIES_B = "A_IMPLIES_B"
B_IMPLIES_A = "B_IMPLIES_A"
NONE = "NONE"
LABELS = (EQUIVALENT, A_IMPLIES_B, B_IMPLIES_A, NONE)

type ListingKey = tuple[str, str]


@dataclasses.dataclass(frozen=True)
class GoldenPair:
    id: str
    a: ListingKey
    b: ListingKey
    expected: str
    flags: tuple[str, ...]
    review: bool
    note: str
    source: str
    raw: dict


@dataclasses.dataclass(frozen=True)
class Prediction:
    a: ListingKey
    b: ListingKey
    type: str
    flags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Scored:
    pair: GoldenPair
    predicted: str
    predicted_flags: tuple[str, ...]


@dataclasses.dataclass
class TypeScore:
    predicted: int = 0
    correct: int = 0
    expected: int = 0
    found: int = 0

    @property
    def precision(self) -> float | None:
        return self.correct / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.found / self.expected if self.expected else None


@dataclasses.dataclass
class Report:
    scored: list[Scored]
    equivalent: TypeScore
    implies: TypeScore
    confusion: Counter

    def wrong(self) -> list[Scored]:
        return [s for s in self.scored if s.predicted != NONE and not _is_correct(s.predicted, s.pair.expected)]

    def missed(self) -> list[Scored]:
        return [s for s in self.scored if s.pair.expected != NONE and s.predicted != s.pair.expected]

    def flag_mismatches(self) -> list[Scored]:
        return [s for s in self.scored
                if s.predicted != NONE and s.pair.expected != NONE and set(s.predicted_flags) != set(s.pair.flags)]

    def passes(self, min_precision: float) -> bool:
        return all(t.precision is None or t.precision >= min_precision for t in (self.equivalent, self.implies))


def _key(side: dict) -> ListingKey:
    return side["exchange"], side["exchange_security_id"]


def load_golden(path: Path = GOLDEN_DIR / "pairs.jsonl") -> list[GoldenPair]:
    pairs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["expected"] not in LABELS:
            raise ValueError(f"golden pair {row['id']}: unknown label {row['expected']!r}")
        pairs.append(GoldenPair(
            id=row["id"], a=_key(row["a"]), b=_key(row["b"]), expected=row["expected"],
            flags=tuple(row.get("flags", ())), review=row.get("review", False), note=row.get("note", ""),
            source=row.get("source", ""), raw=row,
        ))
    return pairs


def load_predictions(path: Path) -> list[Prediction]:
    predictions = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        predictions.append(Prediction(_key(row["a"]), _key(row["b"]), row["type"], tuple(row.get("flags", ()))))
    return predictions


def predicted_label(pair: GoldenPair, predictions: list[Prediction]) -> tuple[str, tuple[str, ...]]:
    forward = backward = equivalent = False
    flags: set[str] = set()
    for p in predictions:
        if {p.a, p.b} != {pair.a, pair.b}:
            continue
        flags.update(p.flags)
        if p.type == EQUIVALENT:
            equivalent = True
        elif p.type == "IMPLIES":
            if (p.a, p.b) == (pair.a, pair.b):
                forward = True
            else:
                backward = True
    if equivalent or (forward and backward):
        return EQUIVALENT, tuple(sorted(flags))
    if forward:
        return A_IMPLIES_B, tuple(sorted(flags))
    if backward:
        return B_IMPLIES_A, tuple(sorted(flags))
    return NONE, ()


def _is_correct(predicted: str, expected: str) -> bool:
    if predicted == EQUIVALENT:
        return expected == EQUIVALENT
    return predicted == expected or expected == EQUIVALENT


def score(golden: list[GoldenPair], predictions: list[Prediction]) -> Report:
    by_pair: dict[frozenset, list[Prediction]] = {}
    for p in predictions:
        by_pair.setdefault(frozenset((p.a, p.b)), []).append(p)

    equivalent, implies, confusion, scored = TypeScore(), TypeScore(), Counter(), []
    for pair in golden:
        predicted, flags = predicted_label(pair, by_pair.get(frozenset((pair.a, pair.b)), []))
        scored.append(Scored(pair, predicted, flags))
        confusion[(pair.expected, predicted)] += 1

        if pair.expected == EQUIVALENT:
            equivalent.expected += 1
            equivalent.found += predicted == EQUIVALENT
        elif pair.expected in (A_IMPLIES_B, B_IMPLIES_A):
            implies.expected += 1
            implies.found += predicted == pair.expected

        if predicted == EQUIVALENT:
            equivalent.predicted += 1
            equivalent.correct += _is_correct(predicted, pair.expected)
        elif predicted in (A_IMPLIES_B, B_IMPLIES_A):
            implies.predicted += 1
            implies.correct += _is_correct(predicted, pair.expected)

    return Report(scored, equivalent, implies, confusion)
