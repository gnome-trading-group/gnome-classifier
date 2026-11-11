"""Scores key judgments against the hand-labelled slots in `tests/fixtures/keys/slots.jsonl`.

Each golden slot holds the slot as the judge sees it (`Slot.to_json`), the groups of its keys that name one entity, and
any pairs the labeller couldn't settle (`unsure`), which count neither way. Scored by pairs of keys: a pair the judge
joins that the labels don't is a wrong join, which makes two different entities' contracts comparable, so join
precision is what the gate checks.
"""
import dataclasses
import json
from pathlib import Path

from classifier.propositions.key_reconcile import KeyJudgment, Slot, held_to_slot

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "keys" / "slots.jsonl"


@dataclasses.dataclass(frozen=True)
class GoldenSlot:
    id: str
    slot: Slot
    expected: tuple[tuple[str, ...], ...]
    note: str
    source: str
    unsure: frozenset[frozenset[str]] = frozenset()


def _pairs(groups) -> set[frozenset[str]]:
    return {frozenset((a, b)) for g in groups for a in g for b in g if a < b}


@dataclasses.dataclass
class KeyReport:
    scored: list[tuple[GoldenSlot, KeyJudgment | None]]

    def _counts(self) -> tuple[int, int, int]:
        predicted = correct = expected = 0
        for golden, judgment in self.scored:
            want = _pairs(golden.expected) - golden.unsure
            got = (_pairs(held_to_slot(golden.slot, judgment).groups) if judgment else set()) - golden.unsure
            predicted += len(got)
            correct += len(got & want)
            expected += len(want)
        return predicted, correct, expected

    @property
    def predicted(self) -> int:
        return self._counts()[0]

    @property
    def correct(self) -> int:
        return self._counts()[1]

    @property
    def expected(self) -> int:
        return self._counts()[2]

    @property
    def precision(self) -> float | None:
        p, c, _ = self._counts()
        return c / p if p else None

    @property
    def recall(self) -> float | None:
        _, c, e = self._counts()
        return c / e if e else None

    @property
    def missing(self) -> int:
        return sum(1 for _, j in self.scored if j is None)

    def wrong_joins(self) -> list[tuple[GoldenSlot, KeyJudgment, set[frozenset[str]]]]:
        out = []
        for golden, judgment in self.scored:
            if judgment is None:
                continue
            wrong = _pairs(held_to_slot(golden.slot, judgment).groups) - _pairs(golden.expected) - golden.unsure
            if wrong:
                out.append((golden, judgment, wrong))
        return out

    def missed_joins(self) -> list[tuple[GoldenSlot, KeyJudgment | None, set[frozenset[str]]]]:
        out = []
        for golden, judgment in self.scored:
            got = _pairs(held_to_slot(golden.slot, judgment).groups) if judgment else set()
            missed = _pairs(golden.expected) - got - golden.unsure
            if missed:
                out.append((golden, judgment, missed))
        return out

    def passes(self, min_precision: float) -> bool:
        return self.missing == 0 and (self.precision is None or self.precision >= min_precision)


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenSlot]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(GoldenSlot(row["id"], Slot.from_json(row["slot"]), tuple(tuple(g) for g in row["expected"]),
                              row.get("note", ""), row.get("source", ""),
                              frozenset(frozenset(p) for p in row.get("unsure") or ())))
    return out


def score(golden: list[GoldenSlot], judgments: dict[str, KeyJudgment | None]) -> KeyReport:
    return KeyReport([(g, judgments.get(g.slot.id)) for g in golden])


def save_judgments(path: Path, judgments: dict[str, KeyJudgment | None]) -> None:
    path.write_text(json.dumps({k: (j.to_json() if j else None) for k, j in judgments.items()}, indent=1))


def load_judgments(path: Path) -> dict[str, KeyJudgment | None]:
    return {k: (KeyJudgment.from_json(v) if v else None) for k, v in json.loads(path.read_text()).items()}
