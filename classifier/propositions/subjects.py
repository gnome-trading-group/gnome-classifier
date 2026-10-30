"""Candidate subjects for the LLM extractor, so a contract on another venue reuses the subject (and the entity keys)
an existing one already has instead of coining a second name for the same thing.

Candidates come from subjects already known (stored ones and those read earlier in the same pass), ranked by word
overlap with the contract's event. The registry's subject embeddings add their nearest neighbours on top of these.
"""
import dataclasses
import math
import re
from collections.abc import Callable
from typing import Protocol

from classifier.propositions.types import Kind, Proposition

DEFAULT_CANDIDATES = 8
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "vs", "v", "of", "in", "on", "a", "an", "will", "be", "to", "by", "and", "at", "for", "game", "match", "who"}


@dataclasses.dataclass
class KnownSubject:
    key: str
    description: str
    entities: set[str] = dataclasses.field(default_factory=set)
    measures: set[str] = dataclasses.field(default_factory=set)

    def words(self) -> set[str]:
        return _words(" ".join([self.key, self.description, *self.entities]))


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def entity_of(measure: str | None) -> str | None:
    """The entity a per-entity measure names (team_points:mil, margin:mil)."""
    if measure and ":" in measure:
        return measure.split(":", 1)[1]
    return None


def add_known(known: dict[str, KnownSubject], proposition: Proposition, description: str) -> None:
    subject = known.setdefault(proposition.subject, KnownSubject(proposition.subject, description))
    if not subject.description:
        subject.description = description
    if proposition.kind == Kind.WINNER and proposition.entity:
        subject.entities.add(proposition.entity)
    if (entity := entity_of(proposition.measure)) is not None:
        subject.entities.add(entity)
    if proposition.measure:
        subject.measures.add(proposition.measure)


def candidates_for(text: str, known: dict[str, KnownSubject], limit: int = DEFAULT_CANDIDATES) -> list[KnownSubject]:
    words = _words(text)
    if not words:
        return []
    scored = []
    for subject in known.values():
        overlap = len(words & subject.words())
        if overlap:
            scored.append((overlap / len(words | subject.words()), subject.key, subject))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [s for _, _, s in scored[:limit]]


def subject_text(subject: KnownSubject) -> str:
    """What a subject is embedded as: its key, description and the entities it covers."""
    parts = [subject.key, subject.description]
    if subject.entities:
        parts.append("entities: " + ", ".join(sorted(subject.entities)))
    return ". ".join(p for p in parts if p)


class SubjectIndex(Protocol):
    def neighbours(self, texts: list[str], limit: int) -> list[list[KnownSubject]]:
        """The nearest known subjects to each text."""

    def add(self, subjects: list[KnownSubject]) -> None:
        """Subjects coined during this pass, so later requests can find them."""


Embed = Callable[[list[str], str], list[list[float]]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemorySubjectIndex:
    """Embeds subjects as they are added and searches them by cosine similarity: the subjects of one pass."""

    def __init__(self, embed: Embed):
        self._embed = embed
        self._subjects: dict[str, KnownSubject] = {}
        self._vectors: dict[str, list[float]] = {}

    def add(self, subjects: list[KnownSubject]) -> None:
        new = [s for s in subjects if s.key not in self._vectors]
        for s in subjects:
            self._subjects[s.key] = s
        if new:
            for subject, vector in zip(new, self._embed([subject_text(s) for s in new], "document")):
                self._vectors[subject.key] = vector

    def neighbours(self, texts: list[str], limit: int) -> list[list[KnownSubject]]:
        if not texts or not self._vectors:
            return [[] for _ in texts]
        out = []
        for query in self._embed(texts, "query"):
            ranked = sorted(self._vectors, key=lambda k: -_cosine(query, self._vectors[k]))
            out.append([self._subjects[k] for k in ranked[:limit]])
        return out


class StoredSubjectIndex:
    """The registry's stored subjects, by pgvector distance over their embeddings."""

    def __init__(self, db, embed: Embed):
        self._db = db
        self._embed = embed

    def add(self, subjects: list[KnownSubject]) -> None:
        # Stored subjects are embedded after their propositions are written (`embed_subjects`).
        return None

    def neighbours(self, texts: list[str], limit: int) -> list[list[KnownSubject]]:
        if not texts:
            return []
        return [self._db.nearest_subjects(vector, limit) for vector in self._embed(texts, "query")]


class CombinedSubjectIndex:
    def __init__(self, *indexes: SubjectIndex):
        self._indexes = indexes

    def add(self, subjects: list[KnownSubject]) -> None:
        for index in self._indexes:
            index.add(subjects)

    def neighbours(self, texts: list[str], limit: int) -> list[list[KnownSubject]]:
        merged: list[list[KnownSubject]] = [[] for _ in texts]
        for index in self._indexes:
            for i, found in enumerate(index.neighbours(texts, limit)):
                seen = {s.key for s in merged[i]}
                merged[i].extend(s for s in found if s.key not in seen)
        return [m[:limit] for m in merged]


def merge_candidates(*lists: list[KnownSubject], limit: int = DEFAULT_CANDIDATES) -> list[KnownSubject]:
    out, seen = [], set()
    for candidates in lists:
        for c in candidates:
            if c.key not in seen:
                seen.add(c.key)
                out.append(c)
    return out[:limit]


def embed_subjects(db, embed: Embed, batch: int = 1000) -> int:
    """Embeds stored subjects that have no embedding yet; returns how many."""
    total = 0
    while subjects := db.subjects_without_embeddings(batch):
        db.put_subject_embeddings(dict(zip((s.key for s in subjects), embed([subject_text(s) for s in subjects], "document"))))
        total += len(subjects)
        if len(subjects) < batch:
            break
    return total
