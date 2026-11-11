"""Which events the paid relationship work (Claude reading propositions, the subject merge judge) is spent on. The code
readers and derivation are free, so they cover every event whatever the scope."""
import dataclasses
from collections.abc import Iterable


@dataclasses.dataclass(frozen=True)
class LLMScope:
    """Events in any listed category or carrying any listed tag; no categories and no tags is every event."""
    categories: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()

    @classmethod
    def of(cls, categories: Iterable[str] = (), tags: Iterable[str] = ()) -> "LLMScope":
        # The registry stores categories upper case and tags lower case; config is typed by hand.
        return cls(frozenset(c.strip().upper() for c in categories if c.strip()),
                   frozenset(t.strip().lower() for t in tags if t.strip()))

    @property
    def unlimited(self) -> bool:
        return not self.categories and not self.tags

    def includes(self, category: str | None, tags: Iterable[str]) -> bool:
        if self.unlimited:
            return True
        if category and category.upper() in self.categories:
            return True
        return any(t.lower() in self.tags for t in tags)

    def fingerprint(self) -> str:
        """Stored on securities parked out of scope, so a scope change looks at them again."""
        if self.unlimited:
            return "*"
        return f"categories={','.join(sorted(self.categories))};tags={','.join(sorted(self.tags))}"


EVERYTHING = LLMScope()
