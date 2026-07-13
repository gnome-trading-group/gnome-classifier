import hashlib
from abc import ABC, abstractmethod


class ClassifierCache(ABC):
    def _canon_hash(self, model: str, exchange_id: int, native_id: str) -> str:
        content = model + "\x00" + str(exchange_id) + "\x00" + native_id
        return hashlib.sha256(content.encode()).hexdigest()

    def _judge_hash(
        self,
        model: str,
        title_a: str,
        labels_a: list[str],
        title_b: str,
        labels_b: list[str],
    ) -> str:
        pair_a = (title_a, "|".join(labels_a))
        pair_b = (title_b, "|".join(labels_b))
        if pair_a > pair_b:
            pair_a, pair_b = pair_b, pair_a
        content = (
            model + "\x00"
            + pair_a[0] + "\x00" + pair_a[1] + "\x00"
            + pair_b[0] + "\x00" + pair_b[1]
        )
        return hashlib.sha256(content.encode()).hexdigest()

    @abstractmethod
    def get_canonicalization(self, model: str, exchange_id: int, native_id: str) -> dict | None:
        ...

    @abstractmethod
    def get_canonicalization_bulk(
        self, model: str, pairs: list[tuple[int, str]]
    ) -> dict[tuple[int, str], dict]:
        ...

    @abstractmethod
    def put_canonicalization(
        self, model: str, exchange_id: int, native_id: str, result: dict
    ) -> None:
        ...

    @abstractmethod
    def get_judgment(
        self,
        model: str,
        title_a: str,
        labels_a: list[str],
        title_b: str,
        labels_b: list[str],
    ) -> tuple[list, bool] | None:
        ...

    @abstractmethod
    def put_judgment(
        self,
        model: str,
        title_a: str,
        labels_a: list[str],
        title_b: str,
        labels_b: list[str],
        items: list,
        a_is_first: bool,
    ) -> None:
        ...

    def get_exchange_event(self, exchange_id: int, native_id: str) -> int | None:
        return None

    def put_exchange_event(self, exchange_id: int, native_id: str, event_id: int) -> None:
        pass

