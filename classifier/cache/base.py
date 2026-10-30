import hashlib
from abc import ABC, abstractmethod


class ClassifierCache(ABC):
    def _canon_hash(self, model: str, exchange_id: int, native_id: str) -> str:
        content = model + "\x00" + str(exchange_id) + "\x00" + native_id
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
