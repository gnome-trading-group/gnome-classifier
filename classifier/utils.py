import dataclasses
import logging
import re
from typing import Iterator

from classifier.constants import DEFAULT_BULK_CREATE_BATCH_SIZE

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def from_dict(cls, data: dict):
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


_SYMBOL_PART_RE = re.compile(r"[^A-Z0-9.]+")


def format_security_symbol(prefix: str, *parts: str) -> str:
    cleaned = [_SYMBOL_PART_RE.sub("-", part.upper()).strip("-") for part in parts]
    return "-".join([prefix, *cleaned])


def strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def bulk_create_chunked(items: list[dict], label: str, batch_size: int = DEFAULT_BULK_CREATE_BATCH_SIZE) -> Iterator[tuple[int, list[dict]]]:
    total_chunks = -(-len(items) // batch_size)
    for chunk_start in range(0, len(items), batch_size):
        chunk = items[chunk_start:chunk_start + batch_size]
        chunk_num = chunk_start // batch_size + 1
        logger.info("Creating %s: chunk %d/%d (%d-%d of %d)", label, chunk_num, total_chunks, chunk_start + 1, chunk_start + len(chunk), len(items))
        yield chunk_start, chunk
