import dataclasses
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Iterator

from classifier.constants import DEFAULT_BULK_CREATE_BATCH_SIZE

logger = logging.getLogger(__name__)


def from_dict(cls, data: dict):
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def expiry_close(a: str | None, b: str | None, tolerance: timedelta) -> bool:
    if a is None or b is None:
        return True
    try:
        return abs((_parse_utc(a) - _parse_utc(b)).total_seconds()) <= tolerance.total_seconds()
    except ValueError:
        return True


def generate_security_symbol(canonical_title: str, outcome_label: str) -> str:
    slug = re.sub(r'[^a-z0-9\s]', '', canonical_title.lower()).strip()
    slug = re.sub(r'\s+', '-', slug)[:80]
    outcome = re.sub(r'[^a-z0-9\s]', '', outcome_label.lower()).strip()
    outcome = re.sub(r'\s+', '-', outcome)
    return f"{slug}-{outcome}".upper()


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
