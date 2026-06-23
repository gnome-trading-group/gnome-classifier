import dataclasses
import math
import re
from datetime import datetime, timedelta

from classifier.types import Embedding, Similarity


def cosine_similarity(a: Embedding, b: Embedding) -> Similarity:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def from_dict(cls, data: dict):
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def expiry_close(a: str | None, b: str | None, tolerance: timedelta) -> bool:
    if a is None or b is None:
        return True
    try:
        da = datetime.fromisoformat(a.replace("Z", "+00:00"))
        db = datetime.fromisoformat(b.replace("Z", "+00:00"))
        return abs((da - db).total_seconds()) <= tolerance.total_seconds()
    except ValueError:
        return True


def generate_security_symbol(canonical_title: str, outcome_label: str) -> str:
    slug = re.sub(r'[^a-z0-9\s]', '', canonical_title.lower()).strip()
    slug = re.sub(r'\s+', '-', slug)[:80]
    outcome = re.sub(r'[^a-z0-9\s]', '', outcome_label.lower()).strip()
    outcome = re.sub(r'\s+', '-', outcome)
    return f"{slug}-{outcome}".upper()
