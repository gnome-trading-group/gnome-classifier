"""Securities that failed to get a proposition, so passes don't keep paying to ask again.

A failed read (the model answered without a valid entry for it) waits 10 minutes, then an hour, 6 hours and a day;
after the fifth failure it waits a week. A security no code reader handles waits a week while the LLM is off, and is
read as soon as it is turned on. Records live in the classifier's Redis: nothing else reads them, and losing one
(eviction, node replacement) only means an early re-check. Each record holds its retry time and expires a week after
it, so the attempt count survives the wait; every key expires because the node evicts only keys that do.

A record names the source and extractor version it was made against; one that no longer matches the security's
current source counts as no record, so a changed source is read straight away. A parked record also names the code
readers' version, so any change to the readers looks at parked securities again (free: no model call).
"""
import dataclasses
import json
import logging
import time
from collections.abc import Iterable
from typing import Protocol

import redis as redis_lib

logger = logging.getLogger(__name__)

FAILED = "FAILED"
NEEDS_LLM = "NEEDS_LLM"

BACKOFF_SECONDS = (600, 3600, 6 * 3600, 24 * 3600)
GIVE_UP_AFTER = 5
WEEK_SECONDS = 7 * 24 * 3600

_KEY = "prop:attempt:{}"


@dataclasses.dataclass(frozen=True)
class Attempt:
    source_hash: str
    extractor_version: str
    reason: str
    attempts: int
    retry_at: float
    readers_version: str = ""


def wait_seconds(attempts: int) -> int:
    return BACKOFF_SECONDS[attempts - 1] if attempts <= len(BACKOFF_SECONDS) else WEEK_SECONDS


class AttemptStore(Protocol):
    def now(self) -> float: ...

    def get_many(self, security_ids: list[int]) -> dict[int, Attempt]: ...

    def put(self, security_id: int, attempt: Attempt, ttl_seconds: int) -> None: ...

    def clear(self, security_ids: list[int]) -> None: ...


class RedisAttemptStore:
    def __init__(self, redis_url: str):
        self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def now(self) -> float:
        return time.time()

    def get_many(self, security_ids: list[int]) -> dict[int, Attempt]:
        if not security_ids:
            return {}
        values = self._redis.mget([_KEY.format(sid) for sid in security_ids])
        return {sid: Attempt(**json.loads(v)) for sid, v in zip(security_ids, values) if v is not None}

    def put(self, security_id: int, attempt: Attempt, ttl_seconds: int) -> None:
        self._redis.set(_KEY.format(security_id), json.dumps(dataclasses.asdict(attempt)), ex=ttl_seconds)

    def clear(self, security_ids: list[int]) -> None:
        if security_ids:
            self._redis.delete(*[_KEY.format(sid) for sid in security_ids])


def _current(attempt: Attempt | None, source_hash: str, extractor_version: str) -> Attempt | None:
    if attempt is None or (attempt.source_hash, attempt.extractor_version) != (source_hash, extractor_version):
        return None
    return attempt


def waiting(store: AttemptStore, sources: dict[int, str], extractor_version: str, llm_enabled: bool,
            readers_version: str) -> set[int]:
    """The securities among `sources` ({security_id: current source hash}) that are waiting out a failure, or, while
    the LLM is off and the readers are unchanged, parked for it."""
    attempts = store.get_many(sorted(sources))
    now = store.now()
    out = set()
    for sid, attempt in attempts.items():
        current = _current(attempt, sources[sid], extractor_version)
        if current is None or current.retry_at <= now:
            continue
        if current.reason == NEEDS_LLM and (llm_enabled or current.readers_version != readers_version):
            continue
        out.add(sid)
    return out


def record(
    store: AttemptStore,
    *,
    failed: Iterable[tuple[int, str]],
    needs_llm: Iterable[tuple[int, str]],
    read: list[int],
    extractor_version: str,
    readers_version: str,
) -> tuple[int, int]:
    """Records failed reads and parked securities ((security_id, source hash) each) and clears read ones. Returns how
    many failures this gave up on (reached the weekly wait) and how many were parked."""
    failed, needs_llm = list(failed), list(needs_llm)
    before = store.get_many(sorted({sid for sid, _ in failed}))
    now = store.now()
    gave_up = 0
    for sid, source_hash in failed:
        previous = _current(before.get(sid), source_hash, extractor_version)
        attempts = (previous.attempts if previous is not None and previous.reason == FAILED else 0) + 1
        wait = wait_seconds(attempts)
        store.put(sid, Attempt(source_hash, extractor_version, FAILED, attempts, now + wait), wait + WEEK_SECONDS)
        if attempts == GIVE_UP_AFTER:
            gave_up += 1
            # Logged once per contract: one the model keeps failing on needs its rules looked at.
            logger.warning("proposition read gave up: security=%d attempts=%d", sid, attempts)
    for sid, source_hash in needs_llm:
        store.put(sid, Attempt(source_hash, extractor_version, NEEDS_LLM, 0, now + WEEK_SECONDS, readers_version),
                  2 * WEEK_SECONDS)
    store.clear(read)
    return gave_up, len(needs_llm)
