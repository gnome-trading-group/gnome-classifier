import json
import logging

import redis as redis_lib

from classifier.cache.base import ClassifierCache

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL = 30 * 86400  # 30 days


class RedisClassifierCache(ClassifierCache):
    def __init__(self, redis_url: str, ttl: int = DEFAULT_CACHE_TTL):
        self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=False)
        self._ttl = ttl

    def get_canonicalization(self, model: str, exchange_id: int, native_id: str) -> dict | None:
        data = self._redis.get(f"canon:{self._canon_hash(model, exchange_id, native_id)}")
        if data is None:
            return None
        try:
            return json.loads(data)
        except Exception as e:
            logger.warning("Canon cache decode failed for %s/%s: %s", exchange_id, native_id, e)
            return None

    def get_canonicalization_bulk(
        self, model: str, pairs: list[tuple[int, str]]
    ) -> dict[tuple[int, str], dict]:
        if not pairs:
            return {}
        keys = [f"canon:{self._canon_hash(model, eid, nid)}" for eid, nid in pairs]
        raw_results = self._redis.mget(keys)
        out: dict[tuple[int, str], dict] = {}
        for (eid, nid), data in zip(pairs, raw_results):
            if data is not None:
                try:
                    out[(eid, nid)] = json.loads(data)
                except Exception:
                    pass
        return out

    def put_canonicalization(
        self, model: str, exchange_id: int, native_id: str, result: dict
    ) -> None:
        self._redis.set(
            f"canon:{self._canon_hash(model, exchange_id, native_id)}",
            json.dumps(result),
            ex=self._ttl,
        )

    def get_judgment(
        self,
        model: str,
        title_a: str,
        labels_a: list[str],
        title_b: str,
        labels_b: list[str],
    ) -> tuple[list, bool] | None:
        data = self._redis.get(f"judge:{self._judge_hash(model, title_a, labels_a, title_b, labels_b)}")
        if data is None:
            return None
        try:
            stored = json.loads(data)
        except Exception as e:
            logger.warning("Judge cache decode failed: %s", e)
            return None
        a_is_first = stored.get("first_title") == title_a
        return stored.get("items", []), a_is_first

    def get_judgment_bulk(
        self,
        model: str,
        keys: list[tuple[str, list[str], str, list[str]]],
    ) -> dict[int, tuple[list, bool]]:
        if not keys:
            return {}
        redis_keys = [
            f"judge:{self._judge_hash(model, title_a, labels_a, title_b, labels_b)}"
            for title_a, labels_a, title_b, labels_b in keys
        ]
        raw_results = self._redis.mget(redis_keys)
        out: dict[int, tuple[list, bool]] = {}
        for i, (data, (title_a, _, _, _)) in enumerate(zip(raw_results, keys)):
            if data is None:
                continue
            try:
                stored = json.loads(data)
            except Exception:
                continue
            a_is_first = stored.get("first_title") == title_a
            out[i] = (stored.get("items", []), a_is_first)
        return out

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
        first_title = title_a if a_is_first else title_b
        self._redis.set(
            f"judge:{self._judge_hash(model, title_a, labels_a, title_b, labels_b)}",
            json.dumps({"first_title": first_title, "items": items}),
            ex=self._ttl,
        )

    def get_exchange_event(self, exchange_id: int, native_id: str) -> int | None:
        val = self._redis.hget("exchange_events", f"{exchange_id}:{native_id}")
        return int(val) if val is not None else None

    def put_exchange_event(self, exchange_id: int, native_id: str, event_id: int) -> None:
        self._redis.hset("exchange_events", f"{exchange_id}:{native_id}", str(event_id))

    def get_exchange_event_bulk(
        self, pairs: list[tuple[int, str]]
    ) -> dict[tuple[int, str], int]:
        if not pairs:
            return {}
        pipeline = self._redis.pipeline()
        for eid, nid in pairs:
            pipeline.hget("exchange_events", f"{eid}:{nid}")
        raw_results = pipeline.execute()
        out: dict[tuple[int, str], int] = {}
        for (eid, nid), val in zip(pairs, raw_results):
            if val is not None:
                out[(eid, nid)] = int(val)
        return out

    def put_exchange_event_bulk(self, mapping: dict[tuple[int, str], int]) -> None:
        if not mapping:
            return
        pipeline = self._redis.pipeline()
        for (eid, nid), event_id in mapping.items():
            pipeline.hset("exchange_events", f"{eid}:{nid}", str(event_id))
        pipeline.execute()
