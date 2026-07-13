import json
import logging

import redis as redis_lib

from classifier.cache.base import ClassifierCache

logger = logging.getLogger(__name__)


class RedisClassifierCache(ClassifierCache):
    def __init__(self, redis_url: str):
        self._redis = redis_lib.Redis.from_url(redis_url, decode_responses=False)

    def get_canonicalization(self, model: str, exchange_id: int, native_id: str) -> dict | None:
        field = self._canon_hash(model, exchange_id, native_id)
        data = self._redis.hget("canon", field)
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
        pipeline = self._redis.pipeline()
        for eid, nid in pairs:
            pipeline.hget("canon", self._canon_hash(model, eid, nid))
        raw_results = pipeline.execute()
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
        field = self._canon_hash(model, exchange_id, native_id)
        self._redis.hset("canon", field, json.dumps(result))

    def get_judgment(
        self,
        model: str,
        title_a: str,
        labels_a: list[str],
        title_b: str,
        labels_b: list[str],
    ) -> tuple[list, bool] | None:
        field = self._judge_hash(model, title_a, labels_a, title_b, labels_b)
        data = self._redis.hget("judge", field)
        if data is None:
            return None
        try:
            stored = json.loads(data)
        except Exception as e:
            logger.warning("Judge cache decode failed: %s", e)
            return None
        a_is_first = stored.get("first_title") == title_a
        return stored.get("items", []), a_is_first

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
        field = self._judge_hash(model, title_a, labels_a, title_b, labels_b)
        first_title = title_a if a_is_first else title_b
        self._redis.hset("judge", field, json.dumps({"first_title": first_title, "items": items}))

    def get_exchange_event(self, exchange_id: int, native_id: str) -> int | None:
        val = self._redis.hget("exchange_events", f"{exchange_id}:{native_id}")
        return int(val) if val is not None else None

    def put_exchange_event(self, exchange_id: int, native_id: str, event_id: int) -> None:
        self._redis.hset("exchange_events", f"{exchange_id}:{native_id}", str(event_id))

