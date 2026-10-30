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
