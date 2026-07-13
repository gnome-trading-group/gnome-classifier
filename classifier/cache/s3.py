import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

from classifier.cache.base import ClassifierCache

logger = logging.getLogger(__name__)


class S3ClassifierCache(ClassifierCache):
    def __init__(self, bucket: str, bypass_reads: bool = False):
        self._bucket = bucket
        self._bypass_reads = bypass_reads
        self._s3 = boto3.client("s3")

    def _s3_canon_key(self, model: str, exchange_id: int, native_id: str) -> str:
        return f"canon/{self._canon_hash(model, exchange_id, native_id)}.json"

    def _s3_judge_key(
        self,
        model: str,
        title_a: str,
        labels_a: list[str],
        title_b: str,
        labels_b: list[str],
    ) -> str:
        return f"judge/{self._judge_hash(model, title_a, labels_a, title_b, labels_b)}.json"

    def list_keys(self, prefix: str) -> set[str]:
        keys: set[str] = set()
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.add(obj["Key"])
        return keys

    def _get(self, key: str) -> dict | None:
        if self._bypass_reads:
            return None
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
            return json.loads(response["Body"].read())
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.warning("Cache get failed for %s: %s", key, e)
            return None

    def _put(self, key: str, value: dict) -> None:
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(value),
                ContentType="application/json",
            )
        except Exception as e:
            logger.warning("Cache put failed for %s: %s", key, e)

    def get_canonicalization(self, model: str, exchange_id: int, native_id: str) -> dict | None:
        return self._get(self._s3_canon_key(model, exchange_id, native_id))

    def get_canonicalization_bulk(
        self, model: str, pairs: list[tuple[int, str]]
    ) -> dict[tuple[int, str], dict]:
        existing = self.list_keys("canon/")
        hits = [
            (eid, nid)
            for eid, nid in pairs
            if self._s3_canon_key(model, eid, nid) in existing
        ]

        results: dict[tuple[int, str], dict] = {}
        if not hits:
            return results

        def _fetch(exchange_id: int, native_id: str) -> tuple[tuple[int, str], dict | None]:
            return (exchange_id, native_id), self._get(
                self._s3_canon_key(model, exchange_id, native_id)
            )

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(_fetch, eid, nid) for eid, nid in hits}
            for future in as_completed(futures):
                pair, cached = future.result()
                if cached is not None:
                    results[pair] = cached
        return results

    def put_canonicalization(
        self, model: str, exchange_id: int, native_id: str, result: dict
    ) -> None:
        self._put(self._s3_canon_key(model, exchange_id, native_id), result)

    def get_judgment(
        self,
        model: str,
        title_a: str,
        labels_a: list[str],
        title_b: str,
        labels_b: list[str],
    ) -> tuple[list, bool] | None:
        key = self._s3_judge_key(model, title_a, labels_a, title_b, labels_b)
        result = self._get(key)
        if result is None:
            return None
        stored_first_title = result.get("first_title")
        a_is_first = stored_first_title == title_a
        return result.get("items", []), a_is_first

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
        key = self._s3_judge_key(model, title_a, labels_a, title_b, labels_b)
        first_title = title_a if a_is_first else title_b
        self._put(key, {"first_title": first_title, "items": items})
