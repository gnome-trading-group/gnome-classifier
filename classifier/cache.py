import hashlib
import json
import logging

import boto3

logger = logging.getLogger(__name__)


class ClassifierCache:
    def __init__(self, bucket: str):
        self._bucket = bucket
        self._s3 = boto3.client("s3")

    def _get(self, key: str) -> dict | None:
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

    def _canon_key(self, model: str, exchange_id: int, native_id: str) -> str:
        content = model + "\x00" + str(exchange_id) + "\x00" + native_id
        digest = hashlib.sha256(content.encode()).hexdigest()
        return f"cache/canon/{digest}.json"

    def _judge_key(self, model: str, title_a: str, labels_a: list[str], title_b: str, labels_b: list[str]) -> str:
        # Normalize ordering so the same pair always maps to the same key
        pair_a = (title_a, "|".join(labels_a))
        pair_b = (title_b, "|".join(labels_b))
        if pair_a > pair_b:
            pair_a, pair_b = pair_b, pair_a
        content = (
            model + "\x00"
            + pair_a[0] + "\x00" + pair_a[1] + "\x00"
            + pair_b[0] + "\x00" + pair_b[1]
        )
        digest = hashlib.sha256(content.encode()).hexdigest()
        return f"cache/judge/{digest}.json"

    def get_canonicalization(self, model: str, exchange_id: int, native_id: str) -> dict | None:
        return self._get(self._canon_key(model, exchange_id, native_id))

    def put_canonicalization(self, model: str, exchange_id: int, native_id: str, result: dict) -> None:
        self._put(self._canon_key(model, exchange_id, native_id), result)

    def get_judgment(
        self,
        model: str,
        title_a: str,
        labels_a: list[str],
        title_b: str,
        labels_b: list[str],
    ) -> tuple[list, bool] | None:
        """Returns (items, a_is_first) or None on miss.
        a_is_first indicates whether stored 'first_label' corresponds to event_a's labels."""
        key = self._judge_key(model, title_a, labels_a, title_b, labels_b)
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
        """Store judgment items using first_label/second_label keyed by outcome label.
        a_is_first indicates whether 'first' refers to event_a."""
        key = self._judge_key(model, title_a, labels_a, title_b, labels_b)
        first_title = title_a if a_is_first else title_b
        self._put(key, {"first_title": first_title, "items": items})
