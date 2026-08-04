import json
from unittest.mock import MagicMock, patch

import pytest

from classifier.workers.fetch import _ACTIVE_EVENTS_KEY, _STALE_TRACKER_KEY, stale_cleanup_handler


def _make_stale_rc(miss_threshold: int = 3):
    from classifier.runtime_config import ClassifierConfig, FeatureFlags, Processing
    rc = MagicMock()
    rc.config = ClassifierConfig(
        feature_flags=FeatureFlags(stale_cleanup_enabled=True),
        processing=Processing(stale_miss_threshold=miss_threshold),
    )
    return rc


def _put_active_events(s3, bucket: str, active_by_exchange: dict[int, list[str]], successful_ids: list[int]):
    payload = {
        "active_by_exchange": {str(eid): nids for eid, nids in active_by_exchange.items()},
        "successful_exchange_ids": successful_ids,
    }
    s3.put_object(
        Bucket=bucket,
        Key=_ACTIVE_EVENTS_KEY,
        Body=json.dumps(payload).encode(),
        ContentType="application/json",
    )


def _put_stale_tracker(s3, bucket: str, tracker: dict):
    s3.put_object(
        Bucket=bucket,
        Key=_STALE_TRACKER_KEY,
        Body=json.dumps(tracker).encode(),
        ContentType="application/json",
    )


class TestStaleCleanupHandlerCache:
    def test_reads_from_s3_cache_not_fetch_all(self, moto_env):
        s3 = moto_env["s3"]
        _put_active_events(s3, "test-cache", {1: ["evt-active"]}, [1])
        _put_stale_tracker(s3, "test-cache", {"1:evt-active": {"exchange_id": 1, "native_event_id": "evt-active", "miss_count": 0}})

        rc = _make_stale_rc()
        with (
            patch("classifier.workers.fetch._get_runtime_config", return_value=rc),
            patch("classifier.workers.fetch.fetch_all") as mock_fetch_all,
        ):
            result = stale_cleanup_handler({}, None)

        mock_fetch_all.assert_not_called()
        assert result["stale_events"] == 0

    def test_increments_miss_count_for_disappeared_event(self, moto_env):
        s3 = moto_env["s3"]
        _put_active_events(s3, "test-cache", {1: []}, [1])
        _put_stale_tracker(s3, "test-cache", {"1:evt-gone": {"exchange_id": 1, "native_event_id": "evt-gone", "miss_count": 0}})

        rc = _make_stale_rc(miss_threshold=3)
        with patch("classifier.workers.fetch._get_runtime_config", return_value=rc):
            result = stale_cleanup_handler({}, None)

        assert result["stale_events"] == 0

        obj = s3.get_object(Bucket="test-cache", Key=_STALE_TRACKER_KEY)
        tracker = json.loads(obj["Body"].read())
        assert tracker["1:evt-gone"]["miss_count"] == 1

    def test_sends_stale_message_when_threshold_reached(self, moto_env):
        s3 = moto_env["s3"]
        _put_active_events(s3, "test-cache", {1: []}, [1])
        _put_stale_tracker(s3, "test-cache", {"1:evt-gone": {"exchange_id": 1, "native_event_id": "evt-gone", "miss_count": 2}})

        rc = _make_stale_rc(miss_threshold=3)
        with patch("classifier.workers.fetch._get_runtime_config", return_value=rc):
            result = stale_cleanup_handler({}, None)

        assert result["stale_events"] == 1

    def test_falls_back_to_fetch_all_when_cache_missing(self, moto_env):
        rc = _make_stale_rc()
        with (
            patch("classifier.workers.fetch._get_runtime_config", return_value=rc),
            patch("classifier.workers.fetch.fetch_exchanges", return_value={}),
            patch("classifier.workers.fetch.init_registry", return_value=MagicMock()),
            patch("classifier.workers.fetch.fetch_all", return_value=([], [])) as mock_fetch_all,
        ):
            result = stale_cleanup_handler({}, None)

        mock_fetch_all.assert_called_once()
        assert result["stale_events"] == 0

    def test_resets_miss_count_for_active_event(self, moto_env):
        s3 = moto_env["s3"]
        _put_active_events(s3, "test-cache", {1: ["evt-back"]}, [1])
        _put_stale_tracker(s3, "test-cache", {"1:evt-back": {"exchange_id": 1, "native_event_id": "evt-back", "miss_count": 2}})

        rc = _make_stale_rc(miss_threshold=5)
        with patch("classifier.workers.fetch._get_runtime_config", return_value=rc):
            stale_cleanup_handler({}, None)

        obj = s3.get_object(Bucket="test-cache", Key=_STALE_TRACKER_KEY)
        tracker = json.loads(obj["Body"].read())
        assert tracker["1:evt-back"]["miss_count"] == 0
