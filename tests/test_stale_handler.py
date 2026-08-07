import json
from unittest.mock import MagicMock, patch

import pytest

from classifier.runtime_config import ClassifierConfig, FeatureFlags, Processing
from classifier.workers.fetch import FetchRunner


def _make_stale_rc(miss_threshold: int = 3):
    rc = MagicMock()
    rc.config = ClassifierConfig(
        feature_flags=FeatureFlags(stale_cleanup_enabled=True),
        processing=Processing(stale_miss_threshold=miss_threshold),
    )
    return rc


def _make_redis(tracker: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.get.return_value = json.dumps(tracker).encode() if tracker is not None else None
    return r


def _saved_tracker(r: MagicMock) -> dict:
    call_args = r.set.call_args
    if call_args is None:
        return {}
    return json.loads(call_args[0][1])


class TestStaleCleanupHandler:
    def test_reads_from_in_memory_active_events(self, moto_env):
        r = _make_redis({"1:evt-active": {"exchange_id": 1, "native_event_id": "evt-active", "miss_count": 0}})
        runner = FetchRunner()
        runner._active_events = ({1: {"evt-active"}}, {1})

        with patch("classifier.workers.fetch.fetch_all") as mock_fetch_all:
            runner._run_stale(_make_stale_rc(), r, moto_env["sqs"], MagicMock())

        mock_fetch_all.assert_not_called()

    def test_increments_miss_count_for_disappeared_event(self, moto_env):
        tracker = {"1:evt-gone": {"exchange_id": 1, "native_event_id": "evt-gone", "miss_count": 0}}
        r = _make_redis(tracker)
        runner = FetchRunner()
        runner._active_events = ({1: set()}, {1})

        runner._run_stale(_make_stale_rc(miss_threshold=3), r, moto_env["sqs"], MagicMock())

        saved = _saved_tracker(r)
        assert saved["1:evt-gone"]["miss_count"] == 1

    def test_sends_stale_message_when_threshold_reached(self, moto_env):
        tracker = {"1:evt-gone": {"exchange_id": 1, "native_event_id": "evt-gone", "miss_count": 2}}
        r = _make_redis(tracker)
        runner = FetchRunner()
        runner._active_events = ({1: set()}, {1})

        runner._run_stale(_make_stale_rc(miss_threshold=3), r, moto_env["sqs"], MagicMock())

        resp = moto_env["sqs"].receive_message(
            QueueUrl=moto_env["contracts_queue"], MaxNumberOfMessages=10, WaitTimeSeconds=0
        )
        messages = resp.get("Messages", [])
        assert len(messages) == 1
        assert json.loads(messages[0]["Body"])["type"] == "stale"

    def test_falls_back_to_fetch_all_when_no_active_events(self, moto_env):
        r = _make_redis({})
        runner = FetchRunner()
        runner._active_events = None

        with (
            patch("classifier.workers.fetch.fetch_exchanges", return_value={}),
            patch("classifier.workers.fetch.fetch_all", return_value=([], [])) as mock_fetch_all,
        ):
            runner._run_stale(_make_stale_rc(), r, moto_env["sqs"], MagicMock())

        mock_fetch_all.assert_called_once()

    def test_resets_miss_count_for_active_event(self, moto_env):
        tracker = {"1:evt-back": {"exchange_id": 1, "native_event_id": "evt-back", "miss_count": 2}}
        r = _make_redis(tracker)
        runner = FetchRunner()
        runner._active_events = ({1: {"evt-back"}}, {1})

        runner._run_stale(_make_stale_rc(miss_threshold=5), r, moto_env["sqs"], MagicMock())

        saved = _saved_tracker(r)
        assert saved["1:evt-back"]["miss_count"] == 0
