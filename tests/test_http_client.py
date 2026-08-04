import time
from unittest.mock import MagicMock, patch

import requests
import requests.exceptions

from classifier.client.http import RateLimitedSession


def _mock_response(status_code: int) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


class TestRateLimitedSession:
    def test_enforces_min_interval_between_requests(self):
        session = RateLimitedSession(min_request_interval=0.1)
        timestamps = []

        def _fake_get(url, **kwargs):
            timestamps.append(time.monotonic())
            return _mock_response(200)

        with patch.object(session._session, "get", side_effect=_fake_get):
            session.get("http://example.com/1")
            session.get("http://example.com/2")

        assert len(timestamps) == 2
        assert timestamps[1] - timestamps[0] >= 0.09

    def test_no_delay_on_first_request(self):
        session = RateLimitedSession(min_request_interval=0.5)
        start = time.monotonic()

        with patch.object(session._session, "get", return_value=_mock_response(200)):
            session.get("http://example.com/")

        assert time.monotonic() - start < 0.4

    def test_session_configured_with_retry_adapter(self):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = RateLimitedSession()
        adapter = session._session.get_adapter("https://example.com")
        assert isinstance(adapter, HTTPAdapter)
        retry = adapter.max_retries
        assert isinstance(retry, Retry)
        assert 429 in retry.status_forcelist
        assert retry.respect_retry_after_header

    def test_post_enforces_interval(self):
        session = RateLimitedSession(min_request_interval=0.1)
        timestamps = []

        def _fake_post(url, **kwargs):
            timestamps.append(time.monotonic())
            return _mock_response(200)

        with patch.object(session._session, "post", side_effect=_fake_post):
            session.post("http://example.com/1")
            session.post("http://example.com/2")

        assert timestamps[1] - timestamps[0] >= 0.09
