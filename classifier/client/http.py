import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


class RateLimitedSession:
    def __init__(self, min_request_interval: float = 0.15):
        self._min_interval = min_request_interval
        self._last_request_at: float = 0.0
        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=_RETRY_STATUS_CODES,
            respect_retry_after_header=True,
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def get(self, url: str, **kwargs) -> requests.Response:
        self._wait()
        return self._session.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        self._wait()
        return self._session.post(url, **kwargs)
