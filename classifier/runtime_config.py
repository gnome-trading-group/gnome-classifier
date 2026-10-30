import base64
import dataclasses
import json
import logging
import time

import requests

from classifier.constants import (
    DEFAULT_CANONICALIZE_BATCH_SIZE,
    DEFAULT_CANONICALIZE_MODEL,
    DEFAULT_CLOSURE_CHECK_MAX_LOOKUPS,
    DEFAULT_DRIFT_INTERVAL_SECONDS,
    DEFAULT_DRIFT_SAMPLE_SIZE,
    DEFAULT_BULK_CREATE_BATCH_SIZE,
    DEFAULT_FEE_SYNC_INTERVAL_SECONDS,
    DEFAULT_VOLUME_INTERVAL_SECONDS,
    DEFAULT_VOLUME_MAX_AGE_SECONDS,
    DEFAULT_VOLUME_MIN_CHANGE,
    DEFAULT_FETCH_INTERVAL_SECONDS,
    DEFAULT_FETCH_MAX_SQS_MESSAGES,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MAX_WAIT_SECONDS,
    DEFAULT_MIN_EVENT_VOLUME,
    DEFAULT_NOTIFY_MAX_MESSAGES,
    DEFAULT_NOTIFY_MAX_WAIT_SECONDS,
    DEFAULT_REACTIVATION_LOOKBACK_DAYS,
    DEFAULT_RESOLUTION_LOOKBACK_DAYS,
    DEFAULT_RESOLVE_INTERVAL_SECONDS,
    DEFAULT_RESOLVE_MAX_SQS_MESSAGES,
    DEFAULT_PROPOSITION_MODELS,
    DEFAULT_PROPOSITION_PROVIDER,
    DEFAULT_RELATIONSHIPS_BACKLOG_LIMIT,
    DEFAULT_RELATIONSHIPS_DERIVE_LIMIT,
    DEFAULT_SETTLE_MAX_CANDIDATES,
    DEFAULT_SETTLE_RECENT_HOURS,
    DEFAULT_SETTLE_SWEEP_INTERVAL_SECONDS,
    DEFAULT_SETTLEMENT_LOOKBACK_DAYS,
    DEFAULT_STALE_INTERVAL_SECONDS,
    DEFAULT_STALE_MAX_SQS_MESSAGES,
    DEFAULT_STALE_MIN_FEED_RATIO,
    DEFAULT_STALE_MISS_THRESHOLD,
    DEFAULT_VOYAGE_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL = 30.0
_REQUEST_TIMEOUT = 5.0


@dataclasses.dataclass
class FeatureFlags:
    fetch_enabled: bool = False
    resolve_enabled: bool = False
    canonicalization_enabled: bool = False
    # Proposition extraction and relationship derivation in the relationships worker.
    relationships_enabled: bool = False
    # Claude reads the securities the venue code readers can't; off, only code-read securities get propositions (the
    # rest are read once it is turned on, while still open). Paid per contract, so off where nobody needs the coverage.
    proposition_llm_enabled: bool = False
    stale_cleanup_enabled: bool = False
    settlement_enabled: bool = False
    reactivation_enabled: bool = False
    fee_sync_enabled: bool = False
    debug: bool = False


@dataclasses.dataclass
class Models:
    canonicalize_model: str = DEFAULT_CANONICALIZE_MODEL
    voyage_embedding_model: str = DEFAULT_VOYAGE_EMBEDDING_MODEL
    proposition_provider: str = DEFAULT_PROPOSITION_PROVIDER
    # Tried once when the first provider is unavailable (refused credentials, out of credit, down).
    proposition_fallback_provider: str | None = None
    proposition_models: dict[str, str] = dataclasses.field(default_factory=lambda: dict(DEFAULT_PROPOSITION_MODELS))

    def proposition_providers(self) -> list[str]:
        providers = [self.proposition_provider]
        if self.proposition_fallback_provider and self.proposition_fallback_provider != self.proposition_provider:
            providers.append(self.proposition_fallback_provider)
        return providers


@dataclasses.dataclass
class Thresholds:
    min_event_volume: float | None = DEFAULT_MIN_EVENT_VOLUME
    # Per-tag / per-category thresholds that replace min_event_volume for matching events; the smallest
    # matching value wins and None ungates. Empty defaults so entries removed in the controller stay removed.
    min_event_volume_by_tag: dict[str, float | None] = dataclasses.field(default_factory=dict)
    min_event_volume_by_category: dict[str, float | None] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Processing:
    canonicalize_batch_size: int = DEFAULT_CANONICALIZE_BATCH_SIZE
    bulk_create_batch_size: int = DEFAULT_BULK_CREATE_BATCH_SIZE
    resolution_lookback_days: int = DEFAULT_RESOLUTION_LOOKBACK_DAYS
    settlement_lookback_days: int = DEFAULT_SETTLEMENT_LOOKBACK_DAYS
    settle_max_candidates: int = DEFAULT_SETTLE_MAX_CANDIDATES
    settle_recent_hours: int = DEFAULT_SETTLE_RECENT_HOURS
    closure_check_max_lookups: int = DEFAULT_CLOSURE_CHECK_MAX_LOOKUPS
    reactivation_lookback_days: int = DEFAULT_REACTIVATION_LOOKBACK_DAYS
    relationships_derive_limit: int = DEFAULT_RELATIONSHIPS_DERIVE_LIMIT
    relationships_backlog_limit: int = DEFAULT_RELATIONSHIPS_BACKLOG_LIMIT
    drift_sample_size: int = DEFAULT_DRIFT_SAMPLE_SIZE
    stale_miss_threshold: int = DEFAULT_STALE_MISS_THRESHOLD
    stale_min_feed_ratio: float = DEFAULT_STALE_MIN_FEED_RATIO
    fetch_max_sqs_messages: int = DEFAULT_FETCH_MAX_SQS_MESSAGES
    resolve_max_sqs_messages: int = DEFAULT_RESOLVE_MAX_SQS_MESSAGES
    stale_max_sqs_messages: int = DEFAULT_STALE_MAX_SQS_MESSAGES


@dataclasses.dataclass
class WorkerParams:
    normalize_max_messages: int = DEFAULT_MAX_MESSAGES
    normalize_max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS
    relationships_max_messages: int = 200
    relationships_max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS
    drift_interval_seconds: int = DEFAULT_DRIFT_INTERVAL_SECONDS
    notify_max_messages: int = DEFAULT_NOTIFY_MAX_MESSAGES
    notify_max_wait_seconds: int = DEFAULT_NOTIFY_MAX_WAIT_SECONDS
    fetch_interval_seconds: int = DEFAULT_FETCH_INTERVAL_SECONDS
    resolve_interval_seconds: int = DEFAULT_RESOLVE_INTERVAL_SECONDS
    stale_interval_seconds: int = DEFAULT_STALE_INTERVAL_SECONDS
    settle_sweep_interval_seconds: int = DEFAULT_SETTLE_SWEEP_INTERVAL_SECONDS
    fee_sync_interval_seconds: int = DEFAULT_FEE_SYNC_INTERVAL_SECONDS
    volume_interval_seconds: int = DEFAULT_VOLUME_INTERVAL_SECONDS
    volume_min_change: float = DEFAULT_VOLUME_MIN_CHANGE
    volume_max_age_seconds: int = DEFAULT_VOLUME_MAX_AGE_SECONDS


@dataclasses.dataclass
class ClassifierConfig:
    feature_flags: FeatureFlags = dataclasses.field(default_factory=FeatureFlags)
    models: Models = dataclasses.field(default_factory=Models)
    thresholds: Thresholds = dataclasses.field(default_factory=Thresholds)
    processing: Processing = dataclasses.field(default_factory=Processing)
    worker_params: WorkerParams = dataclasses.field(default_factory=WorkerParams)


def _from_dict(cls, data: dict):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if dataclasses.is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(f.type, value)
        elif not dataclasses.is_dataclass(f.type):
            kwargs[f.name] = value
    return cls(**{k: v for k, v in kwargs.items() if k in fields})


class RuntimeConfig:
    def __init__(self, controller_api_url: str, api_key: str):
        self._url = controller_api_url.rstrip("/") + "/config/classifier"
        self._api_key = api_key
        self._config = ClassifierConfig()
        self._last_refresh: float = 0.0
        self._defaults_header = base64.b64encode(
            json.dumps(dataclasses.asdict(ClassifierConfig())).encode()
        ).decode()

    @property
    def config(self) -> ClassifierConfig:
        return self._config

    def refresh(self) -> None:
        now = time.monotonic()
        if now - self._last_refresh < _REFRESH_INTERVAL:
            return
        self._last_refresh = now
        try:
            response = requests.get(
                self._url,
                headers={"x-api-key": self._api_key, "x-config-defaults": self._defaults_header},
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            self._config = _from_dict(ClassifierConfig, data["config"])
        except Exception:
            logger.warning("Failed to refresh runtime config, keeping last-known config", exc_info=True)
