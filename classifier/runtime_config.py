import base64
import dataclasses
import json
import logging
import time

import requests

from classifier.constants import (
    DEFAULT_CANONICALIZE_BATCH_SIZE,
    DEFAULT_CANONICALIZE_MODEL,
    DEFAULT_BULK_CREATE_BATCH_SIZE,
    DEFAULT_DEDUP_EXPIRY_TOLERANCE_HOURS,
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    DEFAULT_FETCH_MAX_SQS_MESSAGES,
    DEFAULT_HEDGEABLE_WITH_CONFIDENCE,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MAX_WAIT_SECONDS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_NOTIFY_MAX_MESSAGES,
    DEFAULT_NOTIFY_MAX_WAIT_SECONDS,
    DEFAULT_NEIGHBOR_SEARCH_LIMIT,
    DEFAULT_RESOLUTION_LOOKBACK_DAYS,
    DEFAULT_RESOLVE_MAX_SQS_MESSAGES,
    DEFAULT_SEMANTIC_JUDGMENT_MODEL,
    DEFAULT_STALE_MAX_SQS_MESSAGES,
    DEFAULT_STALE_MISS_THRESHOLD,
    DEFAULT_STRUCTURAL_CONFIDENCE,
    DEFAULT_SYNC_THRESHOLD,
    DEFAULT_VOYAGE_EMBED_CHUNK_SIZE,
    DEFAULT_VOYAGE_EMBEDDING_MODEL,
    STANDARDIZED_CATEGORIES,
)

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL = 30.0
_REQUEST_TIMEOUT = 5.0


@dataclasses.dataclass
class FeatureFlags:
    fetch_enabled: bool = False
    resolve_enabled: bool = False
    canonicalization_enabled: bool = False
    semantic_judgements_enabled: bool = False
    stale_cleanup_enabled: bool = False


@dataclasses.dataclass
class CategoryFilter:
    enabled: bool = False
    allowed_categories: list[str] = dataclasses.field(
        default_factory=lambda: sorted(STANDARDIZED_CATEGORIES)
    )


@dataclasses.dataclass
class Models:
    canonicalize_model: str = DEFAULT_CANONICALIZE_MODEL
    semantic_judgment_model: str = DEFAULT_SEMANTIC_JUDGMENT_MODEL
    voyage_embedding_model: str = DEFAULT_VOYAGE_EMBEDDING_MODEL


@dataclasses.dataclass
class Thresholds:
    embedding_similarity_threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    structural_confidence: float = DEFAULT_STRUCTURAL_CONFIDENCE
    hedgeable_with_confidence: float = DEFAULT_HEDGEABLE_WITH_CONFIDENCE


@dataclasses.dataclass
class Processing:
    canonicalize_batch_size: int = DEFAULT_CANONICALIZE_BATCH_SIZE
    bulk_create_batch_size: int = DEFAULT_BULK_CREATE_BATCH_SIZE
    dedup_expiry_tolerance_hours: int = DEFAULT_DEDUP_EXPIRY_TOLERANCE_HOURS
    resolution_lookback_days: int = DEFAULT_RESOLUTION_LOOKBACK_DAYS
    neighbor_search_limit: int = DEFAULT_NEIGHBOR_SEARCH_LIMIT
    voyage_embed_chunk_size: int = DEFAULT_VOYAGE_EMBED_CHUNK_SIZE
    anthropic_sync_threshold: int = DEFAULT_SYNC_THRESHOLD
    stale_miss_threshold: int = DEFAULT_STALE_MISS_THRESHOLD
    fetch_max_sqs_messages: int = DEFAULT_FETCH_MAX_SQS_MESSAGES
    resolve_max_sqs_messages: int = DEFAULT_RESOLVE_MAX_SQS_MESSAGES
    stale_max_sqs_messages: int = DEFAULT_STALE_MAX_SQS_MESSAGES


@dataclasses.dataclass
class WorkerParams:
    normalize_max_messages: int = DEFAULT_MAX_MESSAGES
    normalize_max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS
    embed_max_messages: int = DEFAULT_VOYAGE_EMBED_CHUNK_SIZE
    embed_max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS
    relationships_max_messages: int = 200
    relationships_max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS
    notify_max_messages: int = DEFAULT_NOTIFY_MAX_MESSAGES
    notify_max_wait_seconds: int = DEFAULT_NOTIFY_MAX_WAIT_SECONDS


@dataclasses.dataclass
class ClassifierConfig:
    feature_flags: FeatureFlags = dataclasses.field(default_factory=FeatureFlags)
    category_filter: CategoryFilter = dataclasses.field(default_factory=CategoryFilter)
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
