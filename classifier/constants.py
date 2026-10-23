STANDARDIZED_CATEGORIES = frozenset({
    "CRYPTO", "POLITICS", "SPORTS", "ECONOMICS", "FINANCE", "ENTERTAINMENT",
    "SCIENCE", "TECHNOLOGY", "HEALTH", "WEATHER", "LEGAL", "OTHER",
})

DEFAULT_CANONICALIZE_BATCH_SIZE = 15
DEFAULT_BULK_CREATE_BATCH_SIZE = 200

DEFAULT_MIN_CONFIDENCE = 0.70

DEFAULT_RESOLUTION_LOOKBACK_DAYS = 3
# How long after a listing is deactivated the settle cycle keeps looking for its final value; disputed results can
# take days to settle.
DEFAULT_SETTLEMENT_LOOKBACK_DAYS = 30
# Outcomes each exchange looks up per settle cycle, so a backlog can't hold up the fetch loop for long.
DEFAULT_SETTLE_MAX_CANDIDATES = 20_000
# Outcomes deactivated this recently are looked up after every fetch; nearly all values arrive within this window, so
# the rest of the lookback only needs the hourly sweep.
DEFAULT_SETTLE_RECENT_HOURS = 6
# Contracts that leave an exchange's feed between two fetches are looked up on the venue straight away. More than this
# in one fetch is a feed fault, not markets closing, and is left to the resolve cycle.
DEFAULT_CLOSURE_CHECK_MAX_LOOKUPS = 500
# How long after a listing is deactivated it is turned back on if its venue lists it as open again. Matches the
# settlement lookback, so nothing still waiting on a value is left retired by a bad feed.
DEFAULT_REACTIVATION_LOOKBACK_DAYS = 30
# A fetch reactivating this many listings of one exchange is a venue that misreported them as closed; it is logged as a
# warning for the ListingsReactivated alarm.
REACTIVATION_WARNING_THRESHOLD = 50



DEFAULT_MAX_MESSAGES = 500
DEFAULT_MAX_WAIT_SECONDS = 60
DEFAULT_NOTIFY_MAX_WAIT_SECONDS = 300
DEFAULT_VISIBILITY_TIMEOUT = 900

# Bedrock model or inference profile id, called through the Converse API.
DEFAULT_CANONICALIZE_MODEL = "us.amazon.nova-lite-v1:0"
DEFAULT_SEMANTIC_JUDGMENT_MODEL = "claude-sonnet-4-6"
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-3"

DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD = 0.85
DEFAULT_HEDGEABLE_WITH_CONFIDENCE = 0.90

DEFAULT_SYNC_THRESHOLD = 10
DEFAULT_NEIGHBOR_SEARCH_LIMIT = 60
DEFAULT_VOYAGE_EMBED_CHUNK_SIZE = 2000
# Stale cycles run hourly, so an event must be gone from its venue's feed for a day before it is retired.
DEFAULT_STALE_MISS_THRESHOLD = 24
# A fetch returning fewer than this share of an exchange's tracked events is treated as an outage, not closures.
DEFAULT_STALE_MIN_FEED_RATIO = 0.5

DEFAULT_FETCH_MAX_SQS_MESSAGES = 100000
DEFAULT_RESOLVE_MAX_SQS_MESSAGES = 100000
DEFAULT_STALE_MAX_SQS_MESSAGES = 100000
DEFAULT_NOTIFY_MAX_MESSAGES = 100000

DEFAULT_FETCH_INTERVAL_SECONDS = 60
DEFAULT_RESOLVE_INTERVAL_SECONDS = 1800
DEFAULT_STALE_INTERVAL_SECONDS = 3600
DEFAULT_SETTLE_SWEEP_INTERVAL_SECONDS = 3600
DEFAULT_FEE_SYNC_INTERVAL_SECONDS = 3600

# 24h dollar volume an event needs before it is ingested; every adapter reports 24h volume.
DEFAULT_MIN_EVENT_VOLUME: float | None = 1000
