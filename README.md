# gnome-classifier

A prediction market contract classifier that ingests contracts from multiple exchanges (Polymarket, Kalshi, Hyperliquid), writes them into the security master as events, securities and listings, generates semantic embeddings, and discovers relationships between contracts.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [Exchange Adapters](#exchange-adapters)
5. [Data Model](#data-model)
6. [Entity Creation Pipeline](#entity-creation-pipeline)
7. [Relationship Discovery](#relationship-discovery)
8. [Caching](#caching)
9. [Runtime Configuration](#runtime-configuration)
10. [Infrastructure](#infrastructure)
11. [Development](#development)

---

## Overview

Each prediction market exchange has its own schema, naming conventions and contract structures. The classifier pulls raw contracts from every exchange, records each native exchange event and its outcomes in the security master, and builds a graph of relationships between the resulting securities.

**Identity model:**
- One `sm.event` per native exchange event, keyed by `(exchange_id, native_event_id)`. Events are never merged across or within exchanges, and the event title is the exchange's title as the adapter reports it.
- One `sm.security` per listing, keyed by its listing `(exchange_id, exchange_security_id)`. Symbols never decide identity.
- Links between events (the same question on two exchanges, or related questions on one) come only from the semantic relationship judge.

**What it produces:**
- `sm.event` rows (exchange title, category, tags, native exchange/event ID and URL)
- `sm.security` rows, one per outcome contract (e.g. `KX-KXWTI15M-26AUG240015-15-YES`)
- `sm.listing` / `sm.listing_spec` rows mapping each security to its exchange identifiers and trading parameters
- `sm.event_contract` rows linking events to their outcome securities
- `sm.contract_relationship` rows (equivalence, implication, mutual exclusion, hedgeability)
- Vector embeddings in `sm.event_embedding` for similarity search

---

## Architecture

```
                        ┌──────────────────────────┐
                        │   FetchRunner (ECS/EC2)  │ ◄─► Redis (fetch:known_contracts,
                        │ fetch / resolve / stale  │           fetch:sent_resolved,
                        │   on runtime intervals   │           fetch:stale_tracker)
                        └────────────┬─────────────┘
                                     │
                              contracts-queue (SQS)
                                     │
                            ┌────────▼────────┐
                            │ NormalizeWorker │ ◄── Redis (canon cache, exchange event cache)
                            │    (ECS/EC2)    │ ──► Claude API (haiku, category + tags)
                            │                 │ ──► Registry API / PostgreSQL (sm.*)
                            └────────┬────────┘
                                     │
                              entities-queue (SQS)
                                     │
                            ┌────────▼────────┐
                            │   EmbedWorker   │ ──► Voyage AI API
                            │    (ECS/EC2)    │ ──► PostgreSQL (sm.event_embedding)
                            └────────┬────────┘
                                     │
                             embeddings-queue (SQS)
                                     │
                         ┌───────────▼───────────┐
                         │  RelationshipsWorker  │ ◄── Redis (judgment cache)
                         │       (ECS/EC2)       │ ──► Claude API (sonnet)
                         │                       │ ──► Registry API (sm.contract_relationship)
                         └───────────────────────┘

      NormalizeWorker / RelationshipsWorker ──► SNS topic ──► slack-queue ──► NotifyWorker ──► Slack
```

**Components:**

| Component | CMD | Trigger |
|---|---|---|
| FetchRunner | `fetch` | Internal loop; fetch / resolve / stale cycles on intervals from runtime config |
| NormalizeWorker | `normalize` | Continuous SQS poll (contracts-queue) |
| EmbedWorker | `embed` | Continuous SQS poll (entities-queue) |
| RelationshipsWorker | `relationships` | Continuous SQS poll (embeddings-queue) |
| NotifyWorker | `notify` | Continuous SQS poll (slack-queue) |

All five run as ECS services on a single EC2 auto-scaling group (spot t3.medium, 1–2 instances) inside the registry VPC, so they can reach PostgreSQL and Redis.

---

## Data Flow

### Stage 1: Fetch (`classifier/workers/fetch.py` → `FetchRunner._run_fetch`)

**Every `fetch_interval_seconds` (default 60s).** Skipped when `fetch_enabled` is false.

1. Loads the last-seen contract hashes from Redis (`fetch:known_contracts`, `"exchange_id:exchange_security_id"` → hash).
2. Looks up exchanges in the registry by `exchange_code` and, for each adapter, streams pages from `adapter.fetch(exchange_id)`.
3. For each page, `diff_contracts()` (`classifier/stages/fetch.py`):
   - Hashes every contract (`contract_hash`: exchange/security IDs, outcome label, event title, native event ID, tick/lot size, min notional, contract multiplier).
   - Groups contracts by `(exchange_id, exchange_event_native_id)`.
   - Drops groups whose `event_volume` is below `min_event_volume` (when set).
   - Emits one `{"type": "new", "contracts": [...]}` message per group in which any contract is new or changed, capped at `fetch_max_sqs_messages` per cycle.
4. Sends messages to the contracts queue page by page.
5. Commits per exchange: on success, that exchange's hashes are replaced with the freshly fetched ones and its active native event IDs are recorded (`active_by_exchange`, `successful_ids`). If an adapter raises (adapters raise on API errors rather than returning partial results), the exchange keeps its previous hashes (plus hashes of groups already sent this cycle) and is left out of `successful_ids`.
6. Saves the merged hashes back to Redis and keeps `(active_by_exchange, successful_ids)` in memory for the stale cycle.

**Why group by event?** NormalizeWorker reconciles an event's listings against the message, so it always needs the complete current contract set of the event, not just the changed contract.

---

### Stage 2: Resolve (`FetchRunner._run_resolve`)

**Every `resolve_interval_seconds` (default 1800s).** Skipped when `resolve_enabled` is false.

1. `fetch_resolved_outcomes()` calls `adapter.fetch_resolved(exchange_id, resolution_lookback_days)` on each adapter. Each returns a set of `exchange_security_id`s, including individually closed markets inside still-open events (see [Exchange Adapters](#exchange-adapters)). A failing adapter is logged and skipped.
2. Loads already-sent IDs from Redis (`fetch:sent_resolved`).
3. Sends `{"type": "resolved", "exchange_id": X, "native_id": <exchange_security_id>}` for each unsent ID, capped at `resolve_max_sqs_messages`.
4. Saves `(previously sent ∩ currently resolved) ∪ newly sent`, so entries drop out once they leave the lookback window.

---

### Stage 3: Stale Cleanup (`FetchRunner._run_stale`)

**Every `stale_interval_seconds` (default 3600s).** Skipped when `stale_cleanup_enabled` is false.

Handles events that an exchange silently stops listing without marking them resolved.

1. Uses the in-memory `(active_by_exchange, successful_ids)` from the last fetch cycle. Exchanges in the tracker that are not in `successful_ids` are treated as failed. If no fetch has completed since startup, it fetches all adapters directly.
2. Loads the tracker from Redis (`fetch:stale_tracker`: `"exchange_id:native_event_id"` → `{exchange_id, native_event_id, miss_count}`).
3. `update_stale_tracker()` (`classifier/stages/stale.py`):
   - Entries on failed exchanges are carried forward unchanged (no miss-counting).
   - Entries present in the active set reset to `miss_count = 0`; absent ones increment.
   - Newly seen events are added with `miss_count = 0`.
   - Entries reaching `stale_miss_threshold` (default 6) become `{"type": "stale", "exchange_id": X, "native_event_id": Y}` messages and leave the tracker. Anything beyond `stale_max_sqs_messages` stays in the tracker at the threshold count for the next cycle.
4. Sends stale messages and saves the tracker.

---

### Stage 4: Normalize (`classifier/workers/normalize.py` → `NormalizeWorker`)

**Reads contracts-queue, writes entities-queue.**

Messages are collected into a batch (20s long-polling) until `normalize_max_messages` is reached or `normalize_max_wait_seconds` has passed since the first message. `process_batch()` then handles all three message types:

**`type: "resolved"`** → `detect_resolved_events()` (`classifier/stages/resolve.py`):
1. Finds active listings matching the resolved `exchange_security_id`s and deactivates them.
2. Deactivates securities left with no active listings.
3. Marks events with no remaining active securities as resolved (`resolved = true`, `resolved_at = now`) and deletes their embeddings.

**`type: "stale"`** → `deactivate_stale_events()` (`classifier/stages/stale.py`): looks up the event by `(exchange_id, native_event_id)` on `sm.event`, deactivates that event's active listings on that exchange, then runs the same security/event cascade as resolution.

Both publish `{"type": "resolved", ...counts, "resolved_event_ids": [...], "resolved_event_names": [...]}` to SNS when anything changed.

**`type: "new"`** → `create_entities()` (see [Entity Creation Pipeline](#entity-creation-pipeline)). Every new security yields `{"type": "new_security", "security_id", "security_symbol"}` on the entities queue; securities belonging to a newly created event also carry `created_event_id` / `created_event_name`.

If `process_batch()` raises, no messages are deleted; they reappear after the 15-minute receive visibility timeout and move to the DLQ after 3 receives.

---

### Stage 5: Embed (`classifier/workers/embed.py` → `EmbedWorker`)

**Reads entities-queue, writes embeddings-queue.**

`embed_and_update()` (`classifier/stages/embed.py`):
1. Loads all unresolved events with no row in `sm.event_embedding`.
2. Embeds them in chunks of `voyage_embed_chunk_size` with the Voyage client: text is `"{title}. {description[:200]}"`, sent in batches of 128 with 5 parallel workers and up to 3 retries.
3. Upserts the vectors into `sm.event_embedding`.
4. Adds every security of the newly embedded events to the set to classify.

Each resulting security is forwarded as `{"type": "security", "security_id", "security_symbol"}` (plus `created_event_*` when present).

---

### Stage 6: Relationships (`classifier/workers/relationships.py` → `RelationshipsWorker`)

**Reads embeddings-queue.** Calls `run_classification_sync()` (`classifier/stages/classify.py`) with the batch's security IDs; see [Relationship Discovery](#relationship-discovery). Afterwards publishes `{"type": "new_events", "created_event_ids", "created_event_names"}` to SNS for any newly created events in the batch.

---

### Stage 7: Notify (`classifier/workers/notify.py` → `NotifyWorker`)

**Reads slack-queue (SNS subscription).**

Unwraps SNS envelopes and collects `new_events` and `resolved` payloads across the batch (default wait 300s so messages accumulate). `format_notification_blocks()` (`classifier/notifications.py`) builds a Slack Block Kit message with a "Contract Classifier" header and sections listing new and resolved events (up to 20 each), each linked to `https://controller.gnometrading.group/predictions/events/{event_id}`. Posts via `chat.postMessage` using `urllib.request`. Skips entirely if the Slack token or channel is not configured.

---

## Exchange Adapters

Adapters live in `classifier/adapters/` and are registered in `ADAPTERS`. Each exposes:

- `exchange_code` — matched against the registry's exchange records
- `symbol_prefix` — prefix of every security symbol it produces
- `fetch(exchange_id)` — generator yielding pages of `AdapterContract`s for active contracts
- `fetch_resolved(exchange_id, lookback_days)` — set of resolved `exchange_security_id`s

API errors are logged and re-raised; adapters never truncate silently.

| Adapter | `exchange_code` | `symbol_prefix` | Symbol format |
|---|---|---|---|
| `PolymarketIntlAdapter` | `POLYMARKET_INTL` | `PM_I` | `PM_I-{market-slug}-{OUTCOME}` (binary), `PM_I-{market-slug}` (neg-risk group) |
| `KalshiAdapter` | `KALSHI` | `KX` | `KX-{market_ticker}-{YES\|NO}` (binary), `KX-{market_ticker}` (multi-outcome) |
| `HyperliquidAdapter` | `HYPERLIQUID` | `HL` | `HL-{outcome_id}-{side}` (binary), `HL-{outcome_id}` (multi-outcome) |

Symbols are built by `classifier.utils.format_security_symbol(prefix, *parts)`, which uppercases each part and replaces runs of characters outside `[A-Z0-9.]` with `-`, e.g. `KX-KXWTI15M-26AUG240015-15-YES`.

### AdapterContract (`classifier/adapters/types.py`)

| Field | Description |
|---|---|
| `exchange_id` | Registry exchange ID |
| `exchange_security_id` | Exchange-specific contract ID (listing identity) |
| `exchange_security_symbol` | Human-readable exchange symbol (`"{title[:60]} -- {outcome}"`) |
| `security_symbol` | Registry security symbol (see table above) |
| `base_currency` / `quote_currency` / `settle_currency` | Always `"USDC"` |
| `security_type` / `asset_class` | Always `EVENT_CONTRACT` / `PREDICTION` |
| `contract_type` | `BINARY` or `MULTI_OUTCOME` |
| `inverse` / `is_quanto` | Always `False` |
| `tick_size`, `lot_size`, `min_notional`, `contract_multiplier` | Listing spec values (scaled integers) |
| `event_title` | Event title as reported by the adapter |
| `outcome_label` | Outcome name (e.g. "Yes", "Trump") |
| `exchange_event_native_id` | Native event ID (event identity and grouping key) |
| `exchange_event_native_url` | Link to the event on the exchange, when available |
| `event_description`, `event_category`, `event_expiry` | Optional metadata |
| `event_volume` | Volume used for `min_event_volume` filtering, when available |

### Polymarket International (`classifier/adapters/polymarket_intl.py`)

- **API:** `https://gamma-api.polymarket.com/events/keyset` with cursor pagination (500 per page). Active: `active=true&closed=false`. Resolved: `active=false&closed=true&end_date_min={lookback}`.
- Closed markets are dropped from active fetches.
- **Neg-risk groups** (all open markets have `negRisk`): one `MULTI_OUTCOME` contract per market using the YES token, `outcome_label = groupItemTitle`, `exchange_event_native_id = event slug`, event title = event title, volume summed across markets.
- **Other markets**: each market is its own binary event with one contract per outcome token. `exchange_event_native_id = conditionId`, event title = market question (sports spread/total/prop markets that reuse the event title get a suffix such as `": Spread -3.5"`).
- `exchange_security_id = "{conditionId}:{tokenId}"`. Native URL is `https://polymarket.com/event/{slug}`.
- Tick size comes from `orderPriceMinTickSize` (default 10,000,000); lot size 10,000; contract multiplier 1e9.
- `fetch_resolved` returns tokens of recently closed events plus closed markets inside active events.

### Kalshi (`classifier/adapters/kalshi.py`)

- **API:** `https://external-api.kalshi.com/trade-api/v2/events?with_nested_markets=true` with cursor pagination (200 per page). Active: `status=open`. Resolved: `status=settled&min_close_ts={lookback}`.
- Markets whose `status != "active"` are dropped from active fetches.
- **Multi-outcome** (`mutually_exclusive` and >1 market): one `MULTI_OUTCOME` contract per market. `exchange_event_native_id = event_ticker`, `exchange_security_id = market ticker`, `outcome_label = yes_sub_title`.
- **Binary with sub-markets** (not mutually exclusive, >1 market): each market is its own event, `exchange_event_native_id = market ticker`, title `"{event title}: {yes_sub_title}"`.
- **Simple binary** (single market): `exchange_event_native_id = event_ticker`; the market's `yes_sub_title` is appended to the title when not already in it.
- Binary contracts are Yes/No pairs with `exchange_security_id = "{ticker}:yes"` / `"{ticker}:no"`.
- Tick size from `price_ranges` (default 10,000,000); lot size 10,000; contract multiplier 1e9. Native URL built from the series ticker and a title slug.
- `fetch_resolved` returns IDs from settled events plus non-active markets inside open events.

### Hyperliquid (`classifier/adapters/hyperliquid.py`)

- **API:** `POST https://api.hyperliquid.xyz/info` with `{"type": "outcomeMeta"}`; one page per fetch.
- **Questions with >1 active outcome** (excluding fallback and settled outcomes): one `MULTI_OUTCOME` contract per outcome, `exchange_event_native_id = "q:{first_active_outcome_id}"`, `exchange_security_id = "@{outcome_id}"`. `class:priceBucket` questions get a generated title (`"{underlying} price range on {expiry}"`) and range labels (`"< $X"`, `"$X - $Y"`, `"> $Y"`).
- **Questions with one active outcome and orphan outcomes**: binary, `exchange_event_native_id = "o:{outcome_id}"`, `exchange_security_id = "@{outcome_id}:0"` / `":1"`, labels from `sideSpecs`. `class:priceBinary` orphans get a generated title (`"Will {underlying} be above ${target}? ({expiry})"`).
- Category and expiry are parsed from structured description fields. Tick and lot size are fixed at 1,000,000; contract multiplier 1e9. No volume or native URL.
- `fetch_resolved` returns `@{oid}`, `@{oid}:0` and `@{oid}:1` for each question's `settledNamedOutcomes` (the lookback is not used).

---

## Data Model

### Database Tables (`sm` schema)

**`sm.event`** — one row per native exchange event

| Column | Type | Description |
|---|---|---|
| `event_id` | int PK | |
| `title` | text | The exchange's title as reported by the adapter |
| `description` | text | Optional description |
| `category` | text | One of the standardized categories |
| `tags` | text[] | Keyword tags |
| `resolved` | bool | True when the event has settled |
| `resolved_at` | timestamptz | When it was resolved |
| `expiry` | timestamptz | Expected resolution time |
| `exchange_id` | int FK → sm.exchange | Exchange the event comes from |
| `native_event_id` | varchar | Exchange's own event identifier |
| `native_url` | varchar | Link to the event on the exchange |

`(exchange_id, native_event_id)` has a unique index. These columns were added by registry migration 019, which folds `sm.exchange_event` into `sm.event`.

**`sm.exchange_event`** — deprecated. The classifier no longer reads or writes it; a follow-up registry migration drops it once the native columns on `sm.event` become `NOT NULL`.

**`sm.security`** — one row per listed outcome contract

| Column | Type | Description |
|---|---|---|
| `security_id` | int PK | |
| `symbol` | text UNIQUE | e.g. `KX-KXWTI15M-26AUG240015-15-YES` |
| `type` | | Always `EVENT_CONTRACT` |
| `contract_type` | | `BINARY` or `MULTI_OUTCOME` |
| `asset_class` | | Always `PREDICTION` |
| `base/quote/settle_currency_id` | int FK | Currency references |
| `inverse`, `quanto` | bool | Always false |
| `expiry` | timestamptz | Event expiry |
| `active` | bool | False when deactivated |

**`sm.listing`** — maps a security to its exchange identifiers; `(exchange_id, exchange_security_id)` is the security's identity

| Column | Type | Description |
|---|---|---|
| `listing_id` | int PK | |
| `security_id` | int FK | |
| `exchange_id` | int FK | |
| `exchange_security_id` | text | Exchange's market/token ID |
| `exchange_security_symbol` | text | Exchange's human-readable symbol |
| `active` | bool | False when deactivated |

**`sm.listing_spec`** — `listing_id`, `tick_size`, `lot_size`, `min_notional`, `contract_multiplier`.

**`sm.event_contract`** — links an event to its outcome securities: `event_contract_id`, `event_id`, `security_id`, `outcome_label`.

**`sm.event_embedding`** — pgvector embeddings (`event_id`, `embedding`) from Voyage AI, searched through an HNSW index.

**`sm.contract_relationship`** — `relationship_id`, `security_id_a`, `security_id_b`, `relationship_type`, `confidence` (0.0–1.0), `method` (`"rule"`, `"embedding"`, or `"manual"`).

**`sm.currency`** — `currency_id`, `symbol`.

**`sm.hedge_keyword`** — `security_id` (a tradeable, non-event security) and `keyword`; a keyword appearing in an event title or outcome label marks that security as a hedge.

---

## Entity Creation Pipeline

**Entry point:** `create_entities()` in `classifier/stages/entities.py`. Every create goes through the registry API (`gnomepy.registry.RegistryClient`) bulk endpoints in chunks of 200.

### Step 1: Partition native events (known vs. new)

`prepare_canonicalization_inputs()`:
1. Groups contracts by `NativeKey = (exchange_id, exchange_event_native_id)`.
2. Looks each key up in the Redis exchange event cache, then in `sm.event` by `(exchange_id, native_event_id)` for cache misses (hits are written back to Redis).
3. Known keys map to their `event_id`; unknown keys become `CanonicalizeInput`s.

### Step 2: Categorize new events

If `canonicalization_enabled`, `canonicalize_events()` (`classifier/stages/canonicalize.py`) asks Claude for a `category` (one of the standardized categories) and 3–8 lowercase `tags` per new event. Titles are not rewritten.

1. Checks the Redis canonicalization cache.
2. Sends uncached events in chunks of `canonicalize_batch_size` (default 15). Each event carries a 6-character key derived from its title, which Claude must echo back; mismatches are treated as misses.
3. Uses parallel synchronous calls when the request count is ≤ `anthropic_sync_threshold`, otherwise the Anthropic Batch API.
4. Retries missed events one at a time, and raises `RuntimeError` if any still fail (the SQS message redelivers).
5. Caches results in Redis.

If disabled, the category is the exchange's category (or `"OTHER"`) and tags are empty.

### Step 3: Create entities

`create_entities_from_canonical()`:

1. **Events**: one `sm.event` per new native key with the adapter's `title`, `description`, `category`, `tags`, `expiry`, `exchange_id`, `native_event_id`, `native_url`. New mappings are written to the Redis exchange event cache.
2. **Existing listings**: looks up `sm.listing` by `(exchange_id, exchange_security_id)`; contracts with a listing already have their security.
3. **Currencies**: creates any missing base/quote/settle currency.
4. **Securities**: one per unlisted contract, using `AdapterContract.security_symbol`. A security with that symbol that has no listing at all (left behind by an earlier failed attempt, found via `get_unlisted_securities`) is reused. Any other symbol clash fails on the registry's unique symbol constraint rather than reusing a security.
5. **Listings**: one `sm.listing` per unlisted contract.
6. **Event contracts**: creates missing `(event_id, security_id)` links with the outcome label.
7. **Listing specs**: creates or updates `sm.listing_spec` when tick size, lot size, min notional or contract multiplier changed.

### Step 4: Reconcile removed contracts

`_reconcile_stale_entities()` covers outcomes an exchange removed from a known event. Because each message carries the full contract group of its native event:

1. Takes the events that already existed before this batch and loads their active listings.
2. Only judges listings on exchanges those pre-existing events belong to.
3. Deactivates any such listing whose `(exchange_id, exchange_security_id)` is not in the batch, then deactivates securities left with no active listings.

---

## Relationship Discovery

`run_classification_sync()` (`classifier/stages/classify.py`) runs two phases for the given security IDs, then deduplicates and writes.

### Relationship Types

| Type | Meaning | Source | Confidence |
|---|---|---|---|
| `EQUIVALENT` | Same question, different wording | Semantic judge (+ complement derivation) | per pair (LLM) |
| `IMPLIES` | If A resolves YES, B must resolve YES | Semantic judge (+ complement derivation) | per pair (LLM) |
| `MUTUALLY_EXCLUSIVE` | Both cannot resolve YES | Semantic judge | per pair (LLM) |
| `HEDGEABLE_WITH` | Event contract can be hedged with a tradeable security | Rule-based keyword match | `hedgeable_with_confidence` (0.90) |

Relationships between outcomes of the same event (complements, mutually exclusive siblings) are not written; they are implied by `sm.event_contract`.

### Phase A: Rule-based (always runs)

`classify_structural()` → `find_hedgeable_pairs()` (`classifier/relationships/rule_based.py`): loads `sm.hedge_keyword` and, for each event contract, checks the event title and outcome label for each keyword as a whole word (case-insensitive). Matches produce `HEDGEABLE_WITH` with method `"rule"`.

### Phase B: Semantic (only if `semantic_judgements_enabled`)

`classifier/relationships/semantic.py`:

1. **Candidates**: `db.find_all_neighbor_pairs()` runs a per-event LATERAL KNN query over `sm.event_embedding` (HNSW) restricted to unresolved events, keeping pairs with cosine similarity ≥ `embedding_similarity_threshold` (default 0.85) and at most `neighbor_search_limit` (default 60) neighbors per event. When the category filter is enabled, events outside `allowed_categories` are not used as search targets.
2. **Contracts sent to the judge**: for a binary event (exactly 2 contracts) only one representative contract is sent; other events send all contracts.
3. **Cache**: pairs already judged (Redis judgment cache) are reused.
4. **Judge**: uncached pairs go to Claude (`semantic_judgment_model`, default `claude-sonnet-4-6`) with a system prompt of worked examples (`_JUDGE_SYSTEM_PROMPT`). The user message lists both events' titles, descriptions, numbered contracts and the embedding similarity. Claude returns `[{"a", "b", "type", "confidence", "direction"?}]` with types `EQUIVALENT`, `IMPLIES` or `MUTUALLY_EXCLUSIVE`. Items below 0.70 confidence are discarded. Sync vs. Batch API follows `anthropic_sync_threshold`.
5. **Complement derivation** (`classifier/relationships/structural.py`): using each binary event's two contracts as complements:
   - `IMPLIES(A→B)` ⇒ `IMPLIES(¬B→¬A)`
   - `EQUIVALENT(A,B)` ⇒ `EQUIVALENT(¬A,¬B)`
   - `MUTUALLY_EXCLUSIVE(A,B)` ⇒ `IMPLIES(A→¬B)` and `IMPLIES(B→¬A)`

Semantic relationships are written with method `"embedding"`.

### Dedup and write

`_dedup_and_write_relationships()`:
- Skips pairs that already have a non-`manual` relationship.
- Skips pairs not touching any of the input securities.
- Keeps the highest-confidence candidate per ordered pair.
- Drops candidates below `min_confidence` (default 0.70).
- Writes via `registry.bulk_create_contract_relationships()` in chunks of 1000.

---

## Caching

### Redis (`classifier/cache/redis.py`, `classifier/workers/fetch.py`)

| Key | Value | TTL | Used by |
|---|---|---|---|
| `canon:{sha256(model, exchange_id, native_event_id)}` | `{"category", "tags"}` | 30 days | NormalizeWorker (categorization) |
| `judge:{sha256(model, titles + labels, order-independent)}` | `{"first_title", "items": [...]}`; empty `items` cached too | 30 days | RelationshipsWorker (semantic judge) |
| `exchange_events` (hash) | field `"{exchange_id}:{native_event_id}"` → `event_id` | none | NormalizeWorker (known-event lookup) |
| `fetch:known_contracts` | `{"eid:esid": hash}` | 7 days | FetchRunner fetch cycle |
| `fetch:sent_resolved` | `[[exchange_id, exchange_security_id], ...]` | 7 days | FetchRunner resolve cycle |
| `fetch:stale_tracker` | `{"eid:native_event_id": {exchange_id, native_event_id, miss_count}}` | 7 days | FetchRunner stale cycle |

Bulk reads use `MGET` or pipelines. A Redis read failure in the FetchRunner starts that state empty.

---

## Runtime Configuration

`RuntimeConfig` (`classifier/runtime_config.py`) fetches `/config/classifier` from the controller API at most every 30 seconds. On failure the last-known config is kept. The current defaults are sent base64-encoded in the `x-config-defaults` header.

**`FeatureFlags`** (all default `false`)

| Field | Effect when true |
|---|---|
| `fetch_enabled` | FetchRunner runs fetch cycles |
| `resolve_enabled` | FetchRunner runs resolve cycles |
| `stale_cleanup_enabled` | FetchRunner runs stale cycles |
| `canonicalization_enabled` | NormalizeWorker calls Claude for category and tags |
| `semantic_judgements_enabled` | RelationshipsWorker runs the semantic phase |
| `debug` | Verbose `[DEBUG]` logging across stages |

**`Models`**

| Field | Default |
|---|---|
| `canonicalize_model` | `claude-haiku-4-5-20251001` |
| `semantic_judgment_model` | `claude-sonnet-4-6` |
| `voyage_embedding_model` | `voyage-3` |

**`Thresholds`**

| Field | Default | Description |
|---|---|---|
| `embedding_similarity_threshold` | `0.85` | Minimum cosine similarity for semantic candidates |
| `min_confidence` | `0.70` | Relationships below this are not written |
| `hedgeable_with_confidence` | `0.90` | Confidence of keyword-matched hedgeable pairs |
| `min_event_volume` | `5000` | Event groups below this volume are not sent; `null` disables |

**`Processing`**

| Field | Default | Description |
|---|---|---|
| `canonicalize_batch_size` | `15` | Events per Claude categorization request |
| `bulk_create_batch_size` | `200` | Items per registry bulk-create call |
| `resolution_lookback_days` | `3` | How far back to query resolved markets |
| `neighbor_search_limit` | `60` | Max pgvector neighbors per event |
| `voyage_embed_chunk_size` | `2000` | Events per Voyage embedding chunk |
| `anthropic_sync_threshold` | `10` | Use sync calls at or below this many requests, Batch API above |
| `stale_miss_threshold` | `6` | Consecutive stale-cycle misses before an event is stale |
| `fetch_max_sqs_messages` | `100000` | Max groups sent per fetch cycle |
| `resolve_max_sqs_messages` | `100000` | Max messages sent per resolve cycle |
| `stale_max_sqs_messages` | `100000` | Max messages sent per stale cycle |

**`WorkerParams`**

| Field | Default | Description |
|---|---|---|
| `normalize_max_messages` / `normalize_max_wait_seconds` | `500` / `60` | NormalizeWorker batch limits |
| `embed_max_messages` / `embed_max_wait_seconds` | `2000` / `60` | EmbedWorker batch limits |
| `relationships_max_messages` / `relationships_max_wait_seconds` | `200` / `60` | RelationshipsWorker batch limits |
| `notify_max_messages` / `notify_max_wait_seconds` | `100000` / `300` | NotifyWorker batch limits |
| `fetch_interval_seconds` | `60` | FetchRunner fetch cycle interval |
| `resolve_interval_seconds` | `1800` | FetchRunner resolve cycle interval |
| `stale_interval_seconds` | `3600` | FetchRunner stale cycle interval |

**`CategoryFilter`**

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Restrict semantic classification to `allowed_categories` |
| `allowed_categories` | all categories | Categories included when enabled |

Standardized categories: `CRYPTO`, `POLITICS`, `SPORTS`, `ECONOMICS`, `ENTERTAINMENT`, `SCIENCE`, `TECHNOLOGY`, `WEATHER`, `LEGAL`, `OTHER`.

---

## Infrastructure

Defined with AWS CDK (TypeScript) in `cdk/`. `ClassifierPipelineStack` is a CodePipeline that deploys a Dev stage and, after manual approval, a Prod stage. Each stage contains `ClassifierStack` (`cdk/lib/stacks/classifier-stack.ts`) and `MonitoringStack` (`cdk/lib/stacks/monitoring-stack.ts`).

### Secrets (AWS Secrets Manager)

| Secret | Used by |
|---|---|
| `anthropic-api-key` | NormalizeWorker, RelationshipsWorker |
| `voyage-api-key` | EmbedWorker (also granted to RelationshipsWorker) |
| `slack-bot-token` | NotifyWorker |
| `registry-database-root-user` | NormalizeWorker, EmbedWorker, RelationshipsWorker |

Registry and controller API keys are read from API Gateway (`apigateway:GET`) at startup.

### SQS and SNS

| Queue | Visibility timeout | DLQ retention | maxReceiveCount |
|---|---|---|---|
| `ContractsQueue` | 5 min | 14 days | 3 |
| `EntitiesQueue` | 30 min | 14 days | 3 |
| `EmbeddingsQueue` | 15 min | 14 days | 3 |
| `SlackQueue` | 2 min | 7 days | 5 |

Workers override the visibility timeout to 15 minutes on each receive. `NotificationsTopic` (SNS, exported as `ClassifierNotificationsTopicArn`) fans out to `SlackQueue`.

### ECS Cluster

- One EC2 (not Fargate) cluster in the `registry-database-vpc`, backed by an auto-scaling group of t3.medium spot instances (1–2, public subnet, public IP).
- One service per worker (`desiredCount: 1`, `minHealthyPercent: 0`, `maxHealthyPercent: 100`), all from the repository's Docker image with the CMD set to the worker name:

| Service | CMD | Memory |
|---|---|---|
| Fetch | `fetch` | 1024 MiB |
| Normalize | `normalize` | 512 MiB |
| Embed | `embed` | 512 MiB |
| Relationships | `relationships` | 512 MiB |
| Notify | `notify` | 128 MiB |

### ElastiCache Redis

Single-node `cache.t3.micro` Redis in private subnets. Port 6379 is open to the worker security group and to the VPC CIDR (for SSM tunnel debugging). The endpoint is a stack output.

### Monitoring

`MonitoringStack` builds a CloudWatch dashboard via `cdk-monitoring-constructs` with alarms routed to Slack: max message age (1 hour) on each queue, any message in a DLQ, and service health for all five workers.

---

## Development

### Prerequisites

- Python 3.13 with [Poetry](https://python-poetry.org/)
- AWS credentials (for tunnels and real infrastructure)

### Setup

```bash
poetry install
```

### Running Tests

```bash
poetry run pytest -x -q
```

Tests use in-memory stubs from `scripts/testing.py`; no AWS services, database or API keys are needed:

- **`StubRegistry`** — in-memory `RegistryClient` with auto-incrementing IDs for bulk creates and in-place bulk patches.
- **`StubDB`** — in-memory `ClassifierDB` that reads `StubRegistry`'s state by reference.
- **`MemoryClassifierCache`** — dict-backed `ClassifierCache`.
- **`no_op_anthropic_client()`** / **`no_op_voyage_client()`** — mocks returning `category="OTHER"`, `tags=[]` and empty embeddings.

### Scripts (`[tool.poetry.scripts]`)

| Command | Purpose |
|---|---|
| `poetry run worker <fetch\|normalize\|embed\|relationships\|notify>` | Run a worker (`classifier/workers/runner.py`) |
| `poetry run dry-run <fetch\|canonicalize\|entities\|classify\|reclassify> ...` | Run pipeline stages locally against stubs, or real Postgres/Redis when `DATABASE_URL` / `REDIS_URL` are set (`scripts/dry_run.py`) |
| `poetry run bootstrap [ADAPTER] [--no-classify] [--with-judgment] [--batch-size N]` | One-time full load of the exchange universe into the real registry (`scripts/bootstrap.py`) |
| `poetry run tunnel [--stage dev\|prod] [--pg] [--redis]` | SSM port-forwarding tunnels to RDS and Redis through the bastion (`scripts/tunnel.py`) |

### Running the Pipeline In-Process

`classifier/pipeline.py` exposes `run_full_pipeline_sync()`, which runs entity creation → embedding → classification synchronously:

```python
from classifier.pipeline import run_full_pipeline_sync

result = run_full_pipeline_sync(
    registry, batch_client, contracts,
    voyage_client=voyage_client,
    cache=cache,
    db=db,
    skip_classify=False,
    skip_semantic=True,  # skip the LLM judgment phase
)
print(result.entity_result)
print(result.classification)
```

### Worker Entry Points

The Docker image's entrypoint is `python -m classifier.workers.runner`, a Click group whose subcommands (`fetch`, `normalize`, `embed`, `relationships`, `notify`) start the matching worker. SQS workers read queue URLs and secret names from environment variables via `WorkerConfig` and loop in `BaseWorker.run()`; `FetchRunner` reads `REDIS_ENDPOINT`, `CONTRACTS_QUEUE_URL` and the registry/controller settings directly. All exit cleanly on SIGTERM/SIGINT.
