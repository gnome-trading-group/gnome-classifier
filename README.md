# gnome-classifier

A prediction market contract classifier that ingests contracts from multiple exchanges (Polymarket, Kalshi, Hyperliquid), normalizes them into a canonical cross-exchange security master, generates semantic embeddings, and discovers relationships between contracts.

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

The classifier solves one problem: prediction market exchanges each have their own schema, naming conventions, and contract structures. The same real-world event appears differently on each exchange — different titles, different outcome labels, different identifiers. This system ingests raw contracts from all exchanges, maps them to canonical events and securities, and builds a graph of relationships between those securities.

**What it produces:**
- A canonical `sm.event` record for each real-world event (with standardized title, category, tags)
- A `sm.security` record for each unique outcome (e.g. "Will Trump win? — Yes")
- `sm.listing` records mapping each security back to its exchange-specific identifiers
- `sm.contract_relationship` records encoding known relationships (complement pairs, mutual exclusion, implication, hedgeability, etc.)
- Vector embeddings in `sm.event_embedding` enabling similarity search

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        EventBridge Schedules                         │
│   FetchLambda (5min)  ResolveLambda (30min)  StaleCleanupLambda (1hr)│
└───────┬───────────────────────┬──────────────────────┬───────────────┘
        │  S3 Cache             │  S3 Cache             │  S3 Cache
        │  (known_contracts)    │  (sent_resolved)      │  (stale_tracker)
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                │
                         contracts-queue (SQS)
                                │
                        ┌───────▼────────┐
                        │ NormalizeWorker│ ◄── Redis (canon cache, exchange event cache)
                        │   (ECS/EC2)    │ ──► PostgreSQL (sm.event, sm.security, ...)
                        └───────┬────────┘
                                │
                         entities-queue (SQS)
                                │
                        ┌───────▼────────┐
                        │  EmbedWorker   │ ──► Voyage AI API
                        │   (ECS/EC2)    │ ──► PostgreSQL (sm.event_embedding)
                        └───────┬────────┘
                                │
                        embeddings-queue (SQS)
                                │
                        ┌───────▼──────────────┐
                        │ RelationshipsWorker   │ ◄── Redis (judgment cache)
                        │      (ECS/EC2)        │ ──► Claude API (sonnet)
                        │                       │ ──► PostgreSQL (sm.contract_relationship)
                        └───────┬───────────────┘
                                │
                          SNS Topic ──► slack-queue ──► NotifyWorker ──► Slack
```

**Components:**

| Component | Type | Schedule / Trigger |
|---|---|---|
| FetchLambda | AWS Lambda (no VPC) | Every 5 minutes |
| ResolveLambda | AWS Lambda (no VPC) | Every 30 minutes |
| StaleCleanupLambda | AWS Lambda (no VPC) | Every 1 hour |
| NormalizeWorker | ECS task (EC2 spot) | Continuous SQS poll |
| EmbedWorker | ECS task (EC2 spot) | Continuous SQS poll |
| RelationshipsWorker | ECS task (EC2 spot) | Continuous SQS poll |
| NotifyWorker | ECS task (EC2 spot) | Continuous SQS poll |

The three Lambdas run outside the VPC (they only call exchange APIs, S3, and SQS). The four ECS workers run inside the VPC on a single EC2 auto-scaling group (spot t3.medium, 1–2 instances) because they need access to PostgreSQL and Redis.

---

## Data Flow

### Stage 1: Fetch (`classifier/workers/fetch.py` → `fetch_handler`)

**Runs every 5 minutes.**

1. Checks the `fetch_enabled` feature flag. If false, exits immediately.
2. Calls `fetch_exchanges(registry)` to get the database's list of exchanges and their IDs.
3. Calls `fetch_all(exchange_by_name)` which iterates all adapters (Polymarket, Kalshi, Hyperliquid) and calls `adapter.fetch(exchange_id)` on each, returning a flat list of `AdapterContract` objects representing every currently active/open contract.
4. Computes a 16-character SHA-256 hash (`_contract_hash`) for each contract based on `exchange_id`, `exchange_security_id`, `outcome_label`, `event_title`, and `exchange_event_native_id`. This detects when exchange metadata changes (e.g. an outcome label changes from "Yes/No" to "Over 5.5/Under 5.5").
5. Groups contracts by `(exchange_id, exchange_event_native_id)` — all contracts belonging to the same event are grouped together.
6. Loads the previous fetch state from S3 at `fetch-cache/known_contracts.json`, a `dict[str, str]` mapping `"exchange_id:exchange_security_id"` to its last-seen hash.
7. For each event group, checks if ANY contract in the group has a new or changed hash. If so, the entire group is sent as a single SQS message. Grouping ensures the NormalizeWorker always sees the complete, current state of an event when processing it.
8. Sends changed/new groups to the contracts queue as `{"type": "new", "contracts": [...]}` messages, each containing the full `AdapterContract` dict for every contract in the event.
9. Replaces the S3 cache entirely with the current hash map.

**Why group by event?** If only contract B changes within a 3-contract event but contracts A and C are unchanged, the NormalizeWorker still needs to see all three contracts to correctly reconcile which listings are current and which are stale. A per-contract message format would not provide that context.

---

### Stage 2: Resolve (`classifier/workers/fetch.py` → `resolve_handler`)

**Runs every 30 minutes.**

1. Calls `fetch_resolved_outcomes(exchange_by_name, lookback_days)` which calls `adapter.fetch_resolved(exchange_id, lookback_days)` on each adapter. Each adapter queries its exchange for recently settled/closed contracts within the lookback window (default 3 days) and returns a set of `exchange_security_id` strings.
2. Loads the previous set of already-sent resolved IDs from S3 at `fetch-cache/sent_resolved.json`.
3. For each resolved ID not already sent, creates a `{"type": "resolved", "exchange_id": X, "native_id": Y}` message and sends it to the contracts queue. Here `native_id` is the `exchange_security_id`.
4. Replaces the S3 cache with the full current resolved set. Entries naturally expire when they fall outside the lookback window on subsequent runs.

---

### Stage 3: Stale Cleanup (`classifier/workers/fetch.py` → `stale_cleanup_handler`)

**Runs every hour.**

Handles the case where an exchange silently removes an event without ever marking it resolved — the event simply stops appearing in the active feed.

1. Checks the `stale_cleanup_enabled` feature flag. If false, exits.
2. Calls `fetch_all()` to get all currently active events across all exchanges. Records which exchanges failed.
3. Extracts the set of active `(exchange_id, native_event_id)` pairs from the results.
4. Loads the stale tracker from S3 at `fetch-cache/stale_tracker.json`. The tracker maps `"{exchange_id}:{native_event_id}"` to a record with `exchange_id`, `native_event_id`, and `miss_count`.
5. Updates the tracker:
   - Keys present in the current active set → reset `miss_count` to 0.
   - Keys absent from the current active set (on exchanges that did NOT fail) → increment `miss_count`.
   - New keys not yet in tracker → add with `miss_count = 0`.
   - Keys whose exchange failed → carry forward unchanged (no miss-counting).
6. Any key with `miss_count >= stale_miss_threshold` (default 6, meaning 6 consecutive hourly runs = ~6 hours) → send `{"type": "stale", "exchange_id": X, "native_event_id": Y}` to the contracts queue and remove the key from the tracker.
7. Saves the updated tracker to S3.

**Tracker size is self-limiting:** the tracker is effectively rebuilt each run from the current active set. Entries are only added when seen in the active API response, and removed once they cross the stale threshold. The maximum tracker size equals the total number of currently active events across all exchanges.

---

### Stage 4: Normalize (`classifier/workers/normalize.py` → `NormalizeWorker`)

**Continuous SQS poll. Reads from contracts-queue, writes to entities-queue.**

The NormalizeWorker collects batches of messages, then processes them all together. The batch collection loop:
- Polls SQS with 20-second long-polling
- Continues accumulating messages until `normalize_max_messages` is reached or `normalize_max_wait_seconds` has elapsed since the first message arrived
- Refreshes runtime config (from controller API) on each poll iteration

After collecting a batch, `process_batch()` runs:

**For `type: "new"` messages:**

Calls `create_entities()` (`classifier/stages/entities.py`). See [Entity Creation Pipeline](#entity-creation-pipeline) for the full breakdown. The result includes a list of new security IDs and their symbols. For each new security, a `{"type": "new_security", "security_id": X, "security_symbol": Y}` message is enqueued to the entities queue for embedding.

If any new entities were created, publishes `{"type": "new_entity", "security_symbol": Y, ...counts}` to SNS for each new security.

**For `type: "resolved"` messages:**

Calls `detect_resolved_events()` (`classifier/stages/resolve.py`):
1. Groups resolved `exchange_security_id`s by exchange.
2. Queries `sm.listing` for active listings matching those IDs on those exchanges.
3. Deactivates matched listings (`active = false`).
4. Checks which candidate securities now have zero active listings. Deactivates those securities.
5. For each deactivated security, finds linked events. For each event with zero remaining active securities, marks the event as resolved (`resolved = true`, `resolved_at = now`).
6. Publishes `{"type": "resolution", "events_resolved": N, "securities_deactivated": N, "listings_deactivated": N}` to SNS.

**For `type: "stale"` messages:**

Calls `deactivate_stale_events()` (`classifier/stages/stale.py`):
1. For each `(exchange_id, native_event_id)`, looks up the `event_id` via `db.get_exchange_event()`.
2. Gets security IDs linked to that event, then all active listings for those securities.
3. Filters to only listings on the stale exchange (other exchanges might still be active).
4. Runs the same cascade as resolution: deactivate listings → deactivate securities → resolve events.
5. Publishes `{"type": "stale_cleanup", ...counts}` to SNS.

After `process_batch()` succeeds, the worker deletes all processed messages from SQS. If `process_batch()` raises an exception, no messages are deleted — they become visible again after the visibility timeout (15 minutes) and are reprocessed. After 3 failed attempts, messages go to the DLQ.

---

### Stage 5: Embed (`classifier/workers/embed.py` → `EmbedWorker`)

**Continuous SQS poll. Reads from entities-queue, writes to embeddings-queue.**

Each message is `{"type": "new_security", "security_id": X, "security_symbol": Y}`.

`process_batch()`:
1. Extracts security IDs and symbols from the message batch.
2. Reconstructs an `EntityResult` with those security IDs.
3. Calls `embed_and_update()` (`classifier/stages/embed.py`):
   - Queries `db.get_events_without_embeddings()`: finds unresolved events with no row in `sm.event_embedding`.
   - Chunks those events (default 2000 at a time) and calls `voyage_client.embed_events()`.
   - The Voyage client builds embedding text as `"{title}. {description[:200]}"` and calls the Voyage API in batches of 128 texts using 5 parallel workers, with up to 3 retries and exponential backoff.
   - Upserts the resulting vectors into `sm.event_embedding` via `db.put_embeddings()`.
   - After embedding, calls `db.get_security_ids_for_events()` on the newly embedded event IDs. This expands the security ID set to include all securities linked to those events — important because an event might have been embedded but some of its securities were already known before this batch.
4. For each security ID in the updated result, sends `{"type": "security", "security_id": X, "security_symbol": Y}` to the embeddings queue.

---

### Stage 6: Relationships (`classifier/workers/relationships.py` → `RelationshipsWorker`)

**Continuous SQS poll. Reads from embeddings-queue, no output queue.**

Each message is `{"type": "security", "security_id": X, "security_symbol": Y}`.

`process_batch()`:
1. Extracts security IDs from messages.
2. Calls `run_classification_sync()` (`classifier/stages/classify.py`), which runs two phases:

**Phase A: Structural classification** (always runs):
- Loads event contracts and events linked to the new security IDs.
- `find_complement_pairs` (structural.py): for every binary event (exactly 2 contracts), creates a `COMPLEMENT` relationship between those two securities. Confidence 1.0.
- `find_mutually_exclusive_pairs` (structural.py): for every event with N contracts (N > 1), creates a `MUTUALLY_EXCLUSIVE` relationship between every ordered pair (i, j) where i ≠ j. Confidence 1.0.
- `find_hedgeable_pairs` (rule_based.py): queries `sm.hedge_keyword` for keyword→tradeable_security mappings. For each event contract, checks if the event title or outcome label contains any keyword (whole-word regex match). Creates `HEDGEABLE_WITH` relationships. Confidence 0.90.

**Phase B: Semantic classification** (only if `semantic_judgements_enabled = true`):
- Loads vector embeddings for the new events.
- `find_semantic_candidates` (semantic.py): for each new event, runs a pgvector cosine similarity search (`1 - (embedding <=> target) >= threshold`) to find similar events within a configurable threshold (default 0.80) and limit (default 50 neighbors). Deduplicates candidate pairs.
- Filters: skips pairs where both events have different (non-null) categories.
- Checks Redis judgment cache for previously judged pairs.
- For uncached pairs, builds Claude API requests. Each request includes both event titles, their contracts with outcome labels, and the embedding similarity score. Uses `claude-sonnet-4-6` by default.
- Submits via `BatchAnthropicClient`: if ≤ 10 requests, uses synchronous parallel API calls (16 threads); otherwise submits as Anthropic Batch API and polls every 30 seconds.
- Parses responses: Claude returns a JSON array of `{"a": idx, "b": idx, "type": "...", "confidence": float, "direction": "A_IMPLIES_B"|"B_IMPLIES_A"}`. Only EQUIVALENT, IMPLIES, CORRELATED, and MUTUALLY_EXCLUSIVE are returned by the judge.
- `derive_complement_relationships`: uses the structural complement map to infer additional relationships:
  - `IMPLIES(A→B)` implies `IMPLIES(comp(B)→comp(A))` (contrapositive)
  - `EQUIVALENT(A,B)` implies `EQUIVALENT(comp(A),comp(B))`
  - `MUTUALLY_EXCLUSIVE(A,B)` implies `IMPLIES(A→comp(B))` and `IMPLIES(B→comp(A))`
- Caches judgments in Redis.

**Dedup and write** (both phases):
- Loads existing relationships for all relevant security IDs.
- Skips pairs that already have a relationship (unless the existing method is "manual").
- For duplicate candidates (same pair, different methods), keeps the highest-confidence one.
- Drops any match below `min_confidence` threshold (default 0.70).
- Writes to `sm.contract_relationship` via `registry.bulk_create_contract_relationships()` in chunks of 1000.

3. For each written relationship, publishes a `{"type": "relationship", ...}` notification to SNS.

---

### Stage 7: Notify (`classifier/workers/notify.py` → `NotifyWorker`)

**Continuous SQS poll. Reads from slack-queue (SNS subscription), no output queue.**

Messages arrive via SNS → SQS. SNS wraps payloads in an envelope with `"Type": "Notification"`. The worker unwraps SNS envelopes and parses the inner payload.

Accumulates across all messages in a batch:
- `type: "new_entity"` → collects new security symbols, accumulates entity counts
- `type: "resolution"` → accumulates resolution counts
- `type: "stale_cleanup"` → accumulates stale cleanup counts
- `type: "relationship"` → increments relationship counter

Calls `format_notification_blocks()` to build a Slack Block Kit message:
- Header block: "Contract Classifier"
- Section block: bullet list of new security symbols (e.g. `• \`WILL-TRUMP-WIN-YES\``)
- Context block: dot-separated summary (e.g. `3 events · 5 securities · 5 listings · 12 relationships · 2 events resolved`)

Posts to Slack via `chat.postMessage` using `urllib.request` (no external HTTP libraries).

---

## Exchange Adapters

### AdapterContract

All three adapters produce `AdapterContract` objects (`classifier/adapters/types.py`):

| Field | Type | Description |
|---|---|---|
| `exchange_id` | int | Registry exchange ID |
| `exchange_security_id` | str | Exchange-specific unique identifier for this contract |
| `exchange_security_symbol` | str | Human-readable symbol (truncated) |
| `base_currency` | str | Always `"USDC"` |
| `quote_currency` | str | Always `"USDC"` |
| `settle_currency` | str | Always `"USDC"` |
| `security_type` | SecurityType | Always `EVENT_CONTRACT` |
| `contract_type` | ContractType | `BINARY` or `MULTI_OUTCOME` |
| `asset_class` | AssetClass | Always `PREDICTION` |
| `inverse` | bool | Always `False` |
| `is_quanto` | bool | Always `False` |
| `tick_size` | float | Minimum price increment |
| `lot_size` | float | Minimum trade size |
| `min_notional` | float | Minimum notional value |
| `contract_multiplier` | float | Payout scale (1e9 = USDC cents to full USDC) |
| `event_title` | str | Raw exchange event title |
| `outcome_label` | str | Outcome name (e.g. "Yes", "Trump", "Over 5.5") |
| `exchange_event_native_id` | str | Exchange's event identifier (used for grouping) |
| `event_description` | str \| None | Raw event description |
| `event_category` | str \| None | Exchange-provided category hint |
| `event_expiry` | str \| None | ISO 8601 timestamp of expected resolution |

---

### Polymarket (`classifier/adapters/polymarket.py`)

- **API base:** `https://gamma-api.polymarket.com`
- **Active fetch:** `GET /events/keyset?active=true&closed=false&limit=500` with cursor-based pagination
- **Resolved fetch:** `GET /events/keyset?active=false&closed=true&end_date_min={lookback_date}&limit=500`

**Binary markets** (single condition): each market has exactly 2 token outcomes. `exchange_event_native_id = conditionId`. `exchange_security_id = "conditionId:tokenId"`. Both Yes and No tokens are returned as separate contracts (`BINARY` contract type).

**Neg-risk multi-outcome markets**: when `len(markets) > 1` and all have `negRisk=True` (mutually exclusive outcomes like "Which team wins?"). `exchange_event_native_id = event_slug`. Each market becomes one `MULTI_OUTCOME` contract using `groupItemTitle` as the outcome label. `exchange_security_id = "conditionId:tokenId"`.

| Parameter | Value |
|---|---|
| `tick_size` | 10,000,000 |
| `lot_size` | 1,000,000 |
| `contract_multiplier` | 1,000,000,000 |

---

### Kalshi (`classifier/adapters/kalshi.py`)

- **API base:** `https://external-api.kalshi.com/trade-api/v2`
- **Active fetch:** `GET /events?with_nested_markets=true&status=open&limit=200` with cursor pagination
- **Resolved fetch:** `GET /events?status=settled&min_close_ts={lookback_ts}&limit=200`

**Multi-outcome events** (`mutually_exclusive=True`, >1 market): the event represents a categorical question. Each market becomes one `MULTI_OUTCOME` contract. `exchange_event_native_id = event_ticker`. `exchange_security_id = market_ticker`. `outcome_label = yes_sub_title`.

**Binary with sub-markets** (not mutually exclusive, >1 market): each sub-market becomes its own independent binary event. `exchange_event_native_id = market_ticker`. Title is `"{event_title}: {sub_title}"`. Yes/No contracts are created per sub-market.

**Simple binary** (single market): `exchange_event_native_id = event_ticker`. Yes/No contracts. `exchange_security_id = "ticker:yes"` / `"ticker:no"`.

Resolved IDs for multi-outcome events are market tickers; for binary events they are `"ticker:yes"` and `"ticker:no"`.

| Parameter | Value |
|---|---|
| `tick_size` | 1,000,000 |
| `lot_size` | 1,000,000 |
| `contract_multiplier` | 1,000,000,000 |

---

### Hyperliquid (`classifier/adapters/hyperliquid.py`)

- **API base:** `https://api.hyperliquid.xyz/info`
- **Active fetch:** `POST /info` with `{"type": "outcomeMeta"}`
- **Resolved fetch:** reads `settledNamedOutcomes` from the same endpoint's `questions` list

The response contains two top-level keys: `outcomes` (individual binary/multi contracts) and `questions` (grouped sets of outcomes).

**Multi-outcome questions**: when a question has >1 active outcome, each outcome becomes a `MULTI_OUTCOME` contract. `exchange_event_native_id = "q:{first_outcome_id}"`. `exchange_security_id = "@{outcome_id}"`.

**Binary questions**: single active outcome. `exchange_security_id = "@{outcome_id}:0"` (Yes) and `"@{outcome_id}:1"` (No). `exchange_event_native_id = "o:{outcome_id}"`.

**Orphan outcomes** (not linked to any question): handled individually. Special `class=priceBinary` outcomes auto-generate a title like "Will {underlying} be above ${targetPrice}?". `class=priceBucket` generates range labels like "< $50,000", "$50,000–$100,000", "> $100,000".

Metadata (category, expiry, underlying asset, price thresholds) is extracted from structured description fields using regex patterns.

Resolved security IDs are formatted as `"@{oid}"`, `"@{oid}:0"`, `"@{oid}:1"`.

| Parameter | Value |
|---|---|
| `tick_size` | (exchange-provided) |
| `lot_size` | (exchange-provided) |
| `contract_multiplier` | 1,000,000,000 |

---

## Data Model

### Database Tables (`sm` schema)

**`sm.event`** — one row per canonical real-world event

| Column | Type | Description |
|---|---|---|
| `event_id` | int PK | |
| `title` | text | Canonicalized, exchange-neutral title |
| `description` | text | Optional description |
| `category` | text | One of the standardized categories |
| `tags` | text[] | 3–8 keyword tags |
| `resolved` | bool | True when the event has settled |
| `resolved_at` | timestamptz | When it was resolved |
| `expiry` | timestamptz | Expected resolution time |

**`sm.exchange_event`** — maps exchange-specific events to canonical events

| Column | Type | Description |
|---|---|---|
| `exchange_event_id` | int PK | |
| `exchange_id` | int FK | Which exchange |
| `event_id` | int FK → sm.event | Canonical event |
| `native_event_id` | text | Exchange's own event identifier |
| `raw_title` | text | Original exchange title before canonicalization |

**`sm.security`** — one row per unique outcome (cross-exchange)

| Column | Type | Description |
|---|---|---|
| `security_id` | int PK | |
| `symbol` | text UNIQUE | e.g. `WILL-TRUMP-WIN-YES` |
| `type` | int | Always `EVENT_CONTRACT` |
| `contract_type` | int | `BINARY` or `MULTI_OUTCOME` |
| `asset_class` | int | Always `PREDICTION` |
| `base/quote/settle_currency_id` | int FK | Currency references |
| `inverse` | bool | Always false |
| `is_quanto` | bool | Always false |
| `expiry` | timestamptz | Inherited from event |
| `active` | bool | False when deactivated |

**`sm.listing`** — maps a security to its exchange-specific identifier

| Column | Type | Description |
|---|---|---|
| `listing_id` | int PK | |
| `security_id` | int FK | Canonical security |
| `exchange_id` | int FK | |
| `exchange_security_id` | text | Exchange's unique market/token ID |
| `exchange_security_symbol` | text | Exchange's human-readable symbol |
| `active` | bool | False when deactivated |

**`sm.listing_spec`** — trading parameters for a listing

| Column | Type | Description |
|---|---|---|
| `listing_id` | int FK | |
| `tick_size` | numeric | Minimum price increment |
| `lot_size` | numeric | Minimum trade size |
| `min_notional` | numeric | |
| `contract_multiplier` | numeric | Payout multiplier |

**`sm.event_contract`** — links an event to its outcome securities

| Column | Type | Description |
|---|---|---|
| `event_contract_id` | int PK | |
| `event_id` | int FK | |
| `security_id` | int FK | |
| `outcome_label` | text | Human-readable outcome (e.g. "Yes", "Trump") |

**`sm.event_embedding`** — pgvector embeddings for semantic search

| Column | Type | Description |
|---|---|---|
| `event_id` | int FK PK | |
| `embedding` | vector | Float array from Voyage AI |

**`sm.contract_relationship`** — known relationships between securities

| Column | Type | Description |
|---|---|---|
| `relationship_id` | int PK | |
| `security_id_a` | int FK | |
| `security_id_b` | int FK | |
| `relationship_type` | text | See relationship types below |
| `confidence` | numeric | 0.0–1.0 |
| `method` | text | `"structural"`, `"rule"`, `"embedding"`, `"manual"` |

**`sm.currency`** — currency reference data

| Column | Type |
|---|---|
| `currency_id` | int PK |
| `symbol` | text |

**`sm.hedge_keyword`** — keyword → tradeable security mapping for hedgeable pair detection

| Column | Type | Description |
|---|---|---|
| `security_id` | int FK | Tradeable (non-event) security |
| `keyword` | text | Word that, if found in an event title/outcome, suggests this security is a hedge |

---

## Entity Creation Pipeline

**Entry point:** `create_entities()` in `classifier/stages/entities.py`

### Step 1: Partition contracts (known vs. new)

`prepare_canonicalization_inputs(contracts, cache, db)`:

1. Groups all incoming contracts by `NativeKey = (exchange_id, exchange_event_native_id)`.
2. Bulk-checks Redis (`cache.get_exchange_event_bulk()`) for previously seen exchange events.
3. For cache misses, queries `db.get_all_exchange_events()` — a JOIN of `sm.exchange_event` with unresolved `sm.event` rows. Writes hits back to Redis.
4. **Known events**: NativeKeys already in the DB are recorded in `seen_exchange_events`. Their event IDs are stored in `event_id_by_native`. Event titles/categories/tags are pre-fetched for dedup.
5. **New events**: NativeKeys not found anywhere become `CanonicalizeInput` records (raw title, description, category, exchange ID, native ID) queued for canonicalization.

### Step 2: Canonicalize new event titles

If `canonicalization_enabled = true` (runtime flag):

`canonicalize_events(batch_client, events, cache, model, batch_size, sync_threshold)` in `classifier/stages/canonicalize.py`:

1. Bulk-checks the Redis canonicalization cache. Already-cached events skip the API call.
2. Chunks uncached events into batches of `canonicalize_batch_size` (default 50).
3. For each chunk, builds a prompt listing all events as `[N] Title: ... | Description: ... | Category: ...`. The prompt instructs Claude to return a JSON array with `title` (clean, exchange-neutral), `category` (one of 10 standardized values), and `tags` (3–8 lowercase keywords).
4. If total request count ≤ `anthropic_sync_threshold` (default 10): submits all requests in parallel using 16 threads. Otherwise: submits as Anthropic Batch API and polls every 30 seconds until complete (up to 4-hour timeout).
5. Parses responses. For any events that failed or were missing from the response (e.g. Claude omitted an item), retries individually with a simpler single-event prompt.
6. If any events permanently fail after individual retry, raises `RuntimeError` (the SQS message will redeliver).
7. Caches all successful results in Redis.
8. Returns `{NativeKey: {"title": str, "category": str, "tags": list[str]}}`.

If `canonicalization_enabled = false`:

Builds the canonical dict directly from raw contract data: `title = event_title`, `category = event_category or "OTHER"`, `tags = []`. No API calls. Cross-exchange title dedup will not work (same event on different exchanges will get separate canonical events) but the rest of the pipeline operates normally.

### Step 3: Title+expiry dedup

Before creating new events, `_title_expiry_dedup()` checks whether the canonicalized title matches an already-existing event:

1. Loads all unresolved events from DB: `(title, expiry, event_id)`.
2. For each new NativeKey, checks if its canonical title matches any existing event title AND their expiry timestamps are within `dedup_expiry_tolerance_hours` (default 1 hour) of each other. If either expiry is `None`, the timestamps are considered close (null = no constraint).
3. If a match is found: reuses the existing `event_id` instead of creating a duplicate event. This is how the same real-world event on two exchanges (Polymarket "Will Trump win?" and Kalshi "Will Trump win?") maps to a single `sm.event` row when canonicalization produces matching titles.
4. If no match: the event is queued for creation in `pending_events`.

### Step 4: Create entities

All creates use the registry API (`gnomepy.registry.RegistryClient`) via bulk endpoints, chunked at 200 items each.

1. **Events** (`registry.bulk_create_events`): creates one `sm.event` per pending event with `title`, `description`, `category`, `tags`, `expiry`.
2. **Exchange events** (`registry.bulk_create_exchange_events`): creates one `sm.exchange_event` per new NativeKey linking the exchange's native event ID to the canonical event ID. Also stores `raw_title`.
3. **Currencies**: checks which currency symbols (base/quote/settle) are missing from `sm.currency` and creates them.
4. **Security symbol generation**: for each contract, generates a symbol via `generate_security_symbol(canonical_title, outcome_label)`:
   - Lowercases the canonical title, removes all non-alphanumeric characters, replaces spaces with hyphens, truncates to 80 characters.
   - Does the same for the outcome label.
   - Concatenates as `"{title_slug}-{outcome_slug}"` and uppercases.
   - Example: "Will Trump win the 2024 election?" + "Yes" → `"WILL-TRUMP-WIN-THE-2024-ELECTION-YES"`
5. **Check existing securities** (`db.get_existing_securities`): avoids recreating securities that already exist (e.g. the same event appears on a second exchange).
6. **Securities** (`registry.bulk_create_securities`): creates `sm.security` rows for symbols not already in the DB.
7. **Listings** (`registry.bulk_create_listings`): creates `sm.listing` rows for each `(exchange_id, exchange_security_id)` pair not already in DB.
8. **Event contracts** (`registry.bulk_create_event_contracts`): creates `sm.event_contract` rows linking `(event_id, security_id)` with the outcome label.
9. **Listing specs** (`registry.bulk_create_listing_specs`): creates `sm.listing_spec` rows with tick size, lot size, min notional, and contract multiplier.

### Step 5: Reconcile stale entities within the batch

`_reconcile_stale_entities()` handles the case where a previously-known event now has fewer contracts than before (e.g. an exchange removed one outcome from a multi-outcome market):

1. Filters to NativeKeys that were already known (`seen_exchange_events`) and had an event ID in the DB.
2. Gets all security IDs linked to those events.
3. Gets all currently active listings for those securities.
4. Builds the set of exchange security IDs present in the current batch.
5. Any active listing whose `exchange_security_id` is NOT in the current batch (for that exchange) is stale — it was removed by the exchange. Deactivates those listings.
6. Securities that lose all active listings are also deactivated.

---

## Relationship Discovery

### Relationship Types

| Type | Meaning | How Discovered | Default Confidence |
|---|---|---|---|
| `COMPLEMENT` | One is the logical negation of the other (binary Yes/No pair) | Structural: event has exactly 2 contracts | 1.0 |
| `MUTUALLY_EXCLUSIVE` | At most one can resolve YES | Structural: contracts within the same event | 1.0 |
| `EQUIVALENT` | Same underlying question, different wording (arbitrage opportunity) | Semantic: Claude judge | per-pair (LLM) |
| `IMPLIES` | If A resolves YES, B must also resolve YES | Semantic: Claude judge | per-pair (LLM) |
| `CORRELATED` | Same underlying driver, tend to move together | Semantic: Claude judge | per-pair (LLM) |
| `HEDGEABLE_WITH` | An event contract can be hedged using a tradeable security | Rule-based: keyword matching | 0.90 |

### Semantic Candidate Selection

The pgvector similarity search (`db.find_neighbors`) runs the query:

```sql
SELECT event_id, 1 - (embedding <=> target_embedding) AS similarity
FROM sm.event_embedding
JOIN sm.event ON event.event_id = event_embedding.event_id
WHERE event.resolved = false
  AND 1 - (embedding <=> target_embedding) >= threshold
ORDER BY embedding <=> target_embedding
LIMIT limit
```

Only pairs where the similarity meets the threshold (default 0.80 cosine similarity) are sent to Claude for judgment. This keeps the LLM judgment budget focused on actually-similar event pairs.

### Claude Judgment Prompt

The judge uses an extensive system prompt (`_JUDGE_SYSTEM_PROMPT` in `relationships/semantic.py`) with worked examples of each relationship type. The user message shows:

```
Event A: Will BTC be above $100,000 by end of 2025?
  Contracts: [1] Yes [2] No
Event B: Will BTC exceed $100k in 2025?
  Contracts: [1] Yes [2] No
Embedding similarity: 0.947
```

Claude returns a JSON array of relationships between specific contract indices, with confidence and direction (for IMPLIES).

### Complement Inference

After Claude's judgments are parsed, the system infers additional relationships by applying logical rules to complement pairs:

- If `A → B` (IMPLIES), then `¬B → ¬A` (contrapositive IMPLIES)
- If `A ≡ B` (EQUIVALENT), then `¬A ≡ ¬B`
- If `A ⊕ B` (MUTUALLY_EXCLUSIVE), then `A → ¬B` and `B → ¬A`

This doubles the coverage without additional API calls.

---

## Caching

### Redis (`classifier/cache/redis.py`)

Three independent hash namespaces in Redis:

**Canonicalization cache (`"canon"` hash)**
- Key: SHA-256 of `(model, exchange_id, native_event_id)`
- Value: JSON `{"title": str, "category": str, "tags": [str]}`
- Written after every successful Claude canonicalization call
- Read before sending any events to Claude

**Judgment cache (`"judge"` hash)**
- Key: SHA-256 of `(model, title_a, labels_a, title_b, labels_b)` where titles are sorted lexicographically to make the key symmetric
- Value: JSON `{"first_title": str, "items": [{relationship JSON}]}`
- Written after every Claude semantic judgment response
- Read before building judgment requests for candidate pairs

**Exchange event cache (`"exchange_events"` hash)**
- Key: `"{exchange_id}:{native_event_id}"`
- Value: `event_id` as string
- Written after DB lookups in `prepare_canonicalization_inputs`
- Read at the start of every `create_entities()` call to avoid repeated DB queries for already-known events
- All reads/writes use Redis pipelines for efficiency

### S3 (`fetch-cache/` prefix in the classifier cache bucket)

| Key | Format | Purpose |
|---|---|---|
| `fetch-cache/known_contracts.json` | `{str: str}` — `"eid:esid"` → hash | Tracks last-seen hash of every contract. Replaced wholesale each FetchLambda run. |
| `fetch-cache/sent_resolved.json` | `[[eid, native_id], ...]` — list of pairs | Tracks resolved IDs already sent to the contracts queue. Replaced wholesale each ResolveLambda run; entries expire naturally when they fall outside the lookback window. |
| `fetch-cache/stale_tracker.json` | `{str: {exchange_id, native_event_id, miss_count}}` — keyed by `"eid:native_id"` | Tracks consecutive miss counts for events that stop appearing in fetch. Updated each StaleCleanupLambda run. |

---

## Runtime Configuration

Configuration is fetched from the controller API endpoint `/config/classifier` every 30 seconds. If the fetch fails, the last-known configuration is retained. The current defaults are sent as a base64-encoded JSON header with each request so the controller can show what defaults are in effect.

**`FeatureFlags`**

| Field | Default | Effect when true |
|---|---|---|
| `fetch_enabled` | `false` | FetchLambda actually fetches and sends contracts |
| `canonicalization_enabled` | `false` | NormalizeWorker calls Claude to canonicalize titles |
| `semantic_judgements_enabled` | `false` | RelationshipsWorker runs the semantic classification phase |
| `stale_cleanup_enabled` | `false` | StaleCleanupLambda actually sends stale messages |

**`Models`**

| Field | Default |
|---|---|
| `canonicalize_model` | `"claude-haiku-4-5-20251001"` |
| `semantic_judgment_model` | `"claude-sonnet-4-6"` |
| `voyage_embedding_model` | `"voyage-3"` |

**`Thresholds`**

| Field | Default | Description |
|---|---|---|
| `embedding_similarity_threshold` | `0.80` | Minimum cosine similarity for semantic candidate pairs |
| `min_confidence` | `0.70` | Relationships below this are dropped before writing |
| `structural_confidence` | `1.0` | Confidence assigned to structural relationships |
| `hedgeable_with_confidence` | `0.90` | Confidence assigned to keyword-matched hedgeable pairs |

**`Processing`**

| Field | Default | Description |
|---|---|---|
| `canonicalize_batch_size` | `50` | Events per Claude API chunk |
| `bulk_create_batch_size` | `200` | Items per registry bulk-create call |
| `dedup_expiry_tolerance_hours` | `1` | Max hour difference to consider two expiries "the same" for title dedup |
| `resolution_lookback_days` | `3` | How far back to query resolved markets |
| `neighbor_search_limit` | `50` | Max pgvector neighbors per event |
| `voyage_embed_chunk_size` | `2000` | Events per Voyage embedding batch |
| `anthropic_sync_threshold` | `10` | Use sync API if request count ≤ this; use Batch API otherwise |
| `stale_miss_threshold` | `6` | Consecutive hourly misses before an event is marked stale (≈ 6 hours) |

**`WorkerParams`**

| Field | Default | Description |
|---|---|---|
| `fetch_interval_seconds` | `300` | Not currently used by workers directly |
| `normalize_max_messages` | `500` | Max SQS messages per NormalizeWorker batch |
| `normalize_max_wait_seconds` | `60` | Max seconds to accumulate messages before processing |
| `embed_max_messages` | `500` | EmbedWorker batch size limit |
| `embed_max_wait_seconds` | `60` | EmbedWorker wait limit |
| `relationships_max_messages` | `200` | RelationshipsWorker batch size limit |
| `relationships_max_wait_seconds` | `60` | RelationshipsWorker wait limit |
| `notify_max_messages` | `500` | NotifyWorker batch size limit |
| `notify_max_wait_seconds` | `300` | NotifyWorker wait limit (5 min, allows message accumulation for batch Slack posts) |

**`CategoryFilter`**

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | If true, only events in `allowed_categories` get semantic classification |
| `allowed_categories` | all 10 categories | Categories to include when filtering is enabled |

Standardized categories: `CRYPTO`, `POLITICS`, `SPORTS`, `ECONOMICS`, `ENTERTAINMENT`, `SCIENCE`, `TECHNOLOGY`, `WEATHER`, `LEGAL`, `OTHER`.

---

## Infrastructure

All infrastructure is defined in `cdk/lib/stacks/classifier-stack.ts` using AWS CDK (TypeScript).

### Secrets (AWS Secrets Manager)

| Secret Name | Used By |
|---|---|
| `anthropic-api-key` | NormalizeWorker, RelationshipsWorker |
| `voyage-api-key` | EmbedWorker, RelationshipsWorker |
| `slack-bot-token` | NotifyWorker |
| `registry-database-root-user` | NormalizeWorker, EmbedWorker, RelationshipsWorker |

API keys for the registry and controller APIs are stored in API Gateway and fetched via `apigateway:GET` at startup.

### S3 Bucket

`gnome-classifier-cache-{stage}` — 90-day lifecycle expiration on all objects.

### SQS Queues

| Queue | Visibility Timeout | DLQ Retention | maxReceiveCount |
|---|---|---|---|
| `ContractsQueue` | 5 minutes | 14 days | 3 |
| `EntitiesQueue` | 30 minutes | 14 days | 3 |
| `EmbeddingsQueue` | 15 minutes | 14 days | 3 |
| `SlackQueue` | 2 minutes | 7 days | 5 |

The visibility timeout determines how long a message is hidden from other consumers while being processed. If `process_batch()` takes longer than the timeout, the message becomes visible again and will be redelivered. After `maxReceiveCount` failed deliveries, messages move to the DLQ for investigation.

### Lambda Functions

All three Lambdas use the same Docker image as the ECS workers. No VPC attachment — they only need internet access (exchange APIs) and AWS service endpoints (SQS, S3).

| Lambda | Handler | Schedule | Timeout | Memory |
|---|---|---|---|---|
| FetchLambda | `classifier.workers.fetch.fetch_handler` | Every 5 min | 10 min | 2048 MB |
| ResolveLambda | `classifier.workers.fetch.resolve_handler` | Every 30 min | 10 min | 2048 MB |
| StaleCleanupLambda | `classifier.workers.fetch.stale_cleanup_handler` | Every 1 hr | 10 min | 2048 MB |

### ECS Cluster

- Single cluster on EC2 (not Fargate) for cost efficiency.
- Auto-scaling group: t3.medium spot instances, 1–2 instances, public subnet with public IP, `allowAllOutbound`.
- All four workers run as separate ECS services (`desiredCount: 1`, `minHealthyPercent: 0`, `maxHealthyPercent: 100`) on this cluster.

### ElastiCache Redis

Single-node Redis cluster (`cache.t3.micro`) in private subnets. Workers connect via the VPC. Access controlled by a security group that allows port 6379 from the worker security group and from within the VPC CIDR (for SSM tunnel debugging).

---

## Development

### Prerequisites

- Python 3.12+ with [Poetry](https://python-poetry.org/)
- AWS credentials configured

### Setup

```bash
poetry install
```

### Running Tests

```bash
poetry run pytest -x -q
```

All tests use in-memory stubs from `scripts/testing.py` — no real AWS services, database, or API keys required.

**`StubRegistry`** — in-memory `RegistryClient` implementation. All bulk-create methods allocate auto-incrementing IDs and store records in lists. All bulk-patch methods update the in-memory records in place. Shares mutable state with `StubDB`.

**`StubDB`** — in-memory `ClassifierDB` that reads directly from `StubRegistry._events`, `._securities`, `._listings`, etc. Because it shares state by reference, writes via `StubRegistry` are immediately visible to `StubDB` queries.

**`MemoryClassifierCache`** — in-memory `ClassifierCache` using Python dicts. No Redis required.

**`no_op_anthropic_client()`** — mock Anthropic client that parses the prompt for event titles and echoes them back with `category="OTHER"` and `tags=[]`. Used in entity creation tests when canonicalization is enabled.

### Running the Full Pipeline Locally

`classifier/pipeline.py` exposes `run_full_pipeline_sync()` which runs the complete entity creation → embedding → classification flow synchronously in a single process. This is useful for scripts and one-off runs:

```python
from classifier.pipeline import run_full_pipeline_sync

result = run_full_pipeline_sync(
    registry, batch_client, contracts,
    voyage_client=voyage_client,
    cache=cache,
    db=db,
    skip_classify=False,
    skip_semantic=True,  # skip LLM judgment phase
)
print(result.entity_result)
print(result.classification)
```

### Worker Entry Points

Each worker is started by the Docker image CMD:

| CMD | Worker |
|---|---|
| `normalize` | `NormalizeWorker` |
| `embed` | `EmbedWorker` |
| `relationships` | `RelationshipsWorker` |
| `notify` | `NotifyWorker` |

Each entry point instantiates a `WorkerConfig` (reads from environment variables), creates the worker, and calls `worker.run()` which enters the infinite poll loop.
