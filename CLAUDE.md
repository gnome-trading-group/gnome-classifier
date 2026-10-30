# CLAUDE.md

The classifier pulls every open market from the prediction venues (Kalshi, Polymarket International, Polymarket US,
Hyperliquid), writes them into the registry's security master as events, securities, listings and event contracts,
reads what each security asks (`sm.contract_proposition`) and derives relationships between them
(`sm.contract_relationship`). It also retires closed markets, records
settlement values, syncs venue fee schedules, and publishes resolved/new events on SNS. Dependents: the launcher (stops
sessions on resolved events via the exported `ClassifierNotificationsTopicArn`), Slack (Notify worker), anything that
trades or records the listings it creates, and the registry's settlement sweeper (reads the settlement values).

## Commands

Python 3.13, Poetry, from the repo root:
- `poetry install`
- `poetry run pytest -x -q`; one file: `poetry run pytest tests/test_fetch.py`; one test: add `::TestLeftFeed::test_a_failed_fetch_reports_nothing`
- `poetry run worker <fetch|normalize|relationships|notify>` - run a worker (needs the ECS env vars, see `classifier/workers/config.py`)
- `poetry run eval-relationships --derive` - score propositions + derivation against the 256 golden pairs (paid model
  calls; `--save-propositions F` / `--propositions F` to re-score free); `poetry run eval-propositions` - score
  extraction (`--venue-only` free, `--stored` replays stored sources, `--stored --drift` the drift check)
- `poetry run dry-run <fetch|canonicalize|entities> ...` - run stages locally against in-memory
  stubs; real Postgres/Redis only when `DATABASE_URL`/`REDIS_URL` are set. Flags in the `scripts/dry_run.py` docstring.
- `poetry run bootstrap ...` - one-time full load into the real registry (writes for real; see script header)
- `poetry run backfill-taxonomy [--apply]` - re-derive categories/tags from the static taxonomy; dry run without `--apply`
- `poetry run tunnel` - currently broken: `<Stage>-DatabaseStack` no longer exports `BastionInstanceId`

CDK (TypeScript), from `cdk/` (`npm ci` needs a GitHub Packages token for `@gnome-trading-group/gnome-shared-cdk`):
- `npx tsc --noEmit` - type-check (`npm run build` also works but drops gitignored `.js`/`.d.ts` next to sources; edit the `.ts`)
- `npm test` - jest (`test/cdk.test.ts`, resource counts on `ClassifierStack`)
- `npx cdk synth` - what the pipeline runs; `npm run diff-substacks` - diff every stage stack

## Layout

- `classifier/workers/` - `runner.py` (Click entrypoint, the Docker `ENTRYPOINT`), `fetch.py` (`FetchRunner`, the
  timed loop), `base.py` (SQS batch loop for the other three), `config.py` (env vars, client init)
- `classifier/stages/` - the logic per step: `fetch` (diff + volume gate), `entities`, `canonicalize`,
  `propositions` (the relationships pass), `resolve`, `stale`, `settle`, `reactivate`, `fees`, `volume` (events' 24h
  volume to the registry)
- `classifier/adapters/` - one per venue (`kalshi`, `polymarket_intl`, `polymarket_us`, `hyperliquid`), registered in
  `ADAPTERS`; `types.py` (`AdapterContract`, `MarketState`), `settlement.py`
- `classifier/propositions/` - `types` (Proposition), `venue/` (code readers for fixed-template families, one module
  per venue: `kalshi`, `polymarket_us`, `polymarket_intl`, `hyperliquid`; shared parsing in `common`), `llm` (Claude
  extractor), `extract` (venue first, LLM for the rest, binary second sides negated), `subjects` (candidate subjects,
  embedding indexes), `derive` (pure derivation), `drift`, `evaluation`; `classifier/relationships/evaluation.py` -
  scores relationships against `tests/fixtures/relationships/pairs.jsonl`
- `classifier/client/` - Bedrock, Voyage, HTTP clients; `classifier/cache/` - Redis cache
- `classifier/runtime_config.py` - `ClassifierConfig` read from the controller; `constants.py` - its defaults
- `classifier/taxonomy.py` - static venue label -> category/tags mapping; `db.py` - direct Postgres reads (pgvector)
- `scripts/testing.py` - `StubRegistry`, `StubDB`, `MemoryClassifierCache` used by tests and dry-run
- `cdk/lib/stacks/classifier-stack.ts` (queues, Redis, ECS, log metrics), `monitoring-stack.ts` (alarms),
  `cdk/lib/classifier-pipeline-stack.ts` (pipeline), `cdk/lib/config.ts` (accounts, Slack channel)
- `README.md` - long reference on every stage, adapter, table and config field. Parts are stale (see TODO below).

## Conventions and invariants

- Identity (2026-10 overhaul): one `sm.event` per native venue event, keyed `(exchange_id, native_event_id)`; never
  merge events across or within venues, and keep the venue's title. A security is identified by its listing
  `(exchange_id, exchange_security_id)`, never by symbol or title. A symbol clash fails on the registry's unique
  constraint by design. Cross-venue links come only from derived relationships (and manual ones).
- Relationships are derived, never judged: a security's proposition is read once per source hash and
  `EXTRACTOR_VERSION`, relationships are derived per subject by `propositions/derive.py` (`DERIVATION_VERSION`) and
  each subject's set is replaced as a whole (`contract-relationships/reconcile`); manual rows always win. Bumping
  either version re-reads or re-derives. Only `EQUIVALENT` and `IMPLIES` exist; exclusion and complements come from
  `contract_type` and event membership. A field the extractor couldn't determine is in `unknown` and blocks every rule
  that needs it.
- An adapter must never move a security to another event; `entities.py` refuses and logs `already in event`
  (`IdentityRegressions`). Changing how an adapter derives `exchange_event_native_id` or `exchange_security_id` splits
  or orphans markets that collectors and sessions already reference.
- Adapters raise on API errors, never return partial results: a failed venue keeps its previous hashes and is
  excluded from stale miss-counting. Don't swallow errors inside an adapter.
- A fetch message carries an event's full contract set, because Normalize deactivates listings missing from it.
- Volume gate: `min_event_volume` (default 1000, overridable per tag/category) only gates events never sent before;
  already-sent events keep flowing. Polymarket US volume is approximated from share buckets.
- Event volume: after a fetch, at most every `worker_params.volume_interval_seconds` (900), `VolumeWriter` patches
  `sm.event.volume_24h` (market-data collection rules select events by it) for events whose reading moved at least
  `volume_min_change` (0.10) or was last written `volume_max_age_seconds` (3600) ago. Only venues whose fetch
  succeeded; what it last wrote is in memory, so a restart rewrites every event once. A new event is also created
  with its volume (`entities._create_events` sends `volume_24h`), so it has one before the next volume pass. Needs
  registry migration 054 applied first (the patch fails on a registry without the column), and gnome-registry
  1.55+ for volume on create (older ones ignore the field).
- Bulk "closed" feeds only ever deactivate listings; an event resolves only when every outcome has a settlement value,
  or after a day missing from its feed (stale). Polymarket US once listed every open market as closed (2026-10-22).
- Feature flags (`feature_flags.*` in the controller service config `classifier`, all default false): `fetch_enabled`,
  `resolve_enabled`, `settlement_enabled`, `reactivation_enabled`, `stale_cleanup_enabled`, `fee_sync_enabled`,
  `canonicalization_enabled`, `relationships_enabled`, `proposition_llm_enabled` (Claude for what the code readers
  can't read, plus the drift check; paid per contract, keep off in dev), `debug`. `RuntimeConfig` polls `/config/classifier` at
  most every 30s and keeps the last good config on failure. Code defaults only apply to keys the controller lacks.
- LLMs: categorization runs on Bedrock Converse (`canonicalize_model`, default `us.amazon.nova-lite-v1:0`) and is billed
  to AWS, so Normalize needs `bedrock:InvokeModel`, not the Anthropic key. Proposition extraction uses Claude through
  `models.proposition_provider` (`anthropic` API key, or `bedrock`; `proposition_fallback_provider` on an outage) with
  `models.proposition_models[provider]`. Dev and prod have no Claude access on Bedrock yet (2026-10-29), so keep
  `anthropic` until it is enabled per account. Embeddings are Voyage (proposition subjects). Model ids are
  cache keys: changing a model re-runs everything once.
- Redis (single `cache.t3.micro`, private subnets): `fetch:known_contracts`, `fetch:sent_resolved`,
  `fetch:stale_tracker`, `exchange_events`, `canon:*`, `prop:attempt:*` (proposition read backoff; every key
  expires, since the node's `volatile-lru` only evicts keys that do). No bastion, SSM or ECS Exec reaches it, and
  `poetry run tunnel` is broken. TODO(verify): current access path; previously a temporary EC2 in the worker subnet/SG
  running a script from user-data.
- `security_symbol` is computed at fetch and baked into ContractsQueue messages, and `contract_hash` excludes it: a
  symbol fix doesn't repair queued or DLQ'd messages, and DLQ'd contracts are not resent on their own (they are
  already in `fetch:known_contracts`).
- Queues: Contracts -> Normalize -> Entities -> Relationships; SNS topic -> SlackQueue -> Notify.
  `process_batch` failure deletes nothing: messages redeliver (workers set a 15 min visibility per receive) and go to
  the DLQ after 3 receives (Slack: 5). Stages must stay idempotent on redelivery.
- Log lines counted by metric filters (`classifier-stack.ts`) must keep their wording: `cycle failed`,
  `Failed to fetch settlements from`, `not counting misses this cycle`, `not checking them this cycle`,
  `retired listings as open again`, `already in event`, `<Name>Worker started`, `venue reader fell back`,
  `proposition drift`.
- Tests use in-memory stubs from `scripts/testing.py` plus `MagicMock` LLM/Voyage clients and moto; no AWS needed.

## Operations

- Deploys: `ClassifierPipeline` (InfraPipelines account) builds `main` on every push: synth, `Dev` stage, manual
  `ApproveProd`, `Prod` stage. Stacks per stage: `Dev-`/`Prod-ClassifierStack` and `-ClassifierMonitoringStack`.
  Dev and prod are separate accounts. Prod Slack channel is set in `cdk/lib/config.ts`; dev posts nothing.
- gnomepy (`RegistryClient`, `Exchange` etc.) is the registry API's client. The image installs from `poetry.lock` and
  `pyproject.toml` only says `>3.0.0`, so a bump is `poetry update gnomepy` plus a deploy. Registry API shape changes
  break the classifier until the matching gnomepy ships here: the exchange `schema_type` removal broke fetch cycles
  until gnomepy 3.1.0.
- What runs: five ECS services (EC2 spot t3.medium ASG, 1-2 instances, registry VPC), one task each, all from the repo
  Docker image. There are no Lambdas besides CDK's ASG drain hook; resolve, stale and settle are Fetch cycles.
  - Fetch (`fetch`, 2048 MiB): loop; fetch every `fetch_interval_seconds` (60) followed by closure check, recent
    settlements, reactivation; resolve every 1800s; stale every 3600s; settle sweep every 3600s; fee sync every 3600s.
    Volume write every 900s after a fetch. Intervals are `worker_params.*` in the service config; each cycle is
    skipped while its flag is off (volume follows `fetch_enabled`).
  - Normalize (512 MiB, ContractsQueue), Relationships (512, EntitiesQueue),
    Notify (128, SlackQueue, batches up to 300s): continuous SQS long polling.
- Logs: one log group per worker, CloudFormation-named
  `Prod-ClassifierStack-<Worker>Task<Worker>ContainerLogGroup<hash>-<suffix>` (e.g. `FetchTaskFetchContainerLogGroup`),
  stream prefix = worker command. Find with `aws logs describe-log-groups --log-group-name-prefix Prod-ClassifierStack-FetchTask --profile prod`.
- Alarms (no actions; the registry's alerting router hears them via EventBridge, severity from the name suffix; nothing pages):
  - `Classifier--<Contracts|Entities|Slack>-Queue-Message-Age-Max-Warning`: oldest message > 1h. The
    consumer is down, crash-looping or slow; check its log group and `WorkerRestarts`.
  - `Classifier--<Queue>Dlq-Queue-Message-Count-Max-Error`: any DLQ message. Find the exception in the consumer's logs,
    fix, then redrive (stale symbols survive a redrive, see above).
  - `Classifier--<Worker>Worker-Memory-Usage-Warning`: > 90% of the task limit; ECS kills at 100%. Fetch was OOM-killed at 1024 MiB.
  - `Classifier--WorkerHealth-<Worker>WorkerRestarts-Error`: 3+ starts in an hour. Look for OOM kills (ECS stopped
    reason) or a startup exception (missing secret, registry/controller unreachable).
  - `Classifier--WorkerHealth-FetchCycleFailures-Error`: 3+ in an hour of `... cycle failed` (fetch, closure check,
    recent settle, reactivation, resolve, stale cleanup, settle, fee sync, volume) or `Failed to fetch settlements from <venue>`.
    Read the traceback in the Fetch log group: a gnomepy/registry API mismatch (TypeError on a gnomepy model) needs a
    gnomepy bump; a venue HTTP error is usually that venue being down.
  - `Classifier--WorkerHealth-IdentityRegressions-Error`: an adapter tried to link a security to a second event. Code
    bug in that adapter's event id; find `already in event` in the Normalize logs.
  - `Classifier--WorkerHealth-VenueFeedGaps-Warning`: 6 hours running a venue returned under `stale_min_feed_ratio`
    (0.5) of its tracked events; stale cleanup is paused for it. Check the venue's API.
  - `Classifier--WorkerHealth-ClosureChecksSkipped-Warning`: > `closure_check_max_lookups` (500) contracts left one feed
    at once; treated as a feed fault, left to the resolve cycle.
  - `Classifier--WorkerHealth-VenueReaderFallbacks-Warning`: 25+ in an hour of `venue reader fell back`: a family the
    proposition code readers handle (`classifier/propositions/venue/`) stopped parsing, usually a venue rewording its
    rules. `poetry run eval-propositions --stored` shows which family; fix the reader and bump `EXTRACTOR_VERSION`.
  - `Classifier--WorkerHealth-PropositionDrift-Warning`: any `proposition drift`: the daily re-read by Claude disagreed
    with a code reader on a field. Read the line (family, field, both values) and the contract's stored source.
  - `Classifier--WorkerHealth-ListingsReactivated-Warning`: 50+ retired listings on one venue listed as open again and
    turned back on; a venue misreporting closures.
- Runbook: gnome-registry `docs/runbooks/cloudwatch/Classifier.md` (it lists fewer alarms than exist here).
- TODO(verify): README drift to fix separately: Fetch memory (says 1024, is 2048), `min_event_volume` default (5000 vs
  1000), Normalize's Anthropic secret (it uses Bedrock), alarms "routed to Slack" (they have no actions), fee sync interval missing.
