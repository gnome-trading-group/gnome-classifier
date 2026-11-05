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
- `poetry run eval-subject-merges` - score the subject merge judge against the 182 golden subject pairs
  (`tests/fixtures/subjects/pairs.jsonl`; paid model calls; `--save F` / `--judgments F` to re-score free); fails
  under 99% SAME precision
- `poetry run dry-run <fetch|canonicalize|entities> ...` - run stages locally against in-memory
  stubs; real Postgres/Redis only when `DATABASE_URL`/`REDIS_URL` are set. Flags in the `scripts/dry_run.py` docstring.
- `poetry run bootstrap ...` - one-time full load into the real registry (writes for real; see script header)
- `poetry run backfill-taxonomy [--llm] [--apply]` - re-derive categories/tags from the static taxonomy, then with
  `--llm` have Claude fill missing categories and thin tags (paid); dry run without `--apply`
- `worker sync-names [--exchange X] [--apply]` - rename outcome labels, security symbols and listing names to what the
  venues list now; dry run without `--apply`
- `worker split-securities [--exchange X] [--apply]` - give each open market its own security where one security holds
  several (left from when symbols decided identity); dry run without `--apply`
- `worker repair-overdue [--exchange X] [--expired-days 7] [--apply]` - settle and resolve unresolved events that expired
  long ago, which no regular cycle revisits; looks Polymarket International up on the CLOB without the hourly budget; on
  Kalshi also resolves (stale-style) events whose markets Kalshi deleted (404); dry run without `--apply`
- All three read the database, so run them as a one-off ECS task on the Fetch task definition with the command overridden
  (`aws ecs run-task ... --overrides '{"containerOverrides":[{"name":"FetchContainer","command":["sync-names"]}]}'`)
- `poetry run tunnel` - currently broken: `<Stage>-DatabaseStack` no longer exports `BastionInstanceId`

CDK (TypeScript), from `cdk/` (`npm ci` needs a GitHub Packages token for `@gnome-trading-group/gnome-shared-cdk`):
- `npx tsc --noEmit` - type-check (`npm run build` also works but drops gitignored `.js`/`.d.ts` next to sources; edit the `.ts`)
- `npm test` - jest (`test/cdk.test.ts`, resource counts on `ClassifierStack`)
- `npx cdk synth` - what the pipeline runs; `npm run diff-substacks` - diff every stage stack

## Layout

- `classifier/workers/` - `runner.py` (Click entrypoint, the Docker `ENTRYPOINT`), `fetch.py` (`FetchRunner`, the
  timed loop), `base.py` (SQS batch loop for the other three), `config.py` (env vars, client init)
- `classifier/stages/` - the logic per step: `fetch` (diff + volume gate), `entities`, `canonicalize`,
  `propositions` (the relationships pass), `subject_merges` (the merge pass), `resolve`, `stale`, `settle`, `reactivate`, `fees`, `volume` (events' 24h
  volume to the registry)
- `classifier/adapters/` - one per venue (`kalshi`, `polymarket_intl`, `polymarket_us`, `hyperliquid`), registered in
  `ADAPTERS`; `types.py` (`AdapterContract`, `MarketState`), `settlement.py`
- `classifier/propositions/` - `types` (Proposition), `venue/` (code readers for fixed-template families, one module
  per venue: `kalshi`, `polymarket_us`, `polymarket_intl`, `hyperliquid`; shared parsing in `common`), `llm` (Claude
  extractor), `claude` (structured Claude requests with the provider fallback), `extract` (venue first, LLM for the
  rest, binary second sides negated), `subjects` (candidate subjects, embedding indexes), `merges` (duplicate subjects
  across venues: candidates, the Claude judge, the consequence check), `quotes` (top of book for the merge price
  check), `derive` (pure derivation), `drift`, `evaluation`,
  `merge_evaluation`; `classifier/relationships/evaluation.py` -
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
  either version re-reads or re-derives. A change to the code readers needs no bump: once per `READERS_VERSION` (a
  hash of `propositions/venue/`, marked in Redis `readers:version`) the Relationships worker walks the open
  securities and re-reads, by code alone, those the readers now read differently, and stores corrected any LLM reading
  a venue module's `correct` now changes (Polymarket Intl `↑`/`↓` level markets, whose shared rules text only says
  "reaches or exceeds"; Kalshi and Polymarket Intl election winners, which settle on the party whatever candidate the
  outcome names; Kalshi head-to-head outcomes ("Will A defeat B"), whose winner and opponent come from the title since
  the model sometimes keys the winner as their own opponent; Polymarket US game spreads, whose title names the team
  the line favours while the market asks whether the event's first team covers it, so a margin keyed to neither team is left unknown; Polymarket Intl game spreads,
  set from the rules' "win the game by N or more" on the team the model keyed, which the model sometimes reads as the
  moneyline; and on every venue, a title naming the round a market counts, "First Round" or "(1st Round)", scopes the
  reading to `first_round`, so it never meets a reading naming no round: topping a first round is not winning, and a
  final round is left unnamed as the deciding count). Exact scores are kind `SCORE`
  (`scores`: each side's entity and value), which decide the game's winner, totals and margins. Before deriving, a
  subject's readings get one spelling per measure and entity (`propositions/canonical.py`: a measure alias table,
  truncated last words, team codes in games; never a key extended by a word). A contest result (`margin`, `seats`,
  `vote_share`, `governors`, ...; not Senate seats, where "held" and "won" differ) is one reading whatever its
  observation, and different deadlines compare flagged `DEADLINE_DIFFERS`. Game pairs that a game not played as
  scheduled can settle differently are flagged `VOID_RULES_DIFFER`: any two venues, and within Polymarket
  International a winner or exact score against a spread or total (a canceled soccer game settles the first as the
  Draw, the second 50-50). Kalshi part-of-game series
  (`KX<league>1QSPREAD`, `2H`, `1P`) share the full game's subject, scoped from the rules (`first_quarter`). Hyperliquid price touches are read from the perp's current mid (`quotes.hyperliquid_mids`,
  builder exchanges included): an open touch is still on the side it started from. Only `EQUIVALENT` and `IMPLIES`
  exist; exclusion and complements come from `contract_type` and event membership. A field the extractor couldn't
  determine is in `unknown` and blocks every rule that needs it.
- An adapter must never move a security to another event; `entities.py` refuses and logs `already in event`
  (`IdentityRegressions`). Changing how an adapter derives `exchange_event_native_id` or `exchange_security_id` splits
  or orphans markets that collectors and sessions already reference.
- Adapters raise on API errors, never return partial results: a failed venue keeps its previous hashes and is
  excluded from stale miss-counting. Don't swallow errors inside an adapter.
- A fetch message carries an event's full contract set, because Normalize deactivates listings missing from it.
- Volume gate: `min_event_volume` (default 1000, overridable per tag/category) only gates events never sent before;
  already-sent events keep flowing. Polymarket US volume is approximated from share buckets. Every venue lists a game
  as several events gated alone (Kalshi each spread and total line, Polymarket the moneyline's outcomes and the exact
  scores separately), so a game's exact scores or moneyline can be in while the markets they relate to aren't. With
  `game_core_markets_enabled`, once any event of a game (`AdapterContract.game_id`) is in, its core markets
  (`game_core`: who wins the game or a map, spreads, totals, exact score; each adapter decides) follow whatever their
  volume, held to the end of the venue's pages since a game's events can arrive on different pages
  (`stages.fetch.GameCore`); props stay gated.
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
  `canonicalization_enabled`, `game_core_markets_enabled` (see Volume gate), `relationships_enabled`, `proposition_llm_enabled` (Claude for what the code readers
  can't read, plus the drift check; paid per contract, keep off in dev), `subject_merge_enabled` (the subject merge
  pass in Relationships, every 15 minutes while idle: Claude judges duplicate subjects, held-back merges go to the
  registry's `proposition-subjects/merge-candidates` queue; paid, about $0.0005 a pair), `subject_auto_merge_enabled`
  (that pass also makes the merges that pass every check, as `classifier:subject-merger`; off, it only logs them),
  `debug`. `RuntimeConfig` polls `/config/classifier` at most every 30s and keeps the last good config on failure. Code defaults only apply to keys the controller lacks.
- LLMs: every Claude call (proposition extraction in Relationships, canonicalization in Normalize, the subject merge
  judge) goes through `classifier/client/claude.py` with one provider, `models.provider` (`anthropic` API key, or
  `bedrock` via Bedrock's Messages endpoint, IAM `bedrock-mantle:*`), and `models.fallback_provider` on an outage.
  Each task has its own model per provider: `models.proposition_models`, `models.canonicalize_models` (default Haiku
  5.5 for both). Dev and prod have no Claude access on Bedrock yet (2026-10-29), so keep `anthropic` until it is
  enabled per account. Embeddings are Voyage (proposition subjects). Model ids are cache keys: changing a model
  re-runs everything once.
- Redis (single `cache.t3.micro`, private subnets): `fetch:known_contracts`, `fetch:sent_resolved`,
  `fetch:stale_tracker`, `fetch:stale_feed_counts` (each venue's event count on its last fetch that counted misses),
  `exchange_events`, `canon:*`, `prop:attempt:*` (proposition read backoff), `merge:judged:*`
  (what the merge pass made of each subject pair: judgment, what it rested on, outcome) and
  `merge:last-sweep:<MERGE_RULES_VERSION>` (30 days each; losing them costs one full re-judging sweep, about $5–8).
  Every key expires, since the node's `volatile-lru` only evicts keys that do. No bastion, SSM or ECS Exec reaches it, and
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
  `proposition drift`, `proposition collision`.
- Venues rename outcomes in place (Polymarket International placeholders "Team A" become the team). The entities stage
  keeps `outcome_label`, the security symbol and the listing name in step with the adapter; ids never change. A rename
  onto a symbol another security holds is skipped (`symbol rename clashes`).
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
  - Fetch (`fetch`, 3072 MiB): loop; fetch every `fetch_interval_seconds` (60) followed by closure check, recent
    settlements, reactivation; resolve every 1800s; stale every 3600s; settle sweep every 3600s; fee sync every 3600s.
    Volume write every 900s after a fetch; start times every 300s (`schedule_interval_seconds`): each event's scheduled
    start from Kalshi's milestones (read at most every 300s; tickers and rules keep the original time), Polymarket's
    `gameStartTime` or Hyperliquid's `scheduledStart`, written only when it moved and never cleared by a missing
    reading; the registry logs every change (`sm.event_schedule_change`) and the fetch log says `event rescheduled`. Intervals are `worker_params.*` in the service config; each cycle is
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
    recent settle, reactivation, resolve, stale cleanup, settle, fee sync, volume, schedule) or `Failed to fetch settlements from <venue>`.
    Read the traceback in the Fetch log group: a gnomepy/registry API mismatch (TypeError on a gnomepy model) needs a
    gnomepy bump; a venue HTTP error is usually that venue being down.
  - `Classifier--WorkerHealth-IdentityRegressions-Error`: an adapter tried to link a security to a second event. Code
    bug in that adapter's event id; find `already in event` in the Normalize logs.
  - `Classifier--WorkerHealth-VenueFeedGaps-Warning`: 6 hours running a venue returned no events, or under
    `stale_min_feed_ratio` (0.5) of what it returned on its last fetch that counted misses; stale cleanup is paused
    for it. Check the venue's API.
  - `Classifier--WorkerHealth-ClosureChecksSkipped-Warning`: > `closure_check_max_lookups` (500) contracts left one feed
    at once; treated as a feed fault, left to the resolve cycle.
  - `Classifier--WorkerHealth-VenueReaderFallbacks-Warning`: 25+ in an hour of `venue reader fell back`: a family the
    proposition code readers handle (`classifier/propositions/venue/`) stopped parsing, usually a venue rewording its
    rules. `poetry run eval-propositions --stored` shows which family; fix the reader (deployed, it re-reads what it
    now reads differently; no version bump).
  - `Classifier--WorkerHealth-PropositionDrift-Warning`: any `proposition drift`: the daily re-read by Claude disagreed
    with a code reader on a field. Read the line (family, field, both values) and the contract's stored source.
  - `Classifier--WorkerHealth-PropositionCollisions-Warning`: two outcomes of one event were read as the same
    proposition (two candidates given one name), so derivation left both out. Read the `proposition collision` line
    and the two contracts' sources; a code reader for a fixed-template family (no version bump), else a prompt fix
    for the next `EXTRACTOR_VERSION`.
  - `Classifier--WorkerHealth-MultiMarketSecurities-Warning`: a security with several open listings, or listings with
    different sources, got no source (`may be several markets`). Run `worker split-securities`.
  - `Classifier--WorkerHealth-ListingsReactivated-Warning`: 50+ retired listings on one venue listed as open again and
    turned back on; a venue misreporting closures.
- Runbook: gnome-registry `docs/runbooks/cloudwatch/Classifier.md` (it lists fewer alarms than exist here).
- TODO(verify): README drift to fix separately: Fetch memory (says 1024, is 3072), `min_event_volume` default (5000 vs
  1000), alarms "routed to Slack" (they have no actions), fee sync interval missing.
