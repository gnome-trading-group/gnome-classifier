"""
Local testing tool for the classifier pipeline.

Subcommands run progressively deeper pipeline stages:

  dry-run fetch ADAPTER [-n N]
      Fetch raw contracts from an adapter and display them grouped by event.

  dry-run canonicalize [ADAPTER] [-n N] [--no-cache]
      Fetch + canonicalize event titles via Claude. Shows raw→canonical mapping
      and reports collisions (multiple raw titles mapping to the same canonical title).
      Requires ANTHROPIC_API_KEY.

  dry-run entities [ADAPTER] [-n N] [--no-canonicalize] [--no-cache] [--verbose]
      Fetch + create entities (events, securities, listings). Prints summary counts.
      --verbose shows every created entity in detail.
      --no-canonicalize skips Claude and keeps raw titles (no API key required).

  dry-run classify [ADAPTER] [-n N] [--no-canonicalize] [--structural-only] [--skip-judgment] [--no-cache]
      Full pipeline: fetch + entities + relationship classification.
      --structural-only skips all semantic work (no embeddings, no Claude calls).
      --skip-judgment runs embedding search but skips Claude judgment calls.
      Requires ANTHROPIC_API_KEY and VOYAGE_API_KEY (unless --no-canonicalize for
      Anthropic; VOYAGE_API_KEY is always required for embeddings).

  dry-run reclassify [--with-semantic] [--no-cache]
      Re-run relationship classification on ALL existing events in the DB.
      Structural + rule-based only by default. --with-semantic adds Claude judgment.
      Requires DATABASE_URL, REGISTRY_API_URL, REGISTRY_API_KEY (and ANTHROPIC_API_KEY
      with --with-semantic).

Common options (on every subcommand):
  --debug          Enable debug logging
  -o / --output    JSON output path (default: dry_run_output.json)
  --no-cache       Ignore cache even if CACHE_BUCKET / REDIS_URL is set

All subcommands use in-memory stubs by default (no DB writes). Set DATABASE_URL
and REDIS_URL to use real Postgres + Redis via `poetry run tunnel`.
"""
import json
import logging
import os
from collections import defaultdict

import anthropic
import click
import voyageai

from classifier.cache import RedisClassifierCache
from classifier.constants import DEFAULT_MIN_EVENT_VOLUME, DEFAULT_RESOLUTION_LOOKBACK_DAYS as RESOLUTION_LOOKBACK_DAYS
from classifier.db import ClassifierDB
from classifier.pipeline import PipelineResult, create_entities_and_embed, fetch_exchanges, run_full_pipeline_sync
from classifier.client import BatchAnthropicClient, BatchVoyageClient
from classifier.stages.canonicalize import canonicalize_events
from classifier.stages.classify import classify_semantic_sync, prepare_semantic_batch, run_classification_sync
from classifier.stages.fetch import diff_contracts, fetch_all, fetch_resolved_outcomes
from classifier.stages.resolve import detect_resolved_events
from classifier.stages.stale import deactivate_stale_events
from classifier.types import CanonicalizeInput
from gnomepy.registry import RegistryClient
from scripts.testing import StubDB, StubRegistry, no_op_anthropic_client, no_op_voyage_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_cache(no_cache: bool):
    if no_cache:
        return None
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        logger.info("Using Redis cache (SSM tunnel mode)")
        return RedisClassifierCache(redis_url=redis_url)
    return None


def _build_clients(*, no_canonicalize: bool, no_cache: bool):
    if no_canonicalize:
        batch_client = BatchAnthropicClient(client=no_op_anthropic_client())
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise click.ClickException("ANTHROPIC_API_KEY not set — pass --no-canonicalize to skip, or set the key")
        batch_client = BatchAnthropicClient(client=anthropic.Anthropic(api_key=api_key))

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    raw_voyage = voyageai.Client(api_key=voyage_key) if voyage_key else no_op_voyage_client()
    voyage_client = BatchVoyageClient(client=raw_voyage)

    return batch_client, voyage_client, _build_cache(no_cache)


def _fetch_contracts(adapter: str | None, max_contracts: int | None):
    registry = StubRegistry()
    try:
        exchange_by_name = fetch_exchanges(registry, adapter)
    except ValueError as e:
        raise click.ClickException(str(e))

    database_url = os.environ.get("DATABASE_URL")
    db = ClassifierDB(dsn=database_url) if database_url else StubDB(registry)

    contracts, failed = fetch_all(exchange_by_name, max_per_adapter=max_contracts)
    if failed:
        logger.warning("Adapter fetch failures: %s", failed)

    return registry, db, contracts, exchange_by_name


def _display_fetch_results(contracts, new_messages: list[dict], adapter: str | None, min_volume: float | None):
    all_native = {(c.exchange_id, c.exchange_event_native_id) for c in contracts}
    total_events = len(all_native)
    filtered_out = total_events - len(new_messages)

    label = (adapter or "ALL ADAPTERS").upper()
    print(f"\n{'='*70}")
    print(f"{label}  ({len(contracts)} contracts, {total_events} events)")
    print(f"{'='*70}")

    volumes = []
    for msg in new_messages:
        group = msg["contracts"]
        c0 = group[0]
        print(f"\n  {c0['event_title']}")
        print(f"    native_id     : {c0['exchange_event_native_id']}")
        print(f"    contract_type : {c0['contract_type']}")
        print(f"    asset_class   : {c0['asset_class']}")
        print(f"    outcomes      : {[c['outcome_label'] for c in group]}")
        if c0.get("event_category"):
            print(f"    category      : {c0['event_category']}")
        if c0.get("event_description"):
            print(f"    description   : {c0['event_description'][:120]}")
        if c0.get("event_expiry"):
            print(f"    expiry        : {c0['event_expiry']}")
        if c0.get("event_volume") is not None:
            print(f"    volume        : ${c0['event_volume']:,.2f}")
            volumes.append(c0["event_volume"])
        else:
            print(f"    volume        : N/A")
        print(f"    currencies    : base={c0['base_currency']}  quote={c0['quote_currency']}  settle={c0['settle_currency']}")

    shown = len(new_messages)
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  events shown    : {shown}")
    if filtered_out:
        print(f"  filtered out    : {filtered_out}  (min_volume=${min_volume:,.2f})")
    print(f"  contracts total : {sum(len(msg['contracts']) for msg in new_messages)}")
    if volumes:
        print(f"  volume range    : ${min(volumes):,.2f} – ${max(volumes):,.2f}")
        print(f"  volume total    : ${sum(volumes):,.2f}")
        print(f"  no volume data  : {shown - len(volumes)}")


def _display_entities_verbose(registry: StubRegistry):
    data = registry.get_dry_run_data()

    if data["events"]:
        print(f"\n{'='*70}")
        print(f"CREATED EVENTS  ({len(data['events'])})")
        print(f"{'='*70}")
        for ev in data["events"]:
            print(f"\n  [{ev['event_id']}] {ev['title']}")
            if ev.get("category"):
                print(f"    category : {ev['category']}")
            if ev.get("tags"):
                print(f"    tags     : {ev['tags']}")
            if ev.get("expiry"):
                print(f"    expiry   : {ev['expiry']}")

    if data["securities"]:
        print(f"\n{'='*70}")
        print(f"CREATED SECURITIES  ({len(data['securities'])})")
        print(f"{'='*70}")
        for sec in data["securities"]:
            print(f"  [{sec['security_id']}] {sec['symbol']}  ({sec.get('contract_type', '?')})")

    if data["event_contracts"]:
        print(f"\n{'='*70}")
        print(f"CREATED EVENT CONTRACTS  ({len(data['event_contracts'])})")
        print(f"{'='*70}")
        for ec in data["event_contracts"]:
            sec = next((s for s in data["securities"] if s["security_id"] == ec["security_id"]), None)
            sym = sec["symbol"] if sec else f"security:{ec['security_id']}"
            print(f"  event:{ec['event_id']}  ×  {sym}  →  outcome: {ec['outcome_label']}")


def _run_canonicalize(contracts, batch_client, cache, output_path: str):
    print(f"\n=== CANONICALIZE ({len(contracts)} contracts) ===\n")

    contracts_by_native: dict[tuple, list] = {}
    for c in contracts:
        nk = (c.exchange_id, c.exchange_event_native_id)
        contracts_by_native.setdefault(nk, []).append(c)

    events_to_canonicalize = [
        CanonicalizeInput(
            group[0].event_title,
            group[0].event_description,
            group[0].event_category,
            exchange_id,
            native_id,
        )
        for (exchange_id, native_id), group in contracts_by_native.items()
    ]

    print(f"Canonicalizing {len(events_to_canonicalize)} unique events...")
    canonical_by_native = canonicalize_events(batch_client, events_to_canonicalize, cache=cache)
    print(f"Done. {len(canonical_by_native)} results.\n")

    raw_titles_by_canonical: dict[str, list[dict]] = defaultdict(list)
    for (exchange_id, native_id), info in canonical_by_native.items():
        group = contracts_by_native[(exchange_id, native_id)]
        raw_titles_by_canonical[info["title"]].append({
            "raw_title": group[0].event_title,
            "native_id": native_id,
            "expiry": group[0].event_expiry,
            "exchange_id": exchange_id,
            "category": info["category"],
            "tags": info["tags"],
        })

    collisions = {k: v for k, v in raw_titles_by_canonical.items() if len(v) > 1}
    print(f"Canonical titles with multiple raw sources (potential false merges): {len(collisions)}")
    for canonical_title, entries in list(collisions.items())[:20]:
        print(f"\n  [{canonical_title}]")
        for e in entries:
            print(f"    expiry={e['expiry']}  exchange={e['exchange_id']}  raw={e['raw_title'][:80]}")

    output = {
        "total_contracts": len(contracts),
        "unique_events": len(contracts_by_native),
        "canonical_results": len(canonical_by_native),
        "collision_count": len(collisions),
        "mapping": {
            f"{exchange_id}:{native_id}": {
                **info,
                "raw_title": contracts_by_native[(exchange_id, native_id)][0].event_title,
                "expiry": contracts_by_native[(exchange_id, native_id)][0].event_expiry,
            }
            for (exchange_id, native_id), info in canonical_by_native.items()
        },
        "collisions": collisions,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull mapping written to {output_path}")


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("-o", "--output", "output_path", default="dry_run_output.json", show_default=True, help="JSON output path")
@click.pass_context
def main(ctx, debug: bool, output_path: str):
    """Local testing tool for the classifier pipeline. Run `dry-run COMMAND --help` for details."""
    ctx.ensure_object(dict)
    ctx.obj["output_path"] = output_path
    ctx.obj["debug"] = debug
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.argument("adapter")
@click.option("-n", "max_contracts", type=int, default=None, help="Limit to first N contracts")
@click.option("--min-volume", type=float, default=DEFAULT_MIN_EVENT_VOLUME, show_default=True, help="Exclude events below this $ volume (0 to disable)")
def fetch(adapter: str, max_contracts: int | None, min_volume: float):
    """Fetch raw contracts from ADAPTER and display them grouped by event."""
    _, _, contracts, exchange_by_name = _fetch_contracts(adapter, max_contracts)
    if not contracts:
        print("No contracts returned.")
        return
    volume_filter = min_volume if min_volume > 0 else None
    new_messages, _, _, _ = diff_contracts(
        contracts, {}, [], exchange_by_name, volume_filter, max_messages=100_000,
    )
    _display_fetch_results(contracts, new_messages, adapter, volume_filter)


@main.command()
@click.argument("adapter", required=False, default=None)
@click.option("-n", "max_contracts", type=int, default=None, help="Limit to first N contracts")
@click.option("--no-cache", is_flag=True, help="Ignore cache even if CACHE_BUCKET / REDIS_URL is set")
@click.pass_context
def canonicalize(ctx, adapter: str | None, max_contracts: int | None, no_cache: bool):
    """Fetch + canonicalize event titles. Shows raw→canonical mapping and collision report."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise click.ClickException("ANTHROPIC_API_KEY not set")
    batch_client = BatchAnthropicClient(client=anthropic.Anthropic(api_key=api_key))

    _, _, contracts, _ = _fetch_contracts(adapter, max_contracts)
    if not contracts:
        print("No contracts returned.")
        return

    _run_canonicalize(contracts, batch_client, _build_cache(no_cache), ctx.obj["output_path"])


@main.command()
@click.argument("adapter", required=False, default=None)
@click.option("-n", "max_contracts", type=int, default=None, help="Limit to first N contracts")
@click.option("--no-canonicalize", is_flag=True, help="Skip Claude — keep raw titles (no API key required)")
@click.option("--no-cache", is_flag=True, help="Ignore cache even if CACHE_BUCKET / REDIS_URL is set")
@click.option("--verbose", is_flag=True, help="Show every created event, security, and event_contract")
@click.pass_context
def entities(ctx, adapter: str | None, max_contracts: int | None, no_canonicalize: bool, no_cache: bool, verbose: bool):
    """Fetch + create entities (events, securities, listings). Prints summary counts."""
    batch_client, voyage_client, cache = _build_clients(no_canonicalize=no_canonicalize, no_cache=no_cache)
    registry, db, contracts, exchange_by_name = _fetch_contracts(adapter, max_contracts)
    if not contracts:
        print("No contracts returned.")
        return

    print(f"\nRunning entity creation ({len(contracts)} contracts)...", flush=True)
    entity_result = create_entities_and_embed(registry, batch_client, contracts, voyage_client=voyage_client, cache=cache, db=db, debug=ctx.obj.get("debug", False))

    print(f"\n{'='*70}")
    print("ENTITY CREATION SUMMARY")
    print(f"{'='*70}")
    for k, v in entity_result.counts.items():
        print(f"  {k:<30}: {v}")
    print(f"  {'new_securities':<30}: {len(entity_result.new_security_ids)}")

    if verbose:
        _display_entities_verbose(registry)

    output = {**registry.get_dry_run_data(), "summary": entity_result.counts}
    with open(ctx.obj["output_path"], "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull output written to {ctx.obj['output_path']}")


@main.command()
@click.argument("adapter", required=False, default=None)
@click.option("-n", "max_contracts", type=int, default=None, help="Limit to first N contracts")
@click.option("--no-canonicalize", is_flag=True, help="Skip Claude canonicalization — keep raw titles")
@click.option("--structural-only", is_flag=True, help="Skip all semantic work — only structural + rule-based relationships")
@click.option("--skip-judgment", is_flag=True, help="Run embedding search but skip Claude judgment calls")
@click.option("--no-cache", is_flag=True, help="Ignore cache even if CACHE_BUCKET / REDIS_URL is set")
@click.pass_context
def classify(ctx, adapter: str | None, max_contracts: int | None, no_canonicalize: bool, structural_only: bool, skip_judgment: bool, no_cache: bool):
    """Full pipeline: fetch + entities + relationship classification."""
    batch_client, voyage_client, cache = _build_clients(
        no_canonicalize=no_canonicalize, no_cache=no_cache,
    )
    registry, db, contracts, exchange_by_name = _fetch_contracts(adapter, max_contracts)
    if not contracts:
        print("No contracts returned.")
        return

    print(f"\nRunning entity creation ({len(contracts)} contracts)...", flush=True)
    print("Running relationship classification...", flush=True)

    debug = ctx.obj.get("debug", False)
    if skip_judgment and not structural_only:
        # Embedding search only — run entity creation + structural, then show candidate count
        result: PipelineResult = run_full_pipeline_sync(
            registry, batch_client, contracts,
            voyage_client=voyage_client, cache=cache, db=db,
            skip_semantic=True, debug=debug,
        )
        if result.classification:
            api_requests, _, _ = prepare_semantic_batch(
                result.entity_result.new_security_ids, cache=cache, db=db, debug=debug,
            )
            if api_requests:
                logger.info("skip_judgment: would call Claude for %d pairs", len(api_requests))
    else:
        result = run_full_pipeline_sync(
            registry, batch_client, contracts,
            voyage_client=voyage_client, cache=cache, db=db,
            skip_semantic=structural_only, debug=debug,
        )

    entity_result = result.entity_result
    classification = result.classification

    relationship_result = {
        "relationships_written": classification.relationships_written if classification else 0,
        "relationships_skipped_low_confidence": classification.relationships_skipped_low_confidence if classification else 0,
    }

    summary = {**entity_result.counts, **relationship_result, "new_security_symbols": entity_result.new_security_symbols}

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {**registry.get_dry_run_data(), "summary": summary}
    with open(ctx.obj["output_path"], "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull output written to {ctx.obj['output_path']}")


@main.command()
@click.argument("adapter", required=False, default=None)
@click.option("--lookback", type=int, default=RESOLUTION_LOOKBACK_DAYS, show_default=True, help="Days to look back for resolved events")
@click.pass_context
def resolve(ctx, adapter: str | None, lookback: int):
    """Detect resolved outcomes and show what would be deactivated (dry-run mode)."""
    registry = StubRegistry()
    try:
        exchange_by_name = fetch_exchanges(registry, adapter)
    except ValueError as e:
        raise click.ClickException(str(e))

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("WARNING: DATABASE_URL not set — results will be empty (no listing data to match against).")
        print("         Set DATABASE_URL (e.g. via `poetry run tunnel`) to see real results.\n")

    if database_url:
        real_db = ClassifierDB(dsn=database_url)
        registry._listings = real_db.get_all_active_listings()
        registry._securities = real_db.get_all_securities()
        registry._events = real_db.get_unresolved_events()
        registry._event_contracts = real_db.get_all_event_contracts()

    db = StubDB(registry)

    print(f"\nFetching resolved outcomes from exchanges (lookback={lookback}d)...", flush=True)
    resolved_by_exchange, failed = fetch_resolved_outcomes(exchange_by_name, lookback_days=lookback)
    if failed:
        print(f"Adapter failures: {failed}")

    for exchange_id, ids in resolved_by_exchange.items():
        exchange_name = next(
            (name for name, ex in exchange_by_name.items() if ex.exchange_id == exchange_id), str(exchange_id)
        )
        print(f"  {exchange_name}: {len(ids)} resolved ids")

    db_label = "real DB (seeded)" if database_url else "stub DB (empty)"
    print(f"\nRunning resolution detection ({db_label}, dry-run writes)...", flush=True)
    result = detect_resolved_events(resolved_by_exchange, registry, db, debug=ctx.obj.get("debug", False))

    print(f"\n{'='*70}")
    print("RESOLUTION SUMMARY")
    print(f"{'='*70}")
    for k, v in result.items():
        print(f"  {k:<30}: {v}")

    with open(ctx.obj["output_path"], "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull output written to {ctx.obj['output_path']}")


@main.command()
@click.argument("adapter", required=False, default=None)
@click.option("--events", type=str, default=None, help="Comma-separated exchange_id:native_event_id pairs to simulate deactivation")
@click.pass_context
def stale(ctx, adapter: str | None, events: str | None):
    """Detect stale events by comparing active exchange listings against the DB.

    Without --events: shows which DB-tracked events are missing from active exchange
    listings (these would accumulate miss counts toward the stale threshold).

    With --events: simulates deactivation for specific exchange_id:native_event_id pairs,
    showing what would be deactivated in the registry (dry-run, no real writes).

    Requires DATABASE_URL (via `poetry run tunnel`).
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise click.ClickException("DATABASE_URL is required — set it via `poetry run tunnel`")

    if events:
        pairs = []
        for part in events.split(","):
            eid_str, nid = part.strip().split(":", 1)
            pairs.append((int(eid_str), nid))

        real_db = ClassifierDB(dsn=database_url)
        registry = StubRegistry()
        registry._listings = real_db.get_all_active_listings()
        registry._securities = real_db.get_all_securities()
        registry._events = real_db.get_unresolved_events()
        registry._event_contracts = real_db.get_all_event_contracts()
        registry._exchange_events = real_db.get_all_active_exchange_events()
        db = StubDB(registry)

        print(f"\nSimulating deactivation of {len(pairs)} event(s) (dry-run)...", flush=True)
        result = deactivate_stale_events(pairs, registry, db, debug=ctx.obj.get("debug", False))

        print(f"\n{'='*70}")
        print("STALE DEACTIVATION SUMMARY (dry-run)")
        print(f"{'='*70}")
        for k, v in result.items():
            print(f"  {k:<30}: {v}")
        return

    stub_registry = StubRegistry()
    try:
        exchange_by_name = fetch_exchanges(stub_registry, adapter)
    except ValueError as e:
        raise click.ClickException(str(e))

    print(f"\nFetching active events from exchanges...", flush=True)
    active_contracts, failed = fetch_all(exchange_by_name)
    if failed:
        logger.warning("Adapter fetch failures: %s", failed)

    active_by_exchange: dict[int, set[str]] = {}
    for c in active_contracts:
        active_by_exchange.setdefault(c.exchange_id, set()).add(c.exchange_event_native_id)

    real_db = ClassifierDB(dsn=database_url)
    db_native_ids = real_db.get_active_exchange_native_ids()

    exchange_name_by_id = {ex.exchange_id: name for name, ex in exchange_by_name.items()}
    missing: dict[int, set[str]] = {}
    for exchange_id, db_ids in db_native_ids.items():
        if exchange_id not in exchange_name_by_id:
            continue
        gone = db_ids - active_by_exchange.get(exchange_id, set())
        if gone:
            missing[exchange_id] = gone

    total_missing = sum(len(ids) for ids in missing.values())

    print(f"\n{'='*70}")
    print("STALE DETECTION SUMMARY")
    print(f"{'='*70}")
    for exchange_id, native_ids in missing.items():
        name = exchange_name_by_id.get(exchange_id, str(exchange_id))
        print(f"\n  {name.upper()} ({len(native_ids)} events missing from active listings):")
        for nid in sorted(native_ids):
            print(f"    {nid}")
    if not total_missing:
        print("  No stale events detected.")
    else:
        print(f"\n  Total events that would accumulate miss counts: {total_missing}")


@main.command()
@click.option("--with-semantic", is_flag=True, help="Also run semantic (embedding search + Claude judgment) in addition to structural")
@click.option("--count-only", is_flag=True, help="Count how many Claude calls would be made across all existing events — no writes, no API calls")
@click.option("--no-cache", is_flag=True, help="Ignore cache even if REDIS_URL is set")
@click.pass_context
def reclassify(ctx, with_semantic: bool, count_only: bool, no_cache: bool):
    """Re-run relationship classification on ALL existing events.

    Structural + rule-based only by default (no Claude calls). Use --with-semantic
    to also run the embedding search + judgment pass. Use --count-only to just
    count how many Claude calls the semantic pass would make without executing them.

    Requires DATABASE_URL, REGISTRY_API_URL, and REGISTRY_API_KEY env vars.
    Requires ANTHROPIC_API_KEY when --with-semantic is set (not needed for --count-only).
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise click.ClickException("DATABASE_URL is required for reclassify")

    if count_only:
        db = ClassifierDB(dsn=database_url)
        cache = _build_cache(no_cache)
        print("\nSearching for semantic candidates across all existing events...", flush=True)
        api_requests, _, cached_results = prepare_semantic_batch(
            new_security_ids=[], db=db, cache=cache,
        )
        total = len(api_requests) + len(cached_results)
        print(f"\n{'='*70}")
        print("SEMANTIC CANDIDATE COUNT")
        print(f"{'='*70}")
        print(f"  {'pairs pending Claude':<30}: {len(api_requests)}")
        print(f"  {'pairs already cached':<30}: {len(cached_results)}")
        print(f"  {'total candidate pairs':<30}: {total}")
        return

    registry_url = os.environ.get("REGISTRY_API_URL")
    registry_key = os.environ.get("REGISTRY_API_KEY")
    if not registry_url or not registry_key:
        raise click.ClickException("REGISTRY_API_URL and REGISTRY_API_KEY are required for reclassify")

    if with_semantic:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise click.ClickException("ANTHROPIC_API_KEY is required with --with-semantic")
        batch_client = BatchAnthropicClient(client=anthropic.Anthropic(api_key=api_key))
    else:
        batch_client = BatchAnthropicClient(client=no_op_anthropic_client())

    registry = RegistryClient(base_url=registry_url, api_key=registry_key)
    db = ClassifierDB(dsn=database_url)
    cache = _build_cache(no_cache)

    mode = "structural + rule-based + semantic" if with_semantic else "structural + rule-based only"
    print(f"\nReclassifying all existing events ({mode})...", flush=True)

    classification = run_classification_sync(
        registry, batch_client, new_security_ids=[],
        cache=cache, db=db, skip_semantic=not with_semantic,
        debug=ctx.obj.get("debug", False),
    )
    print(f"  Structural: {classification.structural.get('relationships_written', 0)} written")
    if classification.semantic:
        print(f"  Semantic: {classification.semantic.get('relationships_written', 0)} written")

    result = {
        "relationships_written": classification.relationships_written,
        "relationships_skipped_low_confidence": classification.relationships_skipped_low_confidence,
    }

    print(f"\n{'='*70}")
    print("RECLASSIFY SUMMARY")
    print(f"{'='*70}")
    for k, v in result.items():
        print(f"  {k:<30}: {v}")

    with open(ctx.obj["output_path"], "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull output written to {ctx.obj['output_path']}")
