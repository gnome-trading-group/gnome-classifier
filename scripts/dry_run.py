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

from classifier.cache import RedisClassifierCache, S3ClassifierCache
from classifier.constants import RESOLUTION_LOOKBACK_DAYS
from classifier.pipeline import (
    PipelineResult,
    classify_semantic_sync,
    create_entities_and_embed,
    fetch_exchanges,
    run_classification_sync,
    run_full_pipeline_sync,
)
from classifier.client import BatchAnthropicClient, BatchVoyageClient
from classifier.db import ClassifierDB
from classifier.stages.canonicalize import canonicalize_events
from classifier.stages.classify import prepare_semantic_batch
from classifier.stages.fetch import fetch_all, fetch_resolved_outcomes
from classifier.stages.resolve import detect_resolved_events
from classifier.types import CanonicalizeInput
from gnomepy.registry import RegistryClient
from scripts.testing import StubDB, StubRegistry, no_op_anthropic_client, no_op_voyage_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_cache(no_cache: bool):
    if no_cache:
        return None
    redis_url = os.environ.get("REDIS_URL")
    database_url = os.environ.get("DATABASE_URL")
    cache_bucket = os.environ.get("CACHE_BUCKET")
    if redis_url and database_url:
        logger.info("Using Redis cache + Postgres (SSM tunnel mode)")
        return RedisClassifierCache(redis_url=redis_url)
    if cache_bucket:
        return S3ClassifierCache(bucket=cache_bucket)
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


def _display_contracts(contracts, adapter: str | None):
    contracts_by_event: dict[str, list] = {}
    for c in contracts:
        contracts_by_event.setdefault(c.event_title, []).append(c)

    label = (adapter or "ALL ADAPTERS").upper()
    print(f"\n{'='*70}")
    print(f"{label}  ({len(contracts)} contracts, {len(contracts_by_event)} events)")
    print(f"{'='*70}")

    for event_title, group in contracts_by_event.items():
        c0 = group[0]
        print(f"\n  {event_title}")
        print(f"    native_id     : {c0.exchange_event_native_id}")
        print(f"    contract_type : {c0.contract_type.name}")
        print(f"    asset_class   : {c0.asset_class.name}")
        print(f"    outcomes      : {[c.outcome_label for c in group]}")
        if c0.event_category:
            print(f"    category      : {c0.event_category}")
        if c0.event_description:
            print(f"    description   : {c0.event_description[:120]}")
        if c0.event_expiry:
            print(f"    expiry        : {c0.event_expiry}")
        print(f"    currencies    : base={c0.base_currency}  quote={c0.quote_currency}  settle={c0.settle_currency}")


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
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.argument("adapter")
@click.option("-n", "max_contracts", type=int, default=None, help="Limit to first N contracts")
def fetch(adapter: str, max_contracts: int | None):
    """Fetch raw contracts from ADAPTER and display them grouped by event."""
    _, _, contracts, _ = _fetch_contracts(adapter, max_contracts)
    if not contracts:
        print("No contracts returned.")
        return
    _display_contracts(contracts, adapter)


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
    entity_result = create_entities_and_embed(registry, batch_client, contracts, voyage_client=voyage_client, cache=cache, db=db)

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

    if skip_judgment and not structural_only:
        # Embedding search only — run entity creation + structural, then show candidate count
        result: PipelineResult = run_full_pipeline_sync(
            registry, batch_client, contracts,
            voyage_client=voyage_client, cache=cache, db=db,
            skip_semantic=True,
        )
        if result.classification:
            api_requests, _, _ = prepare_semantic_batch(
                result.entity_result.new_security_ids, cache=cache, db=db,
            )
            if api_requests:
                logger.info("skip_judgment: would call Claude for %d pairs", len(api_requests))
    else:
        result = run_full_pipeline_sync(
            registry, batch_client, contracts,
            voyage_client=voyage_client, cache=cache, db=db,
            skip_semantic=structural_only,
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
    result = detect_resolved_events(resolved_by_exchange, registry, db)

    print(f"\n{'='*70}")
    print("RESOLUTION SUMMARY")
    print(f"{'='*70}")
    for k, v in result.items():
        print(f"  {k:<30}: {v}")

    with open(ctx.obj["output_path"], "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull output written to {ctx.obj['output_path']}")


@main.command()
@click.option("--with-semantic", is_flag=True, help="Also run semantic (embedding search + Claude judgment) in addition to structural")
@click.option("--no-cache", is_flag=True, help="Ignore cache even if REDIS_URL is set")
@click.pass_context
def reclassify(ctx, with_semantic: bool, no_cache: bool):
    """Re-run relationship classification on ALL existing events.

    Structural + rule-based only by default (no Claude calls). Use --with-semantic
    to also run the embedding search + judgment pass.

    Requires DATABASE_URL, REGISTRY_API_URL, and REGISTRY_API_KEY env vars.
    Requires ANTHROPIC_API_KEY when --with-semantic is set.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise click.ClickException("DATABASE_URL is required for reclassify")

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
