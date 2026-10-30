"""
Local testing tool for the classifier pipeline.

Subcommands run progressively deeper pipeline stages:

  dry-run fetch ADAPTER [-n N]
      Fetch raw contracts from an adapter and display them grouped by event.

  dry-run canonicalize [ADAPTER] [-n N] [--no-cache]
      Fetch + categorize events via Bedrock (category and tags). Shows the per-event
      result and the category distribution.
      Requires AWS credentials with Bedrock access (e.g. AWS_PROFILE=dev).

  dry-run entities [ADAPTER] [-n N] [--no-canonicalize] [--no-cache] [--verbose]
      Fetch + create entities (events, securities, listings). Prints summary counts.
      --verbose shows every created entity in detail.
      --no-canonicalize skips Bedrock and uses the exchange category with no tags (no AWS credentials required).

Propositions and relationships are scored offline by `poetry run eval-propositions` and
`poetry run eval-relationships --derive` instead of a dry run.

Common options (on every subcommand):
  --debug          Enable debug logging
  -o / --output    JSON output path (default: dry_run_output.json)
  --no-cache       Ignore the Redis cache even if REDIS_URL is set

All subcommands use in-memory stubs by default (no DB writes). Set DATABASE_URL
and REDIS_URL to use real Postgres + Redis via `poetry run tunnel`.
"""
import json
import logging
import os
from collections import Counter
from datetime import timedelta

import click

from classifier.adapters import ADAPTERS
from classifier.cache import RedisClassifierCache
from classifier.constants import (
    DEFAULT_MIN_EVENT_VOLUME,
    DEFAULT_RESOLUTION_LOOKBACK_DAYS as RESOLUTION_LOOKBACK_DAYS,
    DEFAULT_SETTLE_MAX_CANDIDATES,
    DEFAULT_SETTLEMENT_LOOKBACK_DAYS,
)
from classifier.db import ClassifierDB
from classifier.stages.entities import create_entities
from classifier.client import BedrockConverseClient, bedrock_runtime_client
from classifier.stages.canonicalize import canonicalize_events
from classifier.stages.fetch import diff_contracts, fetch_all, fetch_exchanges, fetch_resolved_outcomes
from classifier.stages.resolve import deactivate_closed_listings
from classifier.stages.settle import record_settlements
from classifier.stages.stale import deactivate_stale_events
from classifier.types import CanonicalizeInput
from gnomepy.registry import RegistryClient
from scripts.testing import FakeConverseClient, StubDB, StubRegistry

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
    canon_client = FakeConverseClient() if no_canonicalize else BedrockConverseClient(client=bedrock_runtime_client())

    return canon_client, _build_cache(no_cache)


def _fetch_contracts(adapter: str | None, max_contracts: int | None):
    registry = StubRegistry()
    try:
        exchange_by_code = fetch_exchanges(registry, adapter)
    except ValueError as e:
        raise click.ClickException(str(e))

    database_url = os.environ.get("DATABASE_URL")
    db = ClassifierDB(dsn=database_url) if database_url else StubDB(registry)

    contracts, failed = fetch_all(exchange_by_code, max_per_adapter=max_contracts)
    if failed:
        logger.warning("Adapter fetch failures: %s", failed)

    return registry, db, contracts, exchange_by_code


def _parse_volume_overrides(ctx, param, values: tuple[str, ...]) -> dict[str, float | None]:
    overrides: dict[str, float | None] = {}
    for item in values:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise click.BadParameter(f"expected KEY=VALUE, got {item!r}")
        try:
            overrides[key] = None if raw.lower() == "none" else float(raw)
        except ValueError:
            raise click.BadParameter(f"invalid volume {raw!r} for {key!r}")
    return overrides


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
        print(f"  filtered out    : {filtered_out}  (min_volume={min_volume})")
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


def _run_canonicalize(contracts, canon_client, cache, output_path: str):
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
            mapped_category=group[0].mapped_category,
            exchange_tags=tuple(group[0].event_tags),
        )
        for (exchange_id, native_id), group in contracts_by_native.items()
    ]

    print(f"Canonicalizing {len(events_to_canonicalize)} unique events...")
    canonical_by_native = canonicalize_events(canon_client, events_to_canonicalize, cache=cache)
    print(f"Done. {len(canonical_by_native)} results.\n")

    category_counts = Counter(info["category"] for info in canonical_by_native.values())
    print("Events per category:")
    for category, count in category_counts.most_common():
        print(f"  {category:<15} {count}")

    output = {
        "total_contracts": len(contracts),
        "unique_events": len(contracts_by_native),
        "canonical_results": len(canonical_by_native),
        "category_counts": dict(category_counts),
        "mapping": {
            f"{exchange_id}:{native_id}": {
                **info,
                "title": contracts_by_native[(exchange_id, native_id)][0].event_title,
                "expiry": contracts_by_native[(exchange_id, native_id)][0].event_expiry,
            }
            for (exchange_id, native_id), info in canonical_by_native.items()
        },
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
@click.option("--tag-volume", "tag_volumes", multiple=True, callback=_parse_volume_overrides, help="Per-tag threshold, e.g. cs2=50 (repeatable; 'none' ungates)")
@click.option("--category-volume", "category_volumes", multiple=True, callback=_parse_volume_overrides, help="Per-category threshold, e.g. SPORTS=200 (repeatable; 'none' ungates)")
def fetch(adapter: str, max_contracts: int | None, min_volume: float, tag_volumes: dict, category_volumes: dict):
    """Fetch raw contracts from ADAPTER and display them grouped by event."""
    _, _, contracts, exchange_by_code = _fetch_contracts(adapter, max_contracts)
    if not contracts:
        print("No contracts returned.")
        return
    volume_filter = min_volume if min_volume > 0 else None
    new_messages, _, _, _ = diff_contracts(
        contracts, {}, [], exchange_by_code, volume_filter, max_messages=100_000,
        min_volume_by_tag=tag_volumes, min_volume_by_category=category_volumes,
    )
    _display_fetch_results(contracts, new_messages, adapter, volume_filter)


@main.command()
@click.argument("adapter", required=False, default=None)
@click.option("-n", "max_contracts", type=int, default=None, help="Limit to first N contracts")
@click.option("--no-cache", is_flag=True, help="Ignore the Redis cache even if REDIS_URL is set")
@click.pass_context
def canonicalize(ctx, adapter: str | None, max_contracts: int | None, no_cache: bool):
    """Fetch + categorize events via Bedrock. Shows category and tags per event."""
    canon_client = BedrockConverseClient(client=bedrock_runtime_client())

    _, _, contracts, _ = _fetch_contracts(adapter, max_contracts)
    if not contracts:
        print("No contracts returned.")
        return

    _run_canonicalize(contracts, canon_client, _build_cache(no_cache), ctx.obj["output_path"])


@main.command()
@click.argument("adapter", required=False, default=None)
@click.option("-n", "max_contracts", type=int, default=None, help="Limit to first N contracts")
@click.option("--no-canonicalize", is_flag=True, help="Skip Bedrock — use exchange category, no tags (no AWS credentials required)")
@click.option("--no-cache", is_flag=True, help="Ignore the Redis cache even if REDIS_URL is set")
@click.option("--verbose", is_flag=True, help="Show every created event, security, and event_contract")
@click.pass_context
def entities(ctx, adapter: str | None, max_contracts: int | None, no_canonicalize: bool, no_cache: bool, verbose: bool):
    """Fetch + create entities (events, securities, listings). Prints summary counts."""
    canon_client, cache = _build_clients(no_canonicalize=no_canonicalize, no_cache=no_cache)
    registry, db, contracts, exchange_by_code = _fetch_contracts(adapter, max_contracts)
    if not contracts:
        print("No contracts returned.")
        return

    print(f"\nRunning entity creation ({len(contracts)} contracts)...", flush=True)
    entity_result = create_entities(registry, canon_client, contracts, cache=cache, db=db, debug=ctx.obj.get("debug", False))

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
@click.option("--lookback", type=int, default=RESOLUTION_LOOKBACK_DAYS, show_default=True, help="Days to look back for resolved events")
@click.pass_context
def resolve(ctx, adapter: str | None, lookback: int):
    """Detect closed outcomes and show what would be deactivated (dry-run mode); events resolve only on settlement."""
    registry = StubRegistry()
    try:
        exchange_by_code = fetch_exchanges(registry, adapter)
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
    resolved_by_exchange, failed = fetch_resolved_outcomes(exchange_by_code, lookback_days=lookback)
    if failed:
        print(f"Adapter failures: {failed}")

    for exchange_id, ids in resolved_by_exchange.items():
        exchange_code = next(
            (name for name, ex in exchange_by_code.items() if ex.exchange_id == exchange_id), str(exchange_id)
        )
        print(f"  {exchange_code}: {len(ids)} resolved ids")

    db_label = "real DB (seeded)" if database_url else "stub DB (empty)"
    print(f"\nRunning closed-listing deactivation ({db_label}, dry-run writes)...", flush=True)
    result = deactivate_closed_listings(resolved_by_exchange, registry, db, debug=ctx.obj.get("debug", False))

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
@click.option("--lookback", type=int, default=DEFAULT_SETTLEMENT_LOOKBACK_DAYS, show_default=True,
              help="Days since deactivation to look for settlement values")
@click.pass_context
def settle(ctx, adapter: str | None, lookback: int):
    """Look up final settlement values for deactivated outcomes and show what would be recorded (dry-run writes)."""
    database_url = os.environ.get("DATABASE_URL")
    registry_url = os.environ.get("REGISTRY_API_URL")
    registry_key = os.environ.get("REGISTRY_API_KEY")
    if not database_url or not registry_url or not registry_key:
        raise click.ClickException("DATABASE_URL, REGISTRY_API_URL and REGISTRY_API_KEY are required for settle")
    # Candidates come from the real database, so exchange ids must too; writes go to the stub.
    registry = StubRegistry()
    try:
        exchange_by_code = fetch_exchanges(RegistryClient(base_url=registry_url, api_key=registry_key), adapter)
    except ValueError as e:
        raise click.ClickException(str(e))
    result = record_settlements(
        ADAPTERS, exchange_by_code, registry, ClassifierDB(dsn=database_url), timedelta(days=lookback),
        DEFAULT_SETTLE_MAX_CANDIDATES, {},
    )

    print(f"\n{'='*70}")
    print("SETTLEMENT SUMMARY")
    print(f"{'='*70}")
    for k, v in result.items():
        print(f"  {k:<30}: {v}")
    output = {**result, "settlements": registry._settlement_by_event_contract}
    with open(ctx.obj["output_path"], "w") as f:
        json.dump(output, f, indent=2)
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
        exchange_by_code = fetch_exchanges(stub_registry, adapter)
    except ValueError as e:
        raise click.ClickException(str(e))

    print(f"\nFetching active events from exchanges...", flush=True)
    active_contracts, failed = fetch_all(exchange_by_code)
    if failed:
        logger.warning("Adapter fetch failures: %s", failed)

    active_by_exchange: dict[int, set[str]] = {}
    for c in active_contracts:
        active_by_exchange.setdefault(c.exchange_id, set()).add(c.exchange_event_native_id)

    real_db = ClassifierDB(dsn=database_url)
    db_native_ids = real_db.get_active_exchange_native_ids()

    exchange_code_by_id = {ex.exchange_id: name for name, ex in exchange_by_code.items()}
    missing: dict[int, set[str]] = {}
    for exchange_id, db_ids in db_native_ids.items():
        if exchange_id not in exchange_code_by_id:
            continue
        gone = db_ids - active_by_exchange.get(exchange_id, set())
        if gone:
            missing[exchange_id] = gone

    total_missing = sum(len(ids) for ids in missing.values())

    print(f"\n{'='*70}")
    print("STALE DETECTION SUMMARY")
    print(f"{'='*70}")
    for exchange_id, native_ids in missing.items():
        name = exchange_code_by_id.get(exchange_id, str(exchange_id))
        print(f"\n  {name.upper()} ({len(native_ids)} events missing from active listings):")
        for nid in sorted(native_ids):
            print(f"    {nid}")
    if not total_missing:
        print("  No stale events detected.")
    else:
        print(f"\n  Total events that would accumulate miss counts: {total_missing}")


