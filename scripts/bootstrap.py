"""
One-time bootstrap for initial load against real infrastructure.

The Lambda pipeline has a 10-minute timeout and is designed for incremental
updates (tens of contracts per run). This script runs without a timeout
constraint to seed the registry with the full exchange universe initially.

Usage:
  poetry run tunnel --pg --redis     # open SSM tunnels to RDS + Redis
  export DATABASE_URL=...            # printed by tunnel command
  export REDIS_URL=...               # printed by tunnel command
  export ANTHROPIC_API_KEY=...
  export VOYAGE_API_KEY=...
  export CACHE_BUCKET=...            # optional S3 bucket (e.g. gnome-classifier-cache-dev)
  export REGISTRY_API_URL=...        # e.g. https://api.example.com
  export REGISTRY_API_KEY=...
  poetry run bootstrap [--no-classify]

Phase 1 — Entity creation (always runs):
  Fetches all adapters, canonicalizes event titles via Claude (cached to S3),
  and writes events, securities, listings, and exchange_event mappings to the
  real registry + Postgres DB.

Phase 2 — Classification (skipped with --no-classify):
  Runs relationship classification with skip_judgment=True. The structural
  finders (complement pairs, mutually exclusive pairs, hedgeable pairs) run
  and write relationships. Voyage embeddings are generated and stored in the
  HNSW index so the Lambda's first incremental run has the full index ready.
  Claude judgment calls are skipped — the cross-product of all initial
  securities is too large to judge at once. The Lambda handles semantic
  relationship judgment incrementally going forward.

Options:
  --no-classify   Skip Phase 2 entirely (entity creation only). Useful when
                  re-seeding entities after a schema migration without wanting
                  to re-derive all structural relationships.
"""
import logging
import os

import anthropic
import click
import voyageai
from gnomepy.registry import RegistryClient

from classifier.cache import RedisClassifierCache, S3ClassifierCache
from classifier.client import BatchAnthropicClient, BatchVoyageClient
from classifier.db import ClassifierDB
from classifier.pipeline import PipelineResult, fetch_exchanges, run_full_pipeline_sync
from classifier.stages.fetch import fetch_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.argument("adapter", required=False, default=None)
@click.option("--no-classify", is_flag=True, help="Skip relationship classification (entity creation only)")
@click.option("--with-judgment", is_flag=True, help="Run Claude judgment calls during classification (slow — use for small adapter runs)")
def main(adapter: str | None, no_classify: bool, with_judgment: bool) -> None:
    for var in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "REGISTRY_API_URL", "REGISTRY_API_KEY", "DATABASE_URL"):
        if not os.environ.get(var):
            raise click.ClickException(f"Missing required env var: {var}")

    registry = RegistryClient(
        base_url=os.environ["REGISTRY_API_URL"],
        api_key=os.environ["REGISTRY_API_KEY"],
    )
    batch_client = BatchAnthropicClient(
        client=anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
    )
    voyage_client = BatchVoyageClient(client=voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"]))
    redis_url = os.environ.get("REDIS_URL")
    cache_bucket = os.environ.get("CACHE_BUCKET")
    if redis_url:
        logger.info("Using Redis cache (SSM tunnel mode)")
        cache = RedisClassifierCache(redis_url=redis_url)
    elif cache_bucket:
        cache = S3ClassifierCache(bucket=cache_bucket)
    else:
        cache = None
    db = ClassifierDB(dsn=os.environ["DATABASE_URL"])

    try:
        exchange_by_name = fetch_exchanges(registry, adapter)
    except ValueError as e:
        raise click.ClickException(str(e))
    contracts, failed_adapters = fetch_all(exchange_by_name)
    if failed_adapters:
        logger.warning("Adapter fetch failures: %s", failed_adapters)
    print(f"Fetched {len(contracts)} contracts from {len(exchange_by_name)} exchanges", flush=True)

    print("\n=== PHASE 1: ENTITY CREATION ===\n", flush=True)
    result: PipelineResult = run_full_pipeline_sync(
        registry, batch_client, contracts,
        voyage_client=voyage_client, cache=cache, db=db,
        skip_classify=no_classify,
        skip_semantic=not with_judgment,
    )

    print("\nEntity creation summary:")
    for k, v in result.entity_result.counts.items():
        print(f"  {k}: {v}")
    print(f"  new_security_ids: {len(result.entity_result.new_security_ids)}")

    if result.classification is None:
        print("\nSkipping relationship classification.")
        return

    print(f"\n=== PHASE 2: RELATIONSHIP CLASSIFICATION ({len(result.entity_result.new_security_ids)} securities) ===\n", flush=True)
    print(f"  Structural: {result.classification.structural.get('relationships_written', 0)} written")
    if result.classification.semantic:
        print(f"  Semantic: {result.classification.semantic.get('relationships_written', 0)} written")

    print(f"\nBootstrap complete. {result.classification.relationships_written} structural relationships written.")
