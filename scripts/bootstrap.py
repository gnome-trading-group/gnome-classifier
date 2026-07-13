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
  export CACHE_BUCKET=...            # S3 bucket name (e.g. gnome-classifier-cache-dev)
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

from classifier.cache import S3ClassifierCache
from classifier.client import BatchAnthropicClient, ModelRateLimit
from classifier.constants import DEFAULT_RATE_LIMITS
from classifier.db import ClassifierDB
from classifier.stages.classify import classify_relationships
from classifier.stages.entities import create_entities
from classifier.stages.fetch import fetch_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--no-classify", is_flag=True, help="Skip relationship classification (entity creation only)")
def main(no_classify: bool) -> None:
    for var in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "CACHE_BUCKET", "REGISTRY_API_URL", "REGISTRY_API_KEY", "DATABASE_URL"):
        if not os.environ.get(var):
            raise click.ClickException(f"Missing required env var: {var}")

    registry = RegistryClient(
        base_url=os.environ["REGISTRY_API_URL"],
        api_key=os.environ["REGISTRY_API_KEY"],
    )
    _rate_limits = {k: ModelRateLimit(**v) for k, v in DEFAULT_RATE_LIMITS.items()}
    batch_client = BatchAnthropicClient(
        client=anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
        rate_limits=_rate_limits,
    )
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    cache = S3ClassifierCache(bucket=os.environ["CACHE_BUCKET"])
    db = ClassifierDB(dsn=os.environ["DATABASE_URL"])

    # ── Phase 1: entity creation ──────────────────────────────────────
    print("\n=== PHASE 1: ENTITY CREATION ===\n", flush=True)

    exchanges = registry.get_exchange()
    exchange_by_name = {e.exchange_name.lower(): e for e in exchanges}
    contracts, failed_adapters = fetch_all(exchange_by_name)
    if failed_adapters:
        logger.warning("Adapter fetch failures: %s", failed_adapters)
    print(f"Fetched {len(contracts)} contracts from {len(exchange_by_name)} exchanges", flush=True)

    entity_result = create_entities(
        registry, batch_client, contracts, cache=cache, db=db,
    )
    new_security_ids: list[int] = entity_result.pop("new_security_ids")
    entity_result.pop("new_security_symbols")

    print("\nEntity creation summary:")
    for k, v in entity_result.items():
        print(f"  {k}: {v}")
    print(f"  new_security_ids: {len(new_security_ids)}")

    if no_classify or not new_security_ids:
        print("\nSkipping relationship classification.")
        return

    # ── Phase 2: structural relationship classification ───────────────
    print(f"\n=== PHASE 2: RELATIONSHIP CLASSIFICATION ({len(new_security_ids)} securities) ===\n", flush=True)

    result = classify_relationships(
        registry, batch_client, voyage_client,
        new_security_ids=new_security_ids,
        skip_judgment=True,
        cache=cache,
        db=db,
    )
    written = result.get("relationships_written", 0)
    print(f"\nBootstrap complete. {written} relationships written total.")
