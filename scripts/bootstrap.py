"""
One-time bootstrap for initial load against real infrastructure.

The ECS worker pipeline is designed for incremental updates (tens of contracts
per run). This script runs without concurrency constraints to seed the registry
with the full exchange universe initially.

Usage:
  poetry run tunnel --pg --redis     # open SSM tunnels to RDS + Redis
  export DATABASE_URL=...            # printed by tunnel command
  export REDIS_URL=...               # printed by tunnel command
  export REGISTRY_API_URL=...        # e.g. https://api.example.com
  export REGISTRY_API_KEY=...
  poetry run bootstrap [ADAPTER] [--batch-size N]

Fetches all adapters, categorizes new events via Bedrock (cached in Redis), and writes events (one per native
exchange event), securities, listings and their venue sources to the real registry + Postgres DB. Propositions and
relationships are not written here: the relationships worker reads every security that has a source and no
proposition, so they follow once it runs with `feature_flags.relationships_enabled`.

Options:
  --batch-size N  Process contracts in chunks of N. Useful for large universes
                  to avoid memory pressure and long uninterruptible runs.
                  Entity creation is idempotent so a failed run can be safely
                  restarted from scratch.
"""
import logging
import os

import click
from gnomepy.registry import RegistryClient

from classifier.cache import RedisClassifierCache
from classifier.client import BedrockConverseClient, bedrock_runtime_client
from classifier.db import ClassifierDB
from classifier.stages.entities import create_entities
from classifier.stages.fetch import fetch_all, fetch_exchanges

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.argument("adapter", required=False, default=None)
@click.option("--batch-size", default=None, type=int, help="Process contracts in batches of this size")
def main(adapter: str | None, batch_size: int | None) -> None:
    for var in ("REGISTRY_API_URL", "REGISTRY_API_KEY", "DATABASE_URL"):
        if not os.environ.get(var):
            raise click.ClickException(f"Missing required env var: {var}")

    registry = RegistryClient(
        base_url=os.environ["REGISTRY_API_URL"],
        api_key=os.environ["REGISTRY_API_KEY"],
    )
    canon_client = BedrockConverseClient(client=bedrock_runtime_client())
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        logger.info("Using Redis cache (SSM tunnel mode)")
        cache = RedisClassifierCache(redis_url=redis_url)
    else:
        cache = None
    db = ClassifierDB(dsn=os.environ["DATABASE_URL"])

    try:
        exchange_by_code = fetch_exchanges(registry, adapter)
    except ValueError as e:
        raise click.ClickException(str(e))
    contracts, failed_adapters = fetch_all(exchange_by_code)
    if failed_adapters:
        logger.warning("Adapter fetch failures: %s", failed_adapters)
    print(f"Fetched {len(contracts)} contracts from {len(exchange_by_code)} exchanges", flush=True)

    batches = (
        [contracts[i:i + batch_size] for i in range(0, len(contracts), batch_size)]
        if batch_size else [contracts]
    )
    n_batches = len(batches)

    totals: dict[str, int] = {}
    for i, batch in enumerate(batches, 1):
        if n_batches > 1:
            print(f"\n=== BATCH {i}/{n_batches} ({len(batch)} contracts) ===\n", flush=True)
        entity_result = create_entities(registry, canon_client, batch, cache=cache, db=db)
        for k, v in entity_result.counts.items():
            totals[k] = totals.get(k, 0) + v
            if n_batches > 1 and v:
                print(f"  {k}: {v}")

    print("\n=== SUMMARY ===", flush=True)
    for k, v in totals.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
