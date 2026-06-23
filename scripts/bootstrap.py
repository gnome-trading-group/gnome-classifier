"""
One-time bootstrap for initial load of ~100k contracts.

The Lambda pipeline (10-min timeout) is designed for incremental updates.
This script runs without a timeout constraint to seed the registry initially.

Usage:
    ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... CACHE_BUCKET=... \\
    REGISTRY_API_URL=... REGISTRY_API_KEY=... \\
    poetry run bootstrap [--no-classify]

    --no-classify     Skip relationship classification (entity creation only)
"""
import logging
import os
import sys

import anthropic
import voyageai
from gnomepy.registry import RegistryClient

from classifier.cache import ClassifierCache
from classifier.stages.classify import classify_relationships
from classifier.stages.entities import create_entities
from classifier.stages.fetch import fetch_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    args = sys.argv[1:]
    no_classify = "--no-classify" in args

    for var in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "CACHE_BUCKET", "REGISTRY_API_URL", "REGISTRY_API_KEY"):
        if not os.environ.get(var):
            print(f"Missing required env var: {var}")
            sys.exit(1)

    registry = RegistryClient(
        base_url=os.environ["REGISTRY_API_URL"],
        api_key=os.environ["REGISTRY_API_KEY"],
    )
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    cache = ClassifierCache(bucket=os.environ["CACHE_BUCKET"])

    # ── Phase 1: entity creation ──────────────────────────────────────
    print("\n=== PHASE 1: ENTITY CREATION ===\n", flush=True)

    exchanges = registry.get_exchange()
    exchange_by_name = {e.exchange_name.lower(): e for e in exchanges}
    contracts = fetch_all(exchange_by_name)
    print(f"Fetched {len(contracts)} contracts from {len(exchange_by_name)} exchanges", flush=True)

    entity_result = create_entities(
        registry, voyage_client, anthropic_client, contracts, exchange_by_name, cache=cache
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
        registry, anthropic_client, voyage_client,
        new_security_ids=new_security_ids,
        skip_semantic=True,
        cache=cache,
    )
    written = result.get("relationships_written", 0)
    print(f"\nBootstrap complete. {written} relationships written total.")


if __name__ == "__main__":
    main()
