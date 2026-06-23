"""
Run the classifier locally without touching the DB.
"""
import json
import logging
import os
from collections import defaultdict
from unittest.mock import MagicMock

import anthropic
import click
import voyageai

from classifier.adapters import ADAPTERS
from classifier.cache import ClassifierCache
from classifier.stages.canonicalize import canonicalize_events
from classifier.stages.classify import classify_relationships
from classifier.stages.entities import create_entities
from classifier.stages.fetch import fetch_all
from classifier.types import CanonicalizeInput
from scripts.testing import StubRegistry

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")


def _no_op_anthropic_client() -> anthropic.Anthropic:
    def _fake_create(*args, **kwargs):
        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        content = messages[0].get("content", "") if messages else ""
        titles = []
        for line in content.splitlines():
            if line.startswith("[") and "] Title: " in line:
                title = line.split("] Title: ", 1)[1].split(" | ")[0].strip()
                titles.append({"title": title, "category": "OTHER"})
            elif line.startswith("Exchange-provided title:"):
                title = line.split(":", 1)[1].strip()
                titles.append({"title": title, "category": "OTHER"})

        if not titles:
            text = json.dumps([])
        else:
            text = json.dumps(titles) if len(titles) != 1 else json.dumps(titles[0])
        mock_content = MagicMock()
        mock_content.text = text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = _fake_create
    return client


def _run_canonicalize_only(contracts, client, cache, output_path):
    print(f"\n=== CANONICALIZE ONLY ({len(contracts)} contracts) ===\n")

    contracts_by_raw_title: dict[str, list] = {}
    for c in contracts:
        contracts_by_raw_title.setdefault(c.event_title, []).append(c)

    events_to_canonicalize = [
        CanonicalizeInput(
            raw_title,
            group[0].event_description,
            group[0].event_category,
            group[0].exchange_id,
            group[0].exchange_event_native_id,
        )
        for raw_title, group in contracts_by_raw_title.items()
    ]

    print(f"Canonicalizing {len(events_to_canonicalize)} unique raw titles...")
    canonical_by_raw = canonicalize_events(client, events_to_canonicalize, cache=cache)
    print(f"Done. {len(canonical_by_raw)} results.\n")

    raw_titles_by_canonical: dict[str, list[dict]] = defaultdict(list)
    for raw_title, info in canonical_by_raw.items():
        group = contracts_by_raw_title[raw_title]
        expiry = group[0].event_expiry
        exchange_id = group[0].exchange_id
        raw_titles_by_canonical[info["title"]].append({
            "raw_title": raw_title,
            "expiry": expiry,
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
        "unique_raw_titles": len(contracts_by_raw_title),
        "canonical_results": len(canonical_by_raw),
        "collision_count": len(collisions),
        "mapping": {
            raw: {**info, "expiry": contracts_by_raw_title[raw][0].event_expiry}
            for raw, info in canonical_by_raw.items()
        },
        "collisions": collisions,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull mapping written to {output_path}")


@click.command()
@click.argument("adapter", required=False, default=None)
@click.option("-n", "max_contracts", type=int, default=None, help="Limit to first N contracts per adapter")
@click.option("-o", "--output", "output_path", default="dry_run_output.json", show_default=True, help="Output JSON path")
@click.option("--no-canonicalize", is_flag=True, help="Skip Claude canonicalization — events keep raw titles/categories")
@click.option("--skip-semantic", is_flag=True, help="Count embedding pairs that would be sent to Claude without calling it")
@click.option("--canonicalize-only", is_flag=True, help="Fetch + canonicalize only; print raw→canonical mapping and exit")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(
    adapter: str | None,
    max_contracts: int | None,
    output_path: str,
    no_canonicalize: bool,
    skip_semantic: bool,
    canonicalize_only: bool,
    debug: bool,
) -> None:
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if no_canonicalize and not canonicalize_only:
        client = _no_op_anthropic_client()
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise click.ClickException("ANTHROPIC_API_KEY not set — pass --no-canonicalize to skip, or set the key")
        client = anthropic.Anthropic(api_key=api_key)

    cache = None
    cache_bucket = os.environ.get("CACHE_BUCKET")
    if cache_bucket:
        cache = ClassifierCache(bucket=cache_bucket)

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_key and not canonicalize_only:
        raise click.ClickException("VOYAGE_API_KEY not set")
    voyage_client = voyageai.Client(api_key=voyage_key) if voyage_key else None

    original = None
    if adapter:
        original = ADAPTERS[:]
        filtered = [a for a in ADAPTERS if a.exchange_name == adapter.lower()]
        if not filtered:
            raise click.ClickException(
                f"Unknown adapter '{adapter}'. Choices: {[a.exchange_name for a in ADAPTERS]}"
            )
        ADAPTERS[:] = filtered

    try:
        registry = StubRegistry()
        exchanges = registry.get_exchange()
        exchange_by_name = {e.exchange_name.lower(): e for e in exchanges}
        contracts = fetch_all(exchange_by_name, max_per_adapter=max_contracts)

        if canonicalize_only:
            _run_canonicalize_only(contracts, client, cache, output_path)
            return

        print("\n=== DRY RUN ===\n")

        entity_result = create_entities(registry, voyage_client, client, contracts, exchange_by_name)
        new_security_ids = entity_result.pop("new_security_ids")
        new_security_symbols = entity_result.pop("new_security_symbols")

        relationship_result = classify_relationships(
            registry, client, voyage_client,
            new_security_ids=new_security_ids,
            skip_judgment=skip_semantic,
        )

        summary = {**entity_result, **relationship_result, "new_security_symbols": new_security_symbols}

        print("\n=== SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        output = {**registry.get_dry_run_data(), "summary": summary}
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nFull output written to {output_path}")
    finally:
        if original is not None:
            ADAPTERS[:] = original
