"""
Fetch live Kalshi data and trace it through the adapter → entity creation pipeline.

Usage:
    ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... poetry run spot-check [-n N] [--no-canonicalize]

    -n N                Limit to first N contracts (default: all)
    --no-canonicalize   Skip Claude — keep raw titles, skip embeddings
"""
import json
import logging
import os
import sys
from unittest.mock import MagicMock

import anthropic
import voyageai

from classifier.adapters.kalshi import KalshiAdapter
from classifier.stages.entities import create_entities
from scripts.testing import StubRegistry

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

EXCHANGE_ID = 2  # kalshi


def _no_op_anthropic_client() -> anthropic.Anthropic:
    def _fake_create(*args, **kwargs):
        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        content = messages[0].get("content", "") if messages else ""
        titles = []
        for line in content.splitlines():
            if line.startswith("[") and "] Title: " in line:
                title = line.split("] Title: ", 1)[1].split(" | ")[0].strip()
                titles.append({"title": title, "category": "OTHER", "tags": []})
        text = json.dumps(titles) if len(titles) != 1 else json.dumps(titles[0])
        mock_content = MagicMock()
        mock_content.text = text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = _fake_create
    return client


def _no_op_voyage_client():
    client = MagicMock()
    result = MagicMock()
    result.embeddings = []
    client.embed.return_value = result
    return client


def main() -> None:
    args = sys.argv[1:]
    no_canonicalize = "--no-canonicalize" in args

    max_contracts = None
    if "-n" in args:
        idx = args.index("-n")
        try:
            max_contracts = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Usage: -n N requires an integer argument")
            sys.exit(1)

    # ── Fetch from Kalshi API ─────────────────────────────────────────
    print("Fetching from Kalshi API...", flush=True)
    registry = StubRegistry()
    exchanges = registry.get_exchange()
    exchange_by_name = {e.exchange_name.lower(): e for e in exchanges}

    contracts = KalshiAdapter().fetch(EXCHANGE_ID)
    if max_contracts:
        contracts = contracts[:max_contracts]

    if not contracts:
        print("No contracts returned from Kalshi.")
        sys.exit(0)

    contracts_by_title: dict[str, list] = {}
    for c in contracts:
        contracts_by_title.setdefault(c.event_title, []).append(c)

    print(f"\n{'='*70}")
    print(f"KALSHI ADAPTER OUTPUT  ({len(contracts)} contracts, {len(contracts_by_title)} events)")
    print(f"{'='*70}")
    for event_title, group in contracts_by_title.items():
        c0 = group[0]
        print(f"\n  {event_title}")
        print(f"    native_id     : {c0.exchange_event_native_id}")
        print(f"    contract_type : {c0.contract_type.name}")
        print(f"    outcomes      : {[c.outcome_label for c in group]}")
        if c0.event_description:
            print(f"    description   : {c0.event_description[:100]}")
        if c0.event_expiry:
            print(f"    expiry        : {c0.event_expiry}")

    # ── Build clients ─────────────────────────────────────────────────
    if no_canonicalize:
        anthropic_client = _no_op_anthropic_client()
        voyage_client = _no_op_voyage_client()
        print("\n[--no-canonicalize: skipping Claude and embeddings]")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        voyage_key = os.environ.get("VOYAGE_API_KEY")
        missing = [k for k, v in [("ANTHROPIC_API_KEY", api_key), ("VOYAGE_API_KEY", voyage_key)] if not v]
        if missing:
            print(f"Missing env vars: {', '.join(missing)}")
            print("Pass --no-canonicalize to skip Claude and embeddings.")
            sys.exit(1)
        anthropic_client = anthropic.Anthropic(api_key=api_key)
        voyage_client = voyageai.Client(api_key=voyage_key)

    # ── Run entity creation ───────────────────────────────────────────
    print("\nRunning entity creation...", flush=True)
    result = create_entities(registry, voyage_client, anthropic_client, contracts, exchange_by_name)

    new_security_ids = result.pop("new_security_ids")
    new_security_symbols = result.pop("new_security_symbols")

    print(f"\n{'='*70}")
    print("ENTITY CREATION SUMMARY")
    print(f"{'='*70}")
    print(f"  events created          : {result['events_created']}")
    print(f"  securities created      : {result['securities_created']}")
    print(f"  listings created        : {result['listings_created']}")
    print(f"  event_contracts created : {result['event_contracts_created']}")
    print(f"  listing_specs created   : {result['listing_specs_created']}")

    # ── Show created entities ─────────────────────────────────────────
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


if __name__ == "__main__":
    main()
