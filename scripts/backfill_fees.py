"""Backfills fee data onto listings the classifier no longer sees: resolved markets, which the fetch cycle never
revisits but backtests replay. Listings the exchanges still list get theirs from the normal pipeline.

- Kalshi: each listing's fee keys, the series (fee group) and event ticker (fee event) its market belongs to. Market
  tickers don't reliably name their event, so both come from the API: the event from /markets, the series from the
  event's ticker when that is a known series, else from the event itself.
- Polymarket International: each market's own fee schedule (Gamma's feeSchedule), holding from when it listed.

Without --apply it only prints what would change.

  export REGISTRY_API_URL=...        # without the /api suffix
  export REGISTRY_API_KEY=...
  poetry run backfill-fees --exchange KALSHI [--apply]
  poetry run backfill-fees --exchange POLYMARKET_INTL [--apply]
"""
import logging
import os
import sys
from collections import Counter

import click
import requests.exceptions
from gnomepy.registry import RegistryClient

from classifier.adapters.kalshi import BASE_URL as KALSHI_URL
from classifier.adapters.polymarket_intl import GAMMA_API_URL
from classifier.adapters.settlement import chunked
from classifier.client.http import RateLimitedSession
from classifier.stages.fees import polymarket_intl_definition, polymarket_intl_fee_since
from classifier.utils import bulk_create_chunked

logger = logging.getLogger(__name__)

KALSHI_TICKER_BATCH = 100
GAMMA_CONDITION_BATCH = 50


def _get(session: RateLimitedSession, url: str, params) -> dict | list:
    res = session.get(url, params=params, timeout=30)
    res.raise_for_status()
    return res.json()


def _kalshi_series(session: RateLimitedSession) -> set[str]:
    series: set[str] = set()
    cursor = ""
    while True:
        data = _get(session, f"{KALSHI_URL}/series", {"limit": 1000, **({"cursor": cursor} if cursor else {})})
        series.update(s["ticker"] for s in data.get("series") or [] if s.get("ticker"))
        cursor = data.get("cursor", "")
        if not cursor:
            return series


def _kalshi_event_by_ticker(session: RateLimitedSession, tickers: list[str]) -> dict[str, str]:
    """Market ticker -> event ticker; markets past Kalshi's historical cutoff are only on /historical/markets."""
    events: dict[str, str] = {}
    for batch in chunked(tickers, KALSHI_TICKER_BATCH):
        for path in ("/markets", "/historical/markets"):
            missing = [t for t in batch if t not in events]
            if not missing:
                break
            data = _get(session, f"{KALSHI_URL}{path}", {"tickers": ",".join(missing), "limit": len(missing)})
            events.update({m["ticker"]: m["event_ticker"] for m in data.get("markets") or [] if m.get("event_ticker")})
    return events


def _kalshi_event_series(session: RateLimitedSession, event: str) -> str | None:
    for path in (f"/events/{event}", f"/historical/events/{event}"):
        try:
            return _get(session, f"{KALSHI_URL}{path}", {})["event"]["series_ticker"]
        except (requests.exceptions.HTTPError, KeyError):
            continue
    return None


def backfill_kalshi(registry: RegistryClient, exchange_id: int, apply_changes: bool) -> None:
    listings = [l for l in registry._get("/listings", {"exchangeId": exchange_id}, dict) if not l.get("fee_event")]
    tickers = sorted({l["exchange_security_id"].split(":")[0] for l in listings})
    logger.info("Kalshi: %d listings without fee keys, %d markets", len(listings), len(tickers))
    session = RateLimitedSession(min_request_interval=0.1)
    known_series = _kalshi_series(session)
    event_by_ticker = _kalshi_event_by_ticker(session, tickers)
    series_by_event: dict[str, str | None] = {}
    for event in sorted(set(event_by_ticker.values())):
        prefix = event.split("-")[0]
        series_by_event[event] = prefix if prefix in known_series else _kalshi_event_series(session, event)

    patches = []
    outcome = Counter()
    for listing in listings:
        event = event_by_ticker.get(listing["exchange_security_id"].split(":")[0])
        if event is None:
            outcome["market not found"] += 1
            continue
        outcome["patched"] += 1
        patches.append(dict(listing_id=listing["listing_id"], fee_group=series_by_event.get(event), fee_event=event))
    logger.info("Kalshi: %s", dict(outcome))
    if apply_changes:
        for _, chunk in bulk_create_chunked(patches, "listing fee keys"):
            registry.bulk_patch_listings(chunk)


def backfill_polymarket_intl(registry: RegistryClient, exchange_id: int, apply_changes: bool) -> None:
    covered = {row["listing_id"] for row in registry._get("/fees", {"exchangeId": exchange_id}, dict) if row.get("listing_id")}
    listings = [l for l in registry._get("/listings", {"exchangeId": exchange_id}, dict) if l["listing_id"] not in covered]
    by_condition: dict[str, list[dict]] = {}
    for listing in listings:
        by_condition.setdefault(listing["exchange_security_id"].split(":")[0], []).append(listing)
    logger.info("Polymarket International: %d listings without a fee schedule, %d markets", len(listings), len(by_condition))

    session = RateLimitedSession(min_request_interval=0.1)
    markets: dict[str, dict] = {}
    for batch in chunked(sorted(by_condition), GAMMA_CONDITION_BATCH):
        for closed in ("true", "false"):
            params = [("condition_ids", c) for c in batch if c not in markets] + [("closed", closed), ("limit", len(batch))]
            for market in _get(session, f"{GAMMA_API_URL}/markets", params):
                markets.setdefault(market.get("conditionId", ""), market)

    rows = []
    outcome = Counter()
    for condition, condition_listings in by_condition.items():
        market = markets.get(condition)
        definition = market and polymarket_intl_definition(market)
        if market is None:
            outcome["market not found"] += len(condition_listings)
            continue
        if definition is None:
            outcome["no fees"] += len(condition_listings)
            continue
        outcome[f"taker {definition['components'][0]['taker']}"] += len(condition_listings)
        for listing in condition_listings:
            rows.append(dict(
                listing_id=listing["listing_id"], effective_from=polymarket_intl_fee_since(market), definition=definition,
                source="BACKFILL", source_ref=f"{exchange_id}:{listing['exchange_security_id']}",
            ))
    logger.info("Polymarket International: %s", dict(outcome))
    if apply_changes:
        for _, chunk in bulk_create_chunked(rows, "listing fee schedules"):
            registry.bulk_create_fee_schedules(chunk)


@click.command()
@click.option("--exchange", "exchange_code", type=click.Choice(["KALSHI", "POLYMARKET_INTL"]), required=True)
@click.option("--apply", "apply_changes", is_flag=True, help="Write the changes to the registry")
def main(exchange_code: str, apply_changes: bool):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    registry = RegistryClient(base_url=os.environ["REGISTRY_API_URL"], api_key=os.environ["REGISTRY_API_KEY"])
    exchange = next((e for e in registry.get_exchange() if e.exchange_code == exchange_code), None)
    if exchange is None:
        sys.exit(f"{exchange_code} isn't in the registry")
    if exchange_code == "KALSHI":
        backfill_kalshi(registry, exchange.exchange_id, apply_changes)
    else:
        backfill_polymarket_intl(registry, exchange.exchange_id, apply_changes)
    if not apply_changes:
        logger.info("Dry run: nothing written. Pass --apply to write.")


if __name__ == "__main__":
    main()
