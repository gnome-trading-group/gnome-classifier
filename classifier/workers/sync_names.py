"""Renames outcome labels, security symbols and listing names to what the venues list now, for markets renamed before
the entities stage kept names in step: a renamed market only reaches that stage again when something else about it
changes. Runs every open market through the stage's own rename rules (see `find_renames`).

Run it where the workers run (it reads the database), as a one-off task on the Fetch task definition with the command
`sync-names` (dry run) or `sync-names --apply`.
"""
import logging
from collections import Counter

from classifier.adapters.types import AdapterContract
from classifier.db import ClassifierDB
from classifier.stages.entities import Renames, apply_renames, find_renames
from classifier.stages.fetch import fetch_all, fetch_exchanges
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000
SAMPLES = 10


def sync_names(registry: RegistryClient, db: ClassifierDB, exchange_code: str | None, apply: bool) -> None:
    contracts, failed = fetch_all(fetch_exchanges(registry, exchange_code))
    if failed:
        logger.warning("Not fetched, left as they are: %s", failed)
    by_exchange: dict[int, list[AdapterContract]] = {}
    for c in {(c.exchange_id, c.exchange_security_id): c for c in contracts}.values():
        by_exchange.setdefault(c.exchange_id, []).append(c)
    for exchange_id, markets in sorted(by_exchange.items()):
        renames = sync_exchange_names(registry, db, markets, apply)
        logger.info("exchange %d: %d markets; %s %d outcome labels, %d securities, %d listings", exchange_id,
                    len(markets), "renamed" if apply else "would rename", len(renames.labels),
                    len(renames.securities), len(renames.listings))
        for kind, found in (("label", renames.labels), ("symbol", renames.securities),
                            ("listing", renames.listings)):
            for r in found[:SAMPLES]:
                logger.info("  %s %r -> %s", kind, r.was, r.patch)
    if not apply:
        logger.info("Dry run: nothing written. Re-run with --apply to patch.")


def sync_exchange_names(registry: RegistryClient, db: ClassifierDB, markets: list[AdapterContract],
                        apply: bool) -> Renames:
    markets = sorted(markets, key=lambda c: (c.exchange_event_native_id, c.exchange_security_id))
    total = Renames([], [], [])
    for start in range(0, len(markets), BATCH_SIZE):
        batch = markets[start:start + BATCH_SIZE]
        event_id_by_native = db.get_exchange_events(sorted({(c.exchange_id, c.exchange_event_native_id)
                                                             for c in batch}))
        listings = db.get_existing_listings([(c.exchange_id, c.exchange_security_id) for c in batch])
        renames = find_renames(db, batch, event_id_by_native, {k: sid for k, (_, sid) in listings.items()},
                               {k: lid for k, (lid, _) in listings.items()})
        if apply:
            apply_renames(registry, renames)
        total.labels += renames.labels
        total.securities += renames.securities
        total.listings += renames.listings
    return total
