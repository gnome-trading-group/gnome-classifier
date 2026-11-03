"""Gives each market its own security where one security holds several open markets (see `split_merged_securities`).
Current code can't make more of these, so this runs once over what is left.

Run it where the workers run (it reads the database), as a one-off task on the Fetch task definition with the command
`split-securities` (dry run) or `split-securities --apply`.
"""
import logging

from classifier.db import ClassifierDB
from classifier.stages.entities import split_merged_securities
from classifier.stages.fetch import fetch_all, fetch_exchanges
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000


def split_securities(registry: RegistryClient, db: ClassifierDB, exchange_code: str | None, apply: bool) -> None:
    contracts, failed = fetch_all(fetch_exchanges(registry, exchange_code))
    if failed:
        logger.warning("Not fetched, left as they are: %s", failed)
    # A security's markets can be anywhere in the fetch, so find them all before splitting any.
    candidates = []
    for start in range(0, len(contracts), BATCH_SIZE):
        batch = contracts[start:start + BATCH_SIZE]
        listings = db.get_existing_listings([(c.exchange_id, c.exchange_security_id) for c in batch])
        merged = db.securities_with_several_open_listings(sorted({sid for _, sid in listings.values()}))
        candidates += [c for c in batch if listings.get((c.exchange_id, c.exchange_security_id), (0, 0))[1] in merged]
    splits = split_merged_securities(registry, db, candidates, apply)
    for split in splits:
        logger.info("%s listing %d off security %d: new security %s, outcome %r", "moved" if apply else "would move",
                    split.listing_id, split.security_id, split.symbol, split.outcome_label)
    logger.info("%s %d listings off %d securities", "Moved" if apply else "Would move", len(splits),
                len({split.security_id for split in splits}))
    if not apply:
        logger.info("Dry run: nothing written. Re-run with --apply to split.")
