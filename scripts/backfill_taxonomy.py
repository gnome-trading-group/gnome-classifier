"""Backfills unresolved events' categories and tags from the static exchange taxonomy (`classifier/taxonomy.py`), without
the LLM. Resolved events are settled history and are left as they are.

Events created before the taxonomy carry the raw exchange label as their category ("Climate and Weather", "Brazil")
and no tags. Events the exchanges still list get the category and tags their adapter now maps. Events they no longer
list keep their tags and get only a category: Kalshi from the stored label plus its series tags, Polymarket US from the
stored label, Polymarket International from the event's full tag list (only its first tag was stored).

A category field value that no mapping table knows (a Kalshi or Polymarket US category, a Hyperliquid template) stops
the run before anything is written, so a new exchange category gets a mapping instead of a silent OTHER. Labels that
span several categories ("Mentions", AI + Politics), Polymarket International tag sets that name no category (new topic
tags appear daily), and events with no label at all keep an already standardized category or become OTHER, for the LLM
pass to settle.

Without --apply it only prints what would change.

  export REGISTRY_API_URL=...        # without the /api suffix
  export REGISTRY_API_KEY=...
  poetry run backfill-taxonomy [--apply]
"""
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

import click
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import Event

from classifier import taxonomy
from classifier.adapters.kalshi import KalshiAdapter
from classifier.adapters.polymarket_intl import GAMMA_API_URL
from classifier.adapters.settlement import chunked
from classifier.adapters.types import AdapterContract
from classifier.client.http import RateLimitedSession
from classifier.constants import STANDARDIZED_CATEGORIES
from classifier.stages.fetch import fetch_all, fetch_exchanges

logger = logging.getLogger(__name__)

GAMMA_SLUG_BATCH = 50
PATCH_LOG_EVERY = 5000


@dataclass(frozen=True)
class Resolution:
    category: str | None
    known: bool
    label: str


def _is_current(event: Event) -> bool:
    return event.category in STANDARDIZED_CATEGORIES and bool(event.tags)


def _url_segment(url: str | None, index: int) -> str | None:
    parts = [p for p in urlparse(url or "").path.split("/") if p]
    return parts[index] if len(parts) > index else None


def _kalshi(label: str | None, series_tags: list[str]) -> Resolution:
    known = label is None or label in taxonomy._KALSHI_CATEGORY
    return Resolution(taxonomy.kalshi_category(label, series_tags), known, str(label))


def _polymarket_us(label: str | None) -> Resolution:
    known = label is None or (isinstance(label, str) and label.lower() in taxonomy._POLYMARKET_US_CATEGORY)
    return Resolution(taxonomy.polymarket_us_category(label), known, str(label))


def _polymarket_intl(tag_labels: list[str]) -> Resolution:
    return Resolution(taxonomy.polymarket_intl_category(tag_labels), True, ", ".join(map(str, tag_labels)))


def _fetch_polymarket_intl_tags(slugs: set[str]) -> dict[str, list[str]]:
    session = RateLimitedSession(min_request_interval=0.1)
    tags_by_slug: dict[str, list[str]] = {}
    for batch in chunked(sorted(slugs), GAMMA_SLUG_BATCH):
        res = session.get(f"{GAMMA_API_URL}/events", params=[("slug", s) for s in batch] + [("limit", len(batch))],
                          timeout=30)
        res.raise_for_status()
        for event in res.json():
            tags_by_slug[event["slug"]] = [t.get("label") for t in event.get("tags") or [] if isinstance(t, dict)]
    return tags_by_slug


def _resolve(
    code: str,
    event: Event,
    contract: AdapterContract | None,
    kalshi_series_tags: dict[str, list[str]],
    polymarket_intl_tags: dict[str, list[str]],
) -> Resolution:
    if contract is not None and contract.mapped_category:
        return Resolution(contract.mapped_category, True, str(contract.event_category))
    label = contract.event_category if contract is not None else event.category
    if code == "KALSHI":
        series = _url_segment(event.native_url, 1) or event.native_event_id.split("-")[0]
        return _kalshi(label, kalshi_series_tags.get(series.upper(), []))
    if code == "POLYMARKET_US":
        return _polymarket_us(label)
    if code == "POLYMARKET_INTL":
        slug = _url_segment(event.native_url, 1)
        return _polymarket_intl(polymarket_intl_tags.get(slug, [event.category]))
    # Hyperliquid events were always stored with a registry category, which stands when the template is no longer listed.
    standardized = event.category in STANDARDIZED_CATEGORIES
    return Resolution(event.category if standardized else None, code == "HYPERLIQUID" and standardized, str(label))


def plan_backfill(registry: RegistryClient) -> tuple[list[dict], Counter, Counter]:
    exchange_by_code = fetch_exchanges(registry)
    code_by_id = {e.exchange_id: code for code, e in exchange_by_code.items()}
    events = [e for e in registry.get_event(resolved=False) if not _is_current(e)]
    logger.info("%d events need a backfill", len(events))

    contracts, failed = fetch_all(exchange_by_code)
    if failed:
        raise click.ClickException(f"Fetch failed for {failed}; their open events would lose their tags")
    live = {(c.exchange_id, c.exchange_event_native_id): c for c in contracts}

    kalshi_series_tags = KalshiAdapter()._fetch_series_tags()
    polymarket_intl_slugs = {
        slug for e in events
        if code_by_id.get(e.exchange_id) == "POLYMARKET_INTL"
        and ((e.exchange_id, e.native_event_id) not in live or not live[(e.exchange_id, e.native_event_id)].mapped_category)
        and (slug := _url_segment(e.native_url, 1))
    }
    polymarket_intl_tags = _fetch_polymarket_intl_tags(polymarket_intl_slugs)

    patches: list[dict] = []
    changes: Counter = Counter()
    unknown: Counter = Counter()
    for event in events:
        code = code_by_id.get(event.exchange_id, str(event.exchange_id))
        contract = live.get((event.exchange_id, event.native_event_id))
        resolution = _resolve(code, event, contract, kalshi_series_tags, polymarket_intl_tags)
        if not resolution.known:
            unknown[(code, resolution.label)] += 1
            continue
        category = resolution.category or (event.category if event.category in STANDARDIZED_CATEGORIES else "OTHER")
        patch: dict = {}
        if category != event.category:
            patch["category"] = category
        if contract is not None and contract.event_tags and not event.tags:
            patch["tags"] = contract.event_tags
        if patch:
            patches.append({"event_id": event.event_id, **patch})
            state = "open" if contract is not None else "closed"
            changes[(code, state, event.category, category, "tags" in patch)] += 1
    return patches, changes, unknown


@click.command()
@click.option("--apply", "apply_changes", is_flag=True, help="Write the changes to the registry")
def main(apply_changes: bool):
    """Backfill event categories and tags from the static exchange taxonomy."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for var in ("REGISTRY_API_URL", "REGISTRY_API_KEY"):
        if not os.environ.get(var):
            raise click.ClickException(f"{var} is required")
    registry = RegistryClient(base_url=os.environ["REGISTRY_API_URL"], api_key=os.environ["REGISTRY_API_KEY"])

    patches, changes, unknown = plan_backfill(registry)

    by_exchange: Counter = Counter()
    for (code, state, _, _, tagged), n in changes.items():
        by_exchange[(code, state, "category + tags" if tagged else "category only")] += n
    click.echo(f"{len(patches)} events to patch")
    for (code, state, kind), n in sorted(by_exchange.items()):
        click.echo(f"  {code:16} {state:6} {kind:16} {n}")
    click.echo("Category changes:")
    moves: Counter = Counter()
    for (code, _, old, new, _), n in changes.items():
        moves[(code, old, new)] += n
    for (code, old, new), n in moves.most_common(40):
        click.echo(f"  {code:16} {old!s:28} -> {new:13} {n}")

    if unknown:
        click.echo(f"\n{sum(unknown.values())} events have labels no mapping knows; add them to classifier/taxonomy.py:")
        for (code, label), n in unknown.most_common(50):
            click.echo(f"  {code:16} {label[:80]:80} {n}")
        click.echo("Nothing was written.")
        sys.exit(1)

    if not apply_changes:
        click.echo("\nDry run; pass --apply to write.")
        return
    for i, batch in enumerate(chunked(patches, PATCH_LOG_EVERY)):
        registry.bulk_patch_events(batch)
        logger.info("Patched %d / %d events", min((i + 1) * PATCH_LOG_EVERY, len(patches)), len(patches))
