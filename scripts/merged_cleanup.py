"""
One-off cleanup of events and securities corrupted by the old title-based identity model.

The old classifier merged distinct native events into one sm.event (title + expiry within 1h)
and reused securities whose title-derived symbols collided. This deletes those rows so the
current classifier recreates any still-live market with correct identity.

Targets:
  - merged events: an sm.event with more than one sm.exchange_event row
  - shared securities: a security on more than one event, or listed on more than one exchange
  - every security of a merged event

  plan   Read-only, runs anywhere with registry API + AWS access. Finds the targets through the
         registry API and excludes any security with recorded market data, coverage or a
         collector (a merged event is kept whole if any of its securities is excluded).
         Prints the exact `apply` invocation.

  apply  Runs where the registry database and Redis are reachable, e.g. as a one-off ECS task on
         the normalize task definition (`worker cleanup ...`). Recomputes the same targets in SQL,
         refuses to run if the counts differ from the plan's, deletes in one transaction, then
         clears the deleted events' exchange-event cache entries and the deleted listings' fetch
         hashes so fetch resends any still-live market. Rolls back unless --commit is passed.

  purge-exchange
         Same delete path for every event of one exchange, so fetch recreates them with the
         current adapter. Used for Hyperliquid after its adapter switched to rendering outcome
         templates and to Hyperliquid's own coin names (`#<outcome><side>`) as listing ids.
         Run only after the new adapter is deployed.

Usage:
  AWS_PROFILE=dev STAGE=dev poetry run python -m scripts.merged_cleanup plan --stage dev
  worker cleanup apply --exclude ... --expect-events N --expect-securities M [--commit]
  worker cleanup purge-exchange --exchange-code HYPERLIQUID --exclude ... --expect-events N --expect-securities M [--commit]
"""
import dataclasses
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
import click
import psycopg2
import redis as redis_lib

from classifier.workers.config import build_dsn
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)

_MARKET_DATA_BUCKETS = (
    "gnome-market-data-raw-{stage}",
    "gnome-market-data-merged-{stage}",
    "gnome-market-data-{stage}",
    "gnome-market-data-exchange-raw-{stage}",
)
_COVERAGE_TABLE = "market-data-coverage"
_COLLECTORS_TABLE = "market-data-collectors"
_REDIS_EXCHANGE_EVENTS_KEY = "exchange_events"
_REDIS_KNOWN_CONTRACTS_KEY = "fetch:known_contracts"
_REDIS_KNOWN_CONTRACTS_TTL = 7 * 86400


@dataclasses.dataclass
class _ExchangeEventRow:
    # gnomepy no longer models sm.exchange_event, but the table and its endpoint exist until
    # this cleanup lets the follow-up migration drop them.
    event_id: int
    exchange_id: int
    native_event_id: str


def _select_targets(
    natives_by_event: dict[int, set[tuple[int, str]]],
    events_by_security: dict[int, set[int]],
    exchanges_by_security: dict[int, set[int]],
    excluded_security_ids: set[int],
) -> tuple[set[int], set[int]]:
    """Returns (event_ids, security_ids) to delete. Shared by plan and apply so both agree."""
    securities_by_event: dict[int, set[int]] = defaultdict(set)
    for sid, eids in events_by_security.items():
        for eid in eids:
            securities_by_event[eid].add(sid)

    merged_event_ids = {eid for eid, natives in natives_by_event.items() if len(natives) > 1}
    shared_security_ids = {sid for sid, eids in events_by_security.items() if len(eids) > 1}
    shared_security_ids |= {sid for sid, exs in exchanges_by_security.items() if len(exs) > 1}

    event_ids = {eid for eid in merged_event_ids if not securities_by_event[eid] & excluded_security_ids}
    security_ids = shared_security_ids | {sid for eid in event_ids for sid in securities_by_event[eid]}
    return event_ids, security_ids - excluded_security_ids


def _has_market_data(s3, dynamodb, stage: str, security_id: int) -> str | None:
    for template in _MARKET_DATA_BUCKETS:
        bucket = template.format(stage=stage)
        if s3.list_objects_v2(Bucket=bucket, Prefix=f"{security_id}/", MaxKeys=1).get("KeyCount", 0):
            return f"s3://{bucket}/{security_id}/"
    resp = dynamodb.query(
        TableName=_COVERAGE_TABLE,
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": {"S": f"SEC#{security_id}"}},
        Limit=1,
    )
    return f"coverage SEC#{security_id}" if resp.get("Count", 0) else None


def _has_collector(dynamodb, listing_id: int) -> bool:
    resp = dynamodb.get_item(TableName=_COLLECTORS_TABLE, Key={"listingId": {"N": str(listing_id)}})
    return "Item" in resp


@click.group()
def main():
    """Clean up events and securities merged by the old title-based identity model."""


@main.command()
@click.option("--stage", required=True, type=click.Choice(["dev", "prod"]))
@click.option("-o", "--output", default="merged_cleanup_plan.json", show_default=True)
def plan(stage: str, output: str):
    """Find targets through the registry API, guard against market data, write the plan."""
    registry = RegistryClient()
    s3 = boto3.client("s3")
    dynamodb = boto3.client("dynamodb")

    click.echo("Loading exchange events, event contracts and listings from the registry...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        ee_f = executor.submit(registry._get, "/exchange-events", {}, _ExchangeEventRow)
        ec_f = executor.submit(registry.get_event_contracts)
        listing_f = executor.submit(registry.get_listing)
        exchange_events, event_contracts, listings = ee_f.result(), ec_f.result(), listing_f.result()

    natives_by_event: dict[int, set[tuple[int, str]]] = defaultdict(set)
    for ee in exchange_events:
        natives_by_event[ee.event_id].add((ee.exchange_id, ee.native_event_id))
    events_by_security: dict[int, set[int]] = defaultdict(set)
    for ec in event_contracts:
        events_by_security[ec.security_id].add(ec.event_id)
    listings_by_security: dict[int, list] = defaultdict(list)
    exchanges_by_security: dict[int, set[int]] = defaultdict(set)
    for listing in listings:
        if listing.security_id in events_by_security:
            listings_by_security[listing.security_id].append(listing)
            exchanges_by_security[listing.security_id].add(listing.exchange_id)

    _, candidates = _select_targets(natives_by_event, events_by_security, exchanges_by_security, set())
    click.echo(f"Checking market data, coverage and collectors for {len(candidates)} candidate securities...")
    candidate_listings = [l for sid in candidates for l in listings_by_security[sid]]
    with ThreadPoolExecutor(max_workers=16) as executor:
        data_hits = dict(zip(candidates, executor.map(lambda sid: _has_market_data(s3, dynamodb, stage, sid), candidates)))
        collector_hits = dict(zip(
            (l.listing_id for l in candidate_listings),
            executor.map(lambda l: _has_collector(dynamodb, l.listing_id), candidate_listings),
        ))
    excluded: dict[int, str] = {sid: reason for sid, reason in data_hits.items() if reason}
    for l in candidate_listings:
        if collector_hits[l.listing_id]:
            excluded.setdefault(l.security_id, f"collector on listing {l.listing_id}")

    event_ids, security_ids = _select_targets(
        natives_by_event, events_by_security, exchanges_by_security, set(excluded),
    )
    listing_count = sum(len(listings_by_security[sid]) for sid in security_ids)
    result = {
        "stage": stage,
        "event_ids": sorted(event_ids),
        "security_ids": sorted(security_ids),
        "listing_count": listing_count,
        "excluded_securities": {str(sid): reason for sid, reason in sorted(excluded.items())},
    }
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    exclude_arg = ",".join(str(sid) for sid in sorted(excluded)) or "none"
    click.echo(
        f"\nPlan: delete {len(event_ids)} events, {len(security_ids)} securities, {listing_count} listings; "
        f"excluded {len(excluded)} securities with data or collectors. Wrote {output}"
    )
    for sid, reason in list(sorted(excluded.items()))[:20]:
        click.echo(f"  excluded {sid}: {reason}")
    click.echo(
        f"\nApply with:\n  worker cleanup apply --exclude {exclude_arg} "
        f"--expect-events {len(event_ids)} --expect-securities {len(security_ids)} [--commit]"
    )


def _load_graph_sql(cur) -> tuple[dict, dict, dict]:
    """Load only the rows _select_targets can act on, so the full graph never sits in memory.

    Events with a single native event and securities on a single event and exchange can never
    be targets, so they are filtered out in SQL with GROUP BY ... HAVING.
    """
    natives_by_event: dict[int, set[tuple[int, str]]] = defaultdict(set)
    cur.execute(
        "SELECT event_id, exchange_id, native_event_id FROM sm.exchange_event"
        " WHERE event_id IN (SELECT event_id FROM sm.exchange_event GROUP BY event_id HAVING count(*) > 1)"
    )
    for eid, xid, nid in cur.fetchall():
        natives_by_event[eid].add((xid, nid))

    events_by_security: dict[int, set[int]] = defaultdict(set)
    cur.execute(
        "SELECT security_id, event_id FROM sm.event_contract"
        " WHERE security_id IN (SELECT security_id FROM sm.event_contract"
        "                       GROUP BY security_id HAVING count(DISTINCT event_id) > 1)"
        " OR event_id = ANY(%s)",
        (list(natives_by_event),),
    )
    for sid, eid in cur.fetchall():
        events_by_security[sid].add(eid)

    exchanges_by_security: dict[int, set[int]] = defaultdict(set)
    cur.execute(
        "SELECT DISTINCT l.security_id, l.exchange_id FROM sm.listing l"
        " WHERE l.security_id IN (SELECT security_id FROM sm.listing"
        "                         GROUP BY security_id HAVING count(DISTINCT exchange_id) > 1)"
        " AND EXISTS (SELECT 1 FROM sm.event_contract ec WHERE ec.security_id = l.security_id)"
    )
    for sid, xid in cur.fetchall():
        exchanges_by_security[sid].add(xid)
    return natives_by_event, events_by_security, exchanges_by_security


def _parse_exclude(exclude: str) -> set[int]:
    return set() if exclude == "none" else {int(x) for x in exclude.split(",")}


def _check_counts(event_ids: set[int], security_ids: set[int], expect_events: int, expect_securities: int) -> None:
    if (len(event_ids), len(security_ids)) != (expect_events, expect_securities):
        raise click.ClickException(
            f"Target counts changed since the plan: {len(event_ids)} events / {len(security_ids)} securities, "
            f"expected {expect_events} / {expect_securities}. Re-run plan."
        )


def _delete_targets(conn, event_ids: set[int], security_ids: set[int], commit: bool) -> None:
    """Delete the given events and securities (with their listings) in one transaction, then
    clear their Redis exchange-event cache entries and fetch hashes so fetch resends them."""
    events, securities = sorted(event_ids), sorted(security_ids)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT listing_id, exchange_id, exchange_security_id FROM sm.listing WHERE security_id = ANY(%s)",
            (securities,),
        )
        listing_rows = cur.fetchall()
        listings = [row[0] for row in listing_rows]
        listing_keys = {f"{row[1]}:{row[2]}" for row in listing_rows}
        cur.execute(
            "SELECT exchange_id, native_event_id FROM sm.exchange_event WHERE event_id = ANY(%(e)s)"
            " UNION SELECT exchange_id, native_event_id FROM sm.event"
            " WHERE event_id = ANY(%(e)s) AND native_event_id IS NOT NULL",
            {"e": events},
        )
        native_keys = [f"{row[0]}:{row[1]}" for row in cur.fetchall()]

        cur.execute("SELECT count(*) FROM pnl.snapshot WHERE listing_id = ANY(%s)", (listings,))
        pnl_refs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM risk.policy WHERE listing_id = ANY(%s)", (listings,))
        risk_refs = cur.fetchone()[0]
        if pnl_refs or risk_refs:
            raise click.ClickException(
                f"Refusing: {pnl_refs} pnl snapshots and {risk_refs} risk policies reference target listings"
            )

        params = {"s": securities, "e": events, "l": listings}
        for table, sql in (
            ("contract_relationship",
             "DELETE FROM sm.contract_relationship WHERE security_id_a = ANY(%(s)s) OR security_id_b = ANY(%(s)s)"),
            ("event_contract", "DELETE FROM sm.event_contract WHERE security_id = ANY(%(s)s) OR event_id = ANY(%(e)s)"),
            ("listing_spec", "DELETE FROM sm.listing_spec WHERE listing_id = ANY(%(l)s)"),
            ("listing", "DELETE FROM sm.listing WHERE listing_id = ANY(%(l)s)"),
            ("security", "DELETE FROM sm.security WHERE security_id = ANY(%(s)s)"),
            ("exchange_event", "DELETE FROM sm.exchange_event WHERE event_id = ANY(%(e)s)"),
            ("event", "DELETE FROM sm.event WHERE event_id = ANY(%(e)s)"),
        ):
            cur.execute(sql, params)
            click.echo(f"  {table:<22} {cur.rowcount} rows deleted")

    if not commit:
        conn.rollback()
        click.echo("Rolled back (dry run). Re-run with --commit to apply.")
        return
    conn.commit()
    click.echo("Committed.")

    r = redis_lib.Redis.from_url(os.environ.get("REDIS_URL") or os.environ["REDIS_ENDPOINT"], decode_responses=False)
    if native_keys:
        click.echo(f"Cleared {r.hdel(_REDIS_EXCHANGE_EVENTS_KEY, *native_keys)} exchange-event cache entries")
    raw = r.get(_REDIS_KNOWN_CONTRACTS_KEY)
    if raw is not None:
        known = json.loads(raw)
        pruned = {k: v for k, v in known.items() if k not in listing_keys}
        r.set(_REDIS_KNOWN_CONTRACTS_KEY, json.dumps(pruned), ex=_REDIS_KNOWN_CONTRACTS_TTL)
        click.echo(f"Pruned {len(known) - len(pruned)} fetch hashes; fetch will resend those contracts")


@main.command()
@click.option("--exclude", required=True, help="Comma-separated security ids from the plan, or 'none'")
@click.option("--expect-events", required=True, type=int)
@click.option("--expect-securities", required=True, type=int)
@click.option("--commit", is_flag=True, help="Commit the deletes and clear Redis; otherwise roll back")
def apply(exclude: str, expect_events: int, expect_securities: int, commit: bool):
    """Recompute merged/shared targets in SQL and delete them."""
    conn = psycopg2.connect(os.environ.get("DATABASE_URL") or build_dsn())
    try:
        with conn.cursor() as cur:
            event_ids, security_ids = _select_targets(*_load_graph_sql(cur), _parse_exclude(exclude))
        _check_counts(event_ids, security_ids, expect_events, expect_securities)
        _delete_targets(conn, event_ids, security_ids, commit)
    finally:
        conn.close()


@main.command("purge-exchange")
@click.option("--exchange-code", required=True, help="e.g. HYPERLIQUID")
@click.option("--exclude", required=True, help="Comma-separated security ids with market data, or 'none'")
@click.option("--expect-events", required=True, type=int)
@click.option("--expect-securities", required=True, type=int)
@click.option("--commit", is_flag=True, help="Commit the deletes and clear Redis; otherwise roll back")
def purge_exchange(exchange_code: str, exclude: str, expect_events: int, expect_securities: int, commit: bool):
    """Delete every event of one exchange so fetch recreates them with the current adapter."""
    excluded = _parse_exclude(exclude)
    conn = psycopg2.connect(os.environ.get("DATABASE_URL") or build_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT e.event_id, ec.security_id FROM sm.event e"
                " JOIN sm.exchange x ON x.exchange_id = e.exchange_id"
                " LEFT JOIN sm.event_contract ec ON ec.event_id = e.event_id"
                " WHERE x.exchange_code = %s",
                (exchange_code,),
            )
            securities_by_event: dict[int, set[int]] = defaultdict(set)
            for eid, sid in cur.fetchall():
                event_securities = securities_by_event[eid]
                if sid is not None:
                    event_securities.add(sid)
            event_ids = {eid for eid, sids in securities_by_event.items() if not sids & excluded}
            security_ids = {sid for eid in event_ids for sid in securities_by_event[eid]}
            cur.execute(
                "SELECT DISTINCT security_id FROM sm.event_contract"
                " WHERE security_id = ANY(%s) AND NOT event_id = ANY(%s)",
                (sorted(security_ids), sorted(event_ids)),
            )
            shared = [row[0] for row in cur.fetchall()]
            if shared:
                raise click.ClickException(f"Refusing: {len(shared)} target securities also belong to other events")
        _check_counts(event_ids, security_ids, expect_events, expect_securities)
        _delete_targets(conn, event_ids, security_ids, commit)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
