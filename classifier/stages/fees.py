"""Venue fee schedules for the registry's sm.fee_schedule.

Each schedule is a definition the registry validates (endpoints/fees.ts): a fill's fee is the sum of its components,
charged at the taker or maker rate by the fill's liquidity, then rounded. A schedule targets the whole exchange, a fee
group (a Kalshi series), a fee event (a Kalshi event ticker) or one listing (a Polymarket International market), and
holds from its effective_from until the next schedule at the same target.
"""
import logging
from datetime import datetime, timezone

from classifier.utils import bulk_create_chunked

logger = logging.getLogger(__name__)

KALSHI_TAKER_RATE = 0.07
# Maker fees as a share of the taker fee, by Kalshi's fee_type (GET /series). 'flat' series follow a fee table no
# definition expresses; none are listed today.
KALSHI_MAKER_SHARE = {
    "quadratic": 0.0,
    "quadratic_with_maker_fees": 0.25,
    "quadratic_with_combo_maker_fees": 0.5,
}
# Kalshi rounded every fee up to the cent until it began rounding balances to $0.0001 and carrying the overpayment
# across an order's fills (docs.kalshi.com/getting_started/fee_rounding). The date is its changelog's, not confirmed.
KALSHI_ACCUMULATOR_FROM = datetime(2026, 3, 3, tzinfo=timezone.utc)
_KALSHI_CENT_ROUNDING = {"mode": "CEIL", "increment": 0.01, "scope": "FILL"}
_KALSHI_ACCUMULATOR_ROUNDING = {"mode": "CEIL", "increment": 0.0001, "scope": "ORDER_ACCUMULATOR"}

# Polymarket International charges fees in collateral, rounded to $0.00001 (docs.polymarket.com/trading/fees), and
# charged none before 2026-01-05: a market listed earlier that carries a schedule now can only have had it since.
POLYMARKET_INTL_FEES_FROM = datetime(2026, 1, 5, tzinfo=timezone.utc)
_POLYMARKET_INTL_ROUNDING = {"mode": "HALF_UP", "increment": 0.00001, "scope": "FILL"}


def _rate(value: float) -> float:
    # 0.07 * 0.5 is 0.035000000000000004 in binary; schedules are compared by value, so keep them exact to 1e-10.
    return round(value, 10)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def kalshi_definition(fee_type: str, multiplier: float, at: datetime) -> dict | None:
    """A Kalshi series or event fee schedule in force from `at`, or None for a fee type no definition expresses."""
    maker_share = KALSHI_MAKER_SHARE.get(fee_type)
    if maker_share is None:
        return None
    taker = _rate(KALSHI_TAKER_RATE * multiplier)
    return {
        "currency": "USD",
        "components": [{"kind": "PRICE_CURVE", "taker": taker, "maker": _rate(taker * maker_share), "exponent": 1}],
        "rounding": dict(_KALSHI_ACCUMULATOR_ROUNDING if at >= KALSHI_ACCUMULATOR_FROM else _KALSHI_CENT_ROUNDING),
    }


def _with_rounding_cutover(rows: list[dict], scope: dict, definition_at, source_ref: str) -> list[dict]:
    """Adds, for a target whose schedule from before the rounding cutover is still in force at it, the same rates
    under the new rounding from the cutover. `rows` are this target's rows in effective order."""
    before = [r for r in rows if _parse_time(r["effective_from"]) < KALSHI_ACCUMULATOR_FROM]
    if not before or any(_parse_time(r["effective_from"]) == KALSHI_ACCUMULATOR_FROM for r in rows):
        return rows
    in_force = before[-1]
    if in_force["definition"] is None:
        return rows
    definition = definition_at(in_force, KALSHI_ACCUMULATOR_FROM)
    if definition is None:
        return rows
    cutover = {**scope, "effective_from": _iso(KALSHI_ACCUMULATOR_FROM), "definition": definition,
               "source_ref": f"{in_force['source_ref']}:{source_ref}"}
    return sorted(rows + [cutover], key=lambda r: _parse_time(r["effective_from"]))


def kalshi_schedules(exchange_id: int, series_changes: list[dict], event_changes: list[dict]) -> list[dict]:
    """Fee schedules from Kalshi's recorded series changes (GET /series/fee_changes?show_historical=true) and event
    overrides (GET /events/fee_changes), each keyed by the change's id so a later sync skips what it already wrote.

    An event override replaces its series' fee type, multiplier or both; one with neither lifts the override (a null
    definition). The part an override leaves out is taken from the series as it stood when the override began.
    """
    by_series: dict[str, list[dict]] = {}
    for change in series_changes:
        by_series.setdefault(change["series_ticker"], []).append(change)
    for changes in by_series.values():
        changes.sort(key=lambda c: _parse_time(c["scheduled_ts"]))

    def series_at(series: str, at: datetime) -> tuple[str, float]:
        state = ("quadratic", 1.0)
        for c in by_series.get(series, []):
            if _parse_time(c["scheduled_ts"]) <= at:
                state = (c["fee_type"], float(c["fee_multiplier"]))
        return state

    schedules: list[dict] = []
    for series, changes in by_series.items():
        scope = {"exchange_id": exchange_id, "fee_group": series}
        rows = []
        for c in changes:
            at = _parse_time(c["scheduled_ts"])
            definition = kalshi_definition(c["fee_type"], float(c["fee_multiplier"]), at)
            if definition is None:
                logger.warning("Kalshi series %s has fee type %s from %s, which no schedule expresses; skipping",
                               series, c["fee_type"], c["scheduled_ts"])
                continue
            rows.append({**scope, "effective_from": _iso(at), "definition": definition,
                         "source_ref": f"kalshi:series_fee_change:{c['id']}", "_type": c["fee_type"],
                         "_multiplier": float(c["fee_multiplier"])})
        rows = _with_rounding_cutover(
            rows, scope, lambda r, at: kalshi_definition(r["_type"], r["_multiplier"], at), "accumulator")
        schedules.extend(rows)

    by_event: dict[str, list[dict]] = {}
    for change in event_changes:
        by_event.setdefault(change["event_ticker"], []).append(change)
    for event, changes in by_event.items():
        scope = {"exchange_id": exchange_id, "fee_event": event}
        rows = []
        for c in sorted(changes, key=lambda c: _parse_time(c["scheduled_ts"])):
            at = _parse_time(c["scheduled_ts"])
            fee_type, multiplier = c.get("fee_type_override"), c.get("fee_multiplier_override")
            row = {**scope, "effective_from": _iso(at), "source_ref": f"kalshi:event_fee_change:{c['id']}"}
            if fee_type is None and multiplier is None:
                rows.append({**row, "definition": None})
                continue
            series_type, series_multiplier = series_at(c["series_ticker"], at)
            fee_type = fee_type if fee_type is not None else series_type
            multiplier = float(multiplier) if multiplier is not None else series_multiplier
            definition = kalshi_definition(fee_type, multiplier, at)
            if definition is None:
                logger.warning("Kalshi event %s has fee type %s from %s, which no schedule expresses; skipping",
                               event, fee_type, c["scheduled_ts"])
                continue
            rows.append({**row, "definition": definition, "_type": fee_type, "_multiplier": multiplier})
        rows = _with_rounding_cutover(
            rows, scope, lambda r, at: kalshi_definition(r["_type"], r["_multiplier"], at), "accumulator")
        schedules.extend(rows)

    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in schedules]


def polymarket_intl_definition(market: dict) -> dict | None:
    """A Polymarket International market's own fee schedule (Gamma's feeSchedule), or None when it charges none."""
    schedule = market.get("feeSchedule")
    if not market.get("feesEnabled") or not isinstance(schedule, dict):
        return None
    rate = schedule.get("rate")
    if not isinstance(rate, (int, float)) or rate == 0:
        return None
    exponent = schedule.get("exponent", 1)
    return {
        "currency": "USD",
        "components": [{
            "kind": "PRICE_CURVE",
            "taker": _rate(rate),
            "maker": 0 if schedule.get("takerOnly", True) else _rate(rate),
            "exponent": exponent,
        }],
        "rounding": dict(_POLYMARKET_INTL_ROUNDING),
    }


def polymarket_intl_fee_since(market: dict) -> str:
    """When a market's schedule first held: its listing, but no earlier than Polymarket charged any fees."""
    created = market.get("createdAt")
    since = POLYMARKET_INTL_FEES_FROM
    if isinstance(created, str):
        try:
            since = max(since, _parse_time(created))
        except ValueError:
            pass
    return _iso(since)


def sync_venue_fee_schedules(adapters, exchange_by_code: dict, registry, db) -> dict:
    """Writes the schedules venues publish through their APIs (adapters with fetch_fee_schedules) that the registry
    doesn't hold yet, matched by source_ref. Schedules are append-only, so a venue's past changes are written once."""
    written = 0
    failed: list[str] = []
    for adapter in adapters:
        fetch = getattr(adapter, "fetch_fee_schedules", None)
        exchange = exchange_by_code.get(adapter.exchange_code)
        if fetch is None or exchange is None:
            continue
        try:
            schedules = fetch(exchange.exchange_id)
        except Exception as e:
            # Counted by the FetchCycleFailures metric filter (cdk/lib/stacks/classifier-stack.ts); keep the wording.
            logger.error("Failed to fetch fee schedules from %s: %s", adapter.exchange_code, e)
            failed.append(adapter.exchange_code)
            continue
        known = db.get_fee_source_refs(exchange.exchange_id)
        pending = [dict(s, source="VENUE_API") for s in schedules if s["source_ref"] not in known]
        for _, chunk in bulk_create_chunked(pending, "fee schedules"):
            written += len(registry.bulk_create_fee_schedules(chunk))
    return {"fee_schedules_written": written, "failed_exchanges": failed}
