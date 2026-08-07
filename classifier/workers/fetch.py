import dataclasses
import hashlib
import json
import logging
import os
import signal
import time

import boto3
import redis as redis_lib

from classifier.adapters.types import AdapterContract
from classifier.stages.fetch import fetch_all, fetch_exchanges, fetch_resolved_outcomes
from classifier.workers.base import sqs_send_batch
from classifier.workers.config import init_registry, init_runtime_config

logger = logging.getLogger(__name__)

# Redis keys
_REDIS_KNOWN_CONTRACTS_KEY = "fetch:known_contracts"
_REDIS_SENT_RESOLVED_KEY = "fetch:sent_resolved"
_REDIS_STALE_TRACKER_KEY = "fetch:stale_tracker"
_REDIS_TTL = 7 * 86400

def _contract_hash(contract: AdapterContract) -> str:
    content = "\x00".join([
        str(contract.exchange_id),
        contract.exchange_security_id,
        contract.outcome_label,
        contract.event_title,
        contract.exchange_event_native_id,
    ])
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _redis_load_known_contracts(r: redis_lib.Redis) -> dict[str, str]:
    try:
        raw = r.get(_REDIS_KNOWN_CONTRACTS_KEY)
        if raw is None:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.exception("Failed to load known_contracts from Redis, starting empty")
        return {}


def _redis_save_known_contracts(r: redis_lib.Redis, data: dict[str, str]):
    try:
        r.set(_REDIS_KNOWN_CONTRACTS_KEY, json.dumps(data), ex=_REDIS_TTL)
    except Exception:
        logger.exception("Failed to save known_contracts to Redis")


def _redis_load_sent_resolved(r: redis_lib.Redis) -> set[tuple[int, str]]:
    try:
        raw = r.get(_REDIS_SENT_RESOLVED_KEY)
        if raw is None:
            return set()
        data = json.loads(raw)
        return {(int(item[0]), str(item[1])) for item in data}
    except Exception:
        logger.exception("Failed to load sent_resolved from Redis, starting empty")
        return set()


def _redis_save_sent_resolved(r: redis_lib.Redis, data: set[tuple[int, str]]):
    try:
        r.set(_REDIS_SENT_RESOLVED_KEY, json.dumps([[eid, nid] for eid, nid in data]), ex=_REDIS_TTL)
    except Exception:
        logger.exception("Failed to save sent_resolved to Redis")


def _redis_load_stale_tracker(r: redis_lib.Redis) -> dict[str, dict]:
    try:
        raw = r.get(_REDIS_STALE_TRACKER_KEY)
        if raw is None:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.exception("Failed to load stale_tracker from Redis, starting empty")
        return {}


def _redis_save_stale_tracker(r: redis_lib.Redis, data: dict[str, dict]):
    try:
        r.set(_REDIS_STALE_TRACKER_KEY, json.dumps(data), ex=_REDIS_TTL)
    except Exception:
        logger.exception("Failed to save stale_tracker to Redis")


# ── FetchRunner ───────────────────────────────────────────────────────────────

class FetchRunner:
    def __init__(self):
        self._running = False
        self._active_events: tuple[dict[int, set[str]], set[int]] | None = None

    def _handle_shutdown(self, signum, frame):
        logger.info("FetchRunner shutting down (signal %d)", signum)
        self._running = False

    def run(self):
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self._running = True

        rc = init_runtime_config()
        r = redis_lib.Redis.from_url(os.environ["REDIS_ENDPOINT"], decode_responses=False)
        sqs = boto3.client("sqs")
        registry = init_registry()

        last_fetch = -float("inf")
        last_resolve = -float("inf")
        last_stale = -float("inf")

        logger.info("FetchRunner started")
        while self._running:
            rc.refresh()
            wp = rc.config.worker_params
            now = time.monotonic()

            if now - last_fetch >= wp.fetch_interval_seconds:
                last_fetch = now
                try:
                    self._active_events = self._run_fetch(rc, r, sqs, registry)
                except Exception:
                    logger.exception("fetch cycle failed")

            if now - last_resolve >= wp.resolve_interval_seconds:
                last_resolve = now
                try:
                    self._run_resolve(rc, r, sqs, registry)
                except Exception:
                    logger.exception("resolve cycle failed")

            if now - last_stale >= wp.stale_interval_seconds:
                last_stale = now
                try:
                    self._run_stale(rc, r, sqs, registry)
                except Exception:
                    logger.exception("stale cleanup cycle failed")

            next_times = [
                last_fetch + wp.fetch_interval_seconds,
                last_resolve + wp.resolve_interval_seconds,
                last_stale + wp.stale_interval_seconds,
            ]
            sleep_secs = max(0.0, min(next_times) - time.monotonic())
            if sleep_secs > 0:
                time.sleep(sleep_secs)

        logger.info("FetchRunner stopped")

    def _run_fetch(self, rc, r, sqs, registry) -> tuple[dict[int, set[str]], set[int]]:
        if not rc.config.feature_flags.fetch_enabled:
            logger.info("fetch_enabled=False, skipping fetch cycle")
            return self._active_events or ({}, set())

        queue_url = os.environ["CONTRACTS_QUEUE_URL"]
        known_contracts = _redis_load_known_contracts(r)
        exchange_by_name = fetch_exchanges(registry)
        active_contracts, failed = fetch_all(exchange_by_name)

        if failed:
            logger.warning("Failed to fetch contracts from: %s", failed)

        contracts_by_native: dict[tuple[int, str], list[AdapterContract]] = {}
        current_hashes: dict[str, str] = {}
        active_by_exchange: dict[int, set[str]] = {}
        for contract in active_contracts:
            nk = (contract.exchange_id, contract.exchange_event_native_id)
            contracts_by_native.setdefault(nk, []).append(contract)
            ck = f"{contract.exchange_id}:{contract.exchange_security_id}"
            current_hashes[ck] = _contract_hash(contract)
            active_by_exchange.setdefault(contract.exchange_id, set()).add(contract.exchange_event_native_id)

        successful_exchange_ids = {ex.exchange_id for name, ex in exchange_by_name.items() if name not in failed}

        min_event_volume = rc.config.thresholds.min_event_volume
        if min_event_volume is not None:
            filtered_count = 0
            for nk, group in list(contracts_by_native.items()):
                vol = group[0].event_volume
                if vol is not None and vol < min_event_volume:
                    filtered_count += 1
                    del contracts_by_native[nk]
                    for c in group:
                        current_hashes.pop(f"{c.exchange_id}:{c.exchange_security_id}", None)
            if filtered_count:
                logger.info("Filtered %d low-volume event groups (min_event_volume=%.2f)", filtered_count, min_event_volume)

        new_messages: list[dict] = []
        for nk, group in contracts_by_native.items():
            has_new_or_changed = any(
                known_contracts.get(f"{c.exchange_id}:{c.exchange_security_id}") != current_hashes[f"{c.exchange_id}:{c.exchange_security_id}"]
                for c in group
            )
            if has_new_or_changed:
                new_messages.append({"type": "new", "contracts": [dataclasses.asdict(c) for c in group]})

        max_send = rc.config.processing.fetch_max_sqs_messages
        if len(new_messages) > max_send:
            logger.info("Capping fetch from %d to %d groups", len(new_messages), max_send)
            new_messages = new_messages[:max_send]

        if new_messages:
            sqs_send_batch(sqs, queue_url, new_messages)
            logger.info("Sent %d contract groups to contracts-queue", len(new_messages))

        sent_contract_keys = set()
        for msg in new_messages:
            for c in msg["contracts"]:
                sent_contract_keys.add(f"{c['exchange_id']}:{c['exchange_security_id']}")

        final_hashes = {}
        for ck in current_hashes:
            if ck in sent_contract_keys:
                final_hashes[ck] = current_hashes[ck]
            elif ck in known_contracts:
                final_hashes[ck] = known_contracts[ck]
        _redis_save_known_contracts(r, final_hashes)
        logger.info("Fetch cycle complete: %d new/changed groups", len(new_messages))
        return active_by_exchange, successful_exchange_ids

    def _run_resolve(self, rc, r, sqs, registry):
        if not rc.config.feature_flags.resolve_enabled:
            logger.info("resolve_enabled=False, skipping resolve cycle")
            return

        queue_url = os.environ["CONTRACTS_QUEUE_URL"]
        lookback_days = rc.config.processing.resolution_lookback_days
        sent_resolved = _redis_load_sent_resolved(r)
        exchange_by_name = fetch_exchanges(registry)
        resolved_by_exchange, failed = fetch_resolved_outcomes(exchange_by_name, lookback_days)

        if failed:
            logger.warning("Failed to fetch resolved from: %s", failed)

        current_resolved: set[tuple[int, str]] = set()
        new_messages: list[dict] = []
        for exchange_id, native_ids in resolved_by_exchange.items():
            for native_id in native_ids:
                key = (exchange_id, native_id)
                current_resolved.add(key)
                if key not in sent_resolved:
                    new_messages.append({"type": "resolved", "exchange_id": exchange_id, "native_id": native_id})

        max_send = rc.config.processing.resolve_max_sqs_messages
        if len(new_messages) > max_send:
            logger.info("Capping resolve from %d to %d messages", len(new_messages), max_send)
            new_messages = new_messages[:max_send]

        if new_messages:
            sqs_send_batch(sqs, queue_url, new_messages)
            logger.info("Sent %d resolved contracts to contracts-queue", len(new_messages))

        sent_keys = {(m["exchange_id"], m["native_id"]) for m in new_messages}
        _redis_save_sent_resolved(r, (sent_resolved & current_resolved) | sent_keys)
        logger.info("Resolve cycle complete: %d newly resolved", len(new_messages))

    def _run_stale(self, rc, r, sqs, registry):
        if not rc.config.feature_flags.stale_cleanup_enabled:
            logger.info("stale_cleanup_enabled=False, skipping stale cycle")
            return

        queue_url = os.environ["CONTRACTS_QUEUE_URL"]
        miss_threshold = rc.config.processing.stale_miss_threshold

        if self._active_events is not None:
            active_by_exchange, successful_ids = self._active_events
            failed_exchange_ids: set[int] = set()
            logger.info("stale_cleanup using in-memory active events (%d exchanges)", len(active_by_exchange))
        else:
            logger.warning("stale_cleanup: no active events in memory, falling back to fetch_all")
            exchange_by_name = fetch_exchanges(registry)
            active_contracts, failed_exchanges = fetch_all(exchange_by_name)
            failed_exchange_ids = set()
            for name in failed_exchanges:
                ex = exchange_by_name.get(name)
                if ex:
                    failed_exchange_ids.add(ex.exchange_id)
            active_by_exchange = {}
            for contract in active_contracts:
                active_by_exchange.setdefault(contract.exchange_id, set()).add(contract.exchange_event_native_id)

        tracker = _redis_load_stale_tracker(r)

        new_tracker: dict[str, dict] = {}
        all_stale: list[dict] = []

        for tk, entry in tracker.items():
            exchange_id = entry["exchange_id"]
            native_event_id = entry["native_event_id"]
            miss_count = entry["miss_count"]

            if exchange_id in failed_exchange_ids:
                new_tracker[tk] = entry
                continue

            if native_event_id in active_by_exchange.get(exchange_id, set()):
                new_tracker[tk] = {"exchange_id": exchange_id, "native_event_id": native_event_id, "miss_count": 0}
            else:
                miss_count += 1
                if miss_count >= miss_threshold:
                    all_stale.append({"type": "stale", "exchange_id": exchange_id, "native_event_id": native_event_id})
                else:
                    new_tracker[tk] = {"exchange_id": exchange_id, "native_event_id": native_event_id, "miss_count": miss_count}

        for exchange_id, native_ids in active_by_exchange.items():
            for native_event_id in native_ids:
                tk = f"{exchange_id}:{native_event_id}"
                if tk not in new_tracker:
                    new_tracker[tk] = {"exchange_id": exchange_id, "native_event_id": native_event_id, "miss_count": 0}

        max_send = rc.config.processing.stale_max_sqs_messages
        stale_messages = all_stale[:max_send]
        if len(all_stale) > max_send:
            logger.info("Capping stale from %d to %d messages", len(all_stale), max_send)

        for msg in all_stale[max_send:]:
            tk = f"{msg['exchange_id']}:{msg['native_event_id']}"
            new_tracker[tk] = {"exchange_id": msg["exchange_id"], "native_event_id": msg["native_event_id"], "miss_count": miss_threshold}

        if stale_messages:
            sqs_send_batch(sqs, queue_url, stale_messages)
            logger.info("Sent %d stale events to contracts-queue", len(stale_messages))

        if failed_exchange_ids:
            logger.warning("Skipped miss-counting for failed exchange ids: %s", failed_exchange_ids)

        _redis_save_stale_tracker(r, new_tracker)
        logger.info("Stale cycle complete: %d stale events", len(stale_messages))

