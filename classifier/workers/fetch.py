import json
import logging
import os
import signal
import time

import boto3
import redis as redis_lib

from classifier.adapters import ADAPTERS
from classifier.stages.fetch import diff_contracts, fetch_exchanges, fetch_resolved_outcomes
from classifier.stages.settle import record_settlements
from classifier.stages.stale import update_stale_tracker
from classifier.workers.base import sqs_send_batch
from classifier.workers.config import init_db, init_registry, init_runtime_config

logger = logging.getLogger(__name__)

# Redis keys
_REDIS_KNOWN_CONTRACTS_KEY = "fetch:known_contracts"
_REDIS_SENT_RESOLVED_KEY = "fetch:sent_resolved"
_REDIS_STALE_TRACKER_KEY = "fetch:stale_tracker"
_REDIS_TTL = 7 * 86400


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
        self._db = None
        self._settle_cursors: dict[str, int] = {}

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
        last_settle = -float("inf")

        # The FetchWorkerStarts metric filter in cdk/lib/stacks/classifier-stack.ts counts this message to catch crash
        # loops; keep its wording in step with the filter.
        logger.info("FetchWorker started")
        # The FetchCycleFailures metric filter counts the "... cycle failed" messages below; keep their wording.
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

            if now - last_settle >= wp.settle_interval_seconds:
                last_settle = now
                try:
                    self._run_settle(rc, registry)
                except Exception:
                    logger.exception("settle cycle failed")

            next_times = [
                last_fetch + wp.fetch_interval_seconds,
                last_resolve + wp.resolve_interval_seconds,
                last_stale + wp.stale_interval_seconds,
                last_settle + wp.settle_interval_seconds,
            ]
            sleep_secs = max(0.0, min(next_times) - time.monotonic())
            if sleep_secs > 0:
                time.sleep(sleep_secs)

        logger.info("FetchRunner stopped")

    def _run_fetch(self, rc, r, sqs, registry) -> tuple[dict[int, set[str]], set[int]]:
        logger.info("Starting fetch cycle")
        if not rc.config.feature_flags.fetch_enabled:
            logger.info("fetch_enabled=False, skipping fetch cycle")
            return self._active_events or ({}, set())

        queue_url = os.environ["CONTRACTS_QUEUE_URL"]
        known_contracts = _redis_load_known_contracts(r)
        exchange_by_code = fetch_exchanges(registry)
        min_event_volume = rc.config.thresholds.min_event_volume
        max_messages = rc.config.processing.fetch_max_sqs_messages
        debug = rc.config.feature_flags.debug

        merged_hashes: dict[str, str] = dict(known_contracts)
        active_by_exchange: dict[int, set[str]] = {}
        successful_ids: set[int] = set()
        failed: list[str] = []
        total_sent = 0

        for adapter in ADAPTERS:
            exchange = exchange_by_code.get(adapter.exchange_code)
            if not exchange:
                logger.warning("No exchange record for adapter '%s' — skipping", adapter.exchange_code)
                continue
            # Accumulate per exchange and only commit on success: a fetch that dies mid-pagination
            # must not drop the unfetched contracts' hashes or count their events as missing.
            exchange_hashes: dict[str, str] = {}
            exchange_active: set[str] = set()
            try:
                adapter_contracts_count = 0
                for page in adapter.fetch(exchange.exchange_id):
                    adapter_contracts_count += len(page)
                    remaining = max(0, max_messages - total_sent)
                    new_msgs, updated, page_active, _ = diff_contracts(
                        page, known_contracts, [], exchange_by_code,
                        min_event_volume=min_event_volume,
                        max_messages=remaining,
                        debug=debug,
                    )
                    if new_msgs:
                        sqs_send_batch(sqs, queue_url, new_msgs)
                        total_sent += len(new_msgs)
                    exchange_hashes.update(updated)
                    exchange_active.update(page_active.get(exchange.exchange_id, set()))
                logger.info("Fetched %d contracts from %s, sent %d groups", adapter_contracts_count, adapter.exchange_code, total_sent)
                prefix = f"{exchange.exchange_id}:"
                merged_hashes = {k: v for k, v in merged_hashes.items() if not k.startswith(prefix)}
                merged_hashes.update(exchange_hashes)
                active_by_exchange[exchange.exchange_id] = exchange_active
                successful_ids.add(exchange.exchange_id)
            except Exception as e:
                logger.error("Failed to fetch from %s: %s", adapter.exchange_code, e)
                failed.append(adapter.exchange_code)
                # Groups already sent this cycle keep their new hashes so they aren't resent.
                merged_hashes.update(exchange_hashes)

        if failed:
            logger.warning("Failed to fetch contracts from: %s", failed)

        _redis_save_known_contracts(r, merged_hashes)
        logger.info("Fetch cycle complete: %d new/changed groups", total_sent)
        return active_by_exchange, successful_ids

    def _run_resolve(self, rc, r, sqs, registry):
        logger.info("Starting resolve cycle")
        if not rc.config.feature_flags.resolve_enabled:
            logger.info("resolve_enabled=False, skipping resolve cycle")
            return

        queue_url = os.environ["CONTRACTS_QUEUE_URL"]
        lookback_days = rc.config.processing.resolution_lookback_days
        sent_resolved = _redis_load_sent_resolved(r)
        exchange_by_code = fetch_exchanges(registry)
        resolved_by_exchange, failed = fetch_resolved_outcomes(exchange_by_code, lookback_days)

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

    def _run_settle(self, rc, registry):
        if not rc.config.feature_flags.settlement_enabled:
            logger.info("settlement_enabled=False, skipping settle cycle")
            return
        # Connected on first use, so the worker runs without database access while settlement is off.
        if self._db is None:
            self._db = init_db()
        processing = rc.config.processing
        counts = record_settlements(
            ADAPTERS, fetch_exchanges(registry), registry, self._db,
            processing.settlement_lookback_days, processing.settle_max_candidates, self._settle_cursors,
        )
        logger.info("Settle cycle complete: %s", counts)

    def _run_stale(self, rc, r, sqs, registry):
        logger.info("Starting stale cycle")
        if not rc.config.feature_flags.stale_cleanup_enabled:
            logger.info("stale_cleanup_enabled=False, skipping stale cycle")
            return

        queue_url = os.environ["CONTRACTS_QUEUE_URL"]

        tracker = _redis_load_stale_tracker(r)

        if self._active_events is not None:
            active_by_exchange, successful_ids = self._active_events
            failed_exchange_ids = {entry["exchange_id"] for entry in tracker.values()} - successful_ids
            logger.info("stale_cleanup using in-memory active events (%d exchanges)", len(active_by_exchange))
        else:
            logger.warning("stale_cleanup: no active events in memory, falling back to fetch_all")
            exchange_by_code = fetch_exchanges(registry)
            failed_exchange_ids = set()
            active_by_exchange = {}
            for adapter in ADAPTERS:
                exchange = exchange_by_code.get(adapter.exchange_code)
                if not exchange:
                    continue
                try:
                    for page in adapter.fetch(exchange.exchange_id):
                        for contract in page:
                            active_by_exchange.setdefault(contract.exchange_id, set()).add(contract.exchange_event_native_id)
                except Exception as e:
                    logger.error("Failed to fetch from %s: %s", adapter.exchange_code, e)
                    failed_exchange_ids.add(exchange.exchange_id)

        stale_messages, new_tracker = update_stale_tracker(
            tracker, active_by_exchange, failed_exchange_ids,
            miss_threshold=rc.config.processing.stale_miss_threshold,
            max_messages=rc.config.processing.stale_max_sqs_messages,
            min_feed_ratio=rc.config.processing.stale_min_feed_ratio,
        )

        if stale_messages:
            sqs_send_batch(sqs, queue_url, stale_messages)
            logger.info("Sent %d stale events to contracts-queue", len(stale_messages))

        if failed_exchange_ids:
            logger.warning("Skipped miss-counting for failed exchange ids: %s", failed_exchange_ids)

        _redis_save_stale_tracker(r, new_tracker)
        logger.info("Stale cycle complete: %d stale events", len(stale_messages))

