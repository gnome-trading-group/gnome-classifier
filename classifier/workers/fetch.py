import dataclasses
import hashlib
import json
import logging
import os

import boto3

from classifier.adapters.types import AdapterContract
from classifier.stages.fetch import fetch_all, fetch_exchanges, fetch_resolved_outcomes
from classifier.utils import setup_logging
from classifier.workers.base import sqs_send_batch
from classifier.workers.config import init_registry, init_runtime_config

logger = logging.getLogger(__name__)

_KNOWN_CONTRACTS_KEY = "fetch-cache/known_contracts.json"
_SENT_RESOLVED_KEY = "fetch-cache/sent_resolved.json"
_STALE_TRACKER_KEY = "fetch-cache/stale_tracker.json"

_runtime_config = None


def _get_runtime_config():
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = init_runtime_config()
    return _runtime_config


def _contract_hash(contract: AdapterContract) -> str:
    content = "\x00".join([
        str(contract.exchange_id),
        contract.exchange_security_id,
        contract.outcome_label,
        contract.event_title,
        contract.exchange_event_native_id,
    ])
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_known_contracts(s3, bucket: str, key: str) -> dict[str, str]:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read())
        return data if isinstance(data, dict) else {}
    except s3.exceptions.NoSuchKey:
        logger.info("No S3 cache found at %s/%s, cold start", bucket, key)
        return {}
    except Exception:
        logger.exception("Failed to load S3 cache at %s/%s, starting empty", bucket, key)
        return {}


def _save_known_contracts(s3, bucket: str, key: str, data: dict[str, str]):
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data).encode(),
            ContentType="application/json",
        )
    except Exception:
        logger.exception("Failed to save S3 cache at %s/%s", bucket, key)


def _load_s3_set(s3, bucket: str, key: str) -> set[tuple[int, str]]:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read())
        return {(int(item[0]), str(item[1])) for item in data}
    except s3.exceptions.NoSuchKey:
        logger.info("No S3 cache found at %s/%s, cold start", bucket, key)
        return set()
    except Exception:
        logger.exception("Failed to load S3 cache at %s/%s, starting empty", bucket, key)
        return set()


def _save_s3_set(s3, bucket: str, key: str, data: set[tuple[int, str]]):
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps([[eid, nid] for eid, nid in data]).encode(),
            ContentType="application/json",
        )
    except Exception:
        logger.exception("Failed to save S3 cache at %s/%s", bucket, key)


def _load_stale_tracker(s3, bucket: str, key: str) -> dict[str, dict]:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read())
        return data if isinstance(data, dict) else {}
    except s3.exceptions.NoSuchKey:
        logger.info("No S3 stale tracker at %s/%s, cold start", bucket, key)
        return {}
    except Exception:
        logger.exception("Failed to load stale tracker at %s/%s, starting empty", bucket, key)
        return {}


def _save_stale_tracker(s3, bucket: str, key: str, data: dict[str, dict]):
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data).encode(),
            ContentType="application/json",
        )
    except Exception:
        logger.exception("Failed to save stale tracker at %s/%s", bucket, key)


def fetch_handler(event, context):
    setup_logging()
    rc = _get_runtime_config()
    rc.refresh()

    if not rc.config.feature_flags.fetch_enabled:
        logger.info("fetch_enabled=False, skipping fetch cycle")
        return {"new_contracts": 0}

    registry = init_registry()
    sqs = boto3.client("sqs")
    s3 = boto3.client("s3")

    bucket = os.environ["CACHE_BUCKET"]
    queue_url = os.environ["CONTRACTS_QUEUE_URL"]

    known_contracts = _load_known_contracts(s3, bucket, _KNOWN_CONTRACTS_KEY)
    exchange_by_name = fetch_exchanges(registry)
    active_contracts, failed = fetch_all(exchange_by_name)

    if failed:
        logger.warning("Failed to fetch contracts from: %s", failed)

    contracts_by_native: dict[tuple[int, str], list[AdapterContract]] = {}
    current_hashes: dict[str, str] = {}
    for contract in active_contracts:
        nk = (contract.exchange_id, contract.exchange_event_native_id)
        contracts_by_native.setdefault(nk, []).append(contract)
        ck = f"{contract.exchange_id}:{contract.exchange_security_id}"
        current_hashes[ck] = _contract_hash(contract)

    new_messages: list[dict] = []
    for nk, group in contracts_by_native.items():
        has_new_or_changed = any(
            known_contracts.get(f"{c.exchange_id}:{c.exchange_security_id}") != current_hashes[f"{c.exchange_id}:{c.exchange_security_id}"]
            for c in group
        )
        if has_new_or_changed:
            new_messages.append({
                "type": "new",
                "contracts": [dataclasses.asdict(c) for c in group],
            })

    if new_messages:
        sqs_send_batch(sqs, queue_url, new_messages)
        logger.info("Sent %d contract groups to contracts-queue", len(new_messages))

    _save_known_contracts(s3, bucket, _KNOWN_CONTRACTS_KEY, current_hashes)
    return {"new_contracts": len(new_messages)}


def resolve_handler(event, context):
    setup_logging()
    rc = _get_runtime_config()
    rc.refresh()

    registry = init_registry()
    sqs = boto3.client("sqs")
    s3 = boto3.client("s3")

    bucket = os.environ["CACHE_BUCKET"]
    queue_url = os.environ["CONTRACTS_QUEUE_URL"]

    lookback_days = rc.config.processing.resolution_lookback_days

    sent_resolved = _load_s3_set(s3, bucket, _SENT_RESOLVED_KEY)
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

    if new_messages:
        sqs_send_batch(sqs, queue_url, new_messages)
        logger.info("Sent %d resolved contracts to contracts-queue", len(new_messages))

    # Replace cache entirely — entries naturally expire when they leave the lookback window
    _save_s3_set(s3, bucket, _SENT_RESOLVED_KEY, current_resolved)
    return {"resolved_contracts": len(new_messages)}


def stale_cleanup_handler(event, context):
    setup_logging()
    rc = _get_runtime_config()
    rc.refresh()

    if not rc.config.feature_flags.stale_cleanup_enabled:
        logger.info("stale_cleanup_enabled=False, skipping stale cleanup")
        return {"stale_events": 0}

    miss_threshold = rc.config.processing.stale_miss_threshold

    registry = init_registry()
    sqs = boto3.client("sqs")
    s3 = boto3.client("s3")

    bucket = os.environ["CACHE_BUCKET"]
    queue_url = os.environ["CONTRACTS_QUEUE_URL"]

    exchange_by_name = fetch_exchanges(registry)
    active_contracts, failed_exchanges = fetch_all(exchange_by_name)

    failed_exchange_ids: set[int] = set()
    for name in failed_exchanges:
        ex = exchange_by_name.get(name)
        if ex:
            failed_exchange_ids.add(ex.exchange_id)

    active_by_exchange: dict[int, set[str]] = {}
    for contract in active_contracts:
        active_by_exchange.setdefault(contract.exchange_id, set()).add(
            contract.exchange_event_native_id
        )

    tracker = _load_stale_tracker(s3, bucket, _STALE_TRACKER_KEY)

    new_tracker: dict[str, dict] = {}
    stale_messages: list[dict] = []

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
                stale_messages.append({"type": "stale", "exchange_id": exchange_id, "native_event_id": native_event_id})
            else:
                new_tracker[tk] = {"exchange_id": exchange_id, "native_event_id": native_event_id, "miss_count": miss_count}

    for exchange_id, native_ids in active_by_exchange.items():
        for native_event_id in native_ids:
            tk = f"{exchange_id}:{native_event_id}"
            if tk not in new_tracker:
                new_tracker[tk] = {"exchange_id": exchange_id, "native_event_id": native_event_id, "miss_count": 0}

    if stale_messages:
        sqs_send_batch(sqs, queue_url, stale_messages)
        logger.info("Sent %d stale events to contracts-queue", len(stale_messages))

    if failed_exchanges:
        logger.warning("Skipped miss-counting for failed exchanges: %s", failed_exchanges)

    _save_stale_tracker(s3, bucket, _STALE_TRACKER_KEY, new_tracker)
    return {"stale_events": len(stale_messages)}
