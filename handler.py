import dataclasses
import json
import logging
import os
import urllib.parse
import urllib.request

import anthropic
import boto3
import voyageai

import classifier.constants as constants
from classifier.adapters.types import AdapterContract
from classifier.cache import RedisClassifierCache
from classifier.client import BatchAnthropicClient, BatchVoyageClient
from classifier.db import ClassifierDB
from classifier.pipeline import embed_and_update, fetch_exchanges
from classifier.stages.canonicalize import parse_canon_results, prepare_canon_batch
from classifier.stages.classify import classify_structural, prepare_semantic_batch, process_semantic_results
from classifier.stages.entities import create_entities_from_canonical, prepare_canonicalization_inputs
from classifier.stages.fetch import fetch_all, fetch_resolved_outcomes
from classifier.stages.resolve import detect_resolved_events
from classifier.types import EntityResult
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)

_slack_token: str | None = None
_clients = None


def _fetch_api_key(key_id: str) -> str:
    client = boto3.client("apigateway")
    response = client.get_api_key(apiKey=key_id, includeValue=True)
    return response["value"]


def _fetch_secret(secret_name: str) -> str:
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return response["SecretString"]


def _build_dsn() -> str:
    secret = json.loads(_fetch_secret(os.environ["DB_SECRET_NAME"]))
    password = urllib.parse.quote(secret["password"], safe="")
    db_name = secret.get("dbname", os.environ.get("DB_NAME", "gnome"))
    return f"postgresql://{secret['username']}:{password}@{secret['host']}:5432/{db_name}"


def _init_clients():
    anthropic_api_key = _fetch_secret(os.environ["ANTHROPIC_API_KEY_SECRET"])
    voyage_api_key = _fetch_secret(os.environ["VOYAGE_API_KEY_SECRET"])
    registry_api_key = _fetch_api_key(os.environ["REGISTRY_API_KEY_ID"])
    registry = RegistryClient(
        base_url=os.environ["REGISTRY_API_URL"],
        api_key=registry_api_key,
    )
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    batch_client = BatchAnthropicClient(client=anthropic_client)
    voyage_client = BatchVoyageClient(client=voyageai.Client(api_key=voyage_api_key))
    cache = RedisClassifierCache(redis_url=os.environ["REDIS_ENDPOINT"])
    db = ClassifierDB(dsn=_build_dsn())
    return registry, batch_client, voyage_client, cache, db


def _get_clients():
    global _clients
    if _clients is None:
        _clients = _init_clients()
    return _clients


def _entity_result_to_dict(entity_result: EntityResult) -> dict:
    return {
        **entity_result.counts,
        "new_security_ids": entity_result.new_security_ids,
        "new_security_symbols": entity_result.new_security_symbols,
        "has_new_entities": entity_result.has_new_entities,
    }


def _serialize_cached_results(cached_results: dict) -> dict:
    return {f"{nk[0]}:{nk[1]}": v for nk, v in cached_results.items()}


def _deserialize_cached_results(serialized: dict) -> dict:
    result = {}
    for key, v in serialized.items():
        parts = key.split(":", 1)
        result[(int(parts[0]), parts[1])] = v
    return result


def fetch_and_prepare(event, context):
    logging.basicConfig(level=logging.INFO)

    state_machine_arn = os.environ.get("STATE_MACHINE_ARN")
    if state_machine_arn:
        sfn_client = boto3.client("stepfunctions")
        running = sfn_client.list_executions(
            stateMachineArn=state_machine_arn,
            statusFilter="RUNNING",
            maxResults=2,
        )
        if len(running["executions"]) > 1:
            logger.info("Another execution is already running, skipping this invocation")
            return {"has_new_contracts": False, "contracts": []}

    registry, batch_client, voyage_client, cache, db = _get_clients()

    exchange_by_name = fetch_exchanges(registry)
    contracts, failed_adapters = fetch_all(exchange_by_name)
    if failed_adapters:
        logger.warning("Adapter fetch failures: %s", failed_adapters)

    lookback_days = int(os.environ.get("RESOLUTION_LOOKBACK_DAYS", constants.RESOLUTION_LOOKBACK_DAYS))
    resolved_by_exchange, failed_resolve = fetch_resolved_outcomes(exchange_by_name, lookback_days=lookback_days)
    if failed_resolve:
        logger.warning("Resolution fetch failures: %s", failed_resolve)
    resolution_result = detect_resolved_events(resolved_by_exchange, registry, db)
    logger.info("Resolution stage complete: %s", resolution_result)

    contracts_json = [dataclasses.asdict(c) for c in contracts]
    return {
        "has_new_contracts": len(contracts_json) > 0,
        "contracts": contracts_json,
        **resolution_result,
    }


def submit_canon_handler(event, context):
    logging.basicConfig(level=logging.INFO)
    registry, batch_client, voyage_client, cache, db = _get_clients()

    contracts = [AdapterContract(**c) for c in event["contracts"]]
    events_to_canon, entity_ctx = prepare_canonicalization_inputs(contracts, cache, db)
    api_requests, canon_context, cached_results = prepare_canon_batch(events_to_canon, cache)
    result = batch_client.submit_batch(api_requests)

    if result.is_complete:
        canonical = parse_canon_results(result.responses, canon_context, cache, batch_client._client)
        canonical.update(cached_results)
        entity_result = create_entities_from_canonical(registry, canonical, entity_ctx, contracts, cache=cache, db=db)
        entity_result = embed_and_update(voyage_client, entity_result, db)
        logger.info("Canon sync complete: %s", entity_result.counts)
        return {"batch_complete": True, **_entity_result_to_dict(entity_result)}

    return {
        "batch_complete": False,
        "batch_id": result.batch_id,
        "canon_context": canon_context,
        "cached_results": _serialize_cached_results(cached_results),
        "contracts": event["contracts"],
    }


def check_batch_handler(event, context):
    logging.basicConfig(level=logging.INFO)
    _, batch_client, _, _, _ = _get_clients()

    batch_id = event.get("batch_id")
    poll_count = event.get("poll_count", 0) + 1

    if batch_id is None:
        return {**event, "batch_complete": True, "poll_count": poll_count}

    status = batch_client.check_batch(batch_id)
    batch_complete = status == "ended" or poll_count >= 60

    logger.info("Batch %s status=%s poll=%d complete=%s", batch_id, status, poll_count, batch_complete)

    return {**event, "batch_complete": batch_complete, "poll_count": poll_count}


def collect_canon_handler(event, context):
    logging.basicConfig(level=logging.INFO)
    registry, batch_client, voyage_client, cache, db = _get_clients()

    contracts = [AdapterContract(**c) for c in event["contracts"]]
    batch_id = event["batch_id"]
    canon_context = event["canon_context"]
    cached_results = _deserialize_cached_results(event["cached_results"])

    responses = batch_client.collect_batch_results(batch_id)
    events_to_canon, entity_ctx = prepare_canonicalization_inputs(contracts, cache, db)
    canonical = parse_canon_results(responses, canon_context, cache, batch_client._client)
    canonical.update(cached_results)
    entity_result = create_entities_from_canonical(registry, canonical, entity_ctx, contracts, cache=cache, db=db)
    entity_result = embed_and_update(voyage_client, entity_result, db)
    logger.info("Canon batch complete: %s", entity_result.counts)
    return _entity_result_to_dict(entity_result)


def classify_structural_handler(event, context):
    logging.basicConfig(level=logging.INFO)
    registry, batch_client, voyage_client, cache, db = _get_clients()

    new_security_ids = event.get("new_security_ids", [])
    skip_semantic = event.get("skip_semantic", True)

    result = classify_structural(registry, new_security_ids, db=db)
    logger.info("Structural classification complete: %s", result)

    return {
        **result,
        "needs_semantic": not skip_semantic,
        "new_security_ids": new_security_ids,
    }


def submit_semantic_batch_handler(event, context):
    logging.basicConfig(level=logging.INFO)
    registry, batch_client, voyage_client, cache, db = _get_clients()

    new_security_ids = event.get("new_security_ids", [])

    api_requests, pending_context, cached_results = prepare_semantic_batch(
        new_security_ids, cache=cache, db=db,
    )
    result = batch_client.submit_batch(api_requests)

    if result.is_complete:
        responses = result.responses
        semantic_result = process_semantic_results(
            registry, responses, pending_context, cached_results,
            new_security_ids, cache=cache, db=db,
        )
        logger.info("Semantic sync complete: %s", semantic_result)
        return {"batch_complete": True, **semantic_result}

    batch_id = result.batch_id
    logger.info("Submitted semantic batch %s with %d requests", batch_id, len(api_requests))
    return {
        "batch_complete": False,
        "batch_id": batch_id,
        "pending_context": pending_context,
        "cached_results": cached_results,
        "new_security_ids": new_security_ids,
    }


def collect_semantic_results_handler(event, context):
    logging.basicConfig(level=logging.INFO)
    registry, batch_client, voyage_client, cache, db = _get_clients()

    batch_id = event.get("batch_id")
    pending_context = event.get("pending_context", [])
    cached_results = event.get("cached_results", [])
    new_security_ids = event.get("new_security_ids", [])

    responses = {}
    if batch_id is not None:
        responses = batch_client.collect_batch_results(batch_id)

    result = process_semantic_results(
        registry, responses, pending_context, cached_results, new_security_ids,
        cache=cache, db=db,
    )

    logger.info("Semantic classification complete: %s", result)
    return result


def send_notification(event, context):
    logging.basicConfig(level=logging.INFO)
    global _slack_token

    channel = os.environ.get("SLACK_CHANNEL", "")
    if not channel:
        logger.info("SLACK_CHANNEL not set, skipping notification")
        return {"notified": False}

    if _slack_token is None:
        secret_name = os.environ.get("SLACK_BOT_TOKEN_SECRET", "")
        if not secret_name:
            logger.info("SLACK_BOT_TOKEN_SECRET not set, skipping notification")
            return {"notified": False}
        _slack_token = _fetch_secret(secret_name)

    new_symbols = event.get("new_security_symbols", [])
    classification = event.get("classification", {})

    blocks: list = [
        {"type": "header", "text": {"type": "plain_text", "text": "Contract Classifier"}},
    ]

    if new_symbols:
        lines = "\n".join(f"• `{s}`" for s in new_symbols)
        count = len(new_symbols)
        noun = "security" if count == 1 else "securities"
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{count} new {noun}*\n{lines}"},
        })

    # TODO: replace with actual gnome UI URL
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "<https://gnome.example.com|View in Gnome UI>"},
    })

    parts = []
    events_created = event.get("events_created", 0)
    securities_created = event.get("securities_created", 0)
    listings_created = event.get("listings_created", 0)
    relationships_written = classification.get("Payload", {}).get("relationships_written", 0)
    if events_created:
        parts.append(f"{events_created} events")
    if securities_created:
        parts.append(f"{securities_created} securities")
    if listings_created:
        parts.append(f"{listings_created} listings")
    if relationships_written:
        parts.append(f"{relationships_written} relationships")
    events_resolved = event.get("events_resolved", 0)
    securities_deactivated = event.get("securities_deactivated", 0)
    if events_resolved:
        parts.append(f"{events_resolved} events resolved")
    if securities_deactivated:
        parts.append(f"{securities_deactivated} securities deactivated")
    if parts:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": " · ".join(parts)}],
        })

    payload = json.dumps({"channel": channel, "blocks": blocks}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=payload,
        headers={
            "Authorization": f"Bearer {_slack_token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    if not body.get("ok"):
        logger.error("Slack notification failed: %s", body.get("error"))
        return {"notified": False, "error": body.get("error")}

    logger.info("Slack notification sent")
    return {"notified": True}
