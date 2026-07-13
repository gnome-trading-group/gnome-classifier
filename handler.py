import json
import logging
import os
import urllib.parse
import urllib.request

import anthropic
import boto3
import voyageai

from classifier.cache import RedisClassifierCache
from classifier.client import BatchAnthropicClient, ModelRateLimit
from classifier.constants import DEFAULT_RATE_LIMITS
from classifier.db import ClassifierDB
from classifier.stages.classify import classify_relationships
from classifier.stages.entities import create_entities
from classifier.stages.fetch import fetch_all
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
    batch_client = BatchAnthropicClient(
        client=anthropic_client,
        rate_limits={k: ModelRateLimit(**v) for k, v in DEFAULT_RATE_LIMITS.items()},
    )
    voyage_client = voyageai.Client(api_key=voyage_api_key)
    cache = RedisClassifierCache(redis_url=os.environ["REDIS_ENDPOINT"])
    db = ClassifierDB(dsn=_build_dsn())
    return registry, batch_client, voyage_client, cache, db


def _get_clients():
    global _clients
    if _clients is None:
        _clients = _init_clients()
    return _clients


def fetch_and_create_entities(event, context):
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
            return {
                "events_created": 0,
                "securities_created": 0,
                "listings_created": 0,
                "event_contracts_created": 0,
                "listing_specs_created": 0,
                "new_security_ids": [],
                "new_security_symbols": [],
                "has_new_entities": False,
            }

    registry, batch_client, voyage_client, cache, db = _get_clients()

    exchanges = registry.get_exchange()
    exchange_by_name = {e.exchange_name.lower(): e for e in exchanges}
    contracts, failed_adapters = fetch_all(exchange_by_name)
    if failed_adapters:
        logger.warning("Adapter fetch failures: %s", failed_adapters)

    entity_result = create_entities(
        registry, batch_client, contracts, cache=cache, db=db,
    )

    new_security_ids = entity_result.pop("new_security_ids")
    new_security_symbols = entity_result.pop("new_security_symbols")

    logger.info("Entity stage complete: %s", entity_result)
    return {
        **entity_result,
        "new_security_ids": new_security_ids,
        "new_security_symbols": new_security_symbols,
        "has_new_entities": len(new_security_ids) > 0,
    }


def classify_relationships_handler(event, context):
    logging.basicConfig(level=logging.INFO)
    registry, batch_client, voyage_client, cache, db = _get_clients()

    new_security_ids = event.get("new_security_ids", [])

    result = classify_relationships(
        registry, batch_client, voyage_client,
        new_security_ids=new_security_ids,
        cache=cache,
        db=db,
    )

    logger.info("Classification complete: %s", result)
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
