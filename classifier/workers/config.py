import dataclasses
import json
import logging
import os
import urllib.parse

import anthropic
import boto3
import voyageai

from classifier.cache import RedisClassifierCache
from classifier.client import BatchAnthropicClient, BatchVoyageClient

from classifier.db import ClassifierDB
from classifier.runtime_config import RuntimeConfig
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)


def fetch_secret(secret_name: str) -> str:
    client = boto3.client("secretsmanager")
    return client.get_secret_value(SecretId=secret_name)["SecretString"]


def fetch_api_key(key_id: str) -> str:
    client = boto3.client("apigateway")
    return client.get_api_key(apiKey=key_id, includeValue=True)["value"]


def build_dsn() -> str:
    secret = json.loads(fetch_secret(os.environ["DB_SECRET_NAME"]))
    password = urllib.parse.quote(secret["password"], safe="")
    db_name = secret.get("dbname", os.environ.get("DB_NAME", "gnome"))
    return f"postgresql://{secret['username']}:{password}@{secret['host']}:5432/{db_name}"


def init_runtime_config() -> RuntimeConfig:
    api_key = fetch_api_key(os.environ["CONTROLLER_API_KEY_ID"])
    return RuntimeConfig(
        controller_api_url=os.environ["CONTROLLER_API_URL"],
        api_key=api_key,
    )


def init_registry() -> RegistryClient:
    registry_api_key = fetch_api_key(os.environ["REGISTRY_API_KEY_ID"])
    return RegistryClient(base_url=os.environ["REGISTRY_API_URL"], api_key=registry_api_key)


def init_anthropic() -> BatchAnthropicClient:
    api_key = fetch_secret(os.environ["ANTHROPIC_API_KEY_SECRET"])
    return BatchAnthropicClient(client=anthropic.Anthropic(api_key=api_key))


def init_voyage() -> BatchVoyageClient:
    api_key = fetch_secret(os.environ["VOYAGE_API_KEY_SECRET"])
    return BatchVoyageClient(client=voyageai.Client(api_key=api_key))


def init_cache() -> RedisClassifierCache:
    return RedisClassifierCache(redis_url=os.environ["REDIS_ENDPOINT"])


def init_db() -> ClassifierDB:
    return ClassifierDB(dsn=build_dsn())


def init_clients() -> tuple:
    """Return (registry, batch_client, voyage_client, cache, db)."""
    return init_registry(), init_anthropic(), init_voyage(), init_cache(), init_db()


@dataclasses.dataclass
class WorkerConfig:
    contracts_queue_url: str = dataclasses.field(default_factory=lambda: os.environ.get("CONTRACTS_QUEUE_URL", ""))
    entities_queue_url: str = dataclasses.field(default_factory=lambda: os.environ.get("ENTITIES_QUEUE_URL", ""))
    embeddings_queue_url: str = dataclasses.field(default_factory=lambda: os.environ.get("EMBEDDINGS_QUEUE_URL", ""))
    notifications_topic_arn: str = dataclasses.field(default_factory=lambda: os.environ.get("NOTIFICATIONS_TOPIC_ARN", ""))
    slack_queue_url: str = dataclasses.field(default_factory=lambda: os.environ.get("SLACK_QUEUE_URL", ""))

    cache_bucket: str = dataclasses.field(default_factory=lambda: os.environ.get("CACHE_BUCKET", ""))
    slack_channel: str = dataclasses.field(default_factory=lambda: os.environ.get("SLACK_CHANNEL", ""))
    slack_bot_token_secret: str = dataclasses.field(
        default_factory=lambda: os.environ.get("SLACK_BOT_TOKEN_SECRET", "")
    )
