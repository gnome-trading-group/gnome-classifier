import dataclasses
import hashlib
import json
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

import anthropic

from classifier.client import BatchAnthropicClient
from scripts.testing import StubDB, StubRegistry
from gnomepy.registry.types import Event, EventContract


@pytest.fixture
def stub_registry():
    return StubRegistry()


@pytest.fixture
def stub_db(stub_registry):
    return StubDB(stub_registry)


@pytest.fixture
def mock_anthropic():
    inner = MagicMock(spec=anthropic.Anthropic)

    def _fake_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        content = messages[0].get("content", "") if messages else ""
        titles = []
        idx = 1
        for line in content.splitlines():
            if line.startswith("[") and "] Title: " in line:
                raw_title = line.split("] Title: ", 1)[1].split(" | ")[0].strip()
                key = hashlib.sha256(raw_title.encode()).hexdigest()[:6]
                titles.append({"id": idx, "key": key, "title": raw_title, "category": "POLITICS", "tags": ["test", "tag", "here"]})
                idx += 1
            elif line.startswith("Exchange-provided title:"):
                title = line.split(":", 1)[1].strip()
                titles.append({"title": title, "category": "OTHER", "tags": ["test", "tag", "here"]})

        if not titles:
            text = "[]"
        elif len(titles) == 1:
            text = json.dumps(titles[0])
        else:
            text = json.dumps(titles)

        mock_content = MagicMock()
        mock_content.text = text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        return mock_response

    inner.messages.create.side_effect = _fake_create
    return BatchAnthropicClient(client=inner, max_batch_wait=0)


@pytest.fixture
def mock_voyage():
    client = MagicMock()

    def _fake_embed(texts, **kwargs):
        result = MagicMock()
        result.embeddings = [[float(i) / 100] * 10 for i in range(len(texts))]
        return result

    client.embed.side_effect = _fake_embed
    return client


@pytest.fixture
def moto_env(aws_credentials, monkeypatch):
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-east-1")
        sns = boto3.client("sns", region_name="us-east-1")
        s3 = boto3.client("s3", region_name="us-east-1")

        contracts_q = sqs.create_queue(QueueName="contracts")["QueueUrl"]
        entities_q = sqs.create_queue(QueueName="entities")["QueueUrl"]
        embeddings_q = sqs.create_queue(QueueName="embeddings")["QueueUrl"]
        slack_q = sqs.create_queue(QueueName="slack")["QueueUrl"]

        topic_arn = sns.create_topic(Name="notifications")["TopicArn"]
        slack_arn = sqs.get_queue_attributes(
            QueueUrl=slack_q, AttributeNames=["QueueArn"]
        )["Attributes"]["QueueArn"]
        sns.subscribe(TopicArn=topic_arn, Protocol="sqs", Endpoint=slack_arn)

        s3.create_bucket(Bucket="test-cache")

        monkeypatch.setenv("CONTRACTS_QUEUE_URL", contracts_q)
        monkeypatch.setenv("ENTITIES_QUEUE_URL", entities_q)
        monkeypatch.setenv("EMBEDDINGS_QUEUE_URL", embeddings_q)
        monkeypatch.setenv("SLACK_QUEUE_URL", slack_q)
        monkeypatch.setenv("NOTIFICATIONS_TOPIC_ARN", topic_arn)
        monkeypatch.setenv("CACHE_BUCKET", "test-cache")
        monkeypatch.setenv("SLACK_CHANNEL", "test-channel")
        monkeypatch.setenv("SLACK_BOT_TOKEN_SECRET", "")
        monkeypatch.setenv("REGISTRY_API_URL", "http://localhost")
        monkeypatch.setenv("REGISTRY_API_KEY_ID", "fake-key-id")

        yield {
            "sqs": sqs,
            "sns": sns,
            "s3": s3,
            "contracts_queue": contracts_q,
            "entities_queue": entities_q,
            "embeddings_queue": embeddings_q,
            "slack_queue": slack_q,
            "topic_arn": topic_arn,
            "bucket": "test-cache",
        }


@pytest.fixture
def s3_bucket(aws_credentials):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        yield "test-bucket"


@pytest.fixture
def aws_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def make_event(event_id: int, title: str, category: str = "POLITICS", expiry: str | None = None) -> Event:
    return Event(
        event_id=event_id,
        title=title,
        description=None,
        category=category,
        tags=None,
        resolved=False,
        resolved_at=None,
        expiry=expiry,
        date_modified="",
        date_created="",
    )


def make_event_contract(
    ec_id: int,
    event_id: int,
    security_id: int,
    outcome_label: str,
) -> EventContract:
    return EventContract(
        event_contract_id=ec_id,
        event_id=event_id,
        security_id=security_id,
        outcome_label=outcome_label,
        date_created="",
    )
