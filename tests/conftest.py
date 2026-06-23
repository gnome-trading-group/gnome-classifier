import dataclasses
import json
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

import anthropic

from scripts.testing import StubRegistry
from gnomepy.registry.types import Event, EventContract


@pytest.fixture
def stub_registry():
    return StubRegistry()


@pytest.fixture
def mock_anthropic():
    client = MagicMock(spec=anthropic.Anthropic)

    def _fake_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        content = messages[0].get("content", "") if messages else ""
        titles = []
        for line in content.splitlines():
            if line.startswith("[") and "] Title: " in line:
                title = line.split("] Title: ", 1)[1].split(" | ")[0].strip()
                titles.append({"title": title, "category": "POLITICS", "tags": ["test", "tag", "here"]})
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

    client.messages.create.side_effect = _fake_create
    return client


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
        resolution_source=None,
        tags=None,
        embedding=None,
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
