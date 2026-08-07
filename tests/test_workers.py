import dataclasses
import json
import types
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from classifier.adapters.types import AdapterContract
from classifier.client import BatchVoyageClient
from classifier.runtime_config import ClassifierConfig, FeatureFlags
from classifier.workers.config import WorkerConfig
from classifier.workers.embed import EmbedWorker
from classifier.workers.normalize import NormalizeWorker
from classifier.workers.notify import NotifyWorker
from classifier.workers.relationships import RelationshipsWorker
from gnomepy.registry.types import AssetClass, ContractType, SecurityType
from scripts.testing import StubDB, StubRegistry


def _make_handler_rc(**feature_flags):
    rc = MagicMock()
    rc.config = ClassifierConfig(feature_flags=FeatureFlags(**feature_flags))
    return rc


def _sqs_msg(body: dict, receipt_handle: str = "fake-receipt") -> dict:
    return {"Body": json.dumps(body), "ReceiptHandle": receipt_handle, "MessageId": "test-id"}


def _drain_queue(sqs, queue_url: str) -> list[dict]:
    messages = []
    while True:
        resp = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10, WaitTimeSeconds=0)
        batch = resp.get("Messages", [])
        if not batch:
            break
        messages.extend(batch)
    return messages


def _make_adapter_contract(
    exchange_id: int = 1,
    native_event_id: str = "evt-1",
    event_title: str = "Will BTC hit $100k?",
    outcome_label: str = "Yes",
) -> AdapterContract:
    return AdapterContract(
        exchange_id=exchange_id,
        exchange_security_id="sec-1",
        exchange_security_symbol="SYM",
        base_currency="USD",
        quote_currency="USD",
        settle_currency="USD",
        security_type=SecurityType.EVENT_CONTRACT,
        contract_type=ContractType.BINARY,
        asset_class=AssetClass.PREDICTION,
        inverse=False,
        is_quanto=False,
        tick_size=0.01,
        lot_size=1.0,
        min_notional=1.0,
        contract_multiplier=1.0,
        event_title=event_title,
        outcome_label=outcome_label,
        exchange_event_native_id=native_event_id,
    )


def _make_no_op_cache():
    cache = MagicMock()
    cache.get_canonicalization_bulk.return_value = {}
    cache.get_exchange_event_bulk.return_value = {}
    return cache


def _make_mock_runtime_config():
    from classifier.runtime_config import ClassifierConfig
    rc = MagicMock()
    rc.config = ClassifierConfig()
    return rc


def _patch_normalize_setup(worker, stub_registry, stub_db, mock_anthropic, sns_client):
    def _setup(self):
        self._runtime_config = _make_mock_runtime_config()
        self._registry = stub_registry
        self._batch_client = mock_anthropic
        self._cache = _make_no_op_cache()
        self._db = stub_db
        self._sns = sns_client
    worker._setup = types.MethodType(_setup, worker)


def _patch_embed_setup(worker, mock_voyage_client, stub_db):
    def _setup(self):
        self._runtime_config = _make_mock_runtime_config()
        self._voyage_client = BatchVoyageClient(client=mock_voyage_client)
        self._db = stub_db
    worker._setup = types.MethodType(_setup, worker)


def _patch_relationships_setup(worker, stub_registry, stub_db, mock_anthropic, mock_voyage_client, sns_client):
    def _setup(self):
        self._runtime_config = _make_mock_runtime_config()
        self._registry = stub_registry
        self._batch_client = mock_anthropic
        self._cache = _make_no_op_cache()
        self._db = stub_db
        self._sns = sns_client
    worker._setup = types.MethodType(_setup, worker)


class TestNormalizeWorker:
    def test_new_contracts_produces_entity_messages(self, moto_env, stub_registry, stub_db, mock_anthropic):
        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = NormalizeWorker(config)
        _patch_normalize_setup(worker, stub_registry, stub_db, mock_anthropic, sns_client)
        worker._setup()

        contract = _make_adapter_contract()
        messages = [_sqs_msg({"type": "new", "contracts": [dataclasses.asdict(contract)]})]
        output = worker.process_batch(messages)

        assert len(output) > 0
        for msg in output:
            assert msg["type"] == "new_security"
            assert "security_id" in msg
            assert "security_symbol" in msg

    def test_new_contracts_enriches_messages_with_event_info(self, moto_env, stub_registry, stub_db, mock_anthropic):
        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = NormalizeWorker(config)
        _patch_normalize_setup(worker, stub_registry, stub_db, mock_anthropic, sns_client)
        worker._setup()

        contract = _make_adapter_contract(event_title="Will BTC hit $100k?")
        messages = [_sqs_msg({"type": "new", "contracts": [dataclasses.asdict(contract)]})]
        output = worker.process_batch(messages)

        assert len(output) > 0
        # At least one message should carry created event info (event was newly created)
        event_msgs = [m for m in output if "created_event_id" in m]
        assert len(event_msgs) > 0
        assert all("created_event_name" in m for m in event_msgs)

    def test_new_contracts_does_not_publish_to_sns(self, moto_env, stub_registry, stub_db, mock_anthropic):
        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = NormalizeWorker(config)
        _patch_normalize_setup(worker, stub_registry, stub_db, mock_anthropic, sns_client)
        worker._setup()

        contract = _make_adapter_contract()
        messages = [_sqs_msg({"type": "new", "contracts": [dataclasses.asdict(contract)]})]
        worker.process_batch(messages)

        delivered = _drain_queue(moto_env["sqs"], moto_env["slack_queue"])
        payloads = [json.loads(json.loads(m["Body"])["Message"]) for m in delivered]
        types_seen = {p["type"] for p in payloads}
        assert "new_entity" not in types_seen
        assert "new_events" not in types_seen

    def test_resolved_contracts_publishes_resolved(self, moto_env, stub_registry, stub_db, mock_anthropic):
        stub_registry.bulk_create_events([{"title": "Test event"}])
        event_id = stub_registry._events[0].event_id
        stub_registry.bulk_create_securities([{"symbol": "SYM", "type": 4, "contract_type": 7, "asset_class": 5, "inverse": False, "quanto": False}])
        sec_id = stub_registry._securities[0].security_id
        stub_registry.bulk_create_event_contracts([{"event_id": event_id, "security_id": sec_id, "outcome_label": "Yes"}])
        stub_registry.bulk_create_listings([{"security_id": sec_id, "exchange_id": 1, "exchange_security_id": "sec-native-1", "active": True}])

        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = NormalizeWorker(config)
        _patch_normalize_setup(worker, stub_registry, stub_db, mock_anthropic, sns_client)
        worker._setup()

        messages = [_sqs_msg({"type": "resolved", "exchange_id": 1, "native_id": "sec-native-1"})]
        worker.process_batch(messages)

        delivered = _drain_queue(moto_env["sqs"], moto_env["slack_queue"])
        payloads = [json.loads(json.loads(m["Body"])["Message"]) for m in delivered]
        types_seen = {p["type"] for p in payloads}
        assert "resolved" in types_seen


class TestEmbedWorker:
    def test_produces_embeddings_queue_messages(self, moto_env, stub_registry, stub_db, mock_voyage):
        stub_registry.bulk_create_events([{"title": "BTC question"}])
        event_id = stub_registry._events[0].event_id
        stub_registry.bulk_create_securities([{"symbol": "SYM", "type": 4, "contract_type": 7, "asset_class": 5, "inverse": False, "quanto": False}])
        sec_id = stub_registry._securities[0].security_id
        stub_registry.bulk_create_event_contracts([{"event_id": event_id, "security_id": sec_id, "outcome_label": "Yes"}])

        config = WorkerConfig()
        worker = EmbedWorker(config)
        _patch_embed_setup(worker, mock_voyage, stub_db)
        worker._setup()

        messages = [_sqs_msg({"type": "new_security", "security_id": sec_id, "security_symbol": "SYM"})]
        output = worker.process_batch(messages)

        assert len(output) > 0
        for msg in output:
            assert msg["type"] == "security"
            assert msg["security_id"] == sec_id
            assert msg["security_symbol"] == "SYM"

    def test_preserves_created_event_info(self, moto_env, stub_registry, stub_db, mock_voyage):
        stub_registry.bulk_create_events([{"title": "BTC question"}])
        event_id = stub_registry._events[0].event_id
        stub_registry.bulk_create_securities([{"symbol": "SYM", "type": 4, "contract_type": 7, "asset_class": 5, "inverse": False, "quanto": False}])
        sec_id = stub_registry._securities[0].security_id
        stub_registry.bulk_create_event_contracts([{"event_id": event_id, "security_id": sec_id, "outcome_label": "Yes"}])

        config = WorkerConfig()
        worker = EmbedWorker(config)
        _patch_embed_setup(worker, mock_voyage, stub_db)
        worker._setup()

        messages = [_sqs_msg({
            "type": "new_security", "security_id": sec_id, "security_symbol": "SYM",
            "created_event_id": event_id, "created_event_name": "BTC question",
        })]
        output = worker.process_batch(messages)

        assert len(output) > 0
        event_msgs = [m for m in output if m["security_id"] == sec_id]
        assert len(event_msgs) == 1
        assert event_msgs[0].get("created_event_id") == event_id
        assert event_msgs[0].get("created_event_name") == "BTC question"

    def test_empty_batch_returns_empty(self, moto_env, stub_registry, stub_db, mock_voyage):
        config = WorkerConfig()
        worker = EmbedWorker(config)
        _patch_embed_setup(worker, mock_voyage, stub_db)
        worker._setup()

        assert worker.process_batch([]) == []


class TestRelationshipsWorker:
    def test_publishes_new_events_to_sns(self, moto_env, stub_registry, stub_db, mock_anthropic, mock_voyage):
        stub_registry.bulk_create_events([{"title": "BTC event"}])
        event_id = stub_registry._events[0].event_id
        for sym in ("SYM-YES", "SYM-NO"):
            stub_registry.bulk_create_securities([{"symbol": sym, "type": 4, "contract_type": 7, "asset_class": 5, "inverse": False, "quanto": False}])
        sid_yes = stub_registry._securities[0].security_id
        sid_no = stub_registry._securities[1].security_id
        for sid, label in [(sid_yes, "Yes"), (sid_no, "No")]:
            stub_registry.bulk_create_event_contracts([{"event_id": event_id, "security_id": sid, "outcome_label": label}])

        stub_db.put_embeddings({sid_yes: [0.1] * 10, sid_no: [0.2] * 10})

        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = RelationshipsWorker(config)
        _patch_relationships_setup(worker, stub_registry, stub_db, mock_anthropic, mock_voyage, sns_client)
        worker._setup()

        messages = [
            _sqs_msg({"type": "security", "security_id": sid_yes, "security_symbol": "SYM-YES", "created_event_id": event_id, "created_event_name": "BTC event"}),
            _sqs_msg({"type": "security", "security_id": sid_no, "security_symbol": "SYM-NO", "created_event_id": event_id, "created_event_name": "BTC event"}),
        ]
        output = worker.process_batch(messages)

        assert output == []

        delivered = _drain_queue(moto_env["sqs"], moto_env["slack_queue"])
        payloads = [json.loads(json.loads(m["Body"])["Message"]) for m in delivered]
        types_seen = {p["type"] for p in payloads}
        assert "new_events" in types_seen
        new_events_payload = next(p for p in payloads if p["type"] == "new_events")
        assert event_id in new_events_payload["created_event_ids"]
        assert "BTC event" in new_events_payload["created_event_names"]

    def test_no_new_events_no_sns_publish(self, moto_env, stub_registry, stub_db, mock_anthropic, mock_voyage):
        stub_registry.bulk_create_events([{"title": "BTC event"}])
        event_id = stub_registry._events[0].event_id
        stub_registry.bulk_create_securities([{"symbol": "SYM-YES", "type": 4, "contract_type": 7, "asset_class": 5, "inverse": False, "quanto": False}])
        sid_yes = stub_registry._securities[0].security_id
        stub_registry.bulk_create_event_contracts([{"event_id": event_id, "security_id": sid_yes, "outcome_label": "Yes"}])

        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = RelationshipsWorker(config)
        _patch_relationships_setup(worker, stub_registry, stub_db, mock_anthropic, mock_voyage, sns_client)
        worker._setup()

        # No created_event_id in message — existing event got a new security
        messages = [_sqs_msg({"type": "security", "security_id": sid_yes, "security_symbol": "SYM-YES"})]
        output = worker.process_batch(messages)

        assert output == []
        delivered = _drain_queue(moto_env["sqs"], moto_env["slack_queue"])
        assert len(delivered) == 0

    def test_no_securities_returns_empty(self, moto_env, stub_registry, stub_db, mock_anthropic, mock_voyage):
        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = RelationshipsWorker(config)
        _patch_relationships_setup(worker, stub_registry, stub_db, mock_anthropic, mock_voyage, sns_client)
        worker._setup()

        assert worker.process_batch([]) == []


class TestNotifyWorker:
    def test_aggregates_and_calls_slack(self, moto_env):
        config = WorkerConfig()
        worker = NotifyWorker(config)

        def _setup(self):
            self._slack_token = "fake-token"
        worker._setup = types.MethodType(_setup, worker)
        worker._setup()

        sns_envelope = lambda payload: _sqs_msg({
            "Type": "Notification",
            "Message": json.dumps(payload),
        })

        messages = [
            sns_envelope({"type": "new_events", "created_event_ids": [42], "created_event_names": ["Will X happen?"]}),
            sns_envelope({"type": "resolved", "events_resolved": 2, "resolved_event_ids": [10, 11], "resolved_event_names": ["Event A", "Event B"]}),
        ]

        with patch("classifier.workers.notify.send_slack_notification") as mock_send:
            mock_send.return_value = True
            worker.process_batch(messages)
            assert mock_send.called
            _, channel, blocks = mock_send.call_args[0]
            assert channel == "test-channel"
            assert len(blocks) > 0
            block_text = json.dumps(blocks)
            assert "Will X happen?" in block_text
            assert "Event A" in block_text
            assert "controller.gnometrading.group" in block_text

    def test_skips_notification_when_no_events(self, moto_env):
        config = WorkerConfig()
        worker = NotifyWorker(config)

        def _setup(self):
            self._slack_token = "fake-token"
        worker._setup = types.MethodType(_setup, worker)
        worker._setup()

        sns_envelope = lambda payload: _sqs_msg({
            "Type": "Notification",
            "Message": json.dumps(payload),
        })

        messages = [
            sns_envelope({"type": "resolved", "events_resolved": 0, "resolved_event_ids": [], "resolved_event_names": []}),
        ]

        with patch("classifier.workers.notify.send_slack_notification") as mock_send:
            worker.process_batch(messages)
            mock_send.assert_not_called()

    def test_no_token_skips_slack(self, moto_env):
        config = WorkerConfig()
        worker = NotifyWorker(config)

        def _setup(self):
            self._slack_token = None
        worker._setup = types.MethodType(_setup, worker)
        worker._setup()

        with patch("classifier.workers.notify.send_slack_notification") as mock_send:
            worker.process_batch([_sqs_msg({"type": "new_events", "created_event_ids": [1], "created_event_names": ["Test"]})])
            mock_send.assert_not_called()


class TestPipelineMessageCompat:
    def test_normalize_output_feeds_embed(self, moto_env, stub_registry, stub_db, mock_anthropic, mock_voyage):
        """Verify that messages NormalizeWorker produces are valid input for EmbedWorker."""
        sns_client = boto3.client("sns", region_name="us-east-1")

        config = WorkerConfig()
        normalize = NormalizeWorker(config)
        _patch_normalize_setup(normalize, stub_registry, stub_db, mock_anthropic, sns_client)
        normalize._setup()

        contract = _make_adapter_contract()
        norm_output = normalize.process_batch([
            _sqs_msg({"type": "new", "contracts": [dataclasses.asdict(contract)]})
        ])

        assert len(norm_output) > 0

        embed = EmbedWorker(config)
        _patch_embed_setup(embed, mock_voyage, stub_db)
        embed._setup()

        embed_input = [_sqs_msg(msg) for msg in norm_output]
        embed_output = embed.process_batch(embed_input)

        assert len(embed_output) > 0
        for msg in embed_output:
            assert msg["type"] == "security"
            assert "security_id" in msg

    def test_created_event_info_flows_normalize_to_embed(self, moto_env, stub_registry, stub_db, mock_anthropic, mock_voyage):
        """Verify created_event_id/created_event_name flows from NormalizeWorker through EmbedWorker."""
        sns_client = boto3.client("sns", region_name="us-east-1")

        config = WorkerConfig()
        normalize = NormalizeWorker(config)
        _patch_normalize_setup(normalize, stub_registry, stub_db, mock_anthropic, sns_client)
        normalize._setup()

        contract = _make_adapter_contract(event_title="Will BTC hit $100k?")
        norm_output = normalize.process_batch([
            _sqs_msg({"type": "new", "contracts": [dataclasses.asdict(contract)]})
        ])

        # At least one output message should carry event info
        event_msgs = [m for m in norm_output if "created_event_id" in m]
        assert len(event_msgs) > 0

        embed = EmbedWorker(config)
        _patch_embed_setup(embed, mock_voyage, stub_db)
        embed._setup()

        embed_input = [_sqs_msg(msg) for msg in norm_output]
        embed_output = embed.process_batch(embed_input)

        # Event info should be preserved through EmbedWorker
        embed_event_msgs = [m for m in embed_output if "created_event_id" in m]
        assert len(embed_event_msgs) > 0
