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
from classifier.workers.fetch import fetch_handler, resolve_handler
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

    def test_new_contracts_publishes_to_sns(self, moto_env, stub_registry, stub_db, mock_anthropic):
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
        assert "new_entity" in types_seen

    def test_resolved_contracts_publishes_resolution(self, moto_env, stub_registry, stub_db, mock_anthropic):
        # Seed a security with a listing whose exchange_security_id matches the native_id in the resolved message
        stub_registry.bulk_create_events([{"title": "Test event"}])
        event_id = stub_registry._events[0].event_id
        stub_registry.bulk_create_securities([{"symbol": "SYM", "type": 4, "contract_type": 7, "asset_class": 5, "inverse": False, "quanto": False}])
        sec_id = stub_registry._securities[0].security_id
        stub_registry.bulk_create_event_contracts([{"event_id": event_id, "security_id": sec_id, "outcome_label": "Yes"}])
        # Listing's exchange_security_id must equal the native_id sent in the resolved message
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
        assert "resolution" in types_seen


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

    def test_empty_batch_returns_empty(self, moto_env, stub_registry, stub_db, mock_voyage):
        config = WorkerConfig()
        worker = EmbedWorker(config)
        _patch_embed_setup(worker, mock_voyage, stub_db)
        worker._setup()

        assert worker.process_batch([]) == []


class TestRelationshipsWorker:
    def test_writes_relationships_and_publishes_sns(self, moto_env, stub_registry, stub_db, mock_anthropic, mock_voyage):
        # Seed two securities in the same event so semantic.py can find candidates
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
            _sqs_msg({"type": "security", "security_id": sid_yes, "security_symbol": "SYM-YES"}),
            _sqs_msg({"type": "security", "security_id": sid_no, "security_symbol": "SYM-NO"}),
        ]
        output = worker.process_batch(messages)

        assert output == []

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
            sns_envelope({"type": "new_entity", "new_symbols": ["SYM"], "events_created": 1, "securities_created": 1, "listings_created": 1, "created_event_ids": [42], "created_event_names": ["Will X happen?"]}),
            sns_envelope({"type": "resolution", "events_resolved": 2, "securities_deactivated": 2, "listings_deactivated": 4, "resolved_event_ids": [10, 11], "resolved_event_names": ["Event A", "Event B"]}),
            sns_envelope({"type": "stale_cleanup", "events_resolved": 1, "securities_deactivated": 1, "resolved_event_ids": [20], "resolved_event_names": ["Event C"]}),
        ]

        with patch("classifier.workers.notify.send_slack_notification") as mock_send:
            mock_send.return_value = True
            worker.process_batch(messages)
            assert mock_send.called
            _, channel, blocks = mock_send.call_args[0]
            assert channel == "test-channel"
            assert len(blocks) > 0
            # All three event sections should appear
            block_text = json.dumps(blocks)
            assert "Will X happen?" in block_text
            assert "Event A" in block_text
            assert "Event C" in block_text
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
            sns_envelope({"type": "new_entity", "new_symbols": ["SYM"], "events_created": 0, "securities_created": 3, "listings_created": 3, "created_event_ids": [], "created_event_names": []}),
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
            worker.process_batch([_sqs_msg({"type": "new_entity", "new_symbols": ["SYM"], "events_created": 1, "created_event_ids": [1], "created_event_names": ["Test"]})])
            mock_send.assert_not_called()


class TestFetchHandler:
    def test_sends_new_contracts_to_queue(self, moto_env, stub_registry, monkeypatch):
        contract = _make_adapter_contract()

        monkeypatch.setenv("REGISTRY_API_KEY_ID", "fake-key-id")

        with (
            patch("classifier.workers.fetch._get_runtime_config", return_value=_make_handler_rc(fetch_enabled=True)),
            patch("classifier.workers.fetch.init_registry", return_value=stub_registry),
            patch("classifier.workers.fetch.fetch_exchanges", return_value={"polymarket": MagicMock(exchange_id=1)}),
            patch("classifier.workers.fetch.fetch_all", return_value=([contract], [])),
        ):
            result = fetch_handler({}, None)

        assert result["new_contracts"] == 1

        delivered = _drain_queue(moto_env["sqs"], moto_env["contracts_queue"])
        bodies = [json.loads(m["Body"]) for m in delivered]
        assert any(b["type"] == "new" for b in bodies)

        obj = moto_env["s3"].get_object(Bucket="test-cache", Key="fetch-cache/known_contracts.json")
        cache = json.loads(obj["Body"].read())
        assert len(cache) == 1

    def test_deduplicates_on_second_call(self, moto_env, stub_registry, monkeypatch):
        contract = _make_adapter_contract()
        monkeypatch.setenv("REGISTRY_API_KEY_ID", "fake-key-id")

        patches = dict(
            _get_runtime_config=patch("classifier.workers.fetch._get_runtime_config", return_value=_make_handler_rc(fetch_enabled=True)),
            init_registry=patch("classifier.workers.fetch.init_registry", return_value=stub_registry),
            fetch_exchanges=patch("classifier.workers.fetch.fetch_exchanges", return_value={"polymarket": MagicMock(exchange_id=1)}),
            fetch_all=patch("classifier.workers.fetch.fetch_all", return_value=([contract], [])),
        )

        with patches["_get_runtime_config"], patches["init_registry"], patches["fetch_exchanges"], patches["fetch_all"]:
            first = fetch_handler({}, None)

        _drain_queue(moto_env["sqs"], moto_env["contracts_queue"])

        with patches["_get_runtime_config"], patches["init_registry"], patches["fetch_exchanges"], patches["fetch_all"]:
            second = fetch_handler({}, None)

        assert first["new_contracts"] == 1
        assert second["new_contracts"] == 0


class TestResolveHandler:
    def test_sends_resolved_contracts(self, moto_env, stub_registry, monkeypatch):
        monkeypatch.setenv("REGISTRY_API_KEY_ID", "fake-key-id")
        resolved = {1: {"evt-1"}}

        with (
            patch("classifier.workers.fetch._get_runtime_config", return_value=_make_handler_rc(resolve_enabled=True)),
            patch("classifier.workers.fetch.init_registry", return_value=stub_registry),
            patch("classifier.workers.fetch.fetch_exchanges", return_value={"polymarket": MagicMock(exchange_id=1)}),
            patch("classifier.workers.fetch.fetch_resolved_outcomes", return_value=(resolved, [])),
        ):
            result = resolve_handler({}, None)

        assert result["resolved_contracts"] == 1

        delivered = _drain_queue(moto_env["sqs"], moto_env["contracts_queue"])
        bodies = [json.loads(m["Body"]) for m in delivered]
        assert any(b["type"] == "resolved" for b in bodies)

    def test_deduplicates_on_second_call(self, moto_env, stub_registry, monkeypatch):
        monkeypatch.setenv("REGISTRY_API_KEY_ID", "fake-key-id")
        resolved = {1: {"evt-1"}}

        patches = dict(
            rc=patch("classifier.workers.fetch._get_runtime_config", return_value=_make_handler_rc(resolve_enabled=True)),
            init=patch("classifier.workers.fetch.init_registry", return_value=stub_registry),
            exchanges=patch("classifier.workers.fetch.fetch_exchanges", return_value={"polymarket": MagicMock(exchange_id=1)}),
            resolved=patch("classifier.workers.fetch.fetch_resolved_outcomes", return_value=(resolved, [])),
        )

        with patches["rc"], patches["init"], patches["exchanges"], patches["resolved"]:
            first = resolve_handler({}, None)

        _drain_queue(moto_env["sqs"], moto_env["contracts_queue"])

        with patches["rc"], patches["init"], patches["exchanges"], patches["resolved"]:
            second = resolve_handler({}, None)

        assert first["resolved_contracts"] == 1
        assert second["resolved_contracts"] == 0


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
