import dataclasses
import json
import types
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from classifier.adapters.types import AdapterContract
from classifier.runtime_config import ClassifierConfig, FeatureFlags
from classifier.workers.config import WorkerConfig
from classifier.workers.normalize import NormalizeWorker
from classifier.workers.notify import NotifyWorker
from classifier.workers.relationships import RelationshipsWorker
from classifier.workers.base import _COMPRESS_ABOVE_BYTES, encode_body, message_body
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
        security_symbol="PM_I-SEC-1",
    )


def _make_no_op_cache():
    cache = MagicMock()
    cache.get_canonicalization_bulk.return_value = {}
    return cache


def _make_mock_runtime_config():
    from classifier.runtime_config import ClassifierConfig
    rc = MagicMock()
    rc.config = ClassifierConfig()
    return rc


def _patch_normalize_setup(worker, stub_registry, stub_db, mock_converse, sns_client):
    def _setup(self):
        self._runtime_config = _make_mock_runtime_config()
        self._registry = stub_registry
        self._canon_client = mock_converse
        self._cache = _make_no_op_cache()
        self._db = stub_db
        self._sns = sns_client
    worker._setup = types.MethodType(_setup, worker)


def _patch_relationships_setup(worker, stub_registry, stub_db, sns_client, relationships_enabled=False, llm_enabled=False):
    def _setup(self):
        self._runtime_config = _make_mock_runtime_config()
        self._runtime_config.config.feature_flags.relationships_enabled = relationships_enabled
        self._runtime_config.config.feature_flags.proposition_llm_enabled = llm_enabled
        self._registry = stub_registry
        self._voyage = MagicMock()
        self._voyage.embed.side_effect = lambda texts, model, input_type: [[1.0, 0.0] for _ in texts]
        self._db = stub_db
        self._sns = sns_client
    worker._setup = types.MethodType(_setup, worker)


class TestNormalizeWorker:
    def test_new_contracts_produces_entity_messages(self, moto_env, stub_registry, stub_db, mock_converse):
        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = NormalizeWorker(config)
        _patch_normalize_setup(worker, stub_registry, stub_db, mock_converse, sns_client)
        worker._setup()

        contract = _make_adapter_contract()
        messages = [_sqs_msg({"type": "new", "contracts": [dataclasses.asdict(contract)]})]
        output = worker.process_batch(messages)

        assert len(output) > 0
        for msg in output:
            assert msg["type"] == "new_security"
            assert "security_id" in msg
            assert "security_symbol" in msg

    def test_new_contracts_enriches_messages_with_event_info(self, moto_env, stub_registry, stub_db, mock_converse):
        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = NormalizeWorker(config)
        _patch_normalize_setup(worker, stub_registry, stub_db, mock_converse, sns_client)
        worker._setup()

        contract = _make_adapter_contract(event_title="Will BTC hit $100k?")
        messages = [_sqs_msg({"type": "new", "contracts": [dataclasses.asdict(contract)]})]
        output = worker.process_batch(messages)

        assert len(output) > 0
        # At least one message should carry created event info (event was newly created)
        event_msgs = [m for m in output if "created_event_id" in m]
        assert len(event_msgs) > 0
        assert all("created_event_name" in m for m in event_msgs)

    def test_new_contracts_does_not_publish_to_sns(self, moto_env, stub_registry, stub_db, mock_converse):
        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = NormalizeWorker(config)
        _patch_normalize_setup(worker, stub_registry, stub_db, mock_converse, sns_client)
        worker._setup()

        contract = _make_adapter_contract()
        messages = [_sqs_msg({"type": "new", "contracts": [dataclasses.asdict(contract)]})]
        worker.process_batch(messages)

        delivered = _drain_queue(moto_env["sqs"], moto_env["slack_queue"])
        payloads = [json.loads(json.loads(m["Body"])["Message"]) for m in delivered]
        types_seen = {p["type"] for p in payloads}
        assert "new_entity" not in types_seen
        assert "new_events" not in types_seen

    def test_closed_contracts_deactivate_listings_without_resolving_or_announcing(
        self, moto_env, stub_registry, stub_db, mock_converse,
    ):
        stub_registry.bulk_create_events([{"title": "Test event", "exchange_id": 1, "native_event_id": "evt-1"}])
        event_id = stub_registry._events[0].event_id
        stub_registry.bulk_create_securities([{"symbol": "SYM", "type": 4, "contract_type": 7, "asset_class": 5, "inverse": False, "quanto": False}])
        sec_id = stub_registry._securities[0].security_id
        stub_registry.bulk_create_event_contracts([{"event_id": event_id, "security_id": sec_id, "outcome_label": "Yes"}])
        stub_registry.bulk_create_listings([{"security_id": sec_id, "exchange_id": 1, "exchange_security_id": "sec-native-1", "active": True}])

        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        worker = NormalizeWorker(config)
        _patch_normalize_setup(worker, stub_registry, stub_db, mock_converse, sns_client)
        worker._setup()

        messages = [_sqs_msg({"type": "closed", "exchange_id": 1, "native_id": "sec-native-1"})]
        worker.process_batch(messages)

        assert stub_registry._listings[0].active is False
        assert stub_registry._securities[0].active is False
        assert stub_registry._events[0].resolved is False
        assert _drain_queue(moto_env["sqs"], moto_env["slack_queue"]) == []


class TestRelationshipsWorker:
    def _worker(self, stub_registry, stub_db, relationships_enabled=False, llm_enabled=False):
        sns_client = boto3.client("sns", region_name="us-east-1")
        worker = RelationshipsWorker(WorkerConfig())
        _patch_relationships_setup(worker, stub_registry, stub_db, sns_client, relationships_enabled, llm_enabled)
        worker._setup()
        return worker

    def _event_with_two_sides(self, stub_registry):
        stub_registry.bulk_create_events([{"title": "BTC event", "exchange_id": 1, "native_event_id": "evt-1"}])
        event_id = stub_registry._events[0].event_id
        for sym in ("SYM-YES", "SYM-NO"):
            stub_registry.bulk_create_securities([{"symbol": sym, "type": 4, "contract_type": 7, "asset_class": 5, "inverse": False, "quanto": False}])
        sids = [s.security_id for s in stub_registry._securities[:2]]
        for sid, label in zip(sids, ("Yes", "No")):
            stub_registry.bulk_create_event_contracts([{"event_id": event_id, "security_id": sid, "outcome_label": label}])
        return event_id, sids

    def test_publishes_new_events_to_sns(self, moto_env, stub_registry, stub_db):
        event_id, (sid_yes, sid_no) = self._event_with_two_sides(stub_registry)
        worker = self._worker(stub_registry, stub_db)
        messages = [
            _sqs_msg({"type": "security", "security_id": sid, "security_symbol": sym, "created_event_id": event_id,
                      "created_event_name": "BTC event"})
            for sid, sym in ((sid_yes, "SYM-YES"), (sid_no, "SYM-NO"))
        ]
        assert worker.process_batch(messages) == []

        payloads = [json.loads(json.loads(m["Body"])["Message"]) for m in _drain_queue(moto_env["sqs"], moto_env["slack_queue"])]
        new_events = next(p for p in payloads if p["type"] == "new_events")
        assert event_id in new_events["created_event_ids"] and "BTC event" in new_events["created_event_names"]

    def test_no_new_events_no_sns_publish(self, moto_env, stub_registry, stub_db):
        _, (sid_yes, _) = self._event_with_two_sides(stub_registry)
        worker = self._worker(stub_registry, stub_db)
        assert worker.process_batch([_sqs_msg({"type": "security", "security_id": sid_yes, "security_symbol": "SYM-YES"})]) == []
        assert _drain_queue(moto_env["sqs"], moto_env["slack_queue"]) == []

    def test_no_securities_returns_empty(self, moto_env, stub_registry, stub_db):
        assert self._worker(stub_registry, stub_db).process_batch([]) == []

    def test_reads_nothing_while_the_flag_is_off(self, moto_env, stub_registry, stub_db):
        _, (sid_yes, _) = self._event_with_two_sides(stub_registry)
        worker = self._worker(stub_registry, stub_db)
        with patch("classifier.workers.relationships.read_propositions") as read:
            worker.process_batch([_sqs_msg({"type": "security", "security_id": sid_yes, "security_symbol": "SYM-YES"})])
            worker.on_idle()
        read.assert_not_called()

    def test_with_the_llm_off_it_reads_by_code_only_and_skips_the_drift_check(self, moto_env, stub_registry, stub_db):
        _, (sid_yes, _) = self._event_with_two_sides(stub_registry)
        worker = self._worker(stub_registry, stub_db, relationships_enabled=True)
        with patch("classifier.workers.relationships.read_propositions") as read, \
                patch("classifier.workers.relationships.derive_due_subjects") as derive, \
                patch("classifier.workers.relationships.run_drift_check") as drift:
            worker.process_batch([_sqs_msg({"type": "security", "security_id": sid_yes, "security_symbol": "SYM-YES"})])
            worker.on_idle()
        assert read.call_args.kwargs["extractor"] is None
        assert derive.called
        drift.assert_not_called()

    def test_with_the_llm_on_it_reads_derives_and_checks_drift_once_a_day(self, moto_env, stub_registry, stub_db):
        _, (sid_yes, _) = self._event_with_two_sides(stub_registry)
        worker = self._worker(stub_registry, stub_db, relationships_enabled=True, llm_enabled=True)
        with patch("classifier.workers.relationships.read_propositions") as read, \
                patch("classifier.workers.relationships.derive_due_subjects") as derive, \
                patch("classifier.workers.relationships.run_drift_check") as drift:
            msg = _sqs_msg({"type": "security", "security_id": sid_yes, "security_symbol": "SYM-YES"})
            worker.process_batch([msg])
            worker.process_batch([msg])
        assert read.call_args.args[0] == [sid_yes]
        assert read.call_args.kwargs["extractor"] is not None
        assert derive.call_count == 2
        assert drift.call_count == 1

    def test_idle_derives_subjects_that_came_due(self, moto_env, stub_registry, stub_db):
        worker = self._worker(stub_registry, stub_db, relationships_enabled=True)
        with patch("classifier.workers.relationships.derive_due_subjects") as derive, \
                patch("classifier.workers.relationships.run_drift_check"):
            worker.on_idle()
            worker.on_idle()
        assert derive.call_count == 1

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
    def test_normalize_output_feeds_relationships(self, moto_env, stub_registry, stub_db, mock_converse):
        """Messages NormalizeWorker produces are the relationships worker's input, created-event info included."""
        sns_client = boto3.client("sns", region_name="us-east-1")
        config = WorkerConfig()
        normalize = NormalizeWorker(config)
        _patch_normalize_setup(normalize, stub_registry, stub_db, mock_converse, sns_client)
        normalize._setup()

        norm_output = normalize.process_batch([
            _sqs_msg({"type": "new", "contracts": [dataclasses.asdict(_make_adapter_contract(event_title="Will BTC hit $100k?"))]})
        ])
        assert norm_output and all("security_id" in m for m in norm_output)
        assert any("created_event_id" in m for m in norm_output)

        relationships = RelationshipsWorker(config)
        _patch_relationships_setup(relationships, stub_registry, stub_db, sns_client)
        relationships._setup()
        relationships.process_batch([_sqs_msg(m) for m in norm_output])

        payloads = [json.loads(json.loads(m["Body"])["Message"]) for m in _drain_queue(moto_env["sqs"], moto_env["slack_queue"])]
        assert any(p["type"] == "new_events" for p in payloads)

def test_large_bodies_travel_gzipped_and_read_back_whole():
    small = {"type": "new", "contracts": [{"a": 1}]}
    large = {"type": "new", "contracts": [{"rules": "x" * 1000, "i": i} for i in range(400)]}
    assert encode_body(small) == json.dumps(small)
    encoded = encode_body(large)
    assert len(encoded) < _COMPRESS_ABOVE_BYTES < len(json.dumps(large))
    assert message_body({"Body": encoded}) == large
    assert message_body({"Body": json.dumps(small)}) == small
