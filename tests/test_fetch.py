import json
from unittest.mock import MagicMock, patch

import pytest

from classifier.adapters.types import AdapterContract
from classifier.runtime_config import ClassifierConfig, FeatureFlags, Thresholds
from classifier.stages.fetch import fetch_all
from classifier.workers.fetch import FetchRunner
from gnomepy.registry.types import AssetClass, ContractType, Exchange, SecurityType


def _make_exchange(name: str, exchange_id: int = 1) -> Exchange:
    return Exchange(exchange_id=exchange_id, exchange_name=name, region="", schema_type="", date_modified="", date_created="")


def test_fetch_all_skips_unknown_adapter():
    exchange_by_name = {"unknown": _make_exchange("unknown")}
    contracts, failed = fetch_all(exchange_by_name)
    assert contracts == []
    assert failed == []


def test_fetch_all_limits_per_adapter():
    from classifier.adapters.types import AdapterContract
    from gnomepy.registry.types import SecurityType, ContractType, AssetClass

    def _make_contract(title: str) -> AdapterContract:
        return AdapterContract(
            exchange_id=1,
            exchange_security_id=title,
            exchange_security_symbol=title,
            base_currency="USDC",
            quote_currency="USDC",
            settle_currency="USDC",
            security_type=SecurityType.EVENT_CONTRACT,
            contract_type=ContractType.BINARY,
            asset_class=AssetClass.PREDICTION,
            inverse=False,
            is_quanto=False,
            tick_size=1.0,
            lot_size=1.0,
            min_notional=0.0,
            contract_multiplier=1.0,
            event_title=title,
            outcome_label="Yes",
            exchange_event_native_id=f"native:{title}",
        )

    mock_adapter = MagicMock()
    mock_adapter.exchange_name = "polymarket"
    mock_adapter.fetch.return_value = iter([[_make_contract(f"Event {i}") for i in range(20)]])

    exchange_by_name = {"polymarket": _make_exchange("polymarket")}

    with patch("classifier.stages.fetch.ADAPTERS", [mock_adapter]):
        contracts, failed = fetch_all(exchange_by_name, max_per_adapter=5)

    assert len(contracts) == 5
    assert failed == []


def test_fetch_all_handles_adapter_error():
    mock_adapter = MagicMock()
    mock_adapter.exchange_name = "polymarket"
    mock_adapter.fetch.side_effect = RuntimeError("API down")

    exchange_by_name = {"polymarket": _make_exchange("polymarket")}

    with patch("classifier.stages.fetch.ADAPTERS", [mock_adapter]):
        contracts, failed = fetch_all(exchange_by_name)

    assert contracts == []
    assert failed == ["polymarket"]


def _make_contract(
    native_id: str,
    security_id: str,
    event_volume: float | None = None,
) -> AdapterContract:
    return AdapterContract(
        exchange_id=1,
        exchange_security_id=security_id,
        exchange_security_symbol=security_id,
        base_currency="USDC",
        quote_currency="USDC",
        settle_currency="USDC",
        security_type=SecurityType.EVENT_CONTRACT,
        contract_type=ContractType.BINARY,
        asset_class=AssetClass.PREDICTION,
        inverse=False,
        is_quanto=False,
        tick_size=1.0,
        lot_size=1.0,
        min_notional=0.0,
        contract_multiplier=1.0,
        event_title="Test Event",
        outcome_label="Yes",
        exchange_event_native_id=native_id,
        event_volume=event_volume,
    )


def _make_fetch_rc(min_event_volume: float | None = None):
    rc = MagicMock()
    rc.config = ClassifierConfig(
        feature_flags=FeatureFlags(fetch_enabled=True),
        thresholds=Thresholds(min_event_volume=min_event_volume),
    )
    return rc


def _run_fetch(moto_env, contracts, min_event_volume=None):
    rc = _make_fetch_rc(min_event_volume)
    r = MagicMock()
    r.get.return_value = None
    runner = FetchRunner()
    mock_adapter = MagicMock()
    mock_adapter.exchange_name = "polymarket"
    mock_adapter.fetch.return_value = iter([contracts] if contracts else [])
    with (
        patch("classifier.workers.fetch.fetch_exchanges", return_value={"polymarket": MagicMock(exchange_id=1)}),
        patch("classifier.workers.fetch.ADAPTERS", [mock_adapter]),
    ):
        runner._run_fetch(rc, r, moto_env["sqs"], MagicMock())
    messages = []
    while True:
        resp = moto_env["sqs"].receive_message(
            QueueUrl=moto_env["contracts_queue"], MaxNumberOfMessages=10, WaitTimeSeconds=0
        )
        batch = resp.get("Messages", [])
        if not batch:
            break
        messages.extend(batch)
    return messages


class TestVolumeFiltering:
    def test_high_volume_event_passes(self, moto_env):
        contract = _make_contract("evt-1", "sec-1", event_volume=5000.0)
        msgs = _run_fetch(moto_env, [contract], min_event_volume=1000.0)
        assert len(msgs) == 1

    def test_low_volume_event_filtered(self, moto_env):
        contract = _make_contract("evt-1", "sec-1", event_volume=50.0)
        msgs = _run_fetch(moto_env, [contract], min_event_volume=1000.0)
        assert len(msgs) == 0

    def test_none_volume_always_passes(self, moto_env):
        contract = _make_contract("evt-1", "sec-1", event_volume=None)
        msgs = _run_fetch(moto_env, [contract], min_event_volume=1000.0)
        assert len(msgs) == 1

    def test_no_threshold_passes_all(self, moto_env):
        contract = _make_contract("evt-1", "sec-1", event_volume=0.01)
        msgs = _run_fetch(moto_env, [contract], min_event_volume=None)
        assert len(msgs) == 1

    def test_mixed_volume_selectively_filters(self, moto_env):
        high = _make_contract("evt-high", "sec-high", event_volume=5000.0)
        low = _make_contract("evt-low", "sec-low", event_volume=100.0)
        no_vol = _make_contract("evt-none", "sec-none", event_volume=None)
        msgs = _run_fetch(moto_env, [high, low, no_vol], min_event_volume=1000.0)
        assert len(msgs) == 2
