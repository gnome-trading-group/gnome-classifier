import json
from unittest.mock import MagicMock, patch

import pytest

from classifier.adapters.types import AdapterContract
from classifier.runtime_config import ClassifierConfig, FeatureFlags, Thresholds
from classifier.stages.fetch import fetch_all
from classifier.workers.fetch import FetchRunner
from gnomepy.registry.types import AssetClass, ContractType, Exchange, SecurityType


def _make_exchange(code: str, exchange_id: int = 1) -> Exchange:
    return Exchange(
        exchange_id=exchange_id, exchange_code=code, exchange_name=code.title(),
        region="", schema_type="", date_modified="", date_created="",
    )


def test_fetch_all_skips_unknown_adapter():
    exchange_by_code = {"unknown": _make_exchange("unknown")}
    contracts, failed = fetch_all(exchange_by_code)
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
            security_symbol=f"PM_I-{title}",
        )

    mock_adapter = MagicMock()
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.return_value = iter([[_make_contract(f"Event {i}") for i in range(20)]])

    exchange_by_code = {"POLYMARKET_INTL": _make_exchange("POLYMARKET_INTL")}

    with patch("classifier.stages.fetch.ADAPTERS", [mock_adapter]):
        contracts, failed = fetch_all(exchange_by_code, max_per_adapter=5)

    assert len(contracts) == 5
    assert failed == []


def test_fetch_all_handles_adapter_error():
    mock_adapter = MagicMock()
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.side_effect = RuntimeError("API down")

    exchange_by_code = {"POLYMARKET_INTL": _make_exchange("POLYMARKET_INTL")}

    with patch("classifier.stages.fetch.ADAPTERS", [mock_adapter]):
        contracts, failed = fetch_all(exchange_by_code)

    assert contracts == []
    assert failed == ["POLYMARKET_INTL"]


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
        security_symbol=f"PM_I-{security_id}",
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
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.return_value = iter([contracts] if contracts else [])
    with (
        patch("classifier.workers.fetch.fetch_exchanges", return_value={"POLYMARKET_INTL": MagicMock(exchange_id=1)}),
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


def _run_fetch_pages(moto_env, pages, known_contracts=None, raise_after_pages=False):
    rc = _make_fetch_rc()
    r = MagicMock()
    r.get.return_value = json.dumps(known_contracts).encode() if known_contracts is not None else None
    runner = FetchRunner()

    def fetch(exchange_id):
        yield from pages
        if raise_after_pages:
            raise RuntimeError("API error at page 2")

    mock_adapter = MagicMock()
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.side_effect = fetch
    with (
        patch("classifier.workers.fetch.fetch_exchanges", return_value={"POLYMARKET_INTL": MagicMock(exchange_id=1)}),
        patch("classifier.workers.fetch.ADAPTERS", [mock_adapter]),
    ):
        active_by_exchange, successful_ids = runner._run_fetch(rc, r, moto_env["sqs"], MagicMock())
    saved_hashes = json.loads(r.set.call_args[0][1])
    return active_by_exchange, successful_ids, saved_hashes


class TestPaginatedFetch:
    def test_active_events_accumulate_across_pages(self, moto_env):
        pages = [
            [_make_contract("evt-1", "sec-1")],
            [_make_contract("evt-2", "sec-2")],
            [_make_contract("evt-3", "sec-3")],
        ]
        active_by_exchange, successful_ids, _ = _run_fetch_pages(moto_env, pages)
        assert active_by_exchange == {1: {"evt-1", "evt-2", "evt-3"}}
        assert successful_ids == {1}

    def test_failure_mid_pagination_keeps_unfetched_hashes(self, moto_env):
        known = {"1:sec-1": "old-hash-1", "1:sec-2": "old-hash-2"}
        pages = [[_make_contract("evt-1", "sec-1")]]
        active_by_exchange, successful_ids, saved_hashes = _run_fetch_pages(
            moto_env, pages, known_contracts=known, raise_after_pages=True,
        )
        assert saved_hashes["1:sec-2"] == "old-hash-2"
        assert saved_hashes["1:sec-1"] != "old-hash-1"

    def test_failure_mid_pagination_marks_exchange_failed(self, moto_env):
        pages = [[_make_contract("evt-1", "sec-1")]]
        active_by_exchange, successful_ids, _ = _run_fetch_pages(moto_env, pages, raise_after_pages=True)
        assert 1 not in active_by_exchange
        assert successful_ids == set()
