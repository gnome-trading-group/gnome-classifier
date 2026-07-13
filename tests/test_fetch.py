from unittest.mock import MagicMock, patch

import pytest

from classifier.stages.fetch import fetch_all
from gnomepy.registry.types import Exchange


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
    mock_adapter.fetch.return_value = [_make_contract(f"Event {i}") for i in range(20)]

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
