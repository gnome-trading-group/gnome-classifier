from unittest.mock import MagicMock

import pytest

from classifier.adapters.types import AdapterContract
from classifier.stages.entities import create_entities
from gnomepy.registry.types import AssetClass, ContractType, SecurityType


def _make_contract(title: str, outcome: str, exchange_id: int = 1) -> AdapterContract:
    return AdapterContract(
        exchange_id=exchange_id,
        exchange_security_id=f"{title}:{outcome}",
        exchange_security_symbol=f"{title} -- {outcome}",
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
        outcome_label=outcome,
        exchange_event_native_id=f"native:{title}",
    )


def test_create_entities_empty(stub_registry, stub_db, mock_anthropic):
    result = create_entities(stub_registry, mock_anthropic, [], db=stub_db)
    assert result["events_created"] == 0
    assert result["securities_created"] == 0


def test_create_entities_new_event(stub_registry, stub_db, mock_anthropic):
    contracts = [
        _make_contract("Will BTC hit 100k?", "Yes"),
        _make_contract("Will BTC hit 100k?", "No"),
    ]
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    assert result["events_created"] == 1
    assert result["securities_created"] == 2
    assert result["listings_created"] == 2
    assert result["event_contracts_created"] == 2
    assert len(result["new_security_ids"]) == 2


def test_create_entities_dedup_same_event(stub_registry, stub_db, mock_anthropic):
    contracts = [
        _make_contract("Will BTC hit 100k?", "Yes", exchange_id=1),
        _make_contract("Will BTC hit 100k?", "No", exchange_id=1),
        _make_contract("Will BTC hit 100k?", "Yes", exchange_id=2),
        _make_contract("Will BTC hit 100k?", "No", exchange_id=2),
    ]
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    assert result["events_created"] == 1
    assert result["securities_created"] == 2
    assert result["listings_created"] == 4
