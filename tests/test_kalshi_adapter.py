import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from classifier.adapters.kalshi import KalshiAdapter
from classifier.stages.entities import create_entities
from gnomepy.registry.types import ContractType

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "kalshi_events.json").read_text())
EVENTS_BY_TICKER = {e["event_ticker"]: e for e in FIXTURE["events"]}

EXCHANGE_ID = 2
adapter = KalshiAdapter()


def _map(ticker: str) -> list:
    return adapter._map_event(EXCHANGE_ID, EVENTS_BY_TICKER[ticker])


# ── Binary ────────────────────────────────────────────────────────────────────

def test_binary_contract_count():
    contracts = _map("KXELONMARS-99")
    assert len(contracts) == 2


def test_binary_contract_types():
    contracts = _map("KXELONMARS-99")
    assert all(c.contract_type == ContractType.BINARY for c in contracts)


def test_binary_native_id():
    contracts = _map("KXELONMARS-99")
    assert all(c.exchange_event_native_id == "KXELONMARS-99" for c in contracts)


def test_binary_outcomes():
    contracts = _map("KXELONMARS-99")
    assert {c.outcome_label for c in contracts} == {"Yes", "No"}


def test_binary_event_title():
    contracts = _map("KXELONMARS-99")
    assert all(c.event_title == "Will Elon Musk visit Mars in his lifetime?" for c in contracts)


def test_binary_security_ids():
    contracts = _map("KXELONMARS-99")
    ids = {c.exchange_security_id for c in contracts}
    assert ids == {"KXELONMARS-99:yes", "KXELONMARS-99:no"}


# ── Multi-outcome ─────────────────────────────────────────────────────────────

def test_multi_outcome_contract_count():
    contracts = _map("KXNEWPOPE-70")
    # 7 markets, mutually_exclusive=true → 7 MULTI_OUTCOME contracts
    assert len(contracts) == 7


def test_multi_outcome_contract_types():
    contracts = _map("KXNEWPOPE-70")
    assert all(c.contract_type == ContractType.MULTI_OUTCOME for c in contracts)


def test_multi_outcome_native_id():
    contracts = _map("KXNEWPOPE-70")
    assert all(c.exchange_event_native_id == "KXNEWPOPE-70" for c in contracts)


def test_multi_outcome_event_title():
    contracts = _map("KXNEWPOPE-70")
    assert all(c.event_title == "Who will the next Pope be?" for c in contracts)


def test_multi_outcome_outcomes():
    contracts = _map("KXNEWPOPE-70")
    outcomes = {c.outcome_label for c in contracts}
    assert "Pierbattista Pizzaballa" in outcomes
    assert "Pietro Parolin" in outcomes
    assert len(outcomes) == 7


def test_multi_outcome_security_ids_are_market_tickers():
    contracts = _map("KXNEWPOPE-70")
    ids = {c.exchange_security_id for c in contracts}
    assert "KXNEWPOPE-70-PPIZ" in ids
    assert "KXNEWPOPE-70-PPAR" in ids


# ── Sub-markets ───────────────────────────────────────────────────────────────

def test_sub_market_contract_count():
    contracts = _map("KXRAMPBREX-40")
    # 2 markets, mutually_exclusive=false → 2 sub-events × 2 sides = 4
    assert len(contracts) == 4


def test_sub_market_contract_types():
    contracts = _map("KXRAMPBREX-40")
    assert all(c.contract_type == ContractType.BINARY for c in contracts)


def test_sub_market_native_ids_are_market_tickers():
    contracts = _map("KXRAMPBREX-40")
    native_ids = {c.exchange_event_native_id for c in contracts}
    assert native_ids == {"KXRAMPBREX-40-RAMP", "KXRAMPBREX-40-BREX"}


def test_sub_market_event_titles():
    contracts = _map("KXRAMPBREX-40")
    titles = {c.event_title for c in contracts}
    assert titles == {
        "Will Ramp or Brex IPO first?: Ramp",
        "Will Ramp or Brex IPO first?: Brex",
    }


def test_sub_market_outcomes():
    ramp = [c for c in _map("KXRAMPBREX-40") if c.exchange_event_native_id == "KXRAMPBREX-40-RAMP"]
    assert {c.outcome_label for c in ramp} == {"Yes", "No"}


# ── Entity creation with fixture data ─────────────────────────────────────────

@pytest.fixture
def kalshi_exchange():
    ex = MagicMock()
    ex.exchange_id = EXCHANGE_ID
    ex.exchange_name = "kalshi"
    return ex


def test_entity_creation_binary(stub_registry, mock_anthropic, mock_voyage, kalshi_exchange):
    contracts = _map("KXELONMARS-99")
    result = create_entities(
        stub_registry, mock_voyage, mock_anthropic, contracts,
        {"kalshi": kalshi_exchange},
    )
    assert result["events_created"] == 1
    assert result["securities_created"] == 2
    assert result["listings_created"] == 2
    assert result["event_contracts_created"] == 2


def test_entity_creation_multi_outcome(stub_registry, mock_anthropic, mock_voyage, kalshi_exchange):
    contracts = _map("KXNEWPOPE-70")
    result = create_entities(
        stub_registry, mock_voyage, mock_anthropic, contracts,
        {"kalshi": kalshi_exchange},
    )
    assert result["events_created"] == 1
    assert result["securities_created"] == 7
    assert result["listings_created"] == 7
    assert result["event_contracts_created"] == 7


def test_entity_creation_sub_markets(stub_registry, mock_anthropic, mock_voyage, kalshi_exchange):
    contracts = _map("KXRAMPBREX-40")
    result = create_entities(
        stub_registry, mock_voyage, mock_anthropic, contracts,
        {"kalshi": kalshi_exchange},
    )
    # 2 sub-events → 2 events, each with Yes+No → 4 securities
    assert result["events_created"] == 2
    assert result["securities_created"] == 4
    assert result["listings_created"] == 4
    assert result["event_contracts_created"] == 4
