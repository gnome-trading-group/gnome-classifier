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


def test_multi_outcome_description_uses_rules_secondary_when_present():
    contracts = _map("KXNFLGAME-26AUG06CARARI")
    expected = EVENTS_BY_TICKER["KXNFLGAME-26AUG06CARARI"]["markets"][0]["rules_secondary"]
    assert all(c.event_description == expected for c in contracts)


def test_multi_outcome_description_falls_back_to_rules_primary_when_rules_secondary_empty():
    contracts = _map("KXNEWPOPE-70")
    expected = EVENTS_BY_TICKER["KXNEWPOPE-70"]["markets"][0]["rules_primary"]
    assert all(c.event_description == expected for c in contracts)


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


def test_sub_market_description_uses_rules_primary():
    contracts = _map("KXRAMPBREX-40")
    ramp = [c for c in contracts if c.exchange_event_native_id == "KXRAMPBREX-40-RAMP"]
    brex = [c for c in contracts if c.exchange_event_native_id == "KXRAMPBREX-40-BREX"]
    assert all(c.event_description == "If Ramp confirms an IPO first, before Jan 1, 2040, then the market resolves to Yes." for c in ramp)
    assert all(c.event_description == "If Brex confirms an IPO first, before Jan 1, 2040, then the market resolves to Yes." for c in brex)


# ── Volume ───────────────────────────────────────────────────────────────────

def test_binary_event_volume():
    contracts = _map("KXELONMARS-99")
    expected = float(EVENTS_BY_TICKER["KXELONMARS-99"]["markets"][0]["volume_fp"])
    assert all(c.event_volume == expected for c in contracts)


def test_multi_outcome_event_volume_is_sum():
    contracts = _map("KXNEWPOPE-70")
    markets = EVENTS_BY_TICKER["KXNEWPOPE-70"]["markets"]
    expected = sum(float(m["volume_fp"]) for m in markets)
    assert all(c.event_volume == expected for c in contracts)


def test_sub_market_event_volume_per_market():
    contracts = _map("KXRAMPBREX-40")
    markets = {m["ticker"]: float(m["volume_fp"]) for m in EVENTS_BY_TICKER["KXRAMPBREX-40"]["markets"]}
    for c in contracts:
        assert c.event_volume == markets[c.exchange_event_native_id]


# ── Entity creation with fixture data ─────────────────────────────────────────

@pytest.fixture
def kalshi_exchange():
    ex = MagicMock()
    ex.exchange_id = EXCHANGE_ID
    ex.exchange_name = "kalshi"
    return ex


def test_entity_creation_binary(stub_registry, stub_db, mock_anthropic, kalshi_exchange):
    contracts = _map("KXELONMARS-99")
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    assert result.events_created == 1
    assert result.securities_created == 2
    assert result.listings_created == 2
    assert result.event_contracts_created == 2


def test_entity_creation_multi_outcome(stub_registry, stub_db, mock_anthropic, kalshi_exchange):
    contracts = _map("KXNEWPOPE-70")
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    assert result.events_created == 1
    assert result.securities_created == 7
    assert result.listings_created == 7
    assert result.event_contracts_created == 7


def test_entity_creation_sub_markets(stub_registry, stub_db, mock_anthropic, kalshi_exchange):
    contracts = _map("KXRAMPBREX-40")
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    # 2 sub-events → 2 events, each with Yes+No → 4 securities
    assert result.events_created == 2
    assert result.securities_created == 4
    assert result.listings_created == 4
    assert result.event_contracts_created == 4


# ── Single binary with threshold yes_sub_title ───────────────────────────────

def test_threshold_binary_event_title_includes_sub_title():
    contracts = _map("KXCS2TOTALMAPS-26AUG04EXG")
    assert all(c.event_title == "ex-GUARA vs. Yawara Esports: Total Maps: Over 2.5 maps" for c in contracts)


def test_threshold_binary_outcome_labels():
    contracts = _map("KXCS2TOTALMAPS-26AUG04EXG")
    assert {c.outcome_label for c in contracts} == {"Yes", "No"}


def test_threshold_binary_native_id_is_event_ticker():
    contracts = _map("KXCS2TOTALMAPS-26AUG04EXG")
    assert all(c.exchange_event_native_id == "KXCS2TOTALMAPS-26AUG04EXG" for c in contracts)


def test_threshold_binary_security_ids():
    contracts = _map("KXCS2TOTALMAPS-26AUG04EXG")
    assert {c.exchange_security_id for c in contracts} == {
        "KXCS2TOTALMAPS-26AUG04EXG-3:yes",
        "KXCS2TOTALMAPS-26AUG04EXG-3:no",
    }


def test_single_binary_description_uses_rules_primary():
    contracts = _map("KXELONMARS-99")
    expected = EVENTS_BY_TICKER["KXELONMARS-99"]["markets"][0]["rules_primary"]
    assert all(c.event_description == expected for c in contracts)


def test_single_binary_description_falls_back_to_sub_title_when_rules_primary_absent():
    contracts = _map("KXCS2TOTALMAPS-26AUG04EXG")
    assert all(c.event_description == "Total Maps: EXG vs. YAW (Aug 4)" for c in contracts)


def test_redundant_sub_title_not_duplicated():
    # "Mars" is already in the event title — should not be appended again
    contracts = _map("KXELONMARS-99")
    assert all(c.event_title == "Will Elon Musk visit Mars in his lifetime?" for c in contracts)


def test_entity_creation_threshold_binary(stub_registry, stub_db, mock_anthropic):
    contracts = _map("KXCS2TOTALMAPS-26AUG04EXG")
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    assert result.events_created == 1
    assert result.securities_created == 2
    assert result.listings_created == 2
    assert result.event_contracts_created == 2


# ── Finalized market filtering ────────────────────────────────────────────────

def test_finalized_market_excluded_from_contracts():
    contracts = _map("KXIPHONERELEASE-IPHONE18")
    tickers = {c.exchange_security_id for c in contracts}
    assert "KXIPHONERELEASE-IPHONE18-26JUL01:yes" not in tickers
    assert "KXIPHONERELEASE-IPHONE18-26JUL01:no" not in tickers


def test_active_markets_preserved_alongside_finalized():
    contracts = _map("KXIPHONERELEASE-IPHONE18")
    # 2 active sub-markets × 2 sides = 4 contracts (finalized one skipped)
    assert len(contracts) == 4
    native_ids = {c.exchange_event_native_id for c in contracts}
    assert native_ids == {"KXIPHONERELEASE-IPHONE18-26OCT01", "KXIPHONERELEASE-IPHONE18-27JAN01"}


def test_finalized_market_volume_included_in_sum():
    contracts = _map("KXIPHONERELEASE-IPHONE18")
    # volume_fp: 5000 (finalized) + 3000 + 2000 = 10000, but sub-markets use per-market volume
    oct_contracts = [c for c in contracts if c.exchange_event_native_id == "KXIPHONERELEASE-IPHONE18-26OCT01"]
    jan_contracts = [c for c in contracts if c.exchange_event_native_id == "KXIPHONERELEASE-IPHONE18-27JAN01"]
    assert all(c.event_volume == 3000.0 for c in oct_contracts)
    assert all(c.event_volume == 2000.0 for c in jan_contracts)


def test_has_sub_markets_preserved_with_finalized():
    # The 3-market event (1 finalized) must still classify as sub-markets,
    # not binary, so surviving markets keep native_id = market.ticker
    contracts = _map("KXIPHONERELEASE-IPHONE18")
    assert all(c.exchange_event_native_id != "KXIPHONERELEASE-IPHONE18" for c in contracts)


def test_sub_market_iphone_description_uses_rules_primary():
    contracts = _map("KXIPHONERELEASE-IPHONE18")
    oct_contracts = [c for c in contracts if c.exchange_event_native_id == "KXIPHONERELEASE-IPHONE18-26OCT01"]
    jan_contracts = [c for c in contracts if c.exchange_event_native_id == "KXIPHONERELEASE-IPHONE18-27JAN01"]
    assert all("before Oct 1, 2026" in c.event_description for c in oct_contracts)
    assert all("before Jan 1, 2027" in c.event_description for c in jan_contracts)
