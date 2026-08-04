import json
from pathlib import Path

import pytest

from classifier.adapters.polymarket import PolymarketAdapter
from classifier.stages.entities import create_entities
from gnomepy.registry.types import ContractType

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "polymarket_events.json").read_text())
EVENTS_BY_SLUG = {e["slug"]: e for e in FIXTURE["events"]}

EXCHANGE_ID = 3
adapter = PolymarketAdapter()


def _map(slug: str) -> list:
    return adapter._map_event(EXCHANGE_ID, EVENTS_BY_SLUG[slug])


# ── Single binary market ──────────────────────────────────────────────────────

def test_single_binary_contract_count():
    contracts = _map("elon-mars")
    assert len(contracts) == 2


def test_single_binary_contract_types():
    contracts = _map("elon-mars")
    assert all(c.contract_type == ContractType.BINARY for c in contracts)


def test_single_binary_outcomes():
    contracts = _map("elon-mars")
    assert {c.outcome_label for c in contracts} == {"Yes", "No"}


def test_single_binary_native_id_is_condition_id():
    contracts = _map("elon-mars")
    condition_id = EVENTS_BY_SLUG["elon-mars"]["markets"][0]["conditionId"]
    assert all(c.exchange_event_native_id == condition_id for c in contracts)


# ── negRisk=false: multiple independent binary markets (ladder) ───────────────

def test_ladder_contract_count():
    contracts = _map("kraken-ipo-by")
    # 3 markets × 2 sides = 6 binary contracts
    assert len(contracts) == 6


def test_ladder_contract_types():
    contracts = _map("kraken-ipo-by")
    assert all(c.contract_type == ContractType.BINARY for c in contracts)


def test_ladder_native_ids_are_per_market_condition_ids():
    contracts = _map("kraken-ipo-by")
    native_ids = {c.exchange_event_native_id for c in contracts}
    market_condition_ids = {m["conditionId"] for m in EVENTS_BY_SLUG["kraken-ipo-by"]["markets"]}
    assert native_ids == market_condition_ids


def test_ladder_event_titles_use_market_questions():
    contracts = _map("kraken-ipo-by")
    titles = {c.event_title for c in contracts}
    assert "Kraken IPO in 2025?" in titles
    assert "Kraken IPO by March 31, 2026?" in titles


# ── negRisk=true: grouped as MULTI_OUTCOME ────────────────────────────────────

def test_neg_risk_contract_count():
    contracts = _map("harvey-weinstein-prison-time")
    # 3 markets, negRisk=true → 3 MULTI_OUTCOME contracts (one per market, Yes side only)
    assert len(contracts) == 3


def test_neg_risk_contract_types():
    contracts = _map("harvey-weinstein-prison-time")
    assert all(c.contract_type == ContractType.MULTI_OUTCOME for c in contracts)


def test_neg_risk_shared_native_id():
    contracts = _map("harvey-weinstein-prison-time")
    # All contracts share the event slug as native_id → all map to one event
    assert all(c.exchange_event_native_id == "harvey-weinstein-prison-time" for c in contracts)


def test_neg_risk_event_title_is_event_title():
    contracts = _map("harvey-weinstein-prison-time")
    assert all(c.event_title == "Harvey Weinstein prison time?" for c in contracts)


def test_neg_risk_outcome_labels_use_group_item_title():
    contracts = _map("harvey-weinstein-prison-time")
    labels = {c.outcome_label for c in contracts}
    assert labels == {"No Prison Time", "<5 years", "5-10 years"}


def test_neg_risk_security_ids_use_yes_token():
    contracts = _map("harvey-weinstein-prison-time")
    # exchange_security_id should be {conditionId}:{yesTokenId} (index 0)
    markets = EVENTS_BY_SLUG["harvey-weinstein-prison-time"]["markets"]
    expected_ids = {
        f"{m['conditionId']}:{json.loads(m['clobTokenIds'])[0]}"
        for m in markets
    }
    assert {c.exchange_security_id for c in contracts} == expected_ids


# ── Volume ───────────────────────────────────────────────────────────────────

def test_binary_event_volume():
    contracts = _map("elon-mars")
    assert all(c.event_volume == 1500.0 for c in contracts)


def test_neg_risk_event_volume_is_sum_of_markets():
    contracts = _map("harvey-weinstein-prison-time")
    assert all(c.event_volume == 5000.0 for c in contracts)


def test_ladder_event_volume_per_market():
    contracts = _map("kraken-ipo-by")
    # Each binary market becomes its own event, so it gets only its market's volumeNum
    market_vols = {m["conditionId"]: m["volumeNum"] for m in EVENTS_BY_SLUG["kraken-ipo-by"]["markets"]}
    for c in contracts:
        assert c.event_volume == market_vols[c.exchange_event_native_id]


# ── Entity creation ───────────────────────────────────────────────────────────

def test_entity_creation_neg_risk(stub_registry, stub_db, mock_anthropic):
    contracts = _map("harvey-weinstein-prison-time")
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    # All 3 contracts share one native_id → 1 event
    assert result.events_created == 1
    assert result.securities_created == 3
    assert result.listings_created == 3
    assert result.event_contracts_created == 3


def test_entity_creation_ladder(stub_registry, stub_db, mock_anthropic):
    contracts = _map("kraken-ipo-by")
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    # 3 markets with different native_ids → 3 events, each with Yes+No
    assert result.events_created == 3
    assert result.securities_created == 6
    assert result.listings_created == 6
    assert result.event_contracts_created == 6
