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

def test_neg_risk_single_market_uses_event_slug():
    """When only one negRisk market remains open, native_id stays as event_slug (not conditionId)."""
    import copy
    event = copy.deepcopy(EVENTS_BY_SLUG["harvey-weinstein-prison-time"])
    event["markets"] = [event["markets"][0]]
    contracts = adapter._map_event(EXCHANGE_ID, event)
    assert len(contracts) == 1
    assert contracts[0].exchange_event_native_id == "harvey-weinstein-prison-time"
    assert contracts[0].contract_type == ContractType.MULTI_OUTCOME


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


# ── Sports markets ───────────────────────────────────────────────────────────

def test_sports_contract_count():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    assert len(contracts) == 8


def test_sports_contract_types():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    assert all(c.contract_type == ContractType.BINARY for c in contracts)


def test_sports_native_ids_are_per_market():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    native_ids = {c.exchange_event_native_id for c in contracts}
    market_condition_ids = {m["conditionId"] for m in EVENTS_BY_SLUG["wnba-sea-nyl-2026-08-05"]["markets"]}
    assert native_ids == market_condition_ids


def test_sports_moneyline_event_title():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    moneyline_cid = next(
        m["conditionId"] for m in EVENTS_BY_SLUG["wnba-sea-nyl-2026-08-05"]["markets"]
        if m.get("sportsMarketType") == "moneyline"
    )
    moneyline = [c for c in contracts if c.exchange_event_native_id == moneyline_cid]
    assert len(moneyline) == 2
    assert all(c.event_title == "Seattle Storm vs. New York Liberty" for c in moneyline)


def test_sports_spread_event_title():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    spread = [c for c in contracts if "Spread" in c.event_title]
    assert len(spread) == 2
    assert all(c.event_title == "Seattle Storm vs. New York Liberty: Spread -9.5" for c in spread)


def test_sports_total_event_title():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    total = [c for c in contracts if "Total" in c.event_title]
    assert len(total) == 2
    assert all(c.event_title == "Seattle Storm vs. New York Liberty: Total 181.5" for c in total)


def test_sports_player_prop_event_title():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    props = [c for c in contracts if "Breanna Stewart" in c.event_title]
    assert len(props) == 2
    assert all(c.event_title == "Seattle Storm vs. New York Liberty: Breanna Stewart Points 22.5" for c in props)


def test_sports_event_titles_are_distinct():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    titles = {c.event_title for c in contracts}
    assert len(titles) == 4


def test_non_sports_binary_event_title_unchanged():
    contracts = _map("elon-mars")
    assert all(c.event_title == "Will Elon Musk visit Mars before 2030?" for c in contracts)


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


def test_entity_creation_sports(stub_registry, stub_db, mock_anthropic):
    contracts = _map("wnba-sea-nyl-2026-08-05")
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    # 4 distinct event_titles → 4 events, each with 2 outcomes
    assert result.events_created == 4
    assert result.securities_created == 8
    assert result.listings_created == 8
    assert result.event_contracts_created == 8


# ── Category from tags ────────────────────────────────────────────────────────

def test_binary_event_category_from_tags():
    contracts = _map("elon-mars")
    assert all(c.event_category == "Science" for c in contracts)


def test_neg_risk_event_category_from_tags():
    contracts = _map("harvey-weinstein-prison-time")
    assert all(c.event_category == "Legal" for c in contracts)


def test_ladder_event_category_from_tags():
    contracts = _map("kraken-ipo-by")
    assert all(c.event_category == "Crypto" for c in contracts)


def test_sports_event_category_from_tags():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    assert all(c.event_category == "Sports" for c in contracts)


def test_no_tags_yields_none_category():
    import copy
    event = copy.deepcopy(EVENTS_BY_SLUG["elon-mars"])
    event.pop("tags", None)
    contracts = adapter._map_event(EXCHANGE_ID, event)
    assert all(c.event_category is None for c in contracts)


# ── Per-market description ────────────────────────────────────────────────────

def test_ladder_contracts_get_per_market_description():
    contracts = _map("kraken-ipo-by")
    market_descs = {m["conditionId"]: m["description"] for m in EVENTS_BY_SLUG["kraken-ipo-by"]["markets"]}
    for c in contracts:
        assert c.event_description == market_descs[c.exchange_event_native_id]


def test_neg_risk_contracts_get_event_level_description():
    # negRisk per-market desc equals event desc in fixture — verifies we use event_description
    contracts = _map("harvey-weinstein-prison-time")
    event_desc = EVENTS_BY_SLUG["harvey-weinstein-prison-time"]["description"]
    assert all(c.event_description == event_desc for c in contracts)


def test_single_binary_description_prefers_market_description():
    contracts = _map("elon-mars")
    market_desc = EVENTS_BY_SLUG["elon-mars"]["markets"][0]["description"]
    assert all(c.event_description == market_desc for c in contracts)


# ── Event-level endDate fallback ──────────────────────────────────────────────

def test_neg_risk_event_expiry_falls_back_to_event_end_date():
    contracts = _map("iowa-governor-2026")
    assert len(contracts) == 2
    assert all(c.event_expiry == "2026-11-03T00:00:00Z" for c in contracts)


def test_binary_event_expiry_falls_back_to_event_end_date():
    contracts = _map("bitcoin-200k")
    assert len(contracts) == 2
    assert all(c.event_expiry == "2027-01-01T00:00:00Z" for c in contracts)


def test_market_end_date_preferred_over_event_level():
    contracts = _map("elon-mars")
    assert all(c.event_expiry == "2030-01-01T00:00:00Z" for c in contracts)


def test_market_end_date_takes_precedence_over_event_end_date():
    import copy
    event = copy.deepcopy(EVENTS_BY_SLUG["elon-mars"])
    event["endDate"] = "2099-01-01T00:00:00Z"
    contracts = adapter._map_event(EXCHANGE_ID, event)
    assert all(c.event_expiry == "2030-01-01T00:00:00Z" for c in contracts)


# ── Tick size and min notional ────────────────────────────────────────────────

def test_binary_tick_size():
    contracts = _map("elon-mars")
    assert all(c.tick_size == 10_000_000 for c in contracts)  # 0.01 * 1e9


def test_binary_min_notional():
    contracts = _map("elon-mars")
    assert all(c.min_notional == 5_000_000_000_000_000 for c in contracts)  # 5 * 1e9 * 1e6


def test_neg_risk_tick_size():
    contracts = _map("harvey-weinstein-prison-time")
    assert all(c.tick_size == 10_000_000 for c in contracts)


def test_neg_risk_min_notional():
    contracts = _map("harvey-weinstein-prison-time")
    assert all(c.min_notional == 5_000_000_000_000_000 for c in contracts)


def test_ladder_per_market_tick_size():
    contracts = _map("kraken-ipo-by")
    markets = EVENTS_BY_SLUG["kraken-ipo-by"]["markets"]
    expected = {
        markets[0]["conditionId"]: 10_000_000,   # 0.01 * 1e9
        markets[1]["conditionId"]: 1_000_000,    # 0.001 * 1e9
        markets[2]["conditionId"]: 10_000_000,
    }
    for c in contracts:
        assert c.tick_size == expected[c.exchange_event_native_id]


def test_ladder_per_market_min_notional():
    contracts = _map("kraken-ipo-by")
    markets = EVENTS_BY_SLUG["kraken-ipo-by"]["markets"]
    expected = {
        markets[0]["conditionId"]: 5_000_000_000_000_000,    # 5 * 1e15
        markets[1]["conditionId"]: 10_000_000_000_000_000,   # 10 * 1e15
        markets[2]["conditionId"]: 5_000_000_000_000_000,
    }
    for c in contracts:
        assert c.min_notional == expected[c.exchange_event_native_id]


def test_sports_tick_size():
    contracts = _map("wnba-sea-nyl-2026-08-05")
    assert all(c.tick_size == 1_000_000 for c in contracts)  # 0.001 * 1e9


def test_missing_tick_size_falls_back_to_default():
    contracts = _map("bitcoin-200k")
    assert all(c.tick_size == 10_000_000 for c in contracts)


def test_missing_min_notional_falls_back_to_zero():
    contracts = _map("bitcoin-200k")
    assert all(c.min_notional == 0 for c in contracts)


def test_lot_size_unchanged():
    for slug in ["elon-mars", "kraken-ipo-by", "harvey-weinstein-prison-time", "wnba-sea-nyl-2026-08-05"]:
        contracts = _map(slug)
        assert all(c.lot_size == 1_000_000 for c in contracts)


# ── Listing spec update detection ─────────────────────────────────────────────

def test_listing_spec_updated_on_tick_size_change(stub_registry, stub_db, mock_anthropic):
    contracts = _map("elon-mars")
    result1 = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    assert result1.listing_specs_created == 2
    assert result1.listing_specs_updated == 0

    for c in contracts:
        c.tick_size = 1_000_000  # changed from 10_000_000
    result2 = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    assert result2.listing_specs_updated == 2
    assert result2.listing_specs_created == 0


def test_listing_spec_not_updated_when_unchanged(stub_registry, stub_db, mock_anthropic):
    contracts = _map("elon-mars")
    create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    result = create_entities(stub_registry, mock_anthropic, contracts, db=stub_db)
    assert result.listing_specs_created == 0
    assert result.listing_specs_updated == 0
