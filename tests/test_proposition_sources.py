import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock

from classifier.adapters import kalshi, polymarket_intl, polymarket_us
from classifier.adapters.hyperliquid import HyperliquidAdapter, OutcomeTemplate
from classifier.adapters.types import proposition_source_hash
from classifier.stages.entities import _listing_key, _sync_contract_sources, create_entities
from classifier.stages.fetch import contract_hash

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def _kalshi_contracts():
    adapter = kalshi.KalshiAdapter()
    return [c for e in _load("kalshi_events.json")["events"] for c in kalshi._with_sources(e, adapter._map_event(2, e))]


def _pm_intl_contracts():
    data = _load("polymarket_intl_events.json")
    events = data if isinstance(data, list) else data["events"]
    adapter = polymarket_intl.PolymarketIntlAdapter()
    return [c for e in events for c in polymarket_intl._with_sources(e, adapter._map_event(1, e))]


def _pm_us_contracts():
    data = _load("polymarket_us_events.json")
    events = data if isinstance(data, list) else data["events"]
    adapter = polymarket_us.PolymarketUsAdapter()
    return [c for e in events for c in polymarket_us._with_sources(e, adapter._map_event(4, e, {}))]


def _hyperliquid_contracts():
    meta = _load("hyperliquid_outcome_meta.json")
    templates = _load("hyperliquid_outcome_templates.json")
    adapter = HyperliquidAdapter()
    outcomes = {o["outcome"]: o for o in meta["outcomes"]}
    return adapter._map_all(3, outcomes, meta["questions"], {t.id: t for t in map(OutcomeTemplate, templates)}, {})


class TestKalshi:
    def test_every_contract_carries_its_market_rules_and_event(self):
        contracts = _kalshi_contracts()
        assert contracts and all(c.proposition_source["venue"] == "KALSHI" for c in contracts)
        for c in contracts:
            source = c.proposition_source
            assert source["market"]["ticker"] == c.exchange_security_id.split(":")[0]
            assert source["event"]["event_ticker"]
            assert source["side"] in ("yes", "no")

    def test_keeps_the_rules_text_and_the_market_close(self):
        source = next(c.proposition_source for c in _kalshi_contracts() if c.exchange_security_id == "KXELONMARS-99:yes")
        assert source["market"]["rules_primary"].startswith("If Elon Musk visits Mars")
        assert source["market"]["close_time"] == "2099-08-01T04:59:00Z"


class TestPolymarket:
    def test_intl_sources_name_their_condition_and_outcome(self):
        contracts = _pm_intl_contracts()
        assert contracts
        for c in contracts:
            assert c.proposition_source["market"]["conditionId"] == c.exchange_security_id.split(":")[0]
            assert c.proposition_source["outcome"] == c.outcome_label

    def test_us_sources_drop_side_prices(self):
        contracts = _pm_us_contracts()
        assert contracts
        for c in contracts:
            assert c.proposition_source["leg"] in ("long", "short")
            assert c.proposition_source["sides"]
            assert all("price" not in side for side in c.proposition_source["sides"])


def test_hyperliquid_sources_keep_the_structured_description():
    contracts = _hyperliquid_contracts()
    price_binary = [c for c in contracts if "class:priceBinary" in c.proposition_source["outcome"].get("description", "")]
    assert price_binary
    assert "targetPrice:" in price_binary[0].proposition_source["outcome"]["description"]


def test_no_source_carries_volume_or_price_fields():
    moving = {"volume", "volume24hr", "volumeNum", "volume_fp", "volume_24h_fp", "price", "last_price_dollars",
              "yes_bid_dollars", "yes_ask_dollars", "liquidity_dollars", "open_interest_fp"}
    for c in _kalshi_contracts() + _pm_intl_contracts() + _pm_us_contracts() + _hyperliquid_contracts():
        text = json.dumps(c.proposition_source)
        assert not any(f'"{key}"' in text for key in moving), c.exchange_security_id


def test_a_source_change_changes_the_contract_hash():
    c = _kalshi_contracts()[0]
    changed = dataclasses.replace(c, proposition_source={**c.proposition_source, "side": "other"})
    assert contract_hash(changed) != contract_hash(c)
    assert proposition_source_hash(c.proposition_source) == proposition_source_hash(json.loads(json.dumps(c.proposition_source)))


def test_entities_store_each_contracts_source(stub_registry, stub_db):
    contracts = _kalshi_contracts()
    create_entities(stub_registry, MagicMock(), contracts, db=stub_db, canonicalize_enabled=False)
    stored = stub_registry._contract_sources
    assert len(stored) == len({c.exchange_security_id for c in contracts})
    one = next(iter(stored.values()))
    assert one["exchange_id"] == 2 and one["source"]["venue"] == "KALSHI" and one["source_hash"]


def test_a_security_listed_twice_stores_one_source_or_none_when_they_differ(stub_registry):
    a, b, c = _kalshi_contracts()[:3]
    same_as_a = dataclasses.replace(a, exchange_security_id=a.exchange_security_id + "-dup")
    security_id_by_key = {_listing_key(a): 1, _listing_key(same_as_a): 1, _listing_key(b): 2, _listing_key(c): 2}
    assert a.proposition_source != b.proposition_source != c.proposition_source

    stored = _sync_contract_sources(stub_registry, [a, same_as_a, b, c], security_id_by_key)

    assert stored == 1
    assert set(stub_registry._contract_sources) == {1}
