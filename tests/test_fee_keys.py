import dataclasses

from classifier.adapters.polymarket_intl import PolymarketIntlAdapter
from classifier.adapters.polymarket_us import _sport_slug
from classifier.adapters.types import AdapterContract
from classifier.stages.entities import create_entities
from gnomepy.registry.types import AssetClass, ContractType, SecurityType

# The keys the registry's fee schedules target (sm.listing.fee_group / fee_event) are venue facts: the registry's fees
# sync writes the schedules, and a key with none pays the exchange default.


def test_polymarket_intl_markets_are_keyed_by_fee_type_when_fees_are_on():
    schedule = {"exponent": 1, "rate": 0.04, "takerOnly": True, "rebateRate": 0.25}
    fee_group = PolymarketIntlAdapter._fee_group
    assert fee_group({"feeType": "politics_fees", "feesEnabled": True, "feeSchedule": schedule}) == "politics_fees"
    assert fee_group({"feeType": "zero_fees", "feesEnabled": True, "feeSchedule": {**schedule, "rate": 0}}) == "zero_fees"
    assert fee_group({"feeType": "politics_fees", "feesEnabled": False}) is None
    assert fee_group({"feeType": None, "feesEnabled": True}) is None


def test_polymarket_us_fee_group_is_the_sport():
    event = {"tags": [{"slug": "sports"}, {"slug": "wtt", "league": {"slug": "wtt"}},
                      {"slug": "table-tennis", "sport": {"slug": "table-tennis"}}]}
    assert _sport_slug(event) == "table-tennis"
    assert _sport_slug({"tags": [{"slug": "politics"}]}) is None


def _contract(outcome: str, **fees) -> AdapterContract:
    return AdapterContract(
        exchange_id=2, exchange_security_id=f"T-1:{outcome.lower()}", exchange_security_symbol=f"T -- {outcome}",
        base_currency="USDC", quote_currency="USDC", settle_currency="USDC",
        security_type=SecurityType.EVENT_CONTRACT, contract_type=ContractType.BINARY, asset_class=AssetClass.PREDICTION,
        inverse=False, is_quanto=False, 
        event_title="T?", outcome_label=outcome, exchange_event_native_id="T-1", security_symbol=f"KX-T-1-{outcome}".upper(),
        **fees,
    )


def test_listings_carry_fee_keys_and_follow_the_venue(stub_registry, stub_db, mock_canonicalizer):
    contracts = [_contract(side, fee_group="KXT", fee_event="KXT-A") for side in ("Yes", "No")]
    create_entities(stub_registry, mock_canonicalizer, contracts, db=stub_db)
    assert set(stub_registry._fee_keys_by_listing.values()) == {("KXT", "KXT-A")}

    moved = [dataclasses.replace(c, fee_event="KXT-B") for c in contracts]
    create_entities(stub_registry, mock_canonicalizer, moved, db=stub_db)
    assert set(stub_registry._fee_keys_by_listing.values()) == {("KXT", "KXT-B")}


def test_a_lifted_event_override_keeps_its_fee_event(stub_registry, stub_db, mock_canonicalizer):
    create_entities(stub_registry, mock_canonicalizer, [_contract("Yes", fee_group="KXT", fee_event="KXT-A")], db=stub_db)
    create_entities(stub_registry, mock_canonicalizer, [_contract("Yes")], db=stub_db)
    assert set(stub_registry._fee_keys_by_listing.values()) == {(None, "KXT-A")}


