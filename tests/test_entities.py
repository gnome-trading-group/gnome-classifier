import dataclasses

import pytest

from classifier.adapters.types import AdapterContract
from classifier.stages.entities import create_entities
from classifier.workers.sync_names import sync_exchange_names
from gnomepy.registry.types import AssetClass, ContractType, Security, SecurityType


def _make_contract(
    title: str,
    outcome: str,
    exchange_id: int = 1,
    native_id: str | None = None,
    symbol: str | None = None,
) -> AdapterContract:
    native_id = native_id or f"native:{title}"
    return AdapterContract(
        exchange_id=exchange_id,
        exchange_security_id=f"{native_id}:{outcome}",
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
        exchange_event_native_id=native_id,
        security_symbol=symbol or f"EX{exchange_id}-{native_id}-{outcome}".upper(),
    )


def _binary(title: str, exchange_id: int = 1, native_id: str | None = None) -> list[AdapterContract]:
    return [_make_contract(title, side, exchange_id, native_id) for side in ("Yes", "No")]


def test_create_entities_empty(stub_registry, stub_db, mock_converse):
    result = create_entities(stub_registry, mock_converse, [], db=stub_db)
    assert result.events_created == 0
    assert result.securities_created == 0


def test_create_entities_new_event(stub_registry, stub_db, mock_converse):
    result = create_entities(stub_registry, mock_converse, _binary("Will BTC hit 100k?"), db=stub_db)
    assert result.events_created == 1
    assert result.securities_created == 2
    assert result.listings_created == 2
    assert result.event_contracts_created == 2
    assert sorted(result.new_security_symbols) == ["EX1-NATIVE:WILL BTC HIT 100K?-NO", "EX1-NATIVE:WILL BTC HIT 100K?-YES"]
    event = stub_registry._events[0]
    assert (event.exchange_id, event.native_event_id) == (1, "native:Will BTC hit 100k?")
    assert event.title == "Will BTC hit 100k?"


def test_a_new_event_is_created_with_its_volume(stub_registry, stub_db, mock_converse, monkeypatch):
    sent = []
    create = stub_registry.bulk_create_events
    monkeypatch.setattr(stub_registry, "bulk_create_events", lambda items: sent.extend(items) or create(items))
    contracts = [dataclasses.replace(c, event_volume=4200.5) for c in _binary("Will BTC hit 100k?")]

    create_entities(stub_registry, mock_converse, contracts, db=stub_db)

    assert [item["volume_24h"] for item in sent] == [4200.5]


def test_identical_titles_on_two_exchanges_stay_separate(stub_registry, stub_db, mock_converse):
    contracts = _binary("Will BTC hit 100k?", exchange_id=1) + _binary("Will BTC hit 100k?", exchange_id=2)
    result = create_entities(stub_registry, mock_converse, contracts, db=stub_db)
    assert result.events_created == 2
    assert result.securities_created == 4
    assert len({l.security_id for l in stub_registry._listings}) == 4


def test_short_dated_windows_with_same_title_stay_separate(stub_registry, stub_db, mock_converse):
    contracts = (
        _binary("WTI Oil 15 min", exchange_id=2, native_id="KXWTI15M-26AUG240015")
        + _binary("WTI Oil 15 min", exchange_id=2, native_id="KXWTI15M-26AUG240045")
    )
    result = create_entities(stub_registry, mock_converse, contracts, db=stub_db)
    assert result.events_created == 2
    assert result.securities_created == 4


def test_resending_known_group_creates_nothing(stub_registry, stub_db, mock_converse):
    contracts = _binary("Will BTC hit 100k?")
    create_entities(stub_registry, mock_converse, contracts, db=stub_db)
    result = create_entities(stub_registry, mock_converse, contracts, db=stub_db)
    assert result.events_created == 0
    assert result.securities_created == 0
    assert result.listings_created == 0
    assert result.event_contracts_created == 0
    assert result.new_security_ids == []


def test_symbol_clash_between_different_listings_raises(stub_registry, stub_db, mock_converse):
    contracts = [
        _make_contract("Event A", "Yes", native_id="a", symbol="EX1-CLASH"),
        _make_contract("Event B", "Yes", native_id="b", symbol="EX1-CLASH"),
    ]
    with pytest.raises(ValueError, match="duplicate security symbol"):
        create_entities(stub_registry, mock_converse, contracts, db=stub_db)


def test_symbol_clash_with_already_listed_security_raises(stub_registry, stub_db, mock_converse):
    create_entities(stub_registry, mock_converse, [_make_contract("Event A", "Yes", native_id="a", symbol="EX1-CLASH")], db=stub_db)
    with pytest.raises(ValueError, match="duplicate security symbol"):
        create_entities(stub_registry, mock_converse, [_make_contract("Event B", "Yes", native_id="b", symbol="EX1-CLASH")], db=stub_db)


def test_unlisted_security_from_failed_attempt_is_reused(stub_registry, stub_db, mock_converse):
    contract = _make_contract("Event A", "Yes", native_id="a")
    stub_registry._securities.append(Security(
        security_id=999, symbol=contract.security_symbol, type=SecurityType.EVENT_CONTRACT,
        contract_type=ContractType.BINARY, asset_class=AssetClass.PREDICTION,
        base_currency_id=None, quote_currency_id=None, settle_currency_id=None,
        inverse=False, is_quanto=False, expiry=None, strike_price=None, active=True,
        underlying_security_id=None, description=None, date_modified="", date_created="",
    ))
    result = create_entities(stub_registry, mock_converse, [contract], db=stub_db)
    assert result.securities_created == 0
    assert result.new_security_ids == [999]
    assert stub_registry._listings[0].security_id == 999


def test_batch_never_deactivates_another_events_listings(stub_registry, stub_db, mock_converse):
    event_a = _binary("Event A", native_id="a")
    event_b = _binary("Event B", native_id="b")
    create_entities(stub_registry, mock_converse, event_a + event_b, db=stub_db)

    create_entities(stub_registry, mock_converse, event_a, db=stub_db)

    assert all(l.active for l in stub_registry._listings)


def test_removed_contract_of_known_event_is_deactivated(stub_registry, stub_db, mock_converse):
    contracts = [_make_contract("Who wins?", outcome, native_id="q") for outcome in ("Alice", "Bob", "Carol")]
    create_entities(stub_registry, mock_converse, contracts, db=stub_db)

    create_entities(stub_registry, mock_converse, contracts[:2], db=stub_db)

    active = {l.exchange_security_id: l.active for l in stub_registry._listings}
    assert active == {"q:Alice": True, "q:Bob": True, "q:Carol": False}


def test_market_mapped_to_a_new_event_is_not_linked_twice(stub_registry, stub_db, mock_converse, caplog):
    original = _binary("Will BTC hit 100k?", native_id="BTC-100K")
    create_entities(stub_registry, mock_converse, original, db=stub_db)
    moved = [
        AdapterContract(**{**c.__dict__, "exchange_event_native_id": "BTC-EVENT", "event_title": "BTC moved"})
        for c in original
    ]
    result = create_entities(stub_registry, mock_converse, moved, db=stub_db)
    assert result.event_contracts_created == 0
    assert sorted(len([ec for ec in stub_registry._event_contracts if ec.security_id == s.security_id])
                  for s in stub_registry._securities) == [1, 1]
    assert "already in event" in caplog.text


def test_event_ids_come_from_the_database_each_time(stub_registry, stub_db, mock_converse):
    # An event re-keyed to a market's native id (as a migration does) is found by that id, and the market stays put.
    create_entities(stub_registry, mock_converse, _binary("Will BTC hit 100k?", native_id="BTC-OLD"), db=stub_db)
    event = stub_registry._events[0]
    stub_registry._events[0] = dataclasses.replace(event, native_event_id="BTC-NEW")
    moved = [AdapterContract(**{**c.__dict__, "exchange_event_native_id": "BTC-NEW"})
             for c in _binary("Will BTC hit 100k?", native_id="BTC-OLD")]
    result = create_entities(stub_registry, mock_converse, moved, db=stub_db)
    assert [result.events_created, result.event_contracts_created] == [0, 0]
    assert {ec.event_id for ec in stub_registry._event_contracts} == {event.event_id}


def _renamed(c: AdapterContract, outcome: str) -> AdapterContract:
    """The same market (same exchange_security_id) after the venue renamed its outcome."""
    return dataclasses.replace(c, outcome_label=outcome, exchange_security_symbol=f"{c.event_title} -- {outcome}",
                               security_symbol=f"EX1-UCL-{outcome}".upper())


class TestRenamedOutcomes:
    def placeholders(self):
        return [_make_contract("UCL Champion", f"Team {x}", native_id="ucl", symbol=f"EX1-UCL-TEAM-{x}") for x in "AB"]

    def test_a_renamed_outcome_relabels_its_link_and_renames_its_security_and_listing(
            self, stub_registry, stub_db, mock_converse):
        team_a, team_b = self.placeholders()
        create_entities(stub_registry, mock_converse, [team_a, team_b], db=stub_db)
        ids = {s.symbol: s.security_id for s in stub_registry._securities}

        result = create_entities(stub_registry, mock_converse, [_renamed(team_a, "Fenerbahce"), team_b], db=stub_db)

        assert (result.securities_created, result.event_contracts_created) == (0, 0)
        assert (result.event_contracts_relabelled, result.securities_renamed, result.listings_renamed) == (1, 1, 1)
        sid = ids["EX1-UCL-TEAM-A"]
        assert next(s.symbol for s in stub_registry._securities if s.security_id == sid) == "EX1-UCL-FENERBAHCE"
        assert next(l.exchange_security_symbol for l in stub_registry._listings
                    if l.security_id == sid) == "UCL Champion -- Fenerbahce"
        assert next(ec.outcome_label for ec in stub_registry._event_contracts if ec.security_id == sid) == "Fenerbahce"
        assert next(ec.outcome_label for ec in stub_registry._event_contracts
                    if ec.security_id == ids["EX1-UCL-TEAM-B"]) == "Team B"

    def test_unchanged_names_are_not_patched(self, stub_registry, stub_db, mock_converse):
        contracts = self.placeholders()
        create_entities(stub_registry, mock_converse, contracts, db=stub_db)
        result = create_entities(stub_registry, mock_converse, contracts, db=stub_db)
        assert (result.event_contracts_relabelled, result.securities_renamed, result.listings_renamed) == (0, 0, 0)

    def test_a_rename_onto_a_symbol_another_security_holds_is_skipped(self, stub_registry, stub_db, mock_converse,
                                                                      caplog):
        team_a, team_b = self.placeholders()
        create_entities(stub_registry, mock_converse, [team_a, team_b], db=stub_db)
        onto_b = dataclasses.replace(_renamed(team_a, "Fenerbahce"), security_symbol=team_b.security_symbol)

        result = create_entities(stub_registry, mock_converse, [onto_b, team_b], db=stub_db)

        assert (result.event_contracts_relabelled, result.securities_renamed, result.listings_renamed) == (1, 0, 1)
        assert sorted(s.symbol for s in stub_registry._securities) == ["EX1-UCL-TEAM-A", "EX1-UCL-TEAM-B"]
        assert "symbol rename clashes" in caplog.text

    def test_a_market_moved_to_another_event_is_not_relabelled(self, stub_registry, stub_db, mock_converse):
        original = _binary("Will BTC hit 100k?", native_id="BTC-100K")
        create_entities(stub_registry, mock_converse, original, db=stub_db)
        moved = [dataclasses.replace(c, exchange_event_native_id="BTC-EVENT", outcome_label=f"{c.outcome_label}!")
                 for c in original]
        result = create_entities(stub_registry, mock_converse, moved, db=stub_db)
        assert result.event_contracts_relabelled == 0
        assert sorted(ec.outcome_label for ec in stub_registry._event_contracts) == ["No", "Yes"]


class TestSyncNames:
    def renamed_before_the_stage_kept_names(self, stub_registry, stub_db, mock_converse):
        team_a, team_b = TestRenamedOutcomes().placeholders()
        create_entities(stub_registry, mock_converse, [team_a, team_b], db=stub_db)
        return [_renamed(team_a, "Fenerbahce"), team_b]

    def test_a_dry_run_finds_the_renames_and_writes_nothing(self, stub_registry, stub_db, mock_converse):
        markets = self.renamed_before_the_stage_kept_names(stub_registry, stub_db, mock_converse)
        renames = sync_exchange_names(stub_registry, stub_db, markets, apply=False)
        assert [(r.was, r.patch["symbol"]) for r in renames.securities] == [("EX1-UCL-TEAM-A", "EX1-UCL-FENERBAHCE")]
        assert [r.was for r in renames.labels] == ["Team A"]
        assert [r.was for r in renames.listings] == ["UCL Champion -- Team A"]
        assert sorted(s.symbol for s in stub_registry._securities) == ["EX1-UCL-TEAM-A", "EX1-UCL-TEAM-B"]

    def test_apply_writes_them(self, stub_registry, stub_db, mock_converse, monkeypatch):
        markets = self.renamed_before_the_stage_kept_names(stub_registry, stub_db, mock_converse)
        monkeypatch.setattr("classifier.workers.sync_names.BATCH_SIZE", 1)
        sync_exchange_names(stub_registry, stub_db, markets, apply=True)
        assert sorted(s.symbol for s in stub_registry._securities) == ["EX1-UCL-FENERBAHCE", "EX1-UCL-TEAM-B"]
        assert sorted(ec.outcome_label for ec in stub_registry._event_contracts) == ["Fenerbahce", "Team B"]
        assert sync_exchange_names(stub_registry, stub_db, markets, apply=False).securities == []
