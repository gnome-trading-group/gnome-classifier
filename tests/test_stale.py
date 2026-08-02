import dataclasses

import pytest

from classifier.stages.stale import deactivate_stale_events
from gnomepy.registry.types import Event, EventContract, ExchangeEvent, Listing, Security
from scripts.testing import StubDB, StubRegistry


def _seed_event(registry: StubRegistry, event_id: int, resolved: bool = False) -> None:
    registry._events.append(Event(
        event_id=event_id, title=f"Event {event_id}", description=None,
        category=None, tags=None, resolved=resolved, resolved_at=None,
        expiry=None, date_modified="", date_created="",
    ))


def _seed_exchange_event(registry: StubRegistry, event_id: int, exchange_id: int, native_event_id: str) -> None:
    registry._exchange_events.append(ExchangeEvent(
        exchange_event_id=event_id * 100 + exchange_id,
        exchange_id=exchange_id,
        event_id=event_id,
        native_event_id=native_event_id,
        raw_title="",
        date_created="",
    ))


def _seed_security(registry: StubRegistry, security_id: int, active: bool = True) -> None:
    registry._securities.append(Security(
        security_id=security_id, symbol=f"SYM-{security_id}", type=4,
        contract_type=7, asset_class=5, base_currency_id=None,
        quote_currency_id=None, settle_currency_id=None, inverse=False,
        is_quanto=False, expiry=None, strike_price=None, active=active,
        underlying_security_id=None, description=None,
        date_modified="", date_created="",
    ))


def _seed_listing(registry: StubRegistry, listing_id: int, security_id: int, exchange_id: int, exchange_security_id: str, active: bool = True) -> None:
    registry._listings.append(Listing(
        listing_id=listing_id, security_id=security_id, exchange_id=exchange_id,
        exchange_security_id=exchange_security_id,
        exchange_security_symbol=exchange_security_id,
        active=active, date_modified="", date_created="",
    ))


def _seed_event_contract(registry: StubRegistry, ec_id: int, event_id: int, security_id: int) -> None:
    registry._event_contracts.append(EventContract(
        event_contract_id=ec_id, event_id=event_id, security_id=security_id,
        outcome_label="Yes", date_created="",
    ))


@pytest.fixture
def registry():
    return StubRegistry()


@pytest.fixture
def db(registry):
    return StubDB(registry)


def test_empty_input(registry, db):
    result = deactivate_stale_events([], registry, db)
    assert result == {"events_resolved": 0, "securities_deactivated": 0, "listings_deactivated": 0, "resolved_event_ids": [], "resolved_event_names": []}


def test_unknown_native_key_is_skipped(registry, db):
    result = deactivate_stale_events([(1, "no-such-event")], registry, db)
    assert result == {"events_resolved": 0, "securities_deactivated": 0, "listings_deactivated": 0, "resolved_event_ids": [], "resolved_event_names": []}


def test_already_resolved_event_is_skipped(registry, db):
    _seed_event(registry, 1, resolved=True)
    _seed_exchange_event(registry, 1, exchange_id=1, native_event_id="evt-1")

    result = deactivate_stale_events([(1, "evt-1")], registry, db)

    assert result["events_resolved"] == 0
    assert result["listings_deactivated"] == 0


def test_single_event_fully_deactivated(registry, db):
    _seed_event(registry, 1)
    _seed_exchange_event(registry, 1, exchange_id=1, native_event_id="evt-1")
    _seed_security(registry, 10)
    _seed_security(registry, 11)
    _seed_listing(registry, 100, 10, exchange_id=1, exchange_security_id="sec-yes")
    _seed_listing(registry, 101, 11, exchange_id=1, exchange_security_id="sec-no")
    _seed_event_contract(registry, 1, event_id=1, security_id=10)
    _seed_event_contract(registry, 2, event_id=1, security_id=11)

    result = deactivate_stale_events([(1, "evt-1")], registry, db)

    assert result["events_resolved"] == 1
    assert result["securities_deactivated"] == 2
    assert result["listings_deactivated"] == 2
    assert result["resolved_event_ids"] == [1]
    assert "Event 1" in result["resolved_event_names"]
    for s in registry._securities:
        assert s.active is False
    for l in registry._listings:
        assert l.active is False
    assert registry._events[0].resolved is True
    assert registry._events[0].resolved_at is not None


def test_stale_on_one_exchange_leaves_other_exchange_intact(registry, db):
    _seed_event(registry, 1)
    _seed_exchange_event(registry, 1, exchange_id=1, native_event_id="evt-1")
    _seed_security(registry, 10)
    _seed_listing(registry, 100, 10, exchange_id=1, exchange_security_id="sec-poly")
    _seed_listing(registry, 101, 10, exchange_id=2, exchange_security_id="sec-kalshi")
    _seed_event_contract(registry, 1, event_id=1, security_id=10)

    result = deactivate_stale_events([(1, "evt-1")], registry, db)

    assert result["listings_deactivated"] == 1
    assert result["securities_deactivated"] == 0
    assert result["events_resolved"] == 0

    lids = {l.listing_id: l for l in registry._listings}
    assert lids[100].active is False
    assert lids[101].active is True
    assert registry._securities[0].active is True
    assert registry._events[0].resolved is False


def test_multiple_stale_events_processed_together(registry, db):
    _seed_event(registry, 1)
    _seed_exchange_event(registry, 1, exchange_id=1, native_event_id="evt-1")
    _seed_security(registry, 10)
    _seed_listing(registry, 100, 10, exchange_id=1, exchange_security_id="sec-1")
    _seed_event_contract(registry, 1, event_id=1, security_id=10)

    _seed_event(registry, 2)
    _seed_exchange_event(registry, 2, exchange_id=1, native_event_id="evt-2")
    _seed_security(registry, 20)
    _seed_listing(registry, 200, 20, exchange_id=1, exchange_security_id="sec-2")
    _seed_event_contract(registry, 2, event_id=2, security_id=20)

    result = deactivate_stale_events([(1, "evt-1"), (1, "evt-2")], registry, db)

    assert result["events_resolved"] == 2
    assert result["securities_deactivated"] == 2
    assert result["listings_deactivated"] == 2


def test_event_not_resolved_while_active_security_remains(registry, db):
    _seed_event(registry, 1)
    _seed_exchange_event(registry, 1, exchange_id=1, native_event_id="evt-1")
    _seed_security(registry, 10)
    _seed_security(registry, 11)
    _seed_listing(registry, 100, 10, exchange_id=1, exchange_security_id="sec-yes")
    _seed_listing(registry, 101, 11, exchange_id=2, exchange_security_id="sec-no-kalshi")
    _seed_event_contract(registry, 1, event_id=1, security_id=10)
    _seed_event_contract(registry, 2, event_id=1, security_id=11)

    # Event on exchange 1 is stale; security 11 still has an active listing on exchange 2
    result = deactivate_stale_events([(1, "evt-1")], registry, db)

    assert result["listings_deactivated"] == 1
    assert result["securities_deactivated"] == 1  # security 10 lost its only listing
    assert result["events_resolved"] == 0  # security 11 is still active
    assert registry._events[0].resolved is False
