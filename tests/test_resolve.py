import dataclasses

import pytest

from classifier.stages.resolve import detect_resolved_events
from gnomepy.registry.types import EventContract, Listing, Security
from scripts.testing import StubDB, StubRegistry


def _make_security(registry: StubRegistry, security_id: int, symbol: str, active: bool = True) -> Security:
    s = Security(
        security_id=security_id,
        symbol=symbol,
        type=4,
        contract_type=7,
        asset_class=5,
        base_currency_id=None,
        quote_currency_id=None,
        settle_currency_id=None,
        inverse=False,
        is_quanto=False,
        expiry=None,
        strike_price=None,
        active=active,
        underlying_security_id=None,
        description=None,
        date_modified="",
        date_created="",
    )
    registry._securities.append(s)
    return s


def _make_listing(registry: StubRegistry, listing_id: int, security_id: int, exchange_id: int, exchange_security_id: str, active: bool = True) -> Listing:
    l = Listing(
        listing_id=listing_id,
        security_id=security_id,
        exchange_id=exchange_id,
        exchange_security_id=exchange_security_id,
        exchange_security_symbol=exchange_security_id,
        active=active,
        date_modified="",
        date_created="",
    )
    registry._listings.append(l)
    return l


def _make_event_contract(registry: StubRegistry, ec_id: int, event_id: int, security_id: int) -> EventContract:
    ec = EventContract(
        event_contract_id=ec_id,
        event_id=event_id,
        security_id=security_id,
        outcome_label="Yes",
        date_created="",
    )
    registry._event_contracts.append(ec)
    return ec


def _seed_event(registry: StubRegistry, event_id: int, title: str = "Test Event") -> None:
    from gnomepy.registry.types import Event
    ev = Event(
        event_id=event_id,
        title=title,
        description=None,
        category=None,
        tags=None,
        resolved=False,
        resolved_at=None,
        expiry=None,
        date_modified="",
        date_created="",
    )
    registry._events.append(ev)


@pytest.fixture
def registry():
    return StubRegistry()


@pytest.fixture
def db(registry):
    return StubDB(registry)


def test_no_resolved_outcomes(registry, db):
    result = detect_resolved_events({}, registry, db)
    assert result == {"events_resolved": 0, "securities_deactivated": 0, "listings_deactivated": 0}


def test_no_matching_listings(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_event_contract(registry, 1, 1, 10)

    result = detect_resolved_events({1: {"OTHER-TICKER:yes"}}, registry, db)

    assert result["securities_deactivated"] == 0
    assert result["listings_deactivated"] == 0
    assert result["events_resolved"] == 0
    assert registry._securities[0].active is True


def test_single_outcome_resolves_not_event(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_security(registry, 11, "NO-COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_listing(registry, 101, 11, 1, "TICKER:no")
    _make_event_contract(registry, 1, 1, 10)
    _make_event_contract(registry, 2, 1, 11)

    result = detect_resolved_events({1: {"TICKER:yes"}}, registry, db)

    assert result["securities_deactivated"] == 1
    assert result["listings_deactivated"] == 1
    assert result["events_resolved"] == 0

    sids = {s.security_id: s for s in registry._securities}
    assert sids[10].active is False
    assert sids[11].active is True
    lids = {l.listing_id: l for l in registry._listings}
    assert lids[100].active is False
    assert lids[101].active is True
    assert registry._events[0].resolved is False


def test_all_outcomes_resolve_event(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_security(registry, 11, "NO-COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_listing(registry, 101, 11, 1, "TICKER:no")
    _make_event_contract(registry, 1, 1, 10)
    _make_event_contract(registry, 2, 1, 11)

    result = detect_resolved_events({1: {"TICKER:yes", "TICKER:no"}}, registry, db)

    assert result["securities_deactivated"] == 2
    assert result["listings_deactivated"] == 2
    assert result["events_resolved"] == 1

    for s in registry._securities:
        assert s.active is False
    for l in registry._listings:
        assert l.active is False
    assert registry._events[0].resolved is True
    assert registry._events[0].resolved_at is not None


def test_multi_exchange_partial_resolve(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_listing(registry, 100, 10, 1, "COND:TOKEN_YES")
    _make_listing(registry, 101, 10, 2, "TICKER:yes")
    _make_event_contract(registry, 1, 1, 10)

    result = detect_resolved_events({1: {"COND:TOKEN_YES"}}, registry, db)

    # listing on exchange 1 deactivated, but exchange 2 still active → security stays active
    assert result["listings_deactivated"] == 1
    assert result["securities_deactivated"] == 0
    assert registry._securities[0].active is True


def test_security_shared_between_events_stays_resolved_if_other_event_unresolved(registry, db):
    _seed_event(registry, 1, "Event A")
    _seed_event(registry, 2, "Event B")
    _make_security(registry, 10, "SHARED-YES")
    _make_security(registry, 11, "B-NO")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_listing(registry, 101, 11, 1, "TICKER-B:no")
    _make_event_contract(registry, 1, 1, 10)
    _make_event_contract(registry, 2, 2, 10)
    _make_event_contract(registry, 3, 2, 11)

    result = detect_resolved_events({1: {"TICKER:yes"}}, registry, db)

    assert result["securities_deactivated"] == 1
    # Event 1 has only security 10 → resolves once 10 is deactivated
    # Event 2 has securities 10 and 11 → stays active because 11 is still active
    assert result["events_resolved"] == 1

    events = {ev.event_id: ev for ev in registry._events}
    assert events[1].resolved is True
    assert events[2].resolved is False


def test_security_stays_active_while_other_exchange_listing_remains(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")   # exchange 1
    _make_listing(registry, 101, 10, 2, "COND:TOKEN")   # exchange 2 — still active
    _make_event_contract(registry, 1, 1, 10)

    result = detect_resolved_events({1: {"TICKER:yes"}}, registry, db)

    assert result["listings_deactivated"] == 1
    assert result["securities_deactivated"] == 0  # listing 101 still active on exchange 2
    assert result["events_resolved"] == 0

    lids = {l.listing_id: l for l in registry._listings}
    assert lids[100].active is False
    assert lids[101].active is True
    assert registry._securities[0].active is True


def test_already_inactive_listing_skipped(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes", active=False)
    _make_event_contract(registry, 1, 1, 10)

    result = detect_resolved_events({1: {"TICKER:yes"}}, registry, db)

    assert result["listings_deactivated"] == 0
    assert result["securities_deactivated"] == 0


def test_empty_resolved_set_for_exchange(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_event_contract(registry, 1, 1, 10)

    result = detect_resolved_events({1: set()}, registry, db)

    assert result["securities_deactivated"] == 0
    assert registry._securities[0].active is True
