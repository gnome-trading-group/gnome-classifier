import dataclasses

import pytest

from classifier.stages.resolve import deactivate_closed_listings, resolve_settled_events
from gnomepy.registry.types import Event, EventContract, Listing, Security
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


def test_nothing_closed(registry, db):
    assert deactivate_closed_listings({}, registry, db) == {"securities_deactivated": 0, "listings_deactivated": 0}


def test_no_matching_listings(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_event_contract(registry, 1, 1, 10)

    result = deactivate_closed_listings({1: {"OTHER-TICKER:yes"}}, registry, db)

    assert result == {"securities_deactivated": 0, "listings_deactivated": 0}
    assert registry._securities[0].active is True


def test_closing_one_outcome_deactivates_only_it(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_security(registry, 11, "NO-COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_listing(registry, 101, 11, 1, "TICKER:no")
    _make_event_contract(registry, 1, 1, 10)
    _make_event_contract(registry, 2, 1, 11)

    result = deactivate_closed_listings({1: {"TICKER:yes"}}, registry, db)

    assert result == {"securities_deactivated": 1, "listings_deactivated": 1}
    assert {s.security_id: s.active for s in registry._securities} == {10: False, 11: True}
    assert {l.listing_id: l.active for l in registry._listings} == {100: False, 101: True}


def test_closing_every_outcome_leaves_the_event_unresolved(registry, db):
    # A venue saying its markets closed can be wrong (Polymarket US once listed every open market as closed), so only
    # settlement values resolve an event.
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_security(registry, 11, "NO-COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_listing(registry, 101, 11, 1, "TICKER:no")
    _make_event_contract(registry, 1, 1, 10)
    _make_event_contract(registry, 2, 1, 11)

    result = deactivate_closed_listings({1: {"TICKER:yes", "TICKER:no"}}, registry, db)

    assert result == {"securities_deactivated": 2, "listings_deactivated": 2}
    assert all(not l.active for l in registry._listings)
    assert registry._events[0].resolved is False


def test_security_stays_active_while_other_exchange_listing_remains(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "YES-COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes")
    _make_listing(registry, 101, 10, 2, "COND:TOKEN")
    _make_event_contract(registry, 1, 1, 10)

    result = deactivate_closed_listings({1: {"TICKER:yes"}}, registry, db)

    assert result == {"securities_deactivated": 0, "listings_deactivated": 1}
    assert {l.listing_id: l.active for l in registry._listings} == {100: False, 101: True}
    assert registry._securities[0].active is True


def test_already_inactive_listing_skipped(registry, db):
    _seed_event(registry, 1)
    _make_security(registry, 10, "COIN")
    _make_listing(registry, 100, 10, 1, "TICKER:yes", active=False)
    _make_event_contract(registry, 1, 1, 10)

    assert deactivate_closed_listings({1: {"TICKER:yes"}}, registry, db) == {
        "securities_deactivated": 0, "listings_deactivated": 0}


# ── Resolving on settlement ───────────────────────────────────────────────────

def _seed_binary(registry: StubRegistry, event_id: int, first_id: int) -> tuple[int, int]:
    _seed_event(registry, event_id, f"Event {event_id}")
    for offset, side in enumerate(("yes", "no")):
        security_id = first_id + offset
        _make_security(registry, security_id, f"S{security_id}")
        _make_listing(registry, security_id, security_id, 1, f"T{event_id}:{side}")
        _make_event_contract(registry, security_id, event_id, security_id)
    return first_id, first_id + 1


def test_event_resolves_once_every_outcome_is_settled(registry, db):
    yes, no = _seed_binary(registry, 1, 10)
    registry.bulk_patch_event_contracts([{"event_contract_id": yes, "settlement_price": 1}])

    assert resolve_settled_events({1}, set(), registry, db) == ([], [])
    assert registry._events[0].resolved is False

    registry.bulk_patch_event_contracts([{"event_contract_id": no, "settlement_price": 0}])
    assert resolve_settled_events({1}, set(), registry, db) == ([1], ["Event 1"])
    assert registry._events[0].resolved is True and registry._events[0].resolved_at is not None


def test_just_settled_counts_before_the_database_shows_it(registry, db):
    yes, no = _seed_binary(registry, 1, 10)
    registry.bulk_patch_event_contracts([{"event_contract_id": yes, "settlement_price": 1}])
    assert resolve_settled_events({1}, {no}, registry, db) == ([1], ["Event 1"])


def test_resolved_events_are_left_alone(registry, db):
    yes, no = _seed_binary(registry, 1, 10)
    registry.bulk_patch_event_contracts([{"event_contract_id": yes, "settlement_price": 1},
                                         {"event_contract_id": no, "settlement_price": 0}])
    registry.bulk_patch_events([{"event_id": 1, "resolved": True, "resolved_at": "earlier"}])
    assert resolve_settled_events({1}, set(), registry, db) == ([], [])
    assert registry._events[0].resolved_at == "earlier"
