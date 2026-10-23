from datetime import datetime, timedelta, timezone

import pytest

from classifier.constants import REACTIVATION_WARNING_THRESHOLD
from classifier.stages.reactivate import reactivate_listings
from gnomepy.registry.types import Event, EventContract, Listing, Security, SecurityType

LOOKBACK = timedelta(days=30)


def _seed(registry, listing_id: int, exchange_security_id: str, event_id: int = 1, exchange_id: int = 6,
          active: bool = False):
    security_id = listing_id + 1000
    registry._securities.append(Security(
        security_id=security_id, symbol=f"S{security_id}", type=SecurityType.EVENT_CONTRACT, description=None,
        contract_type=7, base_currency_id=None, quote_currency_id=None, settle_currency_id=None, inverse=False,
        is_quanto=False, expiry=None, strike_price=None, active=active, underlying_security_id=None, asset_class=5,
        date_modified="", date_created=""))
    registry._listings.append(Listing(
        listing_id=listing_id, security_id=security_id, exchange_id=exchange_id,
        exchange_security_id=exchange_security_id, exchange_security_symbol="", date_modified="", date_created="",
        active=active))
    registry._event_contracts.append(EventContract(
        event_contract_id=security_id, event_id=event_id, security_id=security_id, outcome_label="Yes",
        date_created=""))
    return security_id


def _seed_event(registry, event_id: int = 1):
    registry._events.append(Event(event_id=event_id, title=f"Event {event_id}", description=None, category=None,
                                  tags=None, resolved=False, resolved_at=None, expiry=None, date_modified="",
                                  date_created=""))


@pytest.fixture
def falsely_closed(stub_registry):
    """A Polymarket US market both of whose legs a bad feed retired, as on 2026-10-22."""
    _seed_event(stub_registry)
    _seed(stub_registry, 1, "aec-cs2:long")
    _seed(stub_registry, 2, "aec-cs2:short")
    return stub_registry


def test_listings_the_feed_lists_as_open_come_back(falsely_closed, stub_db):
    result = reactivate_listings({6}, {"6:aec-cs2:long": "h", "6:aec-cs2:short": "h"}, falsely_closed, stub_db,
                                 LOOKBACK)
    assert result == {"listings_reactivated": 2, "securities_reactivated": 2, "events_unresolved": 0}
    assert all(l.active for l in falsely_closed._listings)
    assert all(s.active for s in falsely_closed._securities)


def test_a_resolved_event_is_unresolved(falsely_closed, stub_db):
    falsely_closed.bulk_patch_events([{"event_id": 1, "resolved": True, "resolved_at": "earlier"}])
    result = reactivate_listings({6}, {"6:aec-cs2:long": "h"}, falsely_closed, stub_db, LOOKBACK)
    assert result["events_unresolved"] == 1
    assert (falsely_closed._events[0].resolved, falsely_closed._events[0].resolved_at) == (False, None)


def test_settled_outcomes_stay_retired(falsely_closed, stub_db):
    falsely_closed.bulk_patch_event_contracts([{"event_contract_id": 1001, "settlement_price": 0}])
    reactivate_listings({6}, {"6:aec-cs2:long": "h", "6:aec-cs2:short": "h"}, falsely_closed, stub_db, LOOKBACK)
    assert {l.exchange_security_id: l.active for l in falsely_closed._listings} == {
        "aec-cs2:long": False, "aec-cs2:short": True}


def test_listings_retired_before_the_lookback_stay_retired(falsely_closed, stub_db):
    falsely_closed._listing_modified_at[1] = datetime.now(timezone.utc) - timedelta(days=31)
    reactivate_listings({6}, {"6:aec-cs2:long": "h", "6:aec-cs2:short": "h"}, falsely_closed, stub_db, LOOKBACK)
    assert {l.exchange_security_id: l.active for l in falsely_closed._listings} == {
        "aec-cs2:long": False, "aec-cs2:short": True}


def test_listings_missing_from_the_fetch_or_of_another_exchange_stay_retired(falsely_closed, stub_db):
    _seed(falsely_closed, 3, "KX-1:yes", exchange_id=5)
    # A failed fetch leaves its exchange out of the ids passed in, even though its old keys are still known.
    result = reactivate_listings({6}, {"6:aec-cs2:long": "h", "5:KX-1:yes": "h"}, falsely_closed, stub_db, LOOKBACK)
    assert result["listings_reactivated"] == 1
    assert {l.exchange_security_id: l.active for l in falsely_closed._listings} == {
        "aec-cs2:long": True, "aec-cs2:short": False, "KX-1:yes": False}


def test_many_reactivations_at_once_are_a_warning(stub_registry, stub_db, caplog):
    keys = {}
    for i in range(REACTIVATION_WARNING_THRESHOLD):
        _seed(stub_registry, 10 + i, f"m{i}:long")
        keys[f"6:m{i}:long"] = "h"
    reactivate_listings({6}, keys, stub_registry, stub_db, LOOKBACK)
    assert "retired listings as open again" in caplog.text


def test_nothing_to_reactivate(falsely_closed, stub_db):
    assert reactivate_listings({6}, {}, falsely_closed, stub_db, LOOKBACK) == {
        "listings_reactivated": 0, "securities_reactivated": 0, "events_unresolved": 0}
