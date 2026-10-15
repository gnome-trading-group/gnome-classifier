import dataclasses

import click
import pytest
from gnomepy.registry.types import AssetClass, ContractType, Event, Exchange, SecurityType

from classifier.adapters.types import AdapterContract
from scripts import backfill_taxonomy

EXCHANGES = {
    code: Exchange(exchange_id=eid, exchange_code=code, exchange_name=code, region="", schema_type="",
                   date_modified="", date_created="")
    for code, eid in {"POLYMARKET_INTL": 1, "KALSHI": 2, "HYPERLIQUID": 3, "POLYMARKET_US": 4}.items()
}


def _event(event_id: int, exchange_id: int, native_id: str, category: str | None, tags=None, url=None) -> Event:
    return Event(event_id=event_id, title=f"event {event_id}", description=None, category=category, tags=tags,
                 resolved=False, resolved_at=None, expiry=None, date_modified="", date_created="",
                 exchange_id=exchange_id, native_event_id=native_id, native_url=url)


def _contract(exchange_id: int, native_id: str, label: str | None, mapped: str | None, tags: list[str]) -> AdapterContract:
    return AdapterContract(
        exchange_id=exchange_id, exchange_security_id=f"{native_id}:yes", exchange_security_symbol=native_id,
        base_currency="USD", quote_currency="USD", settle_currency="USD", security_type=SecurityType.EVENT_CONTRACT,
        contract_type=ContractType.BINARY, asset_class=AssetClass.PREDICTION, inverse=False, is_quanto=False,
        tick_size=1.0, lot_size=1.0, min_notional=0.0, contract_multiplier=1.0, event_title=native_id,
        outcome_label="yes", exchange_event_native_id=native_id, security_symbol=native_id,
        event_category=label, mapped_category=mapped, event_tags=tags,
    )


class _Registry:
    def __init__(self, events):
        self._events = events

    def get_event(self, resolved=None):
        return [e for e in self._events if resolved is None or e.resolved == resolved]


@pytest.fixture
def plan(monkeypatch):
    def _plan(events, contracts=(), series_tags=None, intl_tags=None, failed=()):
        monkeypatch.setattr(backfill_taxonomy, "fetch_exchanges", lambda registry: EXCHANGES)
        monkeypatch.setattr(backfill_taxonomy, "fetch_all", lambda exchanges: (list(contracts), list(failed)))
        monkeypatch.setattr(backfill_taxonomy.KalshiAdapter, "_fetch_series_tags", lambda self: series_tags or {})
        monkeypatch.setattr(backfill_taxonomy, "_fetch_polymarket_intl_tags", lambda slugs: intl_tags or {})
        return backfill_taxonomy.plan_backfill(_Registry(events))
    return _plan


def test_open_event_gets_mapped_category_and_exchange_tags(plan):
    events = [_event(1, 2, "KXHIGHNY-26OCT15", "Climate and Weather")]
    contracts = [_contract(2, "KXHIGHNY-26OCT15", "Climate and Weather", "WEATHER", ["daily temperature"])]
    patches, _, unknown = plan(events, contracts)
    assert patches == [{"event_id": 1, "category": "WEATHER", "tags": ["daily temperature"]}]
    assert not unknown


def test_closed_event_gets_category_only(plan):
    events = [_event(1, 2, "KXHIGHNY-26OCT01", "Climate and Weather", url="https://kalshi.com/markets/kxhighny/x")]
    patches, _, _ = plan(events)
    assert patches == [{"event_id": 1, "category": "WEATHER"}]


def test_closed_kalshi_ambiguous_label_uses_series_tags(plan):
    events = [
        _event(1, 2, "KXEARNINGSMENTIONAAPL-26OCT", "Mentions", url="https://kalshi.com/markets/kxearningsmentionaapl/x"),
        _event(2, 2, "KXTRUMPMENTION-26OCT", "Mentions", url="https://kalshi.com/markets/kxtrumpmention/x"),
    ]
    patches, _, unknown = plan(events, series_tags={"KXEARNINGSMENTIONAAPL": ["Earnings"]})
    assert patches == [{"event_id": 1, "category": "FINANCE"}, {"event_id": 2, "category": "OTHER"}]
    assert not unknown


def test_closed_polymarket_intl_uses_full_tag_list(plan):
    events = [_event(1, 1, "0xabc", "Brazil", url="https://polymarket.com/event/brazil-election")]
    patches, _, unknown = plan(events, intl_tags={"brazil-election": ["Brazil", "Elections", "Politics"]})
    assert patches == [{"event_id": 1, "category": "POLITICS"}]
    assert not unknown


def test_ambiguous_label_keeps_standardized_category(plan):
    events = [_event(1, 2, "KXMENTION-1", "TECHNOLOGY", url="https://kalshi.com/markets/kxmention/x")]
    contracts = [_contract(2, "KXMENTION-1", "Mentions", None, [])]
    patches, _, unknown = plan(events, contracts)
    assert patches == []
    assert not unknown


def test_unknown_category_values_are_reported_and_not_patched(plan):
    events = [
        _event(1, 2, "KXNEW-1", "Brand New Category", url="https://kalshi.com/markets/kxnew/x"),
        _event(2, 4, "slug", "astrology"),
        _event(3, 2, "KXHIGHNY-1", "Climate and Weather", url="https://kalshi.com/markets/kxhighny/x"),
    ]
    patches, _, unknown = plan(events)
    assert unknown == {("KALSHI", "Brand New Category"): 1, ("POLYMARKET_US", "astrology"): 1}
    assert patches == [{"event_id": 3, "category": "WEATHER"}]


def test_labels_naming_no_category_become_other(plan):
    events = [
        _event(1, 1, "0xdef", "United Nations", url="https://polymarket.com/event/next-un-secretary-general"),
        _event(2, 1, "0xghi", None, url="https://polymarket.com/event/untagged"),
        _event(3, 2, "KXNONE-1", None, url="https://kalshi.com/markets/kxnone/x"),
    ]
    patches, _, unknown = plan(events, intl_tags={"next-un-secretary-general": ["United Nations"], "untagged": []})
    assert not unknown
    assert patches == [{"event_id": i, "category": "OTHER"} for i in (1, 2, 3)]


def test_current_events_are_skipped(plan):
    events = [_event(1, 2, "KXHIGHNY-1", "WEATHER", tags=["daily temperature"])]
    contracts = [_contract(2, "KXHIGHNY-1", "Climate and Weather", "WEATHER", ["hourly temperature"])]
    patches, changes, _ = plan(events, contracts)
    assert patches == [] and not changes


def test_existing_tags_are_not_replaced(plan):
    events = [_event(1, 2, "KXHIGHNY-1", "Climate and Weather", tags=["weather"])]
    contracts = [_contract(2, "KXHIGHNY-1", "Climate and Weather", "WEATHER", ["daily temperature"])]
    patches, _, _ = plan(events, contracts)
    assert patches == [{"event_id": 1, "category": "WEATHER"}]


def test_closed_hyperliquid_event_keeps_its_category(plan):
    patches, _, unknown = plan([_event(1, 3, "#123", "OTHER")])
    assert patches == [] and not unknown


def test_resolved_events_are_left_alone(plan):
    resolved = dataclasses.replace(_event(1, 2, "KXHIGHNY-1", "Climate and Weather"), resolved=True)
    patches, changes, unknown = plan([resolved])
    assert patches == [] and not changes and not unknown


def test_failed_fetch_aborts(plan):
    with pytest.raises(click.ClickException):
        plan([_event(1, 2, "KXHIGHNY-1", "Climate and Weather")], failed=["KALSHI"])
