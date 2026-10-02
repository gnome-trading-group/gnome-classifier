import json
from pathlib import Path

from classifier.adapters import polymarket_us
from classifier.adapters.polymarket_us import PolymarketUsAdapter
from classifier.stages.entities import create_entities
from gnomepy.registry.types import ContractType

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "polymarket_us_events.json").read_text())
EVENTS_BY_SLUG = {e["slug"]: e for e in FIXTURE["events"]}

EXCHANGE_ID = 6
adapter = PolymarketUsAdapter()

FUTURES = "mlb-nlchamp-2026-09-27"
GAME = "cfb-librty-del-2026-10-02"
POLITICS = "usho-midterms-2026-11-03"
RESOLVED_ONLY = "uecl-ggk-zil-2026-07-30"


def _map(slug: str, volume_by_slug: dict[str, float] | None = None) -> list:
    return adapter._map_event(EXCHANGE_ID, EVENTS_BY_SLUG[slug], volume_by_slug or {})


def _by_market(contracts: list) -> dict[str, list]:
    out: dict[str, list] = {}
    for c in contracts:
        out.setdefault(c.exchange_event_native_id, []).append(c)
    return out


# ── Contract shape ────────────────────────────────────────────────────────────

def test_resolved_markets_excluded():
    contracts = _map(FUTURES)
    # 6 markets, 2 resolved → 4 open markets x long/short
    assert len(contracts) == 8
    assert not any("chc" in c.exchange_security_id or "phi" in c.exchange_security_id for c in contracts)


def test_event_with_only_resolved_markets_yields_nothing():
    assert _map(RESOLVED_ONLY) == []


def test_all_contracts_binary():
    contracts = [c for slug in EVENTS_BY_SLUG for c in _map(slug)]
    assert all(c.contract_type == ContractType.BINARY for c in contracts)


def test_security_ids_are_slug_long_short():
    contracts = _map(POLITICS)
    assert {c.exchange_security_id for c in contracts} == {
        "paccc-usho-midterms-2026-11-03-dem:long",
        "paccc-usho-midterms-2026-11-03-dem:short",
        "paccc-usho-midterms-2026-11-03-rep:long",
        "paccc-usho-midterms-2026-11-03-rep:short",
    }


def test_native_id_is_market_slug():
    for slug, group in _by_market(_map(GAME)).items():
        assert all(c.exchange_event_native_id == slug for c in group)
        assert len(group) == 2


def test_outcome_labels_come_from_market_sides():
    by_market = _by_market(_map(GAME))
    labels = {slug: {c.exchange_security_id.rsplit(":", 1)[1]: c.outcome_label for c in group} for slug, group in by_market.items()}
    assert labels["aec-cfb-librty-del-2026-10-02"] == {"long": "Flames", "short": "Fightin' Blue Hens"}
    assert labels["asc-cfb-librty-del-2026-10-02-pos-17pt5"] == {"long": "+17.50", "short": "-17.50"}
    assert labels["tsc-cfb-librty-del-2026-10-02-2q-10pt5"] == {"long": "Over", "short": "Under"}


def test_currency_is_usd():
    contracts = _map(POLITICS)
    assert all(c.base_currency == c.quote_currency == c.settle_currency == "USD" for c in contracts)


# ── Titles ────────────────────────────────────────────────────────────────────

def test_market_title_appended_to_event_title():
    contracts = [c for c in _map(FUTURES) if c.exchange_event_native_id.endswith("-atl")]
    assert all(c.event_title == "National League Champion: Atlanta Braves" for c in contracts)


def test_moneyline_title_not_duplicated():
    contracts = [c for c in _map(GAME) if c.exchange_event_native_id == "aec-cfb-librty-del-2026-10-02"]
    assert all(c.event_title == "Liberty vs. Delaware" for c in contracts)


def test_game_event_titles_are_distinct_per_market():
    titles = {c.event_title for c in _map(GAME)}
    assert len(titles) == 4


def test_native_url_uses_event_slug():
    assert all(c.exchange_event_native_url == f"https://polymarket.us/event/{GAME}" for c in _map(GAME))


# ── Specs ─────────────────────────────────────────────────────────────────────

def test_tick_size_per_market():
    ticks = {c.exchange_event_native_id: c.tick_size for c in _map(GAME)}
    assert ticks["aec-cfb-librty-del-2026-10-02"] == 5_000_000
    assert ticks["tsc-cfb-librty-del-2026-10-02-2q-10pt5"] == 10_000_000


def test_lot_size_from_minimum_trade_qty():
    assert all(c.lot_size == 1_000_000 for c in _map(POLITICS))
    assert all(c.lot_size == 10_000 for c in _map(GAME))


def test_missing_tick_size_falls_back_to_default():
    event = json.loads(json.dumps(EVENTS_BY_SLUG[POLITICS]))
    for m in event["markets"]:
        m.pop("orderPriceMinTickSize")
    assert all(c.tick_size == polymarket_us.TICK_SIZE for c in adapter._map_event(EXCHANGE_ID, event, {}))


# ── Symbols ───────────────────────────────────────────────────────────────────

def test_symbol_uses_prefix_slug_and_leg():
    contracts = [c for c in _map(POLITICS) if c.exchange_security_id == "paccc-usho-midterms-2026-11-03-dem:long"]
    assert contracts[0].security_symbol == "PM_US-PACCC-USHO-MIDTERMS-2026-11-03-DEM-LONG"


def test_symbols_unique_across_fixture():
    contracts = [c for slug in EVENTS_BY_SLUG for c in _map(slug)]
    assert len({c.security_symbol for c in contracts}) == len({c.exchange_security_id for c in contracts}) == len(contracts)


# ── Volume ────────────────────────────────────────────────────────────────────

def test_volume_comes_from_bucket_lookup():
    contracts = _map(POLITICS, {"paccc-usho-midterms-2026-11-03-dem": 50_000.0})
    vols = {c.exchange_event_native_id: c.event_volume for c in contracts}
    assert vols == {"paccc-usho-midterms-2026-11-03-dem": 50_000.0, "paccc-usho-midterms-2026-11-03-rep": 0.0}


def test_volume_buckets_are_disjoint_ranges(monkeypatch):
    calls: list[dict] = []

    def fake_paginate(resource, params):
        calls.append(params)
        if params["volumeNumMin"] == "2000":
            yield [{"slug": "a"}]
        elif params["volumeNumMin"] == "1000000":
            yield [{"slug": "b"}]

    monkeypatch.setattr(adapter, "_paginate", fake_paginate)
    volumes = adapter._fetch_volume_buckets()

    assert volumes == {"a": 2_000 * polymarket_us.ASSUMED_PRICE, "b": 1_000_000 * polymarket_us.ASSUMED_PRICE}
    assert calls[0] == {"active": "true", "closed": "false", "volumeNumMin": "500", "volumeNumMax": "1000"}
    assert "volumeNumMax" not in calls[-1]
    assert len(calls) == len(polymarket_us.VOLUME_BUCKETS_SHARES)


# ── Resolved detection ────────────────────────────────────────────────────────

def test_fetch_resolved_covers_closed_and_early_resolved_markets(monkeypatch):
    def fake_paginate(resource, params):
        if resource == "markets":
            yield [{"slug": "closed-market"}]
        else:
            yield [EVENTS_BY_SLUG[FUTURES], EVENTS_BY_SLUG[RESOLVED_ONLY]]

    monkeypatch.setattr(adapter, "_paginate", fake_paginate)
    assert adapter.fetch_resolved(EXCHANGE_ID, 3) == {
        "closed-market:long", "closed-market:short",
        "tec-mlb-nlchamp-2026-09-27-chc:long", "tec-mlb-nlchamp-2026-09-27-chc:short",
        "tec-mlb-nlchamp-2026-09-27-phi:long", "tec-mlb-nlchamp-2026-09-27-phi:short",
        "atc-uecl-ggk-zil-2026-07-30-ggk:long", "atc-uecl-ggk-zil-2026-07-30-ggk:short",
    }


# ── Entity creation ───────────────────────────────────────────────────────────

def test_entity_creation_futures(stub_registry, stub_db, mock_anthropic):
    result = create_entities(stub_registry, mock_anthropic, _map(FUTURES), db=stub_db)
    assert result.events_created == 4
    assert result.securities_created == 8
    assert result.listings_created == 8
    assert result.event_contracts_created == 8
