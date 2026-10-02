import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from classifier.adapters.hyperliquid import HyperliquidAdapter, OutcomeTemplate
from gnomepy.registry.types import ContractType

FIXTURES = Path(__file__).parent / "fixtures"
OUTCOME_META = json.loads((FIXTURES / "hyperliquid_outcome_meta.json").read_text())
RAW_TEMPLATES = json.loads((FIXTURES / "hyperliquid_outcome_templates.json").read_text())
TEMPLATES = {t.id: t for t in map(OutcomeTemplate, RAW_TEMPLATES)}
OUTCOMES = {o["outcome"]: o for o in OUTCOME_META["outcomes"]}
EXCHANGE_ID = 1
adapter = HyperliquidAdapter()


VOLUME_BY_COIN = {"#62070": 100.0, "#62071": 50.0, "#65200": 10.0, "#65211": 5.0}


@pytest.fixture(scope="module")
def contracts():
    return adapter._map_all(EXCHANGE_ID, OUTCOMES, OUTCOME_META["questions"], TEMPLATES, VOLUME_BY_COIN)


def _event(contracts, native_id: str) -> list:
    group = [c for c in contracts if c.exchange_event_native_id == native_id]
    assert group, f"no contracts for {native_id}"
    return group


# ── Whole-payload invariants ──────────────────────────────────────────────────

def test_no_template_markup_leaks(contracts):
    for c in contracts:
        assert "template" not in c.event_title.lower() and "{" not in c.event_title, c.event_title
        assert "template" not in c.outcome_label.lower() and "{" not in c.outcome_label, c.outcome_label
        assert "{" not in (c.event_description or ""), c.event_description
        assert "TEMPLATE" not in c.security_symbol, c.security_symbol


def test_every_event_has_an_expiry(contracts):
    assert all(c.event_expiry for c in contracts)


def test_symbols_are_unique(contracts):
    assert len({c.security_symbol for c in contracts}) == len(contracts)


def test_fetch_requests_meta_templates_and_volumes():
    session = MagicMock()
    responses = {
        "outcomeMeta": OUTCOME_META,
        "outcomeTemplates": RAW_TEMPLATES,
        "spotMetaAndAssetCtxs": [{}, [{"coin": "#62070", "dayNtlVlm": "120.5"}, {"coin": "PURR/USDC", "dayNtlVlm": "9"}]],
    }
    session.post.side_effect = lambda url, json, timeout: MagicMock(json=lambda: responses[json["type"]])
    page = next(HyperliquidAdapter(session=session).fetch(EXCHANGE_ID))
    assert {call.kwargs["json"]["type"] for call in session.post.call_args_list} == set(responses)
    assert len(page) == 416
    assert {c.event_volume for c in page if c.exchange_event_native_id == "o:6207"} == {120.5}


def test_exchange_security_ids_are_hyperliquid_coins(contracts):
    # Hyperliquid names each tradeable side `#<outcome><side>`; the gateway subscribes with it.
    for c in contracts:
        assert c.exchange_security_id.startswith("#") and c.exchange_security_id[1:].isdigit(), c.exchange_security_id


def test_resolved_ids_use_coin_names():
    session = MagicMock()
    meta = {"questions": [{"settledNamedOutcomes": [1473]}]}
    session.post.return_value = MagicMock(json=lambda: meta)
    assert HyperliquidAdapter(session=session).fetch_resolved(EXCHANGE_ID, 3) == {"#14730", "#14731"}


def test_event_volume_sums_all_coins_of_the_event(contracts):
    assert {c.event_volume for c in _event(contracts, "o:6207")} == {150.0}
    assert {c.event_volume for c in _event(contracts, "q:357")} == {15.0}
    assert {c.event_volume for c in _event(contracts, "o:5371")} == {0.0}


# ── Template rendering ────────────────────────────────────────────────────────

def test_template_formats_datetime_keywords():
    template = OutcomeTemplate({"id": "t", "name": "{asset} at {time} UTC", "keywords": [["asset", "string"], ["time", "dateTime"]]})
    assert template.render(template.name, {"asset": "BTC", "time": "20261002-0800"}) == "BTC at 2026-10-02 08:00 UTC"


def test_template_leaves_missing_keywords_visible():
    template = OutcomeTemplate({"id": "t", "name": "{asset} above {threshold}", "keywords": [["asset", "string"]]})
    assert template.render(template.name, {"asset": "BTC"}) == "BTC above {threshold}"


def test_template_expiry_is_latest_datetime_keyword():
    template = OutcomeTemplate({"id": "t", "name": "x", "keywords": [["start", "dateTime"], ["deadline", "dateTime"]]})
    assert template.expiry({"start": "20261002-0015", "deadline": "20261003-0015"}).isoformat() == "2026-10-03T00:15:00"


# ── Standalone template outcomes ──────────────────────────────────────────────

def test_binary_price(contracts):
    group = _event(contracts, "o:5371")
    assert group[0].event_title == "BTC above 83365 at 2026-10-02 08:00?"
    assert group[0].event_description.startswith(
        "The market resolves to Yes if the BTC price is above 83365 at 2026-10-02 08:00"
    )
    assert group[0].event_expiry == "2026-10-02T08:00:00Z"
    assert [c.outcome_label for c in group] == ["Yes", "No"]
    assert [c.security_symbol for c in group] == ["HL-5371-YES", "HL-5371-NO"]


def test_price_touch(contracts):
    assert _event(contracts, "o:7173")[0].event_title == "BTC touches 87500 by 2026-11-01 00:00"


def test_sports_contest_winner_fills_short_names_into_sides(contracts):
    group = _event(contracts, "o:6207")
    assert group[0].event_title == "NFL Regular Season: Cleveland Browns v Pittsburgh Steelers"
    assert [c.outcome_label for c in group] == ["CLE", "PIT"]
    assert [c.security_symbol for c in group] == ["HL-6207-CLE", "HL-6207-PIT"]
    assert group[0].event_expiry == "2026-10-03T00:15:00Z"
    assert "scheduled for 2026-10-02 00:15 UTC (" in group[0].event_description


def test_sports_spread_and_total(contracts):
    assert _event(contracts, "o:7007")[0].event_title == "Pittsburgh Steelers v Cleveland Browns: Steelers -2.5"
    total = _event(contracts, "o:7009")
    assert total[0].event_title == "Pittsburgh Steelers v Cleveland Browns: total 38.5"
    assert [c.outcome_label for c in total] == ["Over", "Under"]


def test_ipo_templates(contracts):
    assert _event(contracts, "o:7360")[0].event_title == "Anthropic first-day market cap above $2000B?"
    assert _event(contracts, "o:2597")[0].event_title == "Anthropic IPO confirmed by 2026-10-31 23:59"


# ── Template questions (multi-outcome events) ────────────────────────────────

def test_tournament_winner_question(contracts):
    group = _event(contracts, "q:198")
    assert group[0].event_title == "2026/2027 English Premier League winner"
    assert group[0].contract_type == ContractType.MULTI_OUTCOME
    assert group[0].outcome_label == "Arsenal"


def test_contest_result_question_labels(contracts):
    group = _event(contracts, "q:357")
    assert group[0].event_title == "UEFA Nations League League A, Matchday 3: Belgium v Turkiye"
    assert [c.outcome_label for c in group] == ["Belgium", "Draw", "Turkiye"]


def test_policy_rate_question_labels(contracts):
    group = _event(contracts, "q:289")
    assert group[0].event_title == "Federal Reserve's Open Market Committee October 2026 rate decision"
    assert [c.outcome_label for c in group] == ["No change", "Decrease", "Increase"]


def test_question_with_one_active_outcome_names_it_in_the_title():
    question = next(q for q in OUTCOME_META["questions"] if q["question"] == 198)
    narrowed = {**question, "settledNamedOutcomes": question["namedOutcomes"][1:]}
    group = adapter._map_all(EXCHANGE_ID, OUTCOMES, [narrowed], TEMPLATES, {})
    title = next(c.event_title for c in group if c.exchange_event_native_id == "q:198")
    assert title == "2026/2027 English Premier League winner: Arsenal"


def test_question_event_id_survives_outcomes_settling():
    question = next(q for q in OUTCOME_META["questions"] if q["question"] == 198)
    narrowed = {**question, "settledNamedOutcomes": question["namedOutcomes"][:1]}
    contracts = adapter._map_all(EXCHANGE_ID, OUTCOMES, [narrowed], TEMPLATES, {})
    group = [c for c in contracts if int(c.exchange_security_id[1:-1]) in question["namedOutcomes"]]
    assert {c.exchange_event_native_id for c in group} == {"q:198"}
    assert question["namedOutcomes"][0] not in {int(c.exchange_security_id[1:-1]) for c in group}


# ── Legacy class: format and unknown templates ───────────────────────────────

def test_recurring_price_binary(contracts):
    group = _event(contracts, "o:7240")
    assert group[0].event_title == "BTC above 84277 at 2026-10-02 06:00?"
    assert group[0].event_category == "CRYPTO"


def test_recurring_price_buckets(contracts):
    group = _event(contracts, "q:365")
    assert group[0].event_title == "BTC price range at 2026-10-02 06:00"
    assert [c.outcome_label for c in group] == ["< 82591", "82591 - 85962", "> 85962"]


def test_unknown_template_falls_back_and_warns(caplog):
    outcome = {
        "outcome": 99999, "name": "template:somethingNew", "description": "time:20261231-2359",
        "sideSpecs": [{"name": "template:Yes"}, {"name": "template:No"}],
    }
    with caplog.at_level(logging.WARNING):
        group = adapter._map_all(EXCHANGE_ID, {99999: outcome}, [], TEMPLATES, {})
    assert group[0].event_title == "somethingNew"
    assert [c.outcome_label for c in group] == ["Yes", "No"]
    assert "Unknown Hyperliquid template" in caplog.text
