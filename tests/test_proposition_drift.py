import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from gnomepy.registry.types import ContractType

from classifier.propositions.drift import check_drift, sample
from classifier.propositions.extract import extract_propositions
from classifier.propositions.inputs import ExtractionInput
from classifier.propositions.llm import PropositionExtractor
from classifier.propositions.venue import claims

BTC_RULES = ("If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time Index (BRTI) before 3 PM EDT "
             "is above 79899.99 at 3 PM EDT on Oct 29, 2026, then the market resolves to Yes.")


def btc(sid, rules=BTC_RULES, side="yes"):
    source = {"venue": "KALSHI", "side": side, "event": {"event_ticker": "KXBTCD-26OCT2915"},
              "market": {"ticker": f"KXBTCD-26OCT2915-T{sid}", "rules_primary": rules}}
    return ExtractionInput(sid, "KALSHI", sid, ContractType.BINARY, "BTC", "Yes", None, None, source, "h")


def llm_entry(sid, **overrides):
    base = {"contract_id": str(sid), "subject": "btc-usd", "subject_description": "Bitcoin", "kind": "THRESHOLD",
            "measure": "price", "comparator": "GT", "value": 79899.99, "unit": "usd", "low": None, "high": None,
            "low_inclusive": True, "high_inclusive": True, "observation": "AT_TIME", "window_start": None,
            "window_end": "2026-10-29T19:00:00Z", "entity": None, "scope": None, "stability": "LIVE",
            "resolution_source": "cf-benchmarks-brti", "unknown": []}
    return {**base, **overrides}


def extractor(entries):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps({"propositions": entries}))])
    return PropositionExtractor(lambda p: client, ["anthropic"], {"anthropic": "m"})


def test_agreeing_readings_report_nothing():
    assert check_drift([btc(1)], extractor([llm_entry(1)])) == []


def test_a_disagreement_is_reported_and_logged(caplog):
    drifts = check_drift([btc(1)], extractor([llm_entry(1, comparator="GTE")]))
    assert [(d.field, d.venue, d.llm) for d in drifts] == [("comparator", "GT", "GTE")]
    assert "proposition drift: family=KALSHI:KXBTCD" in caplog.text


def test_sample_takes_code_readable_first_sides_and_is_stable_within_a_day():
    items = [btc(i) for i in range(1, 30)] + [btc(99, side="no"), btc(100, rules="new wording")]
    day = date(2026, 10, 29)
    picked = sample(items, 10, day)
    assert len(picked) == 10 and {i.security_id for i in picked} <= set(range(1, 30))
    assert picked == sample(items, 10, day)


def test_a_claimed_family_that_stops_parsing_is_logged(caplog):
    assert claims(btc(1).source)
    extract_propositions([btc(1, rules="Kalshi reworded this.")], None)
    assert "venue reader fell back: family=KALSHI:KXBTCD security=1" in caplog.text


def test_long_tail_markets_are_not_claimed():
    assert not claims({"venue": "KALSHI", "market": {"ticker": "KXELONMARS-99"}})
    assert not claims({"venue": "POLYMARKET_INTL", "market": {}})
