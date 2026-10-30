from unittest.mock import MagicMock

from classifier.adapters import kalshi
from classifier.propositions.derive import IMPLIES
from classifier.propositions.types import Kind, Proposition
from classifier.stages.entities import create_entities
from classifier.stages.propositions import derive_due_subjects, read_backlog, read_propositions

RULE = ("If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time Index (BRTI) before 3 PM EDT is "
        "above {} at 3 PM EDT on Oct 29, 2026, then the market resolves to Yes.")


def btc_ladder(strikes=(70000, 71000, 72000)) -> dict:
    return {
        "event_ticker": "KXBTCD-26OCT2915", "series_ticker": "KXBTCD", "title": "BTC price on Oct 29, 2026 at 3pm EDT?",
        "mutually_exclusive": False, "settlement_sources": [{"name": "CF Benchmarks"}],
        "markets": [{"ticker": f"KXBTCD-26OCT2915-T{v - 0.01}", "status": "active", "yes_sub_title": f"${v:,} or above",
                     "rules_primary": RULE.format(v - 0.01), "strike_type": "greater", "floor_strike": v - 0.01,
                     "close_time": "2026-10-29T19:00:00Z"} for v in strikes],
    }


def mention_market() -> dict:
    """A family no code reader handles, so only the LLM can read it."""
    return {
        "event_ticker": "KXTRUMPMENTION-26OCT29", "series_ticker": "KXTRUMPMENTION", "title": "What will Trump say?",
        "mutually_exclusive": False, "settlement_sources": [],
        "markets": [{"ticker": "KXTRUMPMENTION-26OCT29-AI", "status": "active", "yes_sub_title": "AI",
                     "rules_primary": "If Donald Trump says AI during the rally, then the market resolves to Yes.",
                     "close_time": "2026-10-30T03:00:00Z"}],
    }


class AnswersEverything:
    """An extractor that reads every contract it is given as one open question."""

    def __init__(self):
        self.asked: list[int] = []

    def extract(self, groups, candidates):
        items = [i for _, (_, chunk) in groups.items() for i in chunk]
        self.asked += [i.security_id for i in items]
        return {i.security_id: (Proposition(subject="trump-rally-words", kind=Kind.OTHER), "Words at a rally") for i in items}


def embed(texts, kind):
    return [[1.0, float(len(t) % 7)] for t in texts]


def load(stub_registry, stub_db, event):
    contracts = kalshi._with_sources(event, kalshi.KalshiAdapter()._map_event(2, event))
    create_entities(stub_registry, MagicMock(), contracts, db=stub_db, canonicalize_enabled=False)
    by_listing = {(src["source"]["market"]["ticker"], src["source"]["side"]): sid
                  for sid, src in stub_registry._contract_sources.items()}
    return {c.exchange_security_id: by_listing[tuple(c.exchange_security_id.split(":"))] for c in contracts}


def test_a_ladder_is_read_by_code_stored_and_derived(stub_registry, stub_db):
    sids = load(stub_registry, stub_db, btc_ladder())
    result = read_propositions(sorted(sids.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    assert (result.read, result.pending, result.stored) == (6, 0, 6)
    assert set(stub_registry._subjects) == {"btc-usd"}
    assert all(s["embedding"] for s in stub_registry._subjects.values())

    derived = derive_due_subjects(db=stub_db, registry=stub_registry)
    assert derived.subjects_derived == 1
    yes = {k.split("-T")[1].split(":")[0]: v for k, v in sids.items() if k.endswith(":yes")}
    rels = stub_registry._relationships
    assert rels[(yes["71999.99"], yes["69999.99"])]["relationship_type"] == IMPLIES
    assert rels[(yes["71999.99"], yes["69999.99"])]["method"] == "derived"
    assert all(r["flags"] == [] for r in rels.values())


def test_a_second_pass_reads_and_derives_nothing(stub_registry, stub_db):
    sids = sorted(load(stub_registry, stub_db, btc_ladder()).values())
    read_propositions(sids, db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    derive_due_subjects(db=stub_db, registry=stub_registry)

    again = read_propositions(sids, db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    assert (again.read, again.stored) == (0, 0)
    assert derive_due_subjects(db=stub_db, registry=stub_registry).subjects_derived == 0


def test_a_new_strike_rederives_the_subject(stub_registry, stub_db):
    first = load(stub_registry, stub_db, btc_ladder((70000, 71000)))
    read_propositions(sorted(first.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    derive_due_subjects(db=stub_db, registry=stub_registry)
    before = len(stub_registry._relationships)

    added = load(stub_registry, stub_db, btc_ladder((70000, 71000, 72000)))
    read_propositions(sorted(set(added.values()) - set(first.values())), db=stub_db, registry=stub_registry,
                      extractor=None, embed=embed)
    assert derive_due_subjects(db=stub_db, registry=stub_registry).subjects_derived == 1
    assert len(stub_registry._relationships) > before


def test_securities_without_a_source_are_skipped(stub_registry, stub_db):
    result = read_propositions([999], db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    assert result.read == 0


class TestBacklog:
    def test_with_the_llm_off_code_reads_are_stored_and_the_rest_waits(self, stub_registry, stub_db):
        ladder = load(stub_registry, stub_db, btc_ladder())
        mention = load(stub_registry, stub_db, mention_market())
        result, following = read_backlog(db=stub_db, registry=stub_registry, extractor=None, embed=embed, limit=100)
        assert following == 0
        assert (result.stored, result.pending) == (6, 2)
        assert set(stub_registry._propositions) == set(ladder.values())
        assert stub_db.securities_needing_propositions("p1", 100) == sorted(mention.values())

    def test_the_backlog_is_walked_rather_than_reread_from_the_start(self, stub_registry, stub_db):
        mention = sorted(load(stub_registry, stub_db, mention_market()).values())
        ladder = sorted(load(stub_registry, stub_db, btc_ladder()).values())
        # The unreadable securities sort first; a pass that always took the first page would never reach the ladder.
        first, after = read_backlog(db=stub_db, registry=stub_registry, extractor=None, embed=embed, limit=2)
        assert (first.stored, after) == (0, mention[-1])
        passes = 1
        while after:
            _, after = read_backlog(db=stub_db, registry=stub_registry, extractor=None, embed=embed, limit=2, after=after)
            passes += 1
        assert set(stub_registry._propositions) == set(ladder)
        assert passes == 5

    def test_turning_the_llm_on_reads_what_was_left(self, stub_registry, stub_db):
        load(stub_registry, stub_db, btc_ladder())
        mention = load(stub_registry, stub_db, mention_market())
        read_backlog(db=stub_db, registry=stub_registry, extractor=None, embed=embed, limit=100)

        extractor = AnswersEverything()
        result, _ = read_backlog(db=stub_db, registry=stub_registry, extractor=extractor, embed=embed, limit=100)
        assert result.stored == 2
        assert extractor.asked == [sid for key, sid in mention.items() if key.endswith(":yes")]
        assert stub_db.securities_needing_propositions("p1", 100) == []
