import dataclasses

from gnomepy.registry.relations import Edge, close

from classifier.adapters import kalshi
from classifier.propositions.derive import IMPLIES
from classifier.propositions.types import EXTRACTOR_VERSION, Comparator, Kind, Observation, Proposition
from classifier.stages.entities import create_entities
from classifier.stages import propositions as stage
from classifier.stages.propositions import (
    derive_due_subjects,
    read_backlog,
    read_propositions,
    recheck_readers,
    rederive_on_new_version,
)

RULE = ("If the simple average of the sixty seconds of CF Benchmarks' Bitcoin Real-Time Index (BRTI) before {h} PM EDT "
        "is above {v} at {h} PM EDT on Oct 29, 2026, then the market resolves to Yes.")


def btc_ladder(strikes=(70000, 71000, 72000), hour=3) -> dict:
    event = f"KXBTCD-26OCT29{12 + hour}"
    return {
        "event_ticker": event, "series_ticker": "KXBTCD", "title": f"BTC price on Oct 29, 2026 at {hour}pm EDT?",
        "mutually_exclusive": False, "settlement_sources": [{"name": "CF Benchmarks"}],
        "markets": [{"ticker": f"{event}-T{v - 0.01}", "status": "active", "yes_sub_title": f"${v:,} or above",
                     "rules_primary": RULE.format(h=hour, v=v - 0.01), "strike_type": "greater", "floor_strike": v - 0.01,
                     "close_time": f"2026-10-29T{16 + hour}:00:00Z"} for v in strikes],
    }


def record_reconciles(stub_registry) -> list[dict]:
    calls: list[dict] = []
    reconcile = stub_registry.reconcile_contract_relationships

    def recording(**kwargs):
        calls.append(kwargs)
        return reconcile(**kwargs)

    stub_registry.reconcile_contract_relationships = recording
    return calls


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


def rebuilt(stub_registry, stub_db) -> dict:
    """The complete relation a consumer rebuilds from the stored rows."""
    rows = [Edge(a, b, r["relationship_type"], tuple(r["flags"])) for (a, b), r in stub_registry._relationships.items()]
    ids = {sid for row in rows for sid in (row.security_id_a, row.security_id_b)}
    complements = {sid: c for sid in ids if (c := stub_db._complement(sid)) is not None}
    return close(rows, complements)


def load(stub_registry, stub_db, event):
    contracts = kalshi._with_sources(event, kalshi.KalshiAdapter()._map_event(2, event))
    create_entities(stub_registry, None, contracts, db=stub_db)
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
    no = {k.split("-T")[1].split(":")[0]: v for k, v in sids.items() if k.endswith(":no")}
    rels = stub_registry._relationships
    # Only the two steps of the Yes ladder are stored; the No side and the skipped strike are rebuilt.
    assert set(rels) == {(yes["71999.99"], yes["70999.99"]), (yes["70999.99"], yes["69999.99"])}
    assert all(r["relationship_type"] == IMPLIES and r["method"] == "derived" and r["flags"] == [] for r in rels.values())
    relation = rebuilt(stub_registry, stub_db)
    assert relation[(yes["71999.99"], yes["69999.99"])] == ()
    assert relation[(no["69999.99"], no["71999.99"])] == ()


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
        assert stub_db.securities_needing_propositions(EXTRACTOR_VERSION, 100) == sorted(mention.values())

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
        assert stub_db.securities_needing_propositions(EXTRACTOR_VERSION, 100) == []


def test_a_new_hour_rederives_only_that_hour(stub_registry, stub_db):
    three = load(stub_registry, stub_db, btc_ladder(hour=3))
    read_propositions(sorted(three.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    derive_due_subjects(db=stub_db, registry=stub_registry)
    kept = dict(stub_registry._relationships)

    four = load(stub_registry, stub_db, btc_ladder(hour=4))
    read_propositions(sorted(four.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    calls = record_reconciles(stub_registry)
    assert derive_due_subjects(db=stub_db, registry=stub_registry).subjects_derived == 1

    assert {sid for call in calls for sid in call["security_ids"]} == set(four.values())
    assert all(stub_registry._relationships[k] == v for k, v in kept.items())
    assert len(stub_registry._relationships) == 2 * len(kept)


def test_a_large_unit_is_sent_in_chunks_with_the_same_result(stub_registry, stub_db, monkeypatch):
    sids = load(stub_registry, stub_db, btc_ladder(tuple(range(70000, 76000, 1000))))
    read_propositions(sorted(sids.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    monkeypatch.setattr(stage, "RECONCILE_CHUNK_ROWS", 2)
    calls = record_reconciles(stub_registry)
    derive_due_subjects(db=stub_db, registry=stub_registry)

    assert len(calls) > 1
    assert [c["subject_id"] is not None for c in calls] == [False] * (len(calls) - 1) + [True]
    chunked = dict(stub_registry._relationships)
    stub_registry._relationships.clear()
    monkeypatch.setattr(stage, "RECONCILE_CHUNK_ROWS", 10_000)
    for subject in stub_registry._subjects.values():
        subject["derived_at"] = None
    derive_due_subjects(db=stub_db, registry=stub_registry)
    assert chunked == stub_registry._relationships


def test_a_proposition_older_than_its_source_takes_no_part(stub_registry, stub_db):
    sids = load(stub_registry, stub_db, btc_ladder())
    read_propositions(sorted(sids.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    derive_due_subjects(db=stub_db, registry=stub_registry)
    changed = next(iter(sids.values()))
    source = stub_registry._contract_sources[changed]
    stub_registry.upsert_contract_sources([{**source, "source": {**source["source"], "reworded": True},
                                            "source_hash": "new"}])

    assert not any(changed in k for k in stub_registry._relationships)
    contracts, _, _ = stub_db.get_subject_contracts(stub_registry._subjects["btc-usd"]["subject_id"])
    assert changed not in {c.security_id for c in contracts}


def test_a_subject_that_fails_to_derive_does_not_hold_up_the_rest(stub_registry, stub_db):
    btc = load(stub_registry, stub_db, btc_ladder())
    words = load(stub_registry, stub_db, mention_market())
    read_propositions(sorted(btc.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    read_propositions(sorted(words.values()), db=stub_db, registry=stub_registry, extractor=AnswersEverything(),
                      embed=embed)
    failing = stub_registry._subjects["btc-usd"]["subject_id"]
    get_contracts = stub_db.get_subject_contracts

    def broken(subject_id):
        if subject_id == failing:
            raise TypeError("cannot compare")
        return get_contracts(subject_id)

    stub_db.get_subject_contracts = broken
    assert derive_due_subjects(db=stub_db, registry=stub_registry).subjects_derived == 1
    assert stub_db.dirty_subjects(10) == [failing]


class DictMarker(dict):
    """The Redis marker's get/set over a dict."""

    def set(self, key, value):
        self[key] = value


def test_a_new_derivation_version_rederives_every_subject_once(stub_registry, stub_db):
    sids = load(stub_registry, stub_db, btc_ladder())
    read_propositions(sorted(sids.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
    derive_due_subjects(db=stub_db, registry=stub_registry)
    stale = (min(sids.values()), max(sids.values()))
    stub_registry._relationships[stale] = {"relationship_type": "EQUIVALENT", "method": "derived", "flags": []}

    marker = DictMarker()
    assert rederive_on_new_version(stub_db, marker) == 1
    assert rederive_on_new_version(stub_db, marker) == 0
    derive_due_subjects(db=stub_db, registry=stub_registry)
    assert stale not in stub_registry._relationships


def fed_meeting() -> dict:
    event = "KXFEDDECISION-27JAN"
    rules = "If the Federal Reserve does a {} on January 27, 2027, then the market resolves to Yes."
    outcomes = [("C26", "Cut >25bps", "Cut of   25bps"), ("C25", "Cut 25bps", "Cut of 25bps"),
                ("H0", "Fed maintains rate", "Hike of 0bps"), ("H25", "Hike 25bps", "Hike of 25bps"),
                ("H26", "Hike >25bps", "Hike of   25bps")]
    return {
        "event_ticker": event, "series_ticker": "KXFEDDECISION", "title": "Fed decision in Jan 2027?",
        "mutually_exclusive": True, "settlement_sources": [{"name": "Federal Reserve"}],
        "markets": [{"ticker": f"{event}-{code}", "status": "active", "yes_sub_title": subtitle,
                     "rules_primary": rules.format(action), "close_time": "2027-01-27T18:59:00Z"}
                    for code, subtitle, action in outcomes],
    }


class ReadsCutsAsHikes:
    """The model's reading before the code reader: "Cut >25bps" as a change of more than +25, like the hike."""

    def extract(self, groups, candidates):
        items = [i for _, (_, chunk) in groups.items() for i in chunk]
        return {i.security_id: (Proposition(subject="fed-funds-target:2027-01-27", kind=Kind.THRESHOLD,
                                            measure="target_rate_change_bps", comparator=Comparator.GT, value=25.0,
                                            unit="bps", observation=Observation.AT_TIME), "The Fed's January decision")
                for i in items}


class TestRecheckReaders:
    def _read_by_the_model(self, stub_registry, stub_db, monkeypatch, event) -> dict:
        contracts = kalshi._with_sources(event, kalshi.KalshiAdapter()._map_event(2, event))
        create_entities(stub_registry, None, contracts, db=stub_db)
        sids = {src["source"]["market"]["ticker"].rsplit("-", 1)[1]: sid
                for sid, src in stub_registry._contract_sources.items()
                if src["source"]["market"]["ticker"].startswith(event["event_ticker"])}
        monkeypatch.setattr("classifier.propositions.extract.extract_venue", lambda source, prices=None: None)
        read_propositions(sorted(sids.values()), db=stub_db, registry=stub_registry, extractor=ReadsCutsAsHikes(),
                          embed=embed)
        monkeypatch.undo()
        return sids

    def test_what_a_reader_now_reads_is_stored_from_code(self, stub_registry, stub_db, monkeypatch):
        sids = self._read_by_the_model(stub_registry, stub_db, monkeypatch, fed_meeting())
        assert {p["method"] for p in stub_registry._propositions.values()} == {"LLM"}

        result, following = recheck_readers(db=stub_db, registry=stub_registry, embed=embed)
        assert following == 0
        assert (result.read, result.stored) == (5, 5)
        assert {p["method"] for p in stub_registry._propositions.values()} == {"VENUE"}
        cut = stub_db.get_stored_propositions([sids["C26"]])[sids["C26"]][0]
        assert (cut.subject, cut.comparator, cut.value) == ("fomc:2027-01-meeting", Comparator.LT, -25.0)
        assert stub_db.securities_needing_propositions(EXTRACTOR_VERSION, 100) == []

    def test_a_second_walk_reads_nothing(self, stub_registry, stub_db, monkeypatch):
        self._read_by_the_model(stub_registry, stub_db, monkeypatch, fed_meeting())
        recheck_readers(db=stub_db, registry=stub_registry, embed=embed)
        again, _ = recheck_readers(db=stub_db, registry=stub_registry, embed=embed)
        assert (again.read, again.stored) == (0, 0)

    def test_the_walk_pages_and_leaves_what_no_reader_reads(self, stub_registry, stub_db, monkeypatch):
        mention = load(stub_registry, stub_db, mention_market())
        read_propositions(sorted(mention.values()), db=stub_db, registry=stub_registry, extractor=AnswersEverything(),
                          embed=embed)
        sids = self._read_by_the_model(stub_registry, stub_db, monkeypatch, fed_meeting())
        stored, after, pages = 0, 0, 0
        while True:
            result, after = recheck_readers(db=stub_db, registry=stub_registry, embed=embed, limit=4, after=after)
            stored, pages = stored + result.stored, pages + 1
            if not after:
                break
        assert (stored, pages) == (5, 2)
        assert {stub_registry._propositions[sid]["method"] for sid in mention.values()} == {"LLM"}
        assert {stub_registry._propositions[sid]["method"] for sid in sids.values()} == {"VENUE"}

    def test_the_rederived_meeting_has_no_collision(self, stub_registry, stub_db, monkeypatch, caplog):
        self._read_by_the_model(stub_registry, stub_db, monkeypatch, fed_meeting())
        derive_due_subjects(db=stub_db, registry=stub_registry)
        assert "proposition collision" in caplog.text

        caplog.clear()
        recheck_readers(db=stub_db, registry=stub_registry, embed=embed)
        assert derive_due_subjects(db=stub_db, registry=stub_registry).subjects_derived >= 1
        assert "proposition collision" not in caplog.text


class ReadsAThreshold:
    def extract(self, groups, candidates):
        items = [i for _, (_, chunk) in groups.items() for i in chunk]
        return {i.security_id: (Proposition(subject="trump-rally-words", kind=Kind.THRESHOLD, measure="mentions",
                                            comparator=Comparator.GTE, value=3.0, observation=Observation.AT_TIME),
                                "Words at a rally") for i in items}


class TestVenueCorrections:
    def _stored_by_the_model(self, stub_registry, stub_db):
        sids = load(stub_registry, stub_db, mention_market())
        read_propositions(sorted(sids.values()), db=stub_db, registry=stub_registry, extractor=ReadsAThreshold(),
                          embed=embed)
        return sorted(sids.values())

    def test_a_stored_model_reading_a_correction_now_changes_is_stored_corrected(self, stub_registry, stub_db,
                                                                               monkeypatch):
        sids = self._stored_by_the_model(stub_registry, stub_db)
        subjects = {sid: stub_registry._propositions[sid]["extracted_subject_id"] for sid in sids}
        monkeypatch.setattr(stage, "correct_venue",
                            lambda source, p: dataclasses.replace(p, comparator=Comparator.LTE))

        result, _ = recheck_readers(db=stub_db, registry=stub_registry, embed=embed)
        assert result.stored == 2
        stored = stub_db.get_stored_propositions(sids)
        assert {p.comparator for p, _, _ in stored.values()} == {Comparator.LTE}
        assert {method for _, method, _ in stored.values()} == {"LLM"}
        assert {sid: stub_registry._propositions[sid]["extracted_subject_id"] for sid in sids} == subjects
        assert stub_db.securities_needing_propositions(EXTRACTOR_VERSION, 100) == []

        again, _ = recheck_readers(db=stub_db, registry=stub_registry, embed=embed)
        assert again.stored == 0

    def test_without_a_correction_nothing_is_stored(self, stub_registry, stub_db):
        self._stored_by_the_model(stub_registry, stub_db)
        result, _ = recheck_readers(db=stub_db, registry=stub_registry, embed=embed)
        assert result.stored == 0
