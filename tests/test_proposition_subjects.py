from classifier.propositions.subjects import (
    CombinedSubjectIndex,
    InMemorySubjectIndex,
    KnownSubject,
    add_known,
    candidates_for,
    embed_subjects,
    entity_of,
    merge_candidates,
)
from classifier.propositions.types import Comparator, Kind, Proposition


def test_venue_read_games_collect_their_entities_and_measures():
    known = {}
    add_known(known, Proposition(subject="game:ncaaf:2026-09-19:uncclem", kind=Kind.WINNER, entity="clem"), "North Carolina vs Clemson")
    add_known(known, Proposition(subject="game:ncaaf:2026-09-19:uncclem", kind=Kind.THRESHOLD, measure="margin:unc",
                                 comparator=Comparator.GT, value=3.5), "")
    subject = known["game:ncaaf:2026-09-19:uncclem"]
    assert subject.entities == {"clem", "unc"} and subject.measures == {"margin:unc"}
    assert subject.description == "North Carolina vs Clemson"


def test_candidates_rank_by_word_overlap():
    known = {
        "game:ncaaf:2026-09-19:uncclem": KnownSubject("game:ncaaf:2026-09-19:uncclem", "North Carolina vs Clemson", {"unc", "clem"}),
        "btc-usd": KnownSubject("btc-usd", "Bitcoin in US dollars"),
    }
    assert [c.key for c in candidates_for("North Carolina vs. Clemson Clemson", known)] == ["game:ncaaf:2026-09-19:uncclem"]
    assert candidates_for("the vs of", known) == []


def test_entity_of_per_entity_measures():
    assert entity_of("margin:mil") == "mil" and entity_of("total_runs") is None


def fake_embed(texts, kind):
    # Two directions: anything about Seattle's game, and everything else.
    return [[1.0, 0.0] if ("seattle" in t.lower() or "seahawks" in t.lower()) else [0.0, 1.0] for t in texts]


def test_in_memory_index_finds_neighbours_words_would_miss():
    index = InMemorySubjectIndex(fake_embed)
    index.add([KnownSubject("game:nfl:2026-08-23:seaten", "Seattle vs Tennessee", {"sea", "ten"}),
               KnownSubject("btc-usd", "Bitcoin")])
    assert candidates_for("Seahawks vs. Titans", {"game:nfl:2026-08-23:seaten": KnownSubject(
        "game:nfl:2026-08-23:seaten", "Seattle vs Tennessee")}) == []
    assert [s.key for s in index.neighbours(["Seahawks vs. Titans"], 1)[0]] == ["game:nfl:2026-08-23:seaten"]


def test_combined_index_merges_without_duplicates():
    first, second = InMemorySubjectIndex(fake_embed), InMemorySubjectIndex(fake_embed)
    subject = KnownSubject("game:nfl:2026-08-23:seaten", "Seattle vs Tennessee")
    first.add([subject])
    second.add([subject, KnownSubject("btc-usd", "Bitcoin")])
    found = CombinedSubjectIndex(first, second).neighbours(["Seahawks"], 5)[0]
    assert [s.key for s in found] == ["game:nfl:2026-08-23:seaten", "btc-usd"]


def test_merge_candidates_keeps_order_and_limit():
    a, b = KnownSubject("a", ""), KnownSubject("b", "")
    assert [s.key for s in merge_candidates([a], [a, b], limit=1)] == ["a"]


def test_embed_subjects_stores_one_vector_per_subject():
    class DB:
        def __init__(self):
            self.pending = [KnownSubject("btc-usd", "Bitcoin")]
            self.stored = {}

        def subjects_without_embeddings(self, limit):
            pending, self.pending = self.pending, []
            return pending

        def put_subject_embeddings(self, embeddings):
            self.stored.update(embeddings)

    db = DB()
    assert embed_subjects(db, fake_embed) == 1 and db.stored == {"btc-usd": [0.0, 1.0]}
