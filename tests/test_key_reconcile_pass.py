from datetime import datetime, timezone

import requests

from classifier.propositions.canonical import StoredAlias
from classifier.propositions.derive import Contract
from classifier.propositions.key_reconcile import FLAG_PARTY_NOMINEE, KeyJudgment
from classifier.propositions.quotes import Quote
from classifier.propositions.scope import LLMScope
from classifier.propositions.types import Kind, Proposition
from classifier.stages.key_reconcile import RECONCILE_ACTOR, Outcome, run_key_reconcile

GAME = "game:nba:2026-11-11:bos-tor"
RACE = "us-senate-2026-texas"
CHANGED = datetime(2026, 11, 10, 12, 0)


def winner(sid, subject, entity, venue, event_id, label, complement=None):
    return Contract(sid, venue, Proposition(subject, Kind.WINNER, measure="winner", entity=entity, scope="match"),
                    complement, event_id, label=label)


def game_contracts():
    return [winner(1, GAME, "bos", "HYPERLIQUID", 10, "BOS", 2), winner(2, GAME, "tor", "HYPERLIQUID", 10, "TOR", 1),
            winner(3, GAME, "celtics", "POLYMARKET_INTL", 20, "Celtics", 4),
            winner(4, GAME, "raptors", "POLYMARKET_INTL", 20, "Raptors", 3)]


class FakeDB:
    """Subjects as {subject_id: (key, contracts, 24h volume)}."""

    def __init__(self, subjects):
        self.subjects = subjects
        self.key_aliases = {}
        self.labels = {}

    def subject_clock(self):
        return datetime(2026, 11, 10, 13, 0)

    def subject_summaries(self):
        return [(sid, key, key, sorted({c.venue for c in contracts}), [], [], [], len(contracts), volume, CHANGED)
                for sid, (key, contracts, volume) in self.subjects.items()]

    def subjects_in_scope(self, categories, tags):
        scope = LLMScope.of(categories, tags)
        return {sid for sid in self.subjects if scope.includes(*self.labels.get(sid, ("OTHER", ())))}

    def get_subject_contracts(self, subject_id):
        return list(self.subjects[subject_id][1]), set(), CHANGED

    def get_subject_key_aliases(self, subject_id):
        return list(self.key_aliases.get(subject_id, ()))

    def event_titles(self, security_ids):
        return {sid: f"event of {sid}" for sid in security_ids}

    def get_active_listings_for_securities(self, security_ids):
        return [(sid, sid, 1, f"m{sid}") for sid in security_ids]


class FakeRegistry:
    def __init__(self, refuse=False):
        self.changes = []
        self.refuse = refuse

    def change_subject_key_aliases(self, *, subject_id, actor, set=None, remove=None):
        if self.refuse:
            response = requests.Response()
            response.status_code, response._content = 409, b"{}"
            raise requests.HTTPError("409 Client Error: Conflict", response=response)
        self.changes.append((subject_id, actor, set or []))
        return []


class FakeJudge:
    """Joins the key pairs it is given wherever a slot holds both."""

    def __init__(self, groups, nominee=()):
        self.groups, self.nominee = groups, set(nominee)
        self.asked = []

    def judge(self, slots):
        self.asked.extend(s.subject for s in slots)
        out = {}
        for slot in slots:
            groups = tuple(g for g in self.groups if set(g) <= slot.key_names)
            out[slot.id] = KeyJudgment(groups, "judged", frozenset(g for g in groups if g in self.nominee))
        return out


class Memory:
    def __init__(self):
        self.rows, self.sweeps = {}, {}

    def judged(self, prints):
        return {p for p in prints if p in self.rows}

    def remember(self, print_, outcome):
        self.rows[print_] = outcome

    def last_sweep(self, scope):
        return self.sweeps.get(scope)

    def set_last_sweep(self, at, scope):
        self.sweeps[scope] = at


def run(db, registry, judge, memory=None, quotes=lambda refs: {}, scope=LLMScope(), judge_limit=100):
    return run_key_reconcile(db=db, registry=registry, judge=judge, memory=memory or Memory(), judge_limit=judge_limit,
                             max_price_gap=0.15, quotes=quotes, scope=scope)


def test_stores_the_judged_joins_as_the_subjects_aliases():
    registry = FakeRegistry()
    result = run(FakeDB({7: (GAME, game_contracts(), 1000.0)}), registry,
                 FakeJudge([("bos", "celtics"), ("raptors", "tor")]))
    assert registry.changes == [(7, RECONCILE_ACTOR, [
        {"kind": "ENTITY", "key": "bos", "canonical_key": "celtics", "flags": [], "origin": "RECONCILE", "reason": "judged"},
        {"kind": "ENTITY", "key": "tor", "canonical_key": "raptors", "flags": [], "origin": "RECONCILE", "reason": "judged"},
    ])]
    assert (result.judged, result.stored, result.aliases_stored) == (1, 1, 2)


def test_a_slot_is_judged_once_until_its_keys_change():
    db, memory, judge = FakeDB({7: (GAME, game_contracts(), 1000.0)}), Memory(), FakeJudge([])
    run(db, FakeRegistry(), judge, memory)
    memory.sweeps.clear()
    run(db, FakeRegistry(), judge, memory)
    assert judge.asked == [GAME]
    db.subjects[7][1].append(winner(5, GAME, "boston", "KALSHI", 30, "Boston"))
    memory.sweeps.clear()
    run(db, FakeRegistry(), judge, memory)
    assert judge.asked == [GAME, GAME]


def test_only_subjects_in_scope_and_changed_since_the_last_sweep_are_looked_at():
    db = FakeDB({7: (GAME, game_contracts(), 1000.0)})
    judge = FakeJudge([])
    run(db, FakeRegistry(), judge, scope=LLMScope.of(tags=["esports"]))
    assert judge.asked == []
    memory = Memory()
    memory.sweeps["*"] = datetime(2026, 11, 11)
    run(db, FakeRegistry(), judge, memory)
    assert judge.asked == []


def test_a_subject_one_venue_lists_has_nothing_to_reconcile():
    db = FakeDB({7: (GAME, [c for c in game_contracts() if c.venue == "HYPERLIQUID"], 1000.0)})
    judge = FakeJudge([])
    run(db, FakeRegistry(), judge)
    assert judge.asked == []


def test_a_party_joined_to_its_nominee_is_stored_flagged():
    race = [winner(1, RACE, "james-talarico", "KALSHI", 10, "James Talarico"),
            winner(2, RACE, "ken-paxton", "KALSHI", 10, "Ken Paxton"),
            winner(3, RACE, "democratic", "POLYMARKET_INTL", 20, "Talarico"),
            winner(4, RACE, "republican", "POLYMARKET_INTL", 20, "Paxton")]
    registry = FakeRegistry()
    group = ("democratic", "james-talarico")
    run(FakeDB({8: (RACE, race, 1000.0)}), registry, FakeJudge([group], nominee=[group]))
    (_, _, changes), = registry.changes
    assert [(c["key"], c["canonical_key"], c["flags"]) for c in changes] == [
        ("james-talarico", "democratic", [FLAG_PARTY_NOMINEE])]


def test_a_join_the_books_contradict_is_held_back():
    quotes = lambda refs: {1: Quote(bid=0.80, ask=0.82), 3: Quote(bid=0.30, ask=0.32)}
    registry, memory = FakeRegistry(), Memory()
    result = run(FakeDB({7: (GAME, game_contracts(), 1000.0)}), registry, FakeJudge([("bos", "celtics")]), memory,
                 quotes=quotes)
    assert registry.changes == [] and result.held_price == 1
    assert list(memory.rows.values()) == [Outcome.HELD_PRICE]


def test_a_join_already_stored_is_not_stored_again():
    db = FakeDB({7: (GAME, game_contracts(), 1000.0)})
    db.key_aliases[7] = [StoredAlias("ENTITY", "bos", "celtics")]
    registry = FakeRegistry()
    result = run(db, registry, FakeJudge([("bos", "celtics"), ("raptors", "tor")]))
    assert [[(c["key"], c["canonical_key"]) for c in changes] for _, _, changes in registry.changes] == [[("tor", "raptors")]]
    assert result.aliases_stored == 1


def test_a_slot_whose_aliases_the_registry_refuses_is_remembered_and_the_pass_goes_on():
    memory = Memory()
    result = run(FakeDB({7: (GAME, game_contracts(), 1000.0)}), FakeRegistry(refuse=True),
                 FakeJudge([("bos", "celtics")]), memory)
    assert (result.stored, result.refused) == (0, 1)
    assert list(memory.rows.values()) == [Outcome.REFUSED]
