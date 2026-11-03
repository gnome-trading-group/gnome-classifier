import json
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx

from classifier.client.claude import ClaudeAsker
from classifier.propositions.merge_evaluation import GOLDEN_PATH, load_golden, load_judgments, save_judgments, score
from classifier.propositions.merges import (
    Judgment,
    SubjectContract,
    SubjectMergeJudge,
    SubjectProfile,
    Verdict,
    candidate_pairs,
    crosses_venues,
    key_tokens,
    parse_answer,
    profile,
)
from classifier.propositions.types import Comparator, Kind, Proposition

ELECTION_DAY = datetime(2026, 11, 4, tzinfo=timezone.utc)


def subject(key, venues=("KALSHI",), entities=(), measures=(), deadlines=()):
    return SubjectProfile(key=key, description="", venues=frozenset(venues), entities=frozenset(entities),
                          measures=frozenset(measures), deadlines=frozenset(deadlines), proposition_count=1, samples=())


def contract(venue, proposition, title="t", outcome="Yes"):
    return SubjectContract(venue, title, outcome, "rules", proposition)


class TestKeyTokens:
    def test_state_names_read_as_their_codes(self):
        assert key_tokens("georgia-governor-2026") == key_tokens("ga-governor-2026") == ("ga", "governor", "2026")

    def test_longer_state_names_win(self):
        assert key_tokens("us-senate-west-virginia-2026") == ("us", "senate", "wv", "2026")

    def test_filler_words_are_dropped(self):
        assert key_tokens("us-house-2026-control") == key_tokens("us-house-2026")

    def test_a_state_inside_a_word_is_left_alone(self):
        assert "ga" not in key_tokens("georgiana-county")


class TestCandidates:
    def pairs(self, *subjects, neighbours=None):
        return candidate_pairs(subjects, neighbours or {})

    def test_keys_alike_after_normalizing(self):
        assert self.pairs(subject("ga-governor-2026", ["POLYMARKET_INTL"]),
                          subject("georgia-governor-2026", ["KALSHI"])) == [("ga-governor-2026", "georgia-governor-2026")]

    def test_one_extra_number(self):
        assert self.pairs(subject("fomc:2026-12", ["KALSHI"]),
                          subject("fomc:2026-12-09", ["HYPERLIQUID"])) == [("fomc:2026-12", "fomc:2026-12-09")]

    def test_one_extra_word_narrows_the_subject(self):
        assert self.pairs(subject("us-senate-2026", ["KALSHI"]), subject("us-senate-2026-texas", ["POLYMARKET_US"])) == []

    def test_order_matters_in_combination_keys(self):
        assert self.pairs(subject("combo:r-ak-ia-d-oh", ["KALSHI"]), subject("combo:r-ak-oh-d-ia", ["POLYMARKET_US"])) == []

    def test_only_pairs_that_would_link_two_venues(self):
        assert self.pairs(subject("ga-governor-2026"), subject("georgia-governor-2026")) == []
        assert crosses_venues(subject("a", ["KALSHI", "POLYMARKET_US"]), subject("b", ["KALSHI"]))

    def test_the_unknown_subject_is_never_a_candidate(self):
        assert self.pairs(subject("unknown", ["KALSHI"]), subject("btc-usd", ["HYPERLIQUID"]),
                          neighbours={"btc-usd": ["unknown"]}) == []

    def test_a_shared_deadline_and_entity(self):
        a = subject("game:nfl:2026-11-02:chisea", ["KALSHI"], entities=["chi", "sea"], deadlines=[date(2026, 11, 3)])
        b = subject("game:nfl:2026-11-02:sea-chi", ["POLYMARKET_US"], entities=["sea"], deadlines=[date(2026, 11, 2)])
        assert self.pairs(a, b) == [("game:nfl:2026-11-02:chisea", "game:nfl:2026-11-02:sea-chi")]

    def test_generic_entities_and_crowded_days_are_ignored(self):
        day = [date(2026, 11, 4)]
        assert self.pairs(subject("x", ["KALSHI"], entities=["yes"], deadlines=day),
                          subject("y", ["POLYMARKET_US"], entities=["yes"], deadlines=day)) == []
        crowd = [subject(f"race-{i}", ["KALSHI" if i % 2 else "POLYMARKET_US"], entities=["democratic"], deadlines=day)
                 for i in range(7)]
        assert self.pairs(*crowd) == []

    def test_embedding_neighbours(self):
        a, b = subject("fed-funds-target:2026-12-09", ["KALSHI"]), subject("fomc:2026-12-09", ["HYPERLIQUID"])
        assert self.pairs(a, b, neighbours={a.key: [b.key]}) == [(a.key, b.key)]


class TestProfile:
    def test_collects_entities_measures_and_deadlines(self):
        p = profile("tx-governor-2026", "Texas", [
            contract("KALSHI", Proposition("tx-governor-2026", Kind.WINNER, measure="winner", entity="republican",
                                           window_end=ELECTION_DAY)),
            contract("POLYMARKET_US", Proposition("tx-governor-2026", Kind.THRESHOLD, measure="margin:democratic",
                                                  comparator=Comparator.GT, value=3)),
        ])
        assert p.entities == {"republican", "democratic"} and p.measures == {"winner", "margin:democratic"}
        assert p.deadlines == {date(2026, 11, 4)} and p.venues == {"KALSHI", "POLYMARKET_US"}

    def test_samples_take_each_venue_in_turn(self):
        winner = Proposition("s", Kind.WINNER, measure="winner", entity="x")
        p = profile("s", "", [contract("KALSHI", winner)] * 5 + [contract("POLYMARKET_US", winner)], samples=2)
        assert [s["venue"] for s in p.samples] == ["KALSHI", "POLYMARKET_US"]

    def test_round_trips_through_json(self):
        p = subject("s", ["KALSHI"], entities=["a"], measures=["m"], deadlines=[date(2026, 11, 4)])
        assert SubjectProfile.from_json(p.to_json()) == p


def answer(*judgments):
    return SimpleNamespace(stop_reason="end_turn",
                           content=[SimpleNamespace(type="text", text=json.dumps({"judgments": list(judgments)}))])


def judgment(pair_id, verdict="SAME", entity_map=()):
    return {"pair_id": str(pair_id), "verdict": verdict, "reason": "r",
            "entity_map": [{"a": a, "b": b} for a, b in entity_map], "measure_map": []}


PAIR = (subject("ga-governor-2026", ["POLYMARKET_INTL"]), subject("georgia-governor-2026", ["KALSHI"]))


class TestJudge:
    def test_parses_verdicts_and_maps(self):
        out = parse_answer(json.dumps({"judgments": [judgment(0, entity_map=[("ggo", "goias"), ("x", "x")])]}), [PAIR])
        assert out[("ga-governor-2026", "georgia-governor-2026")] == Judgment(Verdict.SAME, "r", (("ggo", "goias"),))

    def test_unparseable_answers_and_entries_are_none(self):
        assert parse_answer("not json", [PAIR]) == {("ga-governor-2026", "georgia-governor-2026"): None}
        assert parse_answer(json.dumps({"judgments": [judgment(0, verdict="MAYBE")]}), [PAIR]) == {
            ("ga-governor-2026", "georgia-governor-2026"): None}

    def test_asks_in_chunks_with_the_schema(self):
        client = MagicMock()
        client.messages.create.side_effect = [answer(judgment(0), judgment(1)), answer(judgment(0, "RELATED"))]
        pairs = [PAIR, (subject("a", ["KALSHI"]), subject("b", ["HYPERLIQUID"])),
                 (subject("c", ["KALSHI"]), subject("d", ["HYPERLIQUID"]))]
        judge = SubjectMergeJudge(ClaudeAsker(lambda p: client, ["anthropic"], {"anthropic": "m"}), max_workers=1,
                                  pairs_per_request=2)
        out = judge.judge(pairs)
        assert [j.verdict for j in out.values()] == [Verdict.SAME, Verdict.SAME, Verdict.RELATED]
        kwargs = client.messages.create.call_args_list[0].kwargs
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        assert [p["pair_id"] for p in json.loads(kwargs["messages"][0]["content"])["pairs"]] == ["0", "1"]

    def test_an_unavailable_provider_leaves_pairs_to_ask_again(self):
        client = MagicMock()
        client.messages.create.side_effect = anthropic.RateLimitError(
            "slow down", response=httpx.Response(429, request=httpx.Request("POST", "https://x")), body=None)
        judge = SubjectMergeJudge(ClaudeAsker(lambda p: client, ["anthropic"], {"anthropic": "m"}))
        assert judge.judge([PAIR]) == {}


class TestEvaluation:
    def test_golden_pairs_load(self):
        golden = load_golden()
        assert len({p.id for p in golden}) == len(golden)
        assert {p.expected for p in golden} == set(Verdict)
        assert all(crosses_venues(p.a, p.b) for p in golden)

    def test_scores_same_precision_and_recall(self, tmp_path):
        assert score([], {}).precision is None

        golden = load_golden(GOLDEN_PATH)
        same, related = golden[0], next(p for p in golden if p.expected == Verdict.RELATED)
        judgments = {same.keys: Judgment(Verdict.SAME, ""), related.keys: Judgment(Verdict.SAME, "")}
        report = score([same, related], judgments)
        assert (report.correct_same, report.predicted_same, report.expected_same) == (1, 2, 1)
        assert not report.passes(0.99) and [p.id for p, _ in report.wrong_merges()] == [related.id]

        path = tmp_path / "j.json"
        save_judgments(path, {**judgments, ("x", "y"): None})
        assert load_judgments(path) == {**judgments, ("x", "y"): None}
