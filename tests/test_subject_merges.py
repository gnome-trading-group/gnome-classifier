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
    check_merge,
    key_tokens,
    parse_answer,
    profile,
)
from classifier.propositions.derive import Contract
from classifier.propositions.types import Comparator, Kind, Observation, Proposition

ELECTION_DAY = datetime(2026, 11, 4, tzinfo=timezone.utc)
SEASON_END = datetime(2027, 1, 11, tzinfo=timezone.utc)


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

    def test_games_at_different_start_times_are_never_candidates(self):
        assert self.pairs(subject("game:cs2:2026-11-03:mayfal-0630", ["POLYMARKET_INTL"]),
                          subject("game:cs2:2026-11-03:mayfal-1300", ["KALSHI"])) == []

    def test_a_game_key_without_a_start_time_is_still_a_candidate(self):
        assert self.pairs(subject("game:dota2:2026-11-03:angic-1100", ["KALSHI"]),
                          subject("game:dota2:2026-11-03:angic", ["POLYMARKET_INTL"])) == [
            ("game:dota2:2026-11-03:angic", "game:dota2:2026-11-03:angic-1100")]

    def test_order_matters_in_combination_keys(self):
        assert self.pairs(subject("combo:r-ak-ia-d-oh", ["KALSHI"]), subject("combo:r-ak-oh-d-ia", ["POLYMARKET_US"])) == []

    def test_one_venues_duplicates_are_candidates_too(self):
        assert self.pairs(subject("a-j-brown:nfl-2026", ["POLYMARKET_US"]),
                          subject("a-j-brown:nfl-2026-27", ["POLYMARKET_US"])) == [
            ("a-j-brown:nfl-2026", "a-j-brown:nfl-2026-27")]

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


def judgment(pair_id, verdict="SAME", entity_map=(), keys=("ga-governor-2026", "georgia-governor-2026")):
    return {"pair_id": str(pair_id), "a_key": keys[0], "b_key": keys[1], "verdict": verdict, "reason": "r",
            "entity_map": [{"a": a, "b": b} for a, b in entity_map], "measure_map": []}


PAIR = (subject("ga-governor-2026", ["POLYMARKET_INTL"]), subject("georgia-governor-2026", ["KALSHI"]))


class TestJudge:
    def test_parses_verdicts_and_maps(self):
        out = parse_answer(json.dumps({"judgments": [judgment(0, entity_map=[("ggo", "goias"), ("x", "x")])]}), [PAIR])
        assert out[("ga-governor-2026", "georgia-governor-2026")] == Judgment(Verdict.SAME, "r", (("ggo", "goias"),))

    def test_an_answer_filed_under_another_pairs_id_is_dropped(self):
        other = (subject("a", ["KALSHI"]), subject("b", ["HYPERLIQUID"]))
        text = json.dumps({"judgments": [judgment(0, keys=("a", "b")), judgment(1, "RELATED", keys=("a", "b"))]})
        assert parse_answer(text, [PAIR, other]) == {
            ("ga-governor-2026", "georgia-governor-2026"): None, ("a", "b"): Judgment(Verdict.RELATED, "r")}

    def test_unparseable_answers_and_entries_are_none(self):
        assert parse_answer("not json", [PAIR]) == {("ga-governor-2026", "georgia-governor-2026"): None}
        assert parse_answer(json.dumps({"judgments": [judgment(0, verdict="MAYBE")]}), [PAIR]) == {
            ("ga-governor-2026", "georgia-governor-2026"): None}

    def test_asks_in_chunks_with_the_schema(self):
        client = MagicMock()
        client.messages.create.side_effect = [answer(judgment(0), judgment(1, keys=("a", "b"))),
                                              answer(judgment(0, "RELATED", keys=("c", "d")))]
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


def winner_contract(sid, subject, entity, venue, event_id, measure="winner"):
    return Contract(sid, venue, Proposition(subject, Kind.WINNER, measure=measure, entity=entity,
                                            window_end=ELECTION_DAY), event_id=event_id)


class TestCheckMerge:
    def test_counts_new_cross_venue_relationships(self):
        a = [winner_contract(1, "georgia-governor-2026", "republican", "KALSHI", 10),
             winner_contract(2, "georgia-governor-2026", "democratic", "KALSHI", 10)]
        b = [winner_contract(3, "ga-governor-2026", "rep", "POLYMARKET_US", 20)]
        check = check_merge(a, b, Judgment(Verdict.SAME, "", (("republican", "rep"),)))
        assert check.gain == 1 and not check.new_collisions
        assert {(d.security_id_a, d.security_id_b, d.relationship_type) for d in check.new_relationships} == {
            (1, 3, "EQUIVALENT")}

    def test_counts_relationships_one_venues_duplicates_add(self):
        def yards(subject, value):
            return Proposition(subject, Kind.THRESHOLD, measure="receiving_yards:a-j-brown", comparator=Comparator.GT,
                               value=value, observation=Observation.CUMULATIVE, window_end=SEASON_END)

        a = [Contract(1, "POLYMARKET_US", yards("a-j-brown:nfl-2026-27", 1000.5), event_id=10)]
        b = [Contract(2, "POLYMARKET_US", yards("a-j-brown:nfl-2026", 1400.5), event_id=20)]
        check = check_merge(a, b)
        assert [(d.security_id_a, d.security_id_b, d.relationship_type) for d in check.new_relationships] == [
            (2, 1, "IMPLIES")]

    def test_unmapped_entities_gain_nothing(self):
        a = [winner_contract(1, "georgia-governor-2026", "republican", "KALSHI", 10)]
        b = [winner_contract(3, "ga-governor-2026", "rep", "POLYMARKET_US", 20)]
        assert check_merge(a, b).gain == 0

    def test_a_merge_that_makes_two_outcomes_alike_collides(self):
        a = [winner_contract(1, "us-house-2026", "democratic", "KALSHI", 10)]
        b = [winner_contract(2, "us-house-2028", "democratic", "POLYMARKET_INTL", 20),
             winner_contract(3, "us-house-2028", "dem", "POLYMARKET_INTL", 20)]
        check = check_merge(a, b, Judgment(Verdict.SAME, "", (("democratic", "dem"),)))
        assert check.new_collisions == {20: frozenset({2, 3})}
