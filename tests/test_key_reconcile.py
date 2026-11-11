import json

from classifier.propositions.key_evaluation import KeyReport, load_golden, score
from classifier.propositions.key_reconcile import (
    FLAG_PARTY_NOMINEE,
    KeyJudgment,
    Reading,
    aliases,
    held_to_slot,
    parse_answer,
    unmatched_slots,
)
from classifier.propositions.types import Comparator, Kind, Observation, Proposition

GAME = "game:nba:2026-11-11:bos-tor"


def winner(entity: str, subject: str = GAME) -> Proposition:
    return Proposition(subject=subject, kind=Kind.WINNER, measure="winner", entity=entity, scope="match")


def reading(sid: int, venue: str, event: int, p: Proposition, label: str | None = None) -> Reading:
    return Reading(sid, venue, event, p, label, f"{venue} event {event}")


class TestUnmatchedSlots:
    def test_two_venues_each_keying_a_team_their_own_way_is_unmatched(self):
        slots = unmatched_slots(GAME, [
            reading(1, "HYPERLIQUID", 10, winner("bos"), "BOS"), reading(2, "HYPERLIQUID", 10, winner("tor"), "TOR"),
            reading(3, "POLYMARKET_INTL", 20, winner("celtics"), "Celtics"),
            reading(4, "POLYMARKET_INTL", 20, winner("raptors"), "Raptors"),
        ])
        assert len(slots) == 1
        slot = slots[0]
        assert slot.key_names == {"bos", "tor", "celtics", "raptors"}
        assert slot.rivals == {frozenset(("bos", "tor")), frozenset(("celtics", "raptors"))}
        assert next(k for k in slot.keys if k.key == "celtics").labels == ("Celtics",)

    def test_keys_every_venue_shares_are_matched(self):
        assert unmatched_slots(GAME, [
            reading(1, "KALSHI", 10, winner("bos")), reading(2, "KALSHI", 10, winner("tor")),
            reading(3, "POLYMARKET_US", 20, winner("bos")), reading(4, "POLYMARKET_US", 20, winner("tor")),
        ]) == []

    def test_a_venue_listing_more_outcomes_is_not_a_spelling_difference(self):
        subject = "nba-mvp:2026-27"
        assert unmatched_slots(subject, [
            reading(1, "KALSHI", 10, winner("nikola-jokic", subject)),
            reading(2, "KALSHI", 10, winner("luka-doncic", subject)),
            reading(3, "POLYMARKET_INTL", 20, winner("nikola-jokic", subject)),
        ]) == []

    def test_numbers_after_a_measure_are_never_entity_keys(self):
        subject = "brentford:epl-2026-27"
        top = lambda n: Proposition(subject=subject, kind=Kind.WINNER, measure=f"top:{n}", entity="brentford",
                                    scope="season")
        assert unmatched_slots(subject, [reading(1, "KALSHI", 10, top(4)), reading(2, "POLYMARKET_INTL", 20, top(6))]) == []

    def test_a_measures_entity_is_its_own_slot(self):
        margin = Proposition(subject=GAME, kind=Kind.THRESHOLD, measure="margin:celtics", comparator=Comparator.GT,
                             value=4.5, unit="points", observation=Observation.CUMULATIVE, scope="match")
        slots = unmatched_slots(GAME, [
            reading(1, "KALSHI", 10, Proposition(subject=GAME, kind=Kind.THRESHOLD, measure="margin:bos",
                                                 comparator=Comparator.GT, value=4.5, unit="points",
                                                 observation=Observation.CUMULATIVE, scope="match")),
            reading(2, "POLYMARKET_INTL", 20, margin),
        ])
        assert [(s.measure, s.per_entity, s.key_names) for s in slots] == [("margin", True, {"bos", "celtics"})]


class TestJudgmentsHeldToTheSlot:
    def slot(self):
        return unmatched_slots(GAME, [
            reading(1, "HYPERLIQUID", 10, winner("bos")), reading(2, "HYPERLIQUID", 10, winner("tor")),
            reading(3, "POLYMARKET_INTL", 20, winner("celtics")), reading(4, "POLYMARKET_INTL", 20, winner("raptors")),
        ])[0]

    def test_a_group_of_one_events_outcomes_is_dropped(self):
        held = held_to_slot(self.slot(), KeyJudgment((("bos", "tor"), ("celtics", "bos")), ""))
        assert held.groups == (("bos", "celtics"),)

    def test_a_group_naming_keys_outside_the_slot_is_dropped(self):
        assert held_to_slot(self.slot(), KeyJudgment((("bos", "boston"),), "")).groups == ()

    def test_a_key_in_two_groups_drops_both(self):
        assert held_to_slot(self.slot(), KeyJudgment((("bos", "celtics"), ("bos", "raptors")), "")).groups == ()

    def test_aliases_point_at_the_most_used_key(self):
        slot = self.slot()
        assert aliases(slot, KeyJudgment((("bos", "celtics"),), ""), uses={"celtics": 3, "bos": 1}) == {"bos": ("celtics", ())}

    def test_a_party_joined_to_its_nominee_is_flagged(self):
        subject = "us-senate-2026-texas"
        margin = lambda e: Proposition(subject=subject, kind=Kind.THRESHOLD, measure="margin", entity=e,
                                       comparator=Comparator.GT, value=3.0, unit="percentage_points")
        slot = unmatched_slots(subject, [
            reading(1, "KALSHI", 10, margin("james-talarico"), "James Talarico, 0-2%"),
            reading(2, "KALSHI", 10, margin("ken-paxton"), "Ken Paxton, 0-2%"),
            reading(3, "POLYMARKET_INTL", 20, margin("democratic"), "Talarico 0-3%"),
            reading(4, "POLYMARKET_INTL", 20, margin("republican"), "Paxton 0-3%"),
        ])[0]
        judgment = KeyJudgment((("democratic", "james-talarico"), ("ken-paxton", "republican")), "",
                               frozenset({("democratic", "james-talarico"), ("ken-paxton", "republican")}))
        assert aliases(slot, judgment, uses={"democratic": 5, "republican": 5}) == {
            "james-talarico": ("democratic", (FLAG_PARTY_NOMINEE,)), "ken-paxton": ("republican", (FLAG_PARTY_NOMINEE,))}

    def test_an_answer_is_matched_to_its_slot_by_id(self):
        slot = self.slot()
        text = json.dumps({"slots": [{"slot_id": "0", "subject": GAME, "measure": "winner", "reason": "x",
                                      "groups": [{"keys": ["bos", "celtics"], "party_nominee": False},
                                                 {"keys": ["tor", "raptors"], "party_nominee": False}]}]})
        judged = parse_answer(text, [slot])[slot.id]
        assert (judged.groups, judged.nominee_groups) == ((("bos", "celtics"), ("raptors", "tor")), frozenset())

    def test_an_answer_naming_another_slot_is_dropped(self):
        slot = self.slot()
        text = json.dumps({"slots": [{"slot_id": "0", "subject": "game:nba:2026-11-12:lal-gsw", "measure": "winner",
                                      "groups": [{"keys": ["bos", "celtics"], "party_nominee": False}], "reason": "x"}]})
        assert parse_answer(text, [slot]) == {slot.id: None}

    def test_an_unreadable_answer_leaves_the_slot_unanswered(self):
        slot = self.slot()
        assert parse_answer("not json", [slot]) == {slot.id: None}


def test_the_golden_slots_load_and_their_labels_are_held_to_their_slots():
    golden = load_golden()
    assert len(golden) > 100
    assert len({g.slot.id for g in golden}) == len(golden)
    for g in golden:
        kept = held_to_slot(g.slot, KeyJudgment(g.expected, "")).groups
        assert set(kept) == {tuple(sorted(x)) for x in g.expected}, g.id


def test_scoring_counts_a_wrong_join_against_precision():
    golden = [g for g in load_golden() if g.id == "c001"]
    wrong = {golden[0].slot.id: KeyJudgment((("bestia", "bestia-academy"),), "")}
    report: KeyReport = score(golden, wrong)
    assert (report.predicted, report.correct, report.precision) == (1, 0, 0.0)
    assert not report.passes(0.99)
