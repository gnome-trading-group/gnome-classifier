from classifier.propositions.canonical import StoredAlias
from classifier.propositions.derive import EQUIVALENT, Contract, derivation_units, derive_unit
from classifier.propositions.key_reconcile import FLAG_PARTY_NOMINEE
from classifier.propositions.types import Comparator, Kind, Observation, Proposition

GAME = "game:nba:2026-11-11:bos-tor"
RACE = "us-senate-2026-texas"


def winner(subject: str, entity: str) -> Proposition:
    return Proposition(subject=subject, kind=Kind.WINNER, measure="winner", entity=entity, scope="match")


def derived(contracts: list[Contract], aliases=()) -> dict[tuple[int, int], tuple[str, tuple[str, ...]]]:
    out = {}
    for unit in derivation_units(contracts, aliases):
        for d in derive_unit(unit):
            out[(d.security_id_a, d.security_id_b)] = (d.relationship_type, d.flags)
    return out


def game() -> list[Contract]:
    return [
        Contract(1, "HYPERLIQUID", winner(GAME, "bos"), 2, 10, label="BOS"),
        Contract(2, "HYPERLIQUID", winner(GAME, "tor"), 1, 10, label="TOR"),
        Contract(3, "POLYMARKET_INTL", winner(GAME, "celtics"), 4, 20, label="Celtics"),
        Contract(4, "POLYMARKET_INTL", winner(GAME, "raptors"), 3, 20, label="Raptors"),
    ]


def test_two_spellings_of_one_team_relate_nothing_until_an_alias_joins_them():
    assert (1, 3) not in derived(game())
    rows = derived(game(), [StoredAlias("ENTITY", "bos", "celtics"), StoredAlias("ENTITY", "tor", "raptors")])
    assert rows[(1, 3)][0] == EQUIVALENT
    assert rows[(2, 4)][0] == EQUIVALENT
    assert FLAG_PARTY_NOMINEE not in rows[(1, 3)][1]


def test_an_alias_between_outcomes_of_one_event_is_left_out():
    rows = derived(game(), [StoredAlias("ENTITY", "bos", "tor")])
    assert all(t != EQUIVALENT for (a, b), (t, _) in rows.items() if {a, b} == {1, 2})


def test_relationships_resting_on_a_party_joined_to_its_nominee_are_flagged():
    contracts = [
        Contract(1, "KALSHI", winner(RACE, "james-talarico"), None, 10, label="James Talarico"),
        Contract(2, "KALSHI", winner(RACE, "ken-paxton"), None, 10, label="Ken Paxton"),
        Contract(3, "POLYMARKET_INTL", winner(RACE, "democratic"), None, 20, label="Talarico"),
        Contract(4, "POLYMARKET_INTL", winner(RACE, "republican"), None, 20, label="Paxton"),
    ]
    rows = derived(contracts, [StoredAlias("ENTITY", "james-talarico", "democratic", (FLAG_PARTY_NOMINEE,))])
    assert rows[(1, 3)] == (EQUIVALENT, tuple(sorted(rows[(1, 3)][1])))
    assert FLAG_PARTY_NOMINEE in rows[(1, 3)][1]
    assert all(FLAG_PARTY_NOMINEE not in flags for pair, (_, flags) in rows.items() if 1 not in pair)


def test_a_measure_alias_joins_two_names_of_one_quantity():
    def yards(sid: int, venue: str, measure: str) -> Contract:
        p = Proposition(subject="josh-allen:nfl-2026-27", kind=Kind.THRESHOLD, measure=measure, comparator=Comparator.GTE,
                        value=10.0, unit="touchdowns", observation=Observation.CUMULATIVE, scope="regular_season")
        return Contract(sid, venue, p, None, sid * 10)
    contracts = [yards(1, "KALSHI", "rushing_scores"), yards(2, "POLYMARKET_INTL", "rushing_touchdowns")]
    assert (1, 2) not in derived(contracts)
    assert derived(contracts, [StoredAlias("MEASURE", "rushing_scores", "rushing_touchdowns")])[(1, 2)][0] == EQUIVALENT


def test_a_chain_reads_through_to_its_end_and_is_still_refused_between_one_events_outcomes():
    rows = derived(game(), [StoredAlias("ENTITY", "bos", "boston"), StoredAlias("ENTITY", "boston", "celtics")])
    assert rows[(1, 3)][0] == EQUIVALENT
    rows = derived(game(), [StoredAlias("ENTITY", "bos", "boston"), StoredAlias("ENTITY", "boston", "tor")])
    assert all(t != EQUIVALENT for (a, b), (t, _) in rows.items() if {a, b} == {1, 2})
