import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from classifier.propositions.types import Kind, Stability
from classifier.propositions.venue import claims, extract_venue
from classifier.propositions.venue.common import team_key

# Sources as the Polymarket adapters stored them, from the live venues (2026-10-31 to 2026-11-10).
SOURCES = json.loads((Path(__file__).parent / "fixtures/relationships/esports_winner_sources.json").read_text())


def source(key: str) -> dict:
    return copy.deepcopy(SOURCES[key])


def read(key_or_source):
    src = source(key_or_source) if isinstance(key_or_source, str) else key_or_source
    assert claims(src)
    return extract_venue(src)


class TestPolymarketIntl:
    def test_a_match_winner_is_the_outcome_winning_the_match(self):
        p = read("pmi-match-isurus")
        assert (p.subject, p.kind, p.measure, p.entity, p.scope, p.negated) == (
            "game:cs2:2026-11-10:isg-reda", Kind.WINNER, "winner", "isurus", "match", False)
        assert p.resolution_source == "hltv"
        assert p.stability == Stability.LIVE
        assert p.window_start == datetime(2026, 11, 10, 19, 20, tzinfo=timezone.utc)

    def test_each_outcome_is_its_own_team(self):
        assert read("pmi-match-red-canids").entity == "red-canids-academy"

    def test_a_map_winner_is_scoped_to_its_map(self):
        p = read("pmi-map2")
        assert (p.subject, p.entity, p.scope) == ("game:cs2:2026-11-10:isg-reda", "red-canids-academy", "map:2")

    def test_names_with_stray_spaces_still_read(self):
        assert read("pmi-spaced-outcome").entity == "saint-sinners"

    def test_the_subject_is_the_slug_so_a_moved_match_keeps_it(self):
        src = source("pmi-match-isurus")
        src["market"]["eventStartTime"] = "2026-11-17T18:00:00Z"
        assert read(src).subject == "game:cs2:2026-11-10:isg-reda"

    @pytest.mark.parametrize("change", [
        lambda m: m.update(question="Counter-Strike: Isurus vs MORROW (BO3) - CCT South America Group Stage"),
        lambda m: m.update(outcomes='["Isurus", "MORROW"]'),
        lambda m: m.update(description=m["description"].replace("win the match against RED Canids Academy", "win")),
        lambda m: m.update(outcomes='["Isurus"]'),
        lambda m: m.update(eventStartTime=None, gameStartTime=None),
    ])
    def test_anything_off_the_template_is_left_to_the_llm(self, change):
        src = source("pmi-match-isurus")
        change(src["market"])
        assert read(src) is None

    def test_a_slug_of_another_form_is_left_to_the_llm(self):
        src = source("pmi-match-isurus")
        src["event"]["slug"] = "counter-strike-isurus-red-canids"
        assert read(src) is None


class TestPolymarketUs:
    def test_a_match_winners_long_leg_is_its_first_team(self):
        p = read("pmus-match-long")
        assert (p.subject, p.kind, p.measure, p.entity, p.scope, p.negated) == (
            "game:cs2:2026-11-10:isg-reda", Kind.WINNER, "winner", "isurus", "match", False)
        assert p.resolution_source == "valve"

    def test_a_match_winners_short_leg_is_its_other_team_winning(self):
        p = read("pmus-match-short")
        assert (p.entity, p.negated) == ("red-canids-academy", False)

    def test_an_event_titled_with_a_map_does_not_rescope_a_match_winner(self):
        src = source("pmus-match-event-titled-map")
        assert "Map 2" in src["event"]["title"]
        assert read(src).scope == "match"

    def test_a_map_winner_is_its_titled_team_winning_that_map(self):
        p = read("pmus-map-long")
        assert (p.entity, p.scope, p.negated) == ("psychoface", "map:1", False)

    def test_a_map_winners_short_leg_is_the_no(self):
        p = read("pmus-map-short")
        assert (p.entity, p.scope, p.negated) == ("psychoface", "map:1", True)

    def test_a_doubleheaders_second_match_has_its_own_subject(self):
        assert read("pmus-doubleheader").subject == "game:cs2:2026-11-03:falf-maybe-dh2"

    def test_venues_listing_one_game_alike_share_its_subject(self):
        assert read("pmus-match-long").subject == read("pmi-match-isurus").subject

    @pytest.mark.parametrize("change", [
        lambda s: s["market"].update(title="Isurus vs MORROW"),
        lambda s: s["sides"][1].update(description="MORROW"),
        lambda s: s["market"].update(description=s["market"]["description"].replace("CS2 match", "Valorant match")),
        lambda s: s["sides"][0].update(long=False),
    ])
    def test_anything_off_the_match_template_is_left_to_the_llm(self, change):
        src = source("pmus-match-long")
        change(src)
        assert read(src) is None

    def test_a_map_title_naming_neither_team_is_left_to_the_llm(self):
        src = source("pmus-map-long")
        src["market"]["title"] = "Somebody wins map 1"
        assert read(src) is None


def test_team_keys_keep_every_word_that_names_a_roster():
    assert team_key("BESTIA Academy") != team_key("BESTIA")
    assert team_key("9INE Academy") == "9ine-academy"
    assert team_key("Sökerorg") == "sokerorg"
