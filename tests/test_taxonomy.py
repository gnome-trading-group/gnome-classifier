import pytest

from classifier import taxonomy
from classifier.constants import STANDARDIZED_CATEGORIES


def test_every_mapped_category_is_standardized():
    tables = [
        taxonomy._KALSHI_CATEGORY, taxonomy._KALSHI_SERIES_TAG_CATEGORY, taxonomy._POLYMARKET_US_CATEGORY,
        taxonomy._POLYMARKET_INTL_TAG_CATEGORY, taxonomy._HYPERLIQUID_TEMPLATE_CATEGORY,
        taxonomy._HYPERLIQUID_AWARD_FIELD_CATEGORY,
    ]
    for table in tables:
        assert {c for c in table.values() if c is not None} <= STANDARDIZED_CATEGORIES


# ── Kalshi ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("category, expected", [
    ("Sports", "SPORTS"), ("Elections", "POLITICS"), ("Financials", "FINANCE"), ("Commodities", "FINANCE"),
    ("Climate and Weather", "WEATHER"), ("Health", "HEALTH"), ("AI", "TECHNOLOGY"),
    ("Mentions", None), ("Science and Technology", None), ("Unknown", None), (None, None),
])
def test_kalshi_category(category, expected):
    assert taxonomy.kalshi_category(category, []) == expected


def test_kalshi_ambiguous_category_uses_series_tags():
    assert taxonomy.kalshi_category("Science and Technology", ["Space"]) == "SCIENCE"
    assert taxonomy.kalshi_category("Mentions", ["Earnings"]) == "FINANCE"
    assert taxonomy.kalshi_category("Science and Technology", ["AI", "Space"]) is None


@pytest.mark.parametrize("series, game", [
    ("KXCS2GAME", "cs2"), ("KXCS2MAP", "cs2"), ("KXCS2TOTALMAPS", "cs2"), ("KXCSGOADVANCE", "cs2"),
    ("KXDOTA2GAME", "dota 2"), ("KXLOLMAP", "league of legends"), ("KXVALORANTGAME", "valorant"),
    ("KXOWGAME", "overwatch"), ("KXR6MAP", "rainbow six siege"), ("KXMLBBGAME", "mobile legends"),
    ("KXESOCCERGAME", "esoccer"), ("KXEBASKETBALLGAME", "ebasketball"), ("KXROCKETLEAGUEGAME", "rocket league"),
])
def test_kalshi_esports_game_from_ticker(series, game):
    assert taxonomy.kalshi_tags(series, ["Esports"]) == [game, "esports"]


def test_kalshi_esports_prefix_needs_esports_tag():
    assert taxonomy.kalshi_tags("KXOWNERSHIP", ["Companies"]) == ["companies"]


# ── Polymarket ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("category, expected", [
    ("sports", "SPORTS"), ("geopolitics", "POLITICS"), ("macro", "ECONOMICS"), ("finance", "FINANCE"),
    ("Culture", "ENTERTAINMENT"), ("science", None), (None, None),
])
def test_polymarket_us_category(category, expected):
    assert taxonomy.polymarket_us_category(category) == expected


@pytest.mark.parametrize("tags, expected", [
    (["Sports", "Games", "Soccer", "UEL"], "SPORTS"),
    (["Fed", "Economy", "Recurring", "US CPI"], "ECONOMICS"),
    (["Crypto", "Pre-Market", "token launch"], "CRYPTO"),
    # Specific tags outrank the broad ones Polymarket adds alongside them.
    (["Culture", "Esports", "Sports"], "SPORTS"),
    (["Economy", "Fed Rates", "Finance"], "ECONOMICS"),
    (["Climate & Science", "Weather"], "WEATHER"),
    (["Culture", "Celebrities"], "ENTERTAINMENT"),
    # Two specific categories conflict, so the LLM decides.
    (["Economy", "Politics"], None),
    (["Recurring", "Hide From New"], None),
    ([], None),
])
def test_polymarket_intl_category(tags, expected):
    assert taxonomy.polymarket_intl_category(tags) == expected


@pytest.mark.parametrize("series, slug, tags, expected", [
    ("counter-strike", "cs2-aaa-inf1-2026-03-10", ["Esports", "counter strike 2", "Games", "Sports"], ["cs2", "esports"]),
    ("cs2", "cs2-re-nem-2026-09-24", ["Sports", "Games", "Esports", "CS2"], ["cs2", "esports"]),
    ("dota-2", "dota2-x-y", ["Esports", "Dota 2"], ["dota 2", "esports"]),
    (None, "lol-sgw-9g-2026-09-24", ["Esports", "League of Legends"], ["league of legends", "esports"]),
    (None, "will-faze-win-a-tier-1-event", ["Esports", "CS2", "counter strike 2"], ["esports", "cs2"]),
])
def test_polymarket_esports_game(series, slug, tags, expected):
    assert taxonomy.polymarket_tags(tags, series, slug) == expected


def test_polymarket_slug_prefix_needs_esports_tag():
    assert taxonomy.polymarket_tags(["Politics"], None, "ow-to-the-moon") == []


# ── Hyperliquid ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("template, values, expected", [
    ("sportsContestWinner", {}, "SPORTS"),
    ("policyRateDecision", {}, "ECONOMICS"),
    ("companyIpoConfirmed", {}, "FINANCE"),
    ("binaryPrice", {"perp": "BTC"}, "CRYPTO"),
    ("binaryPrice", {"perp": "xyz:GOLD"}, "FINANCE"),
    ("priceTouch", {"perp": "xyz:TSM"}, "FINANCE"),
    ("awardWinner", {"field": "Soccer"}, "SPORTS"),
    ("awardWinner", {"field": "Poetry"}, None),
    ("somethingNew", {}, None),
    (None, {}, None),
])
def test_hyperliquid_category(template, values, expected):
    assert taxonomy.hyperliquid_category(template, values) == expected


def test_hyperliquid_tags():
    values = {"sport": "football/soccer", "competition": "UEFA Europa League", "participantA": "Lyon",
              "participantB": "Crystal Palace", "season": "2026/27"}
    assert taxonomy.hyperliquid_tags(values) == ["soccer", "europa league", "lyon", "crystal palace"]
    assert taxonomy.hyperliquid_tags({"perp": "xyz:SP500"}) == ["sp500"]


# ── Tag cleanup ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tag", [
    "Games", "Hide From New", "Recurring", "5M", "Earn 3.25%", "Rewards 50, 4.5, 20", "Election Combos", "Week 6",
    "KPIs", "Sports", "Finance Updown", "86232", "2026", "above 8.0%",
])
def test_clean_tags_drops_noise(tag):
    assert taxonomy.clean_tags([tag]) == []


@pytest.mark.parametrize("raw, expected", [
    ("Counter Strike 2", "cs2"), ("NFL (All)", "nfl"), ("EPL", "premier league"), ("UCL", "champions league"),
    ("BTC", "bitcoin"), ("Fed Rates", "fed"), ("International elections", "global elections"),
])
def test_clean_tags_synonyms(raw, expected):
    assert taxonomy.clean_tags([raw]) == [expected]


def test_merge_tags_keeps_exchange_tags_first():
    assert taxonomy.merge_tags(["nba"], ["steph curry", "NBA", "retirement"]) == ["nba", "steph curry", "retirement"]
