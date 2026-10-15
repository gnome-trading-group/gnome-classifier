"""Static category and tag mappings from exchange metadata.

Exchanges already label their events: Kalshi and Polymarket US with a category, Polymarket International and Kalshi
series with tags, Hyperliquid through the template an event is rendered from. Mapping those labels onto the registry's
categories is deterministic and keeps every event of a series alike; the LLM only fills in what a label leaves open
(`None` from a category function, or fewer than MIN_STATIC_TAGS tags).
"""
import re
from collections.abc import Iterable

from classifier.constants import STANDARDIZED_CATEGORIES

MAX_TAGS = 6
MIN_STATIC_TAGS = 2

# Exchange category labels that span several registry categories map to None, which sends the event to the LLM.
_KALSHI_CATEGORY: dict[str, str | None] = {
    "Sports": "SPORTS",
    "Elections": "POLITICS",
    "Politics": "POLITICS",
    "Financials": "FINANCE",
    "Companies": "FINANCE",
    "Business": "FINANCE",
    "Commodities": "FINANCE",
    "Economics": "ECONOMICS",
    "Crypto": "CRYPTO",
    "Entertainment": "ENTERTAINMENT",
    "Climate and Weather": "WEATHER",
    "AI": "TECHNOLOGY",
    "Health": "HEALTH",
    "Science and Technology": None,
    "Mentions": None,
    "Social": None,
    "World": None,
    "Transportation": None,
}
# Consulted only when the Kalshi category itself is ambiguous.
_KALSHI_SERIES_TAG_CATEGORY = {
    "AI": "TECHNOLOGY",
    "Tech": "TECHNOLOGY",
    "Product launches": "TECHNOLOGY",
    "Earnings": "FINANCE",
    "Companies": "FINANCE",
    "Business": "FINANCE",
    "IPOs": "FINANCE",
    "Space": "SCIENCE",
    "Physics & Math": "SCIENCE",
    "Medicine": "HEALTH",
    "Public Health": "HEALTH",
    "Climate change": "WEATHER",
}

_POLYMARKET_US_CATEGORY: dict[str, str | None] = {
    "sports": "SPORTS",
    "politics": "POLITICS",
    "geopolitics": "POLITICS",
    "culture": "ENTERTAINMENT",
    "crypto": "CRYPTO",
    "macro": "ECONOMICS",
    "climate": "WEATHER",
    "technology": "TECHNOLOGY",
    "finance": "FINANCE",
    # Mixes science with public health (measles counts).
    "science": None,
}

# Polymarket International has no category field; its tags imply one. Keys are lowercased tag labels.
_POLYMARKET_INTL_TAG_CATEGORY = {
    **dict.fromkeys([
        "sports", "esports", "soccer", "football", "nfl (all)", "cfb (all)", "basketball", "hockey", "tennis",
        "cricket", "table tennis", "baseball", "golf", "mma", "ufc", "boxing", "f1", "formula 1", "combat sports",
        "olympics", "chess",
    ], "SPORTS"),
    **dict.fromkeys([
        "politics", "elections", "geopolitics", "us election", "global elections", "world elections", "midterms",
        "house elections", "senate elections",
    ], "POLITICS"),
    **dict.fromkeys(["crypto", "crypto prices", "token launch", "fdv", "stablecoins", "airdrops"], "CRYPTO"),
    **dict.fromkeys([
        "finance", "equities", "stocks", "business", "ipos", "ipo", "earnings", "commodities", "big tech", "indices",
    ], "FINANCE"),
    **dict.fromkeys([
        "economy", "fed", "fed rates", "economic policy", "macro indicators", "global rates", "inflation", "us cpi",
        "gdp", "jobs",
    ], "ECONOMICS"),
    **dict.fromkeys(["tech", "ai", "ai regulation"], "TECHNOLOGY"),
    **dict.fromkeys(["science", "space", "climate & science"], "SCIENCE"),
    **dict.fromkeys([
        "weather", "daily temperature", "highest temperature", "lowest temperature", "climate", "hurricanes",
    ], "WEATHER"),
    **dict.fromkeys(["health", "pandemics", "pandemic", "covid", "measles", "bird flu"], "HEALTH"),
    **dict.fromkeys([
        "culture", "pop culture", "movies", "music", "awards", "tv", "celebrities", "oscars", "grammys",
    ], "ENTERTAINMENT"),
    **dict.fromkeys(["courts", "scotus", "supreme court", "legal", "trials"], "LEGAL"),
}
# Broad tags Polymarket adds alongside a specific one (esports props are also tagged Culture, Fed markets Finance);
# they decide the category only when no specific tag does.
_POLYMARKET_INTL_WEAK_TAGS = frozenset({
    "culture", "pop culture", "business", "tech", "awards", "finance", "climate & science",
})

_HYPERLIQUID_TEMPLATE_CATEGORY: dict[str, str | None] = {
    "sportsContestResult": "SPORTS",
    "sportsContestWinner": "SPORTS",
    "sportsOverUnderMarket": "SPORTS",
    "sportsScalarMarket": "SPORTS",
    "sportsSpread": "SPORTS",
    "sportsTotal": "SPORTS",
    "sportsTournamentWinner": "SPORTS",
    "policyRateDecision": "ECONOMICS",
    "policyRateLadder": "ECONOMICS",
    "companyIpoConfirmed": "FINANCE",
    "companyIpoFirstDayMarketCap": "FINANCE",
    "aiModelHeadToHead": "TECHNOLOGY",
}
_HYPERLIQUID_PRICE_TEMPLATES = frozenset({"binaryPrice", "binaryPriceExternal", "priceTouch", "scalarPrice"})
_HYPERLIQUID_AWARD_TEMPLATES = frozenset({"awardWinner", "awardEntryWins"})
# HIP-3 perps from the xyz deployer are equities, indices and commodities rather than crypto.
_HYPERLIQUID_TRADFI_PREFIX = "xyz:"
_HYPERLIQUID_AWARD_FIELD_CATEGORY = {
    "soccer": "SPORTS", "football": "SPORTS", "basketball": "SPORTS", "tennis": "SPORTS",
    "film": "ENTERTAINMENT", "music": "ENTERTAINMENT", "television": "ENTERTAINMENT",
}
_HYPERLIQUID_TAG_KEYS = (
    "sport", "competition", "participantA", "participantB", "company", "institution", "award", "underlying", "perp",
    "shortName",
)

# Esports games are identified from slugs and tickers because exchanges label them inconsistently ("counter strike 2",
# "CS2", "counter-strike") or not at all (Kalshi tags every esports series just "Esports").
_POLYMARKET_ESPORTS_GAME = {
    "counter-strike": "cs2",
    "cs2": "cs2",
    "dota-2": "dota 2",
    "dota2": "dota 2",
    "league-of-legends": "league of legends",
    "lol": "league of legends",
    "valorant": "valorant",
    "overwatch": "overwatch",
    "ow": "overwatch",
    "rainbow-six-siege": "rainbow six siege",
    "r6": "rainbow six siege",
    "r6siege": "rainbow six siege",
    "mobile-legends-bang-bang": "mobile legends",
    "mlbb": "mobile legends",
    "honor-of-kings": "honor of kings",
    "hok": "honor of kings",
    "starcraft-2": "starcraft 2",
    "call-of-duty": "call of duty",
    "rocket-league": "rocket league",
}
_KALSHI_ESPORTS_TICKER_PREFIX = (
    ("KXCS2", "cs2"),
    ("KXCSGO", "cs2"),
    ("KXVALORANT", "valorant"),
    ("KXLOL", "league of legends"),
    ("KXDOTA2", "dota 2"),
    ("KXR6", "rainbow six siege"),
    ("KXCOD", "call of duty"),
    ("KXOW", "overwatch"),
    ("KXROCKETLEAGUE", "rocket league"),
    ("KXRL", "rocket league"),
    ("KXMLBB", "mobile legends"),
    ("KXESOCCER", "esoccer"),
    ("KXEBASKETBALL", "ebasketball"),
)
_KALSHI_ESPORTS_TAG = "Esports"

_TAG_SYNONYMS = {
    "counter strike 2": "cs2",
    "counter-strike": "cs2",
    "counter-strike 2": "cs2",
    "csgo": "cs2",
    "cs:go": "cs2",
    "lol": "league of legends",
    "val": "valorant",
    "dota2": "dota 2",
    "mobile legends: bang bang": "mobile legends",
    "mlbb": "mobile legends",
    "nfl (all)": "nfl",
    "usa national football league (nfl)": "nfl",
    "cfb (all)": "college football",
    "cfb": "college football",
    "ncaaf": "college football",
    "football/soccer": "soccer",
    "epl": "premier league",
    "english premier league": "premier league",
    "ucl": "champions league",
    "uefa champions league": "champions league",
    "uel": "europa league",
    "uefa europa league": "europa league",
    "uecl": "conference league",
    "europa conference league": "conference league",
    "us election": "us elections",
    "fed rates": "fed",
    "federal reserve": "fed",
    "federal reserve's open market committee": "fed",
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "ripple": "xrp",
    "scotus & courts": "scotus",
    "international elections": "global elections",
    "world elections": "global elections",
    "foreign elections": "global elections",
    "acquisitions": "acquisition",
    "ipos": "ipo",
}
# Tags that say nothing about the subject: exchange UI and market-mechanics labels, and words that fit any event.
_IGNORED_TAGS = frozenset({
    "games", "hide from new", "recurring", "up or down", "daily", "weekly", "monthly", "props", "season stats",
    "pre-market", "daily-close", "finance updown", "pyth finance", "main election", "margin of victory",
    "midterm mov", "kpis", "other", "series", "champion", "champions", "playoffs", "postseason", "conferences",
    "top 25", "stat leaders", "season stat leaders", "player stats", "hit price", "new", "featured", "trending",
    "breaking", "roster changes", "leagues", "win totals", "fantasy season", "race props",
    "international election props", "nov 4 elections", "macro election 2",
    "price", "prices", "market", "markets", "prediction", "predictions", "prediction market", "winner", "winners",
    "settlement", "resolution", "forecast", "outcome", "event", "odds", "betting", "twap", "oracle",
    "prediction markets", "binary", "touch", "spread", "total points", "over/under",
    *(c.lower() for c in STANDARDIZED_CATEGORIES),
})
# Tags unique to one event (a threshold, a date, a time window) can't group events.
_IGNORED_TAG_PATTERNS = tuple(re.compile(p) for p in (
    r"^(?=.*\d)[\d\s$%.,:/\-+]*(k|m|b|bn|t)?$",
    r"\b(19|20)\d{2}\b",
    r"\d{1,2}:\d{2}",
    r"^(above|below|over|under|at least|at most|more than|less than)\b",
    r"^earn \d",
    r"^rewards\b",
    r"\bcombos?$",
    r"^week \d+$",
    r"^\d+[mhd]$",
    r"\bupdown\b",
))


def clean_tags(raw_tags: Iterable, limit: int = MAX_TAGS) -> list[str]:
    """Lowercase, apply synonyms, drop subject-less and event-specific tags, dedupe; keeps the first `limit`."""
    tags: list[str] = []
    for raw in raw_tags:
        if not isinstance(raw, str):
            continue
        tag = " ".join(raw.lower().strip(" #").split())
        tag = _TAG_SYNONYMS.get(tag, tag)
        if not tag or tag in _IGNORED_TAGS or tag in tags or any(p.search(tag) for p in _IGNORED_TAG_PATTERNS):
            continue
        tags.append(tag)
    return tags[:limit]


def merge_tags(primary: Iterable[str], extra: Iterable[str]) -> list[str]:
    return clean_tags([*primary, *extra])


def _with_game(tags: list[str], game: str | None) -> list[str]:
    if game is None:
        return tags
    return clean_tags([game, "esports", *tags])


def kalshi_category(category: str | None, series_tags: list[str]) -> str | None:
    mapped = _KALSHI_CATEGORY.get(category) if isinstance(category, str) else None
    if mapped:
        return mapped
    implied = {_KALSHI_SERIES_TAG_CATEGORY[t] for t in series_tags if t in _KALSHI_SERIES_TAG_CATEGORY}
    return implied.pop() if len(implied) == 1 else None


def kalshi_tags(series_ticker: str | None, series_tags: list[str]) -> list[str]:
    game = None
    if series_ticker and _KALSHI_ESPORTS_TAG in series_tags:
        game = next((g for prefix, g in _KALSHI_ESPORTS_TICKER_PREFIX if series_ticker.startswith(prefix)), None)
    return _with_game(clean_tags(series_tags), game)


def polymarket_us_category(category: str | None) -> str | None:
    return _POLYMARKET_US_CATEGORY.get(category.lower()) if isinstance(category, str) else None


def polymarket_intl_category(tag_labels: list[str]) -> str | None:
    labels = [t.strip().lower() for t in tag_labels if isinstance(t, str)]
    mapped = [(t, _POLYMARKET_INTL_TAG_CATEGORY[t]) for t in labels if t in _POLYMARKET_INTL_TAG_CATEGORY]
    specific = {c for t, c in mapped if t not in _POLYMARKET_INTL_WEAK_TAGS}
    candidates = specific or {c for _, c in mapped}
    return candidates.pop() if len(candidates) == 1 else None


def polymarket_tags(tag_labels: list[str], series_slug: str | None, slug: str | None) -> list[str]:
    """Tags for a Polymarket International or US event, with its esports game when it has one."""
    game = _POLYMARKET_ESPORTS_GAME.get(series_slug.lower()) if isinstance(series_slug, str) else None
    if game is None and isinstance(slug, str) and any(isinstance(t, str) and t.strip().lower() == "esports" for t in tag_labels):
        # Esports match slugs lead with the game: "cs2-aaa-inf1-2026-03-10".
        game = _POLYMARKET_ESPORTS_GAME.get(slug.lower().split("-")[0])
    return _with_game(clean_tags(tag_labels), game)


def hyperliquid_category(template_id: str | None, values: dict[str, str]) -> str | None:
    if not isinstance(template_id, str):
        return None
    if template_id in _HYPERLIQUID_PRICE_TEMPLATES:
        asset = values.get("perp") or values.get("shortName") or ""
        return "FINANCE" if asset.lower().startswith(_HYPERLIQUID_TRADFI_PREFIX) else "CRYPTO"
    if template_id in _HYPERLIQUID_AWARD_TEMPLATES:
        return _HYPERLIQUID_AWARD_FIELD_CATEGORY.get((values.get("field") or "").lower())
    return _HYPERLIQUID_TEMPLATE_CATEGORY.get(template_id)


def hyperliquid_tags(values: dict[str, str]) -> list[str]:
    raw = [values.get(key) for key in _HYPERLIQUID_TAG_KEYS]
    return clean_tags(v.removeprefix(_HYPERLIQUID_TRADFI_PREFIX) for v in raw if isinstance(v, str))
