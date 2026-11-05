"""Parsing helpers the venue readers share."""
import dataclasses
import re
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from classifier.propositions.canonical import GAME_SUBJECT_PREFIX, ROUND_WORDS, round_scope
from classifier.propositions.types import Comparator, Kind, Observation, Proposition, Stability

EASTERN = ZoneInfo("America/New_York")
CMP_WORDS = {"above": Comparator.GT, "below": Comparator.LT, "at or above": Comparator.GTE, "at or below": Comparator.LTE}

# Asset -> (subject, resolution source) for the CF Benchmarks real-time indices Kalshi and Polymarket US settle on.
CF_INDICES = {
    "BTC": ("btc-usd", "cf-benchmarks-brti"),
    "ETH": ("eth-usd", "cf-benchmarks-erti"),
    "SOL": ("sol-usd", "cf-benchmarks-solusd-rti"),
    "XRP": ("xrp-usd", "cf-benchmarks-xrpusd-rti"),
    "DOGE": ("doge-usd", "cf-benchmarks-dogeusd-rti"),
    "HYPE": ("hype-usd", "cf-benchmarks-hypeusd-rti"),
}
# Venues name the same index several ways: BRTI, ERTI, ETHUSDRTI, ETHUSD_RTI.
_INDEX_CODE_ASSETS = {"BRTI": "BTC", "ERTI": "ETH"}


def cf_index_asset(code: str) -> str | None:
    code = code.replace("_", "").upper()
    asset = _INDEX_CODE_ASSETS.get(code) or (code.removesuffix("USDRTI") if code.endswith("USDRTI") else None)
    return asset if asset in CF_INDICES else None


def number(text: str) -> float:
    return float(text.replace(",", ""))


def eastern(date_text: str, time_text: str) -> datetime:
    """A US Eastern date ("Oct 29, 2026" / "October 29, 2026") and time ("3 PM", "3:30PM") in UTC, summer or winter."""
    time_text = re.sub(r"\s*([ap])\.?m\.?$", r" \1M", time_text.strip(), flags=re.IGNORECASE).upper()
    time_format = "%I:%M %p" if ":" in time_text else "%I %p"
    for date_format in ("%b %d, %Y", "%B %d, %Y"):
        try:
            local = datetime.strptime(f"{date_text} {time_text}", f"{date_format} {time_format}")
        except ValueError:
            continue
        return local.replace(tzinfo=EASTERN).astimezone(timezone.utc)
    raise ValueError(f"not an Eastern date and time: {date_text!r} {time_text!r}")


def comparison(match: re.Match) -> dict:
    """Proposition fields from a match with `cmp` (above, below, at or above, at or below, between), `v1` and `v2`."""
    words = match.group("cmp")
    if words == "between":
        return {"kind": Kind.RANGE, "low": number(match.group("v1")), "high": number(match.group("v2"))}
    return {"kind": Kind.THRESHOLD, "comparator": CMP_WORDS[words], "value": number(match.group("v1"))}


def price_rise(subject: str, start: datetime, end: datetime, resolution_source: str) -> Proposition:
    """"Up or down" markets: the price at `end` is at least the price at `start`. Neither price is known when the market
    lists, so the question is the change over the window, not a threshold on the price."""
    return Proposition(
        subject=subject, kind=Kind.THRESHOLD, measure="change", unit="usd", comparator=Comparator.GTE, value=0.0,
        observation=Observation.AT_TIME, window_start=start, window_end=end, stability=Stability.LIVE,
        resolution_source=resolution_source,
    )


@dataclasses.dataclass(frozen=True)
class RateBank:
    subject: str
    measure: str
    resolution_source: str
    zone: ZoneInfo


def _bank(subject: str, measure: str, resolution_source: str, zone: str) -> RateBank:
    return RateBank(subject, measure, resolution_source, ZoneInfo(zone))


# The central banks whose rate decisions the venues list as fixed templates. Every venue's reading of one meeting gets
# the same subject (`{date}` the meeting day where the bank sits, `{month}` its month), measure and time, so the
# venues' outcomes compare. The names are the ones the model gave Polymarket's meetings before code read them.
RATE_BANKS = {
    "fed": _bank("fomc:{month}-meeting", "target_rate_change_bps", "federal-reserve", "America/New_York"),
    "ecb": _bank("ecb-deposit-rate:{date}", "deposit_rate_change_bps", "european-central-bank", "Europe/Berlin"),
    "boe": _bank("boe-bank-rate:{date}", "bank_rate_change_bps", "bank-of-england", "Europe/London"),
    "boj": _bank("boj-policy-rate:{date}", "policy_rate_change_bps", "bank-of-japan", "Asia/Tokyo"),
    "boc": _bank("boc-overnight-rate:{date}", "policy_rate_change_bps", "bank-of-canada", "America/Toronto"),
    "banxico": _bank("banxico-rate:{date}", "policy_rate_change_bps", "banco-de-mexico", "America/Mexico_City"),
    "bcb": _bank("brazil-selic:copom-{date}", "selic_rate_change_bps", "banco-central-do-brasil", "America/Sao_Paulo"),
    "rba": _bank("rba-cash-rate:{date}", "cash_rate_change", "reserve-bank-of-australia", "Australia/Sydney"),
    "rbnz": _bank("rbnz-official-cash-rate:{date}", "policy_rate_change_bps", "reserve-bank-of-new-zealand",
                  "Pacific/Auckland"),
    "bcch": _bank("chile-policy-rate:{date}", "policy_rate_change_bps", "central-bank-of-chile", "America/Santiago"),
    "banrep": _bank("banrep-rate:{date}", "policy_rate_change_bps", "banco-de-la-republica", "America/Bogota"),
    "cnb": _bank("cnb-policy-rate:{date}", "policy_rate_change_bps", "czech-national-bank", "Europe/Prague"),
    "nbp": _bank("nbp-policy-rate:{date}", "policy_rate_change_bps", "narodowy-bank-polski", "Europe/Warsaw"),
    "boi": _bank("bank-of-israel-rate:{date}", "policy_rate_change_bps", "bank-of-israel", "Asia/Jerusalem"),
    "bok": _bank("bok-base-rate:{date}", "base_rate_change_bps", "bank-of-korea", "Asia/Seoul"),
    "cbr": _bank("bank-of-russia-rate:{date}", "policy_rate_change_bps", "bank-of-russia", "Europe/Moscow"),
    "rbi": _bank("rbi-policy-repo:{date}", "policy_repo_rate_change_bps", "reserve-bank-of-india", "Asia/Kolkata"),
    "snb": _bank("snb-policy-rate:{date}", "policy_rate_change_bps", "swiss-national-bank", "Europe/Zurich"),
}


def rate_decision(bank: RateBank, meeting: date, fields: dict) -> Proposition:
    """A rate decision as announced on `meeting` (the bank's own date), observed at the end of that day where the bank
    sits: one instant every venue's reading shares, and fixed once announced."""
    at = datetime.combine(meeting, time(23, 59), tzinfo=bank.zone).astimezone(timezone.utc)
    return Proposition(
        subject=bank.subject.format(date=meeting.isoformat(), month=meeting.strftime("%Y-%m")), measure=bank.measure,
        unit="bps", observation=Observation.AT_TIME, window_end=at, stability=Stability.SETTLED,
        resolution_source=bank.resolution_source, **fields,
    )


# Party names and the letters venues put after a candidate ("Pete Sessions (R)"), as the entity keys the extractor uses.
PARTIES = {"democratic": "democratic", "democrat": "democratic", "republican": "republican",
           "independent": "independent", "libertarian": "libertarian", "green": "green"}
PARTY_LETTERS = {"D": "democratic", "R": "republican", "I": "independent", "L": "libertarian", "G": "green"}


def as_party_winner(proposition: Proposition, party: str | None) -> Proposition:
    """A winner proposition keyed by the party a market settles on, when it was keyed otherwise (by the candidate)."""
    if party is None or proposition.kind != Kind.WINNER or proposition.measure != "winner" or proposition.entity == party:
        return proposition
    return dataclasses.replace(proposition, entity=party, unknown=proposition.unknown - {"entity"})


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _abbreviates(key: str, name: str) -> bool:
    """Whether a key is an abbreviation of a name: its letters in order, starting with the name's first (`orst`, Oregon
    State), a trailing u for University dropped (`bsu`, Boise State)."""
    for short in {key, key.removesuffix("u") if len(key) > 2 else key}:
        rest = iter(name)
        if short[0] == name[0] and all(ch in rest for ch in short):
            return True
    return False


def names_team(team: str, code: str | None, key: str) -> int:
    """How surely an entity key the model chose stands for a team: 2 for the start of its venue code (`gt`, `gtech`)
    or of its name, its initials (`gt`, Georgia Tech) or one of its words (`lakers`, LA Lakers); 1 for an
    abbreviation of it; 0 for neither."""
    flat, name = _flat(key), _flat(team)
    words = re.findall(r"[a-z0-9]+", team.lower())
    if len(flat) < 2 or not name:
        return 0
    if (code and (code.startswith(flat) or flat.startswith(code)) or name.startswith(flat) or flat.startswith(name)
            or flat == "".join(w[0] for w in words) or flat in words):
        return 2
    return 1 if _abbreviates(flat, name) else 0


# A round of a count or contest named in a market's title: "(First Round)", "1st Round", "First Round Winner",
# "End of Round 1". A final round is left unnamed: in a runoff or instant-runoff count it is the count that decides
# the winner, which is what a market naming no round settles on.
_NAMED_ROUND = re.compile(r"\b(?:(?P<word>1st|first|2nd|second|3rd|third)[ -]round|round (?P<number>[1-3])\b)", re.I)
# "advance to the second round" names the round reached, not the one counted.
_ROUND_REACHED = re.compile(r"\b(?:to|reach(?:es)?|make|makes|into) the\s*$", re.I)
_OUTRIGHT = re.compile(r"\boutright\b", re.I)


def _titles(source: dict) -> list[str]:
    market, event = source.get("market") or {}, source.get("event") or {}
    return [t for t in (source.get("title"), market.get("title"), market.get("question"), market.get("groupItemTitle"),
                        event.get("title")) if t]


def _named_round(source: dict) -> str | None:
    found = set()
    for title in _titles(source):
        for m in _NAMED_ROUND.finditer(title):
            if not _ROUND_REACHED.search(title[:m.start()]):
                found.add(ROUND_WORDS[(m.group("word") or m.group("number")).lower()])
    return f"{found.pop()}_round" if len(found) == 1 else None


def with_named_round(source: dict, proposition: Proposition) -> Proposition:
    """A reading of a market that counts one named round (an election's first round, a golf round) scoped to that
    round, so it only compares with readings of the same round: topping a first round is not winning a runoff. An
    "outright" first-round win (a majority, no runoff needed) is its own measure, not topping the first round. Games are
    left alone: their rounds are the tournament's, not the game's."""
    if proposition.subject.startswith(GAME_SUBJECT_PREFIX) or proposition.kind == Kind.OTHER:
        return proposition
    named = _named_round(source)
    scope = round_scope(proposition.scope)
    if named is None:
        return proposition if scope in (None, proposition.scope) else dataclasses.replace(proposition, scope=scope)
    if proposition.scope is not None and scope is None:
        return proposition
    fixed = dataclasses.replace(proposition, scope=named, unknown=proposition.unknown - {"scope"})
    if fixed.kind == Kind.WINNER and fixed.measure == "winner" and any(_OUTRIGHT.search(t) for t in _titles(source)):
        fixed = dataclasses.replace(fixed, measure="outright_winner")
    return fixed
