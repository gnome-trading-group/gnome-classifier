"""Propositions read by Claude for contracts the venue readers don't cover, through the Anthropic API or Amazon
Bedrock as the service config says (`models.proposition_provider`).

One request covers one event's contracts (in chunks, so the answer stays well inside max_tokens), with the event's
rules and the venue's resolution fields, and the nearest existing subjects as candidates. The answer is constrained to
a JSON schema, then validated into Proposition objects; a contract whose answer doesn't validate gets None and stays
pending for the next pass, rather than a guess.
"""
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import anthropic

from classifier.propositions.inputs import ExtractionInput
from classifier.propositions.subjects import KnownSubject, add_known
from classifier.propositions.types import FIELDS, Comparator, Kind, Observation, Proposition, Stability

logger = logging.getLogger(__name__)

CHUNK_SIZE = 25
MAX_TOKENS = 16000
MAX_WORKERS = 8
EFFORT = "medium"

_SYSTEM = """You read prediction-market contracts into structured propositions: exactly what has to happen for the \
contract to pay out. A program compares propositions to find contracts that are equivalent or imply each other, and \
real money trades on those comparisons, so precision matters more than coverage.

For each contract, describe what makes THAT contract pay out (its outcome label says which side it is).

Fields:
- subject: the entity or event the question is about, as a lowercase key of letters, digits and : . - (e.g. \
btc-usd, spx, game:nfl:2026-11-01:ind-jax, us-senate-2026-control, super-troopers-3, josh-allen:nfl-2026-27, \
uk-gdp). Never put the measure, threshold or deadline in the subject: "Josh Allen 30+ passing TDs in 2026-27" and \
"Josh Allen 28.5+ passing TDs in 2026-27" share the subject josh-allen:nfl-2026-27.
  When a candidate subject is the same thing (same asset, same game on the same date, same race, same person and \
season), use its key exactly, and use its listed entity keys for the same teams, players or parties, even if this \
venue names them differently (a candidate game:ncaaf:2026-09-19:uncclem with entities unc, clem is the same game as \
"North Carolina vs. Clemson" on that date: use that key and entity clem for Clemson). Otherwise make a new key in the \
same style.
  League codes: nfl, nba, wnba, mlb, nhl, mls, ncaaf (college football), ncaab (college basketball), epl, \
laliga, seriea, bundesliga, ligue1, ucl, atp, wta, cs2, lol, valorant, dota2; other leagues by their short name. A game \
key is game:<league>:<date the game starts, US Eastern Time>:<teams> (a 8pm ET kickoff is that day, not the next UTC \
day). A team's or player's season is <entity>:<league>-<season> (liverpool:epl-2026-27, josh-allen:nfl-2026-27).
  Crypto assets are <asset>-usd whatever the quote currency (USDT, USDC); the venue's price feed goes in \
resolution_source.
- subject_description: one plain sentence saying what the subject is.
- kind: THRESHOLD (a number compared with a value), RANGE (a number between two values), OCCURS_BY (something \
happens by a deadline), WINNER (an entity wins a contest), OTHER. A market asking which of several candidates is \
chosen, wins or comes first (the next UN Secretary-General, a nominee, a winner) is WINNER with that entity, even \
when it names a deadline: OCCURS_BY is for whether one thing happens at all. A draw or tie in a match is WINNER with \
entity tie.
- measure: the quantity within the subject (price, total_points, team_points:<team>, margin:<team>, rt_score, \
seats, rate, daily_views_peak). A quantity of one team, player or party is <measure>:<entity> with the entity key \
(margin:democratic, receptions:darnell-mooney, team_points:mil). When you use a candidate subject, use its listed \
measure name for the same quantity. For OCCURS_BY, the event that must happen, as a short key (ipo, \
announce_ipo, make_playoffs, reach_nfc_championship, recognize_palestine): announcing something is a different event \
from it happening. Reuse a candidate's measure only when the event that must happen is exactly the same; a narrower \
or more specific event (win_super_bowl, compared with make_playoffs; a merger closing, compared with it being \
approved) gets its own key, even when one implies the other. Null for WINNER.
- comparator: GT, GTE, LT, LTE (THRESHOLD only). Use the rules' exact wording: "above"/"more than" is GT, "at least"/ \
"or above" is GTE.
- value, unit; low, high, low_inclusive, high_inclusive (RANGE). Copy each number exactly as the rules state it: \
"above 7574.9999" is GT 7574.9999, never GTE 7575; do not round or shift a boundary into the next comparator.
- observation: AT_TIME (value at one moment), ANY_TIME (reached at any point in the window, or its maximum/minimum \
over it), CUMULATIVE (built up over a game, season or other window: totals, a team's or player's points, a winning \
margin or spread). Null for OCCURS_BY. For WINNER: null for a contest decided once (a game, an election, a season \
title); AT_TIME when the entity must hold the position at window_end ("the leader on December 31"); ANY_TIME when it \
must hold it at any point by window_end ("leads Iran at some point in 2026").
- window_start, window_end: ISO 8601 UTC, the period in which the thing must happen or be measured (window_end \
is the decisive moment or deadline). Not when the market closes or the result is announced: "a recession in 2026" \
ends 2026-12-31 even if the data comes out in 2027. Convert local times; a date with no time in a Kalshi rule \
is Eastern Time ("before Jan 1, 2027" is 2027-01-01T05:00:00Z).
- entity: who must win (WINNER), as a short lowercase key: a candidate's entity key when it names the same team, \
player or party; otherwise a short name (clemson, democratic, republican, zohran-mamdani).
- scope: the part of the contest when it matters: match (the whole fixture: one game, or a whole best-of series), \
map:<n> (one game or map within a series, whatever the sport calls it: "Game 2 Winner" and "Map 2 Winner" are both \
map:2), regulation, first_half, playoff_series (several fixtures, as an NBA playoff series), regular_season, season. Null when not a sports contest (elections, awards, charts have no scope). A soccer result with a draw outcome is decided after 90 minutes plus stoppage time: \
regulation.
- stability: SETTLED if the measured value can no longer move around the observation time (a review score after \
release, published official statistics), otherwise LIVE.
- resolution_source: the named source that settles it, as a lowercase key (cf-benchmarks-brti, ap-race-call, \
rotten-tomatoes, binance-btcusdt), or null if none is named.
- unknown: every field above you could not determine from the text. Never guess a value; list it here instead. \
Placeholder outcomes (Person A, Other, E) are kind OTHER with subject and entity unknown.

Return one entry per contract id given, in any order."""

_NULLABLE_STRING = {"type": ["string", "null"]}
_NULLABLE_NUMBER = {"type": ["number", "null"]}


def _nullable_enum(values: list[str]) -> dict:
    return {"anyOf": [{"type": "string", "enum": values}, {"type": "null"}]}


_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "contract_id": {"type": "string"},
        "subject": {"type": "string"},
        "subject_description": {"type": "string"},
        "kind": {"type": "string", "enum": [k.value for k in Kind]},
        "measure": _NULLABLE_STRING,
        "comparator": _nullable_enum([c.value for c in Comparator]),
        "value": _NULLABLE_NUMBER,
        "unit": _NULLABLE_STRING,
        "low": _NULLABLE_NUMBER,
        "high": _NULLABLE_NUMBER,
        "low_inclusive": {"type": "boolean"},
        "high_inclusive": {"type": "boolean"},
        "observation": _nullable_enum([o.value for o in Observation]),
        "window_start": _NULLABLE_STRING,
        "window_end": _NULLABLE_STRING,
        "entity": _NULLABLE_STRING,
        "scope": _NULLABLE_STRING,
        "stability": {"type": "string", "enum": [s.value for s in Stability]},
        "resolution_source": _NULLABLE_STRING,
        "unknown": {"type": "array", "items": {"type": "string", "enum": list(FIELDS)}},
    },
    "required": ["contract_id", "subject", "subject_description", "kind", "measure", "comparator", "value", "unit",
                 "low", "high", "low_inclusive", "high_inclusive", "observation", "window_start", "window_end",
                 "entity", "scope", "stability", "resolution_source", "unknown"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"propositions": {"type": "array", "items": _ENTRY_SCHEMA}},
    "required": ["propositions"],
    "additionalProperties": False,
}


def _contract_block(item: ExtractionInput) -> dict:
    return {
        "contract_id": str(item.security_id),
        "outcome": item.outcome_label,
        "contract_type": item.contract_type.name,
        "expiry": item.expiry,
        "venue_source": item.source,
    }


def build_prompt(event_title: str, items: list[ExtractionInput], candidates: list[KnownSubject]) -> str:
    return json.dumps({
        "event_title": event_title,
        "candidate_subjects": [
            {"subject": c.key, "description": c.description, "entities": sorted(c.entities), "measures": sorted(c.measures)}
            for c in candidates
        ],
        "contracts": [_contract_block(i) for i in items],
    }, ensure_ascii=False, default=str)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"time {value!r} has no zone")
    return parsed


def to_proposition(entry: dict) -> tuple[Proposition, str]:
    """A validated proposition and its subject description, or ValueError. Fields that have no meaning for the kind
    are cleared rather than trusted, so two reads of the same contract can't differ on them."""
    unknown = frozenset(entry.get("unknown") or ())
    if entry["kind"] in (Kind.OCCURS_BY.value, Kind.WINNER.value):
        cleared = {"comparator", "value"}
        if entry["kind"] == Kind.WINNER.value:
            cleared.add("measure")
            # A winner's observation says when it must hold the position; a cumulative one has no meaning.
            if entry.get("observation") not in (Observation.AT_TIME.value, Observation.ANY_TIME.value):
                cleared.add("observation")
        else:
            cleared.add("observation")
        entry = {**entry, **{f: None for f in cleared}}
        unknown = unknown - cleared
    subject = entry["subject"] if "subject" not in unknown else "unknown"
    proposition = Proposition(
        subject=subject,
        kind=Kind(entry["kind"]),
        measure=entry.get("measure"),
        comparator=Comparator(entry["comparator"]) if entry.get("comparator") else None,
        value=entry.get("value"),
        low=entry.get("low"),
        high=entry.get("high"),
        low_inclusive=entry.get("low_inclusive", True),
        high_inclusive=entry.get("high_inclusive", True),
        unit=entry.get("unit"),
        observation=Observation(entry["observation"]) if entry.get("observation") else None,
        window_start=_parse_time(entry.get("window_start")),
        window_end=_parse_time(entry.get("window_end")),
        entity=entry.get("entity"),
        scope=entry.get("scope"),
        stability=Stability(entry["stability"]),
        resolution_source=entry.get("resolution_source"),
        unknown=unknown,
    )
    return proposition, entry.get("subject_description") or ""


def parse_answer(text: str, items: list[ExtractionInput]) -> dict[int, tuple[Proposition, str] | None]:
    by_id = {str(i.security_id): i.security_id for i in items}
    out: dict[int, tuple[Proposition, str] | None] = {i.security_id: None for i in items}
    try:
        entries = json.loads(text)["propositions"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Proposition answer is not the schema's JSON: %s", e)
        return out
    for entry in entries:
        security_id = by_id.get(str(entry.get("contract_id")))
        if security_id is None:
            continue
        try:
            out[security_id] = to_proposition(entry)
        except (ValueError, KeyError) as e:
            logger.warning("Proposition for security %s does not validate: %s", security_id, e)
    return out


def is_unavailable(error: Exception) -> bool:
    """Errors another provider might not have. A 400 is usually our request's fault, except running out of credit,
    which the API also reports as a 400."""
    if isinstance(error, (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                          anthropic.InternalServerError, anthropic.APIConnectionError, anthropic.RateLimitError)):
        return True
    return isinstance(error, anthropic.BadRequestError) and "credit balance" in str(error).lower()


class PropositionExtractor:
    """Asks each provider in order (the configured one, then the fallback when the first is unavailable). Both
    clients (`anthropic.Anthropic`, `anthropic.AnthropicBedrockMantle`) take the same Messages request."""

    def __init__(self, client_for: Callable[[str], object], providers: list[str], models: dict[str, str],
                 max_workers: int = MAX_WORKERS):
        missing = [p for p in providers if not models.get(p)]
        if not providers or missing:
            raise ValueError(f"proposition providers {providers} need a model each (missing {missing})")
        self._client_for = client_for
        self._providers = providers
        self._models = models
        self._max_workers = max_workers

    def _ask(self, prompt: str) -> str:
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                return self._ask_one(provider, prompt)
            except Exception as error:
                if not is_unavailable(error):
                    raise
                logger.warning("Proposition provider %s unavailable: %s", provider, error)
                last_error = error
        raise last_error

    def _ask_one(self, provider: str, prompt: str) -> str:
        response = self._client_for(provider).messages.create(
            model=self._models[provider],
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        )
        if response.stop_reason in ("max_tokens", "refusal"):
            raise RuntimeError(f"proposition request stopped: {response.stop_reason}")
        return next(b.text for b in response.content if b.type == "text")

    def extract(
        self,
        groups: dict[str, tuple[str, list[ExtractionInput]]],
        candidates: dict[str, list[KnownSubject]],
    ) -> dict[int, tuple[Proposition, str] | None]:
        """{request group: (title, inputs)} -> {security_id: (proposition, subject description) or None}. A contract
        the model answered without a valid entry maps to None; one whose request no provider could take (an outage,
        rate limit or exhausted credit) is left out, since nothing was learned about it.

        A group longer than one request is read first chunk first: the subjects that chunk chose are offered to the
        rest as candidates, so one event's contracts (36 Champions League teams) land on one subject rather than one
        per request."""
        first, rest = [], []
        for group, (title, items) in groups.items():
            chunks = [items[start:start + CHUNK_SIZE] for start in range(0, len(items), CHUNK_SIZE)]
            first.append((group, title, chunks[0]))
            rest.extend((group, title, chunk) for chunk in chunks[1:])
        results = self._ask_chunks(first, candidates)
        if rest:
            chosen: dict[str, dict[str, KnownSubject]] = {}
            for group, _, chunk in first:
                for item in chunk:
                    answer = results.get(item.security_id)
                    if answer is not None:
                        add_known(chosen.setdefault(group, {}), answer[0], answer[1])
            results.update(self._ask_chunks(rest, {
                group: list(chosen.get(group, {}).values()) + candidates.get(group, []) for group, _, _ in rest
            }))
        return results

    def _ask_chunks(
        self,
        jobs: list[tuple[str, str, list[ExtractionInput]]],
        candidates: dict[str, list[KnownSubject]],
    ) -> dict[int, tuple[Proposition, str] | None]:
        results: dict[int, tuple[Proposition, str] | None] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._ask, build_prompt(title, chunk, candidates.get(group, []))): chunk
                       for group, title, chunk in jobs}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    results.update(parse_answer(future.result(), chunk))
                except Exception as e:
                    # The PropositionExtractionFailures alarm counts this line.
                    logger.warning("Proposition request for %d contracts failed: %s", len(chunk), e)
                    if not is_unavailable(e):
                        results.update({i.security_id: None for i in chunk})
        return results
