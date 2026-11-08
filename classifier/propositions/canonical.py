"""One spelling per measure and per entity within a subject, so readings that name the same thing differently compare.

The extractor reads each contract on its own, so one subject collects spellings of one thing: `rushing_tds` and
`rushing_touchdowns`, `wins:new-york-j` and `wins:new-york-jets`, Hull as `hul` in one game market and `hull-city` in
another. Derivation sees them as different measures and entities and relates nothing.

Only two kinds of entity key are taken as one:
- the same two or more words with the last one cut short (`new-york-j`, `new-york-jets`): a venue's truncated label;
- in a game, a short code that begins the hyphenless key (`hul`, `hull-city`): a venue's team code.
- in a game, keys the winner markets list under one outcome name (`missouri` and `mizz`, both "Missouri"): one
  team's keys from two venues' readers.
A key is never extended by a word (`iowa` is not `iowa-state`, `miami` is not `miami-oh`), two keys an event lists as
different outcomes are never joined, and a key that could be either of two others is left alone.

In a game, a reading naming no part of the game is about the whole game (`match`): the venue readers scope the parts
their titles name (`with_named_part`), so a reading left unscoped is not a part of the game.
"""
import dataclasses
import re
from collections.abc import Iterable

from classifier.propositions.types import Kind, Proposition

GAME_SUBJECT_PREFIX = "game:"
FULL_GAME_SCOPE = "match"
# Outcome names and key words that are no team: a binary's sides, a game's other outcomes.
NOT_TEAMS = frozenset({"yes", "no", "over", "under", "tie", "draw", "other", "none", "neither"})
# Measure names the extractor spells two ways for one quantity.
MEASURE_ALIASES = {
    "regular_season_wins": "wins",
    "passing_tds": "passing_touchdowns",
    "rushing_tds": "rushing_touchdowns",
    "receiving_tds": "receiving_touchdowns",
    "fantasy_points_per_game": "fantasy_ppg",
    "vote_percent": "vote_share",
}
# A map's winner as the extractor spells it (`map_winner`, `map_2_winner`) is the map's `winner`, as the venue readers
# read it, when its scope names that map.
_MAP_WINNER = re.compile(r"^map_(?:(?P<map>\d+)_)?winner$")
_MAX_CODE_LENGTH = 4
ROUND_WORDS = {"1st": "first", "first": "first", "2nd": "second", "second": "second", "3rd": "third",
               "third": "third", "1": "first", "2": "second", "3": "third"}
# Spellings the extractor gives a round's scope.
_ROUND_SCOPE = re.compile(
    r"^(?:(?P<word>first|second|third)_round|round[_:]?(?P<number>[1-3])|(?P<nth>[1-3])(?:st|nd|rd)_round)$"
)


def _measure(p: Proposition) -> str | None:
    measure = p.measure
    if not measure:
        return measure
    m = _MAP_WINNER.match(measure)
    if p.kind == Kind.WINNER and m and (p.scope or "").startswith("map:") \
            and m.group("map") in (None, p.scope.partition(":")[2]):
        return "winner"
    head, sep, tail = measure.partition(":")
    return MEASURE_ALIASES.get(head, head) + sep + tail


def _keys(p: Proposition) -> list[str]:
    keys = [p.entity] if p.entity else []
    if p.measure and ":" in p.measure:
        keys.append(p.measure.partition(":")[2])
    keys.extend(entity for entity, _ in p.scores)
    return keys


def _same_entity(short: str, long: str, game: bool) -> bool:
    if short == long:
        return False
    a, b = short.split("-"), long.split("-")
    if len(a) == len(b) > 1 and a[:-1] == b[:-1] and len(a[-1]) < len(b[-1]) and b[-1].startswith(a[-1]):
        return True
    flat = long.replace("-", "")
    return game and len(short) <= _MAX_CODE_LENGTH and "-" not in short and len(flat) > len(short) \
        and flat.startswith(short)


def entity_aliases(readings: Iterable[tuple[int | None, Proposition, str | None]]) -> dict[str, str]:
    """{key: the key it is another spelling of} among one subject's (event_id, proposition, outcome label) readings."""
    readings = list(readings)
    keys: set[str] = set()
    listed: dict[int, set[str]] = {}
    game = False
    for event_id, p, _ in readings:
        found = _keys(p)
        keys.update(found)
        game = game or p.subject.startswith(GAME_SUBJECT_PREFIX)
        if event_id is not None:
            listed.setdefault(event_id, set()).update(found)
    rivals = {frozenset((a, b)) for found in listed.values() for a in found for b in found if a != b}
    aliases = {}
    for short in keys:
        longer = [k for k in keys if frozenset((short, k)) not in rivals and _same_entity(short, k, game)]
        if len(longer) == 1:
            aliases[short] = longer[0]
    # A chain (`a` -> `ab` -> `abc`) resolves to its end.
    aliases = {k: _resolve(k, aliases) for k in aliases}
    if game:
        aliases = _named_teams(readings, aliases)
    return aliases


def _named_teams(readings: list[tuple[int | None, Proposition, str | None]], aliases: dict[str, str]) -> dict[str, str]:
    """The aliases with keys the game's winner markets list under one outcome name joined, on top of the spellings
    already joined, as the longest of them (the key prefix aliasing would pick)."""
    resolve = lambda k: aliases.get(k, k)
    parent: dict[str, str] = {}

    def root(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    opponents: dict[int, set[str]] = {}
    for event_id, p, label in readings:
        if p.kind != Kind.WINNER or p.negated or not p.entity:
            continue
        if event_id is not None:
            opponents.setdefault(event_id, set()).add(resolve(p.entity))
        if not label:
            continue
        name = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        if name and name not in NOT_TEAMS and p.entity not in NOT_TEAMS:
            parent[root(resolve(p.entity))] = root("name:" + name)
    teams: dict[str, set[str]] = {}
    for x in list(parent):
        if not x.startswith("name:"):
            teams.setdefault(root(x), set()).add(x)
    out = dict(aliases)
    for members in teams.values():
        # Two keys one event lists as opponents are two teams, whatever their names say.
        if len(members) < 2 or any(len(members & found) > 1 for found in opponents.values()):
            continue
        target = max(members, key=lambda k: (len(k), k))
        for k in members - {target}:
            out[k] = target
        for short, long in aliases.items():
            if long in members:
                out[short] = target
    return out


def _resolve(key: str, aliases: dict[str, str]) -> str:
    seen = {key}
    while key in aliases and aliases[key] not in seen:
        key = aliases[key]
        seen.add(key)
    return key


def canonical(p: Proposition, aliases: dict[str, str]) -> Proposition:
    """The proposition with its measure's one spelling and every entity key replaced by the one it is short for."""
    def key(k: str | None) -> str | None:
        return aliases.get(k, k) if k else k

    measure = _measure(p)
    if measure and ":" in measure:
        head, _, tail = measure.partition(":")
        measure = f"{head}:{key(tail)}"
    scores = tuple((key(entity), value) for entity, value in p.scores)
    scope = p.scope
    if scope is None and p.subject.startswith(GAME_SUBJECT_PREFIX) and p.kind not in (Kind.OCCURS_BY, Kind.OTHER):
        scope = FULL_GAME_SCOPE
    if measure == p.measure and key(p.entity) == p.entity and scores == p.scores and scope == p.scope:
        return p
    return dataclasses.replace(p, measure=measure, entity=key(p.entity), scores=scores, scope=scope)


def canonical_readings(readings: list[tuple[int | None, Proposition, str | None]]) -> list[Proposition]:
    aliases = entity_aliases(readings)
    return [canonical(p, aliases) for _, p, _ in readings]


def round_scope(scope: str | None) -> str | None:
    """A round scope in one spelling (`round_1`, `round:1` and `1st_round` are `first_round`); None for any other."""
    m = _ROUND_SCOPE.match(scope or "")
    if m is None:
        return None
    return f"{ROUND_WORDS[m.group('word') or m.group('number') or m.group('nth')]}_round"
