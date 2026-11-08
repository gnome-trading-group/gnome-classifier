"""A game subject's contracts split by fixture: the one match each contract is about.

A game subject is a pair of teams on a date, so it can hold two matches: esports and table-tennis teams meet twice
in a day, and a venue lists one slug for both. Contracts of different matches must never compare (the first match's
winner says nothing about the second's).

Each venue names its own game (`Contract.game`: Kalshi's event-ticker game segment, Polymarket's game slug), but
the ids don't match across venues, so games are matched by when they start:

- One venue's game id is one game, except where its events start more than `SPLIT` apart (one slug reused for two
  of a day's matches). One venue's games starting within `SAME` of each other are one game listed twice (Hyperliquid
  lists each question as its own event; Kalshi sometimes lists one game under two dates).
- When no venue has two games, the subject is one fixture, however far apart the venues' start times are: venues
  disagree by hours on tennis (a scheduled slot vs the actual start) and on delayed matches.
- Otherwise the venue with the most games anchors, and every other game joins the anchor game nearest its start, if
  that one is nearer by at least `CLEAR` than the next. Games that aren't, and games with no start time when there
  are several anchors, are left out: a missed relationship rather than a wrong one.
"""
import dataclasses
from collections import Counter
from datetime import datetime, timedelta
from typing import Protocol

# A venue's events for one match start up to ~1.5 h apart (Kalshi's map markets after the series market); a slug
# reused for two matches of one day has had them 4 h or more apart.
SPLIT = timedelta(hours=3)
SAME = timedelta(minutes=30)
CLEAR = timedelta(hours=1)


class Listed(Protocol):
    venue: str
    game: str | None
    start: datetime | None
    event_id: int | None


@dataclasses.dataclass
class _Game:
    venue: str
    members: list
    start: datetime | None


def split_fixtures[T: Listed](contracts: list[T]) -> tuple[list[list[T]], list[T]]:
    """(fixtures, contracts left out as ambiguous)."""
    games, left_out = _venue_games(contracts)
    timed: dict[str, list[_Game]] = {}
    untimed: list[_Game] = []
    for game in games:
        (timed.setdefault(game.venue, []) if game.start else untimed).append(game)
    if all(len(listed) <= 1 for listed in timed.values()):
        return [[c for c in contracts if c not in left_out]], left_out
    anchor_venue = max(timed, key=lambda v: (len(timed[v]), v == "KALSHI", v))
    anchors = sorted(timed[anchor_venue], key=lambda g: g.start)
    fixtures = [list(g.members) for g in anchors]
    for venue, listed in timed.items():
        if venue == anchor_venue:
            continue
        for game in listed:
            nearest = sorted((abs(game.start - a.start), i) for i, a in enumerate(anchors))
            if len(nearest) > 1 and nearest[1][0] - nearest[0][0] < CLEAR:
                left_out.extend(game.members)
            else:
                fixtures[nearest[0][1]].extend(game.members)
    for game in untimed:
        left_out.extend(game.members)
    return fixtures, left_out


def _venue_games[T: Listed](contracts: list[T]) -> tuple[list[_Game], list[T]]:
    """Each venue's games, and the contracts no start time places within an id listing two."""
    by_id: dict[tuple[str, str], list[T]] = {}
    for c in contracts:
        by_id.setdefault((c.venue, c.game or f"event:{c.event_id}"), []).append(c)
    games: list[_Game] = []
    left_out: list[T] = []
    for (venue, _), members in by_id.items():
        runs: list[list[datetime]] = []
        for t in sorted({c.start for c in members if c.start}):
            if runs and t - runs[-1][-1] <= SPLIT:
                runs[-1].append(t)
            else:
                runs.append([t])
        if len(runs) <= 1:
            games.append(_Game(venue, members, _usual_start(members)))
            continue
        for run in runs:
            in_run = [c for c in members if c.start in run]
            games.append(_Game(venue, in_run, _usual_start(in_run)))
        left_out.extend(c for c in members if c.start is None)
    return _listed_twice(games), left_out


def _usual_start(members: list) -> datetime | None:
    times = Counter(c.start for c in members if c.start)
    return times.most_common(1)[0][0] if times else None


def _listed_twice(games: list[_Game]) -> list[_Game]:
    by_venue: dict[str, list[_Game]] = {}
    for game in games:
        by_venue.setdefault(game.venue, []).append(game)
    out: list[_Game] = []
    for listed in by_venue.values():
        merged: list[_Game] = []
        for game in sorted((g for g in listed if g.start), key=lambda g: g.start):
            if merged and game.start - merged[-1].start <= SAME:
                merged[-1].members.extend(game.members)
            else:
                merged.append(_Game(game.venue, list(game.members), game.start))
        out.extend(merged + [g for g in listed if not g.start])
    return out
