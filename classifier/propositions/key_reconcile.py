"""Finds one entity spelled two ways on one subject, and has Claude say which keys name the same entity.

Derivation compares readings by their entity and measure keys, so one team, candidate or song keyed two ways (`phi` and
`philadelphia`, `bos` and `celtics`, `beauty-and-a-beat` and `beauty-and-a-beat-justin-bieber-nicki-minaj`) relates
nothing. Canonicalization (`canonical.py`) joins the spellings it can tell apart by rule; what is left is found here.

A slot is the readings of one kind, measure and scope ("winner of the match", "wins in the regular season"). A slot is
unmatched when two venues each use keys the other doesn't: one venue merely listing more outcomes than another (one
lists ten candidates, another six) is not a spelling difference. Only the unmatched keys of a slot go to the judge,
each with the outcome labels, event titles and a sample reading it was read from.

What the judge returns is held to the slot in code, whatever the model says: a group joins only keys of that slot, and
never two keys one event lists as different outcomes (two candidates on one ballot, a game's two teams).

A party and a candidate are one entity only once the venues show the candidate is the party's nominee for that contest
(a party's margin market labelled "Talarico 0-3%"): "Democrats win the Texas Senate seat" and "Talarico wins it" then
settle alike unless the party replaces him. Such a join is marked, so the relationships resting on it are flagged.
"""
import dataclasses
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from classifier.client.claude import ClaudeAsker, is_unavailable
from classifier.propositions.canonical import NOT_TEAMS, StoredAlias
from classifier.propositions.types import Proposition
from classifier.propositions.venue.common import PARTIES

logger = logging.getLogger(__name__)

SLOTS_PER_REQUEST = 4
MAX_TOKENS = 8000
# A relationship resting on a party joined to its nominee: the two settle apart if the party replaces its candidate.
FLAG_PARTY_NOMINEE = "PARTY_NOMINEE"
MAX_WORKERS = 8
EFFORT = "medium"
SAMPLES_PER_KEY = 2


@dataclasses.dataclass(frozen=True)
class Reading:
    """One security's canonical reading, with what it was read from."""
    security_id: int
    venue: str
    event_id: int | None
    proposition: Proposition
    label: str | None = None
    title: str | None = None


@dataclasses.dataclass(frozen=True)
class SlotKey:
    key: str
    venues: tuple[str, ...]
    labels: tuple[str, ...]
    titles: tuple[str, ...]
    samples: tuple[dict, ...]

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, row: dict) -> "SlotKey":
        return cls(row["key"], tuple(row["venues"]), tuple(row["labels"]), tuple(row["titles"]),
                   tuple(row["samples"]))


@dataclasses.dataclass(frozen=True)
class Slot:
    subject: str
    kind: str
    measure: str
    scope: str | None
    # Whether the keys sit in the measure (`margin:x`) rather than the entity.
    per_entity: bool
    keys: tuple[SlotKey, ...]
    # Pairs of keys one event lists as different outcomes: never one entity.
    rivals: frozenset[frozenset[str]] = frozenset()

    @property
    def id(self) -> str:
        return f"{self.subject}|{self.kind}|{self.measure}|{self.scope}|{'per-entity' if self.per_entity else 'entity'}"

    @property
    def key_names(self) -> frozenset[str]:
        return frozenset(k.key for k in self.keys)

    def to_json(self) -> dict:
        return {"subject": self.subject, "kind": self.kind, "measure": self.measure, "scope": self.scope,
                "per_entity": self.per_entity, "keys": [k.to_json() for k in self.keys],
                "rivals": sorted(sorted(r) for r in self.rivals)}

    @classmethod
    def from_json(cls, row: dict) -> "Slot":
        return cls(row["subject"], row["kind"], row["measure"], row.get("scope"), row["per_entity"],
                   tuple(SlotKey.from_json(k) for k in row["keys"]), frozenset(frozenset(r) for r in row["rivals"]))


def _names_entity(key: str | None) -> bool:
    # A number after a measure's colon is a parameter of the question (`top:4`, a top-four finish), not an entity.
    return bool(key) and key not in NOT_TEAMS and not key.isdigit()


def _keys_of(p: Proposition) -> list[tuple[tuple, str, bool]]:
    """(slot, key, per-entity) for each key a reading names: its entity, and the entity its measure names."""
    head, _, tail = (p.measure or "").partition(":")
    out = []
    if _names_entity(p.entity):
        out.append(((p.kind.value, head, p.scope, False), p.entity, False))
    if _names_entity(tail):
        out.append(((p.kind.value, head, p.scope, True), tail, True))
    return out


def _sample(r: Reading) -> dict:
    fields = r.proposition.to_fields()
    return {k: v for k, v in fields.items() if k in ("kind", "measure", "entity", "scope", "negated", "comparator",
                                                    "value", "low", "high", "unit") and v not in (None, False, [], "")}


def unmatched_slots(subject: str, readings: Iterable[Reading]) -> list[Slot]:
    """The subject's slots where two venues each use keys the other doesn't, with only those keys."""
    readings = list(readings)
    by_slot: dict[tuple, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    seen: dict[tuple, dict[str, list[Reading]]] = defaultdict(lambda: defaultdict(list))
    events: dict[tuple, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in readings:
        for slot, key, _ in _keys_of(r.proposition):
            by_slot[slot][r.venue].add(key)
            seen[slot][key].append(r)
            if r.event_id is not None and not r.proposition.negated:
                events[slot][r.event_id].add(key)
    out = []
    for slot, per_venue in by_slot.items():
        venues = sorted(per_venue)
        unmatched: set[str] = set()
        for i, a in enumerate(venues):
            for b in venues[i + 1:]:
                only_a, only_b = per_venue[a] - per_venue[b], per_venue[b] - per_venue[a]
                if only_a and only_b:
                    unmatched |= only_a | only_b
        if not unmatched:
            continue
        keys = []
        for key in sorted(unmatched):
            rs = seen[slot][key]
            keys.append(SlotKey(
                key=key,
                venues=tuple(sorted({r.venue for r in rs})),
                labels=tuple(sorted({r.label for r in rs if r.label})),
                titles=tuple(sorted({r.title for r in rs if r.title}))[:SAMPLES_PER_KEY],
                samples=tuple(_sample(r) for r in rs[:SAMPLES_PER_KEY]),
            ))
        rivals = frozenset(frozenset((x, y)) for ks in events[slot].values() for x in ks for y in ks
                           if x != y and x in unmatched and y in unmatched)
        kind, measure, scope, per_entity = slot
        out.append(Slot(subject, kind, measure, scope, per_entity, tuple(keys), rivals))
    return sorted(out, key=lambda s: s.id)


@dataclasses.dataclass(frozen=True)
class KeyJudgment:
    # Groups of the slot's keys that name one entity; a key in no group names an entity of its own.
    groups: tuple[tuple[str, ...], ...]
    reason: str
    # The groups that join a party to its nominee.
    nominee_groups: frozenset[tuple[str, ...]] = frozenset()

    def to_json(self) -> dict:
        return {"groups": [list(g) for g in self.groups], "reason": self.reason,
                "nominee_groups": sorted(list(g) for g in self.nominee_groups)}

    @classmethod
    def from_json(cls, row: dict) -> "KeyJudgment":
        return cls(tuple(tuple(g) for g in row.get("groups") or ()), row.get("reason", ""),
                   frozenset(tuple(g) for g in row.get("nominee_groups") or ()))


def held_to_slot(slot: Slot, judgment: KeyJudgment) -> KeyJudgment:
    """The judgment with every group the slot can't take dropped: one naming a key the slot doesn't hold, or two keys
    one event lists as different outcomes."""
    nominee = {tuple(sorted(set(g))) for g in judgment.nominee_groups}
    kept = []
    for group in judgment.groups:
        keys = tuple(sorted(set(group)))
        if len(keys) < 2:
            continue
        if not set(keys) <= slot.key_names:
            logger.warning("Key judgment for %s names keys outside the slot: %s; dropped", slot.id, keys)
            continue
        if any(frozenset((a, b)) in slot.rivals for a in keys for b in keys if a < b):
            logger.warning("Key judgment for %s joins outcomes of one event: %s; dropped", slot.id, keys)
            continue
        kept.append(keys)
    # A key in two groups would join both groups through it; the judge saying so is not a judgment to keep.
    counts = defaultdict(int)
    for keys in kept:
        for k in keys:
            counts[k] += 1
    kept = [keys for keys in kept if all(counts[k] == 1 for k in keys)]
    return KeyJudgment(tuple(kept), judgment.reason, frozenset(k for k in kept if k in nominee))


def aliases(slot: Slot, judgment: KeyJudgment, uses: dict[str, int] | None = None) -> dict[str, tuple[str, tuple[str, ...]]]:
    """{key: (the key its group is known by, flags)} for a judgment held to its slot. The group's key with the most
    readings, then the longest, stands for the entity everywhere; a party joined to its nominee is known by the party,
    as election winners are keyed, and flagged."""
    uses = uses or {}
    held = held_to_slot(slot, judgment)
    out = {}
    for group in held.groups:
        nominee = group in held.nominee_groups
        target = max(group, key=lambda k: (nominee and k in PARTIES, uses.get(k, 0), len(k), k))
        flags = (FLAG_PARTY_NOMINEE,) if nominee else ()
        out.update({k: (target, flags) for k in group if k != target})
    return out


def resolved(new: tuple[StoredAlias, ...], existing: list[StoredAlias]) -> list[StoredAlias]:
    """The aliases to store so every key the new aliases join, with the stored ones, reads as one key, and no alias
    names another alias as its target (the registry refuses a chain).

    Keys joined by stored or new aliases form a group. The group keeps the key its stored aliases already point at, so
    stored rows, which may belong to another merge, are never re-pointed: a new alias whose key is that stored target
    is turned round (stored `rounds -> rgg` and new `rgg -> roundsgg` store `roundsgg -> rgg`). With nothing stored,
    the group is known by the key the new aliases end at. A group whose stored aliases already point at two keys, or
    whose new aliases end nowhere (a cycle), is left out: joining it would mean re-pointing stored rows."""
    out = []
    for kind in sorted({a.kind for a in new}):
        stored = {a.key: a.canonical_key for a in existing if a.kind == kind}
        fresh = [a for a in new if a.kind == kind and a.key != a.canonical_key]
        parent: dict[str, str] = {}

        def find(k: str) -> str:
            parent.setdefault(k, k)
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        for key, canonical in list(stored.items()) + [(a.key, a.canonical_key) for a in fresh]:
            parent[find(key)] = find(canonical)
        groups: dict[str, set[str]] = defaultdict(set)
        for k in list(parent):
            groups[find(k)].add(k)
        stored_targets = set(stored.values()) - set(stored)
        new_keys = {a.key for a in fresh}
        for members in groups.values():
            mine = [a for a in fresh if a.key in members]
            if not mine:
                continue
            kept = members & stored_targets
            ends = {a.canonical_key for a in mine} - new_keys
            if len(kept) > 1 or (not kept and len(ends) != 1):
                logger.warning("key aliases left out, they would re-point stored aliases or form a cycle: %s; stored %s",
                               [(a.key, a.canonical_key) for a in mine],
                               sorted((k, v) for k, v in stored.items() if k in members))
                continue
            root = next(iter(kept or ends))
            flags = tuple(sorted({f for a in mine for f in a.flags}))
            for k in sorted(members - {root}):
                if stored.get(k) != root:
                    out.append(StoredAlias(kind, k, root, flags))
    return out


_SYSTEM = """You decide which keys in a prediction-market database name the same entity.

Contracts from several venues were read into propositions. Each names its entity with a key: a team, player, \
candidate, party, country, company, song or other outcome. Venues and readers spell one entity differently (a code \
and a full name, a nickname, with or without a suffix, a song with or without its artists), and a program can only \
compare two contracts when their keys match. Real money trades on those comparisons, so joining two different \
entities is far worse than leaving one entity split.

Each slot holds keys from one subject (one game, election, award, season...) used for one kind of question. For each \
slot, list groups of keys that name exactly the same entity, judging from the outcome labels, event titles and sample \
readings, not from the keys alone. Leave a key out of every group when it names an entity no other key names, or when \
you are not sure.

Never the same entity:
- a team and its academy, junior, youth, female or reserve roster ("BESTIA" and "BESTIA Academy");
- two schools or clubs sharing a word ("Iowa" and "Iowa State", "Miami" and "Miami (OH)");
- a party and a candidate, unless the venues' own labels show the candidate is that party's nominee in this contest \
(a party's market labelled with the candidate's name, "Talarico 0-3%" on a Democratic margin market): then they are \
one entity, and the group is marked party_nominee. Before a nomination is settled (a primary still to come, a \
presidential race years out) never join them;
- a candidate and their running mate, two people sharing a surname;
- two songs with one title by different artists; a company and its subsidiary or product;
- keys listed as different outcomes of one event.

reason: one short sentence a reviewer can check, about that slot's keys only. Return one entry per slot id given, echoing the slot's subject and measure exactly as given."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "slots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string"},
                    "subject": {"type": "string"},
                    "measure": {"type": "string"},
                    "groups": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"keys": {"type": "array", "items": {"type": "string"}},
                                       "party_nominee": {"type": "boolean"}},
                        "required": ["keys", "party_nominee"],
                        "additionalProperties": False,
                    }},
                    "reason": {"type": "string"},
                },
                "required": ["slot_id", "subject", "measure", "groups", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["slots"],
    "additionalProperties": False,
}


def build_prompt(slots: list[Slot]) -> str:
    return json.dumps({"slots": [{"slot_id": str(i), "subject": s.subject, "kind": s.kind, "measure": s.measure,
                                  "scope": s.scope, "keys_name": "the measure's entity" if s.per_entity else "the entity",
                                  "keys": [k.to_json() for k in s.keys]}
                                 for i, s in enumerate(slots)]}, ensure_ascii=False, default=str)


def parse_answer(text: str, slots: list[Slot]) -> dict[str, KeyJudgment | None]:
    out: dict[str, KeyJudgment | None] = {s.id: None for s in slots}
    try:
        entries = json.loads(text)["slots"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Key judgment answer is not the schema's JSON: %s", e)
        return out
    for entry in entries:
        try:
            slot = slots[int(entry["slot_id"])]
            # An answer filed under another slot's id would join keys the judge was reasoning about elsewhere.
            if (entry.get("subject"), entry.get("measure")) != (slot.subject, slot.measure):
                logger.warning("Key judgment for slot %s names %s/%s, not %s/%s; dropped", entry.get("slot_id"),
                               entry.get("subject"), entry.get("measure"), slot.subject, slot.measure)
                continue
            groups = [tuple(g["keys"]) for g in entry.get("groups") or ()]
            nominee = frozenset(tuple(sorted(set(g["keys"]))) for g in entry.get("groups") or () if g.get("party_nominee"))
            judgment = KeyJudgment(tuple(groups), entry.get("reason") or "", nominee)
            out[slot.id] = held_to_slot(slot, judgment)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning("Key judgment %r does not validate: %s", entry, e)
    return out


class KeyJudge:
    def __init__(self, asker: ClaudeAsker, max_workers: int = MAX_WORKERS, slots_per_request: int = SLOTS_PER_REQUEST):
        self._asker = asker
        self._max_workers = max_workers
        self._slots_per_request = slots_per_request
        # Tokens the judge has used, for the logs and the eval's cost line.
        self.input_tokens = 0
        self.output_tokens = 0

    def judge(self, slots: list[Slot]) -> dict[str, KeyJudgment | None]:
        """{slot id: judgment held to its slot}. A slot the model answered without a valid entry maps to None; one whose
        request no provider could take is left out, so it is asked again next pass."""
        chunks = [slots[i:i + self._slots_per_request] for i in range(0, len(slots), self._slots_per_request)]
        results: dict[str, KeyJudgment | None] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._asker.ask_with_usage, _SYSTEM, build_prompt(chunk), RESPONSE_SCHEMA, MAX_TOKENS,
                                   EFFORT): chunk for chunk in chunks}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    answer = future.result()
                    self.input_tokens += answer.input_tokens
                    self.output_tokens += answer.output_tokens
                    results.update(parse_answer(answer.text, chunk))
                except Exception as e:
                    logger.warning("Key judgment request for %d slots failed: %s", len(chunk), e)
                    if not is_unavailable(e):
                        results.update({s.id: None for s in chunk})
        return results
