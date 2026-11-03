"""Re-judges subject merges one pair at a time and lists the ones the judge no longer calls the same.

The merge pass judges several pairs per request; this asks about each merge in effect on its own, from the registry
as it is now. A merge moves its propositions onto the kept subject, but each proposition still names the subject it
was read onto (`extracted_subject_id`), so both sides are rebuilt from the kept subject's propositions. A side with no
open contracts left (a game that has finished) is shown its resolved ones. Paid: one request per merge, about
$0.0005 with Haiku.

Writes nothing; undoing a merge is a person's call, from the subject's page in the controller.

  STAGE=prod AWS_PROFILE=prod ANTHROPIC_API_KEY_SECRET=anthropic-api-key poetry run audit-subject-merges
  ... audit-subject-merges --actor any --out /tmp/audit.json
"""
import collections
import dataclasses
import json
import logging
from pathlib import Path

import click
from gnomepy.registry import RegistryClient

from classifier.client.claude import ClaudeAsker
from classifier.constants import DEFAULT_SUBJECT_MERGE_MODELS
from classifier.propositions.merges import Judgment, SubjectContract, SubjectMergeJudge, SubjectProfile, Verdict, profile
from classifier.propositions.types import Proposition
from classifier.stages.subject_merges import MERGE_ACTOR
from classifier.workers.config import claude_client


@dataclasses.dataclass
class _Alias:
    alias_subject_id: int
    alias_key: str
    canonical_subject_id: int
    canonical_key: str
    actor: str
    date_created: str


@dataclasses.dataclass
class _Proposition:
    security_id: int
    subject_id: int
    extracted_subject_id: int
    fields: dict
    exchange_code: str | None = None
    event_title: str | None = None
    outcome_label: str | None = None
    event_resolved: bool | None = None


def split_sides(rows: list[_Proposition], alias_id: int, alias_key: str,
                canonical_key: str) -> dict[str, list[SubjectContract]]:
    """{key: contracts} for the merged subject (what was read onto it) and the kept subject (everything else), open
    contracts only unless a side has none."""
    sides: dict[str, list[tuple[bool, SubjectContract]]] = collections.defaultdict(list)
    for r in rows:
        key = alias_key if r.extracted_subject_id == alias_id else canonical_key
        try:
            proposition = Proposition.from_fields(key, r.fields)
        except (ValueError, KeyError):
            continue
        sides[key].append((bool(r.event_resolved), SubjectContract(r.exchange_code or "?", r.event_title or "",
                                                                    r.outcome_label or "", "", proposition)))
    out = {}
    for key, contracts in sides.items():
        open_ones = [c for resolved, c in contracts if not resolved]
        out[key] = open_ones or [c for _, c in contracts]
    return out


@click.command()
@click.option("--actor", default=MERGE_ACTOR, show_default=True, help="Only merges made by this actor; 'any' for all.")
@click.option("--provider", type=click.Choice(sorted(DEFAULT_SUBJECT_MERGE_MODELS)), default="anthropic", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Write every merge's judgment as JSON.")
def main(actor: str, provider: str, out: Path | None):
    logging.basicConfig(level=logging.WARNING)
    registry = RegistryClient()
    aliases = [a for a in registry._get("/proposition-subjects/merges", {}, _Alias) if actor == "any" or a.actor == actor]
    click.echo(f"{len(aliases)} merges in effect" + ("" if actor == "any" else f" by {actor}"))
    if not aliases:
        return

    subjects = {s.subject_id: s for s in registry.get_proposition_subjects(
        subject_ids=sorted({a.alias_subject_id for a in aliases} | {a.canonical_subject_id for a in aliases}))}
    by_canonical: dict[int, list[_Proposition]] = {}
    pairs, merges = [], []
    for a in aliases:
        if a.canonical_subject_id not in by_canonical:
            by_canonical[a.canonical_subject_id] = registry._get(
                "/contract-propositions", {"subjectId": a.canonical_subject_id, "denormalize": "true"}, _Proposition)
        sides = split_sides(by_canonical[a.canonical_subject_id], a.alias_subject_id, a.alias_key, a.canonical_key)
        if a.alias_key not in sides or a.canonical_key not in sides:
            click.echo(f"  skipped {a.alias_key} -> {a.canonical_key}: a side has no contracts")
            continue
        profiles = {k: profile(k, subjects[i].description if i in subjects else "", sides[k])
                    for k, i in ((a.alias_key, a.alias_subject_id), (a.canonical_key, a.canonical_subject_id))}
        pairs.append((profiles[a.alias_key], profiles[a.canonical_key]))
        merges.append(a)

    judge = SubjectMergeJudge(ClaudeAsker(claude_client, [provider], DEFAULT_SUBJECT_MERGE_MODELS), pairs_per_request=1)
    judgments: dict[tuple[str, str], Judgment | None] = judge.judge(pairs)
    doubtful = []
    for a in merges:
        j = judgments.get((a.alias_key, a.canonical_key))
        if j is None or j.verdict != Verdict.SAME:
            doubtful.append((a, j))
    click.echo(f"{len(merges)} re-judged, {len(doubtful)} not judged the same:")
    for a, j in doubtful:
        click.echo(f"  {a.alias_key} -> {a.canonical_key} ({a.actor}, {a.date_created[:16]}): "
                   f"{j.verdict.value if j else 'no answer'} - {j.reason if j else ''}")
    if out:
        out.write_text(json.dumps([
            {"alias": a.alias_key, "canonical": a.canonical_key, "actor": a.actor, "merged_at": a.date_created,
             "judgment": judgments[(a.alias_key, a.canonical_key)].to_json()
             if judgments.get((a.alias_key, a.canonical_key)) else None}
            for a in merges], indent=1))


if __name__ == "__main__":
    main()
