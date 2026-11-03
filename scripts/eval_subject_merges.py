"""Scores the subject merge judge against the golden subject pairs (`classifier/propositions/merge_evaluation.py`).

Asks the judge about every golden pair (paid model calls, unless --judgments names a saved run), then reports SAME
precision and recall and the confusion of verdicts. Exits 1 when SAME precision is under --min-precision: a wrong SAME
merges two different contests.

  AWS_PROFILE=dev ANTHROPIC_API_KEY_SECRET=anthropic-api-key poetry run eval-subject-merges --save /tmp/judgments.json
  poetry run eval-subject-merges --judgments /tmp/judgments.json --show all   # re-score a saved run, free
"""
import logging
import sys
from pathlib import Path

import click

from classifier.constants import DEFAULT_PROPOSITION_MODELS
from classifier.client.claude import ClaudeAsker
from classifier.propositions.merge_evaluation import (
    GOLDEN_PATH,
    MISSING,
    GoldenSubjectPair,
    load_golden,
    load_judgments,
    save_judgments,
    score,
)
from classifier.propositions.merges import Judgment, SubjectMergeJudge, Verdict
from classifier.workers.config import claude_client

DEFAULT_MIN_PRECISION = 0.99
_COLUMNS = [v.value for v in Verdict] + [MISSING]


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def _print_pairs(title: str, rows: list[tuple[GoldenSubjectPair, Judgment | None]]) -> None:
    if not rows:
        return
    click.echo(f"\n{title} ({len(rows)}):")
    for pair, judgment in rows:
        verdict = judgment.verdict.value if judgment else MISSING
        click.echo(f"  {pair.id} [{pair.source}] expected {pair.expected.value}, judged {verdict}: "
                   f"{pair.a.key}  <>  {pair.b.key}")
        if judgment and judgment.reason:
            click.echo(f"    reason: {judgment.reason}")
        if pair.note:
            click.echo(f"    note: {pair.note}")


@click.command()
@click.option("--golden", type=click.Path(exists=True, path_type=Path), default=GOLDEN_PATH)
@click.option("--source", multiple=True, help="Only score golden pairs from these candidate sources (token, shared, embedding, known).")
@click.option("--show", multiple=True, type=click.Choice(["wrong", "missed", "all"]), default=["wrong"])
@click.option("--min-precision", type=float, default=DEFAULT_MIN_PRECISION, show_default=True)
@click.option("--provider", type=click.Choice(sorted(DEFAULT_PROPOSITION_MODELS)), default="anthropic", show_default=True)
@click.option("--model", default=None, help="The provider's model to judge with (default: the configured one).")
@click.option("--judgments", "saved", type=click.Path(exists=True, path_type=Path), default=None,
              help="A saved run to re-score without model calls.")
@click.option("--save", type=click.Path(path_type=Path), default=None)
def main(golden: Path, source: tuple[str, ...], show: tuple[str, ...], min_precision: float, provider: str,
         model: str | None, saved: Path | None, save: Path | None):
    logging.basicConfig(level=logging.WARNING)
    pairs = load_golden(golden)
    if source:
        pairs = [p for p in pairs if p.source in source]
    if saved:
        judgments = load_judgments(saved)
    else:
        asker = ClaudeAsker(claude_client, [provider], {**DEFAULT_PROPOSITION_MODELS, **({provider: model} if model else {})})
        judgments = SubjectMergeJudge(asker).judge([(p.a, p.b) for p in pairs])
        if save:
            save_judgments(save, judgments)

    report = score(pairs, judgments)
    click.echo(f"{len(pairs)} golden subject pairs, {report.expected_same} SAME")
    click.echo(f"  SAME precision {_pct(report.precision)} ({report.correct_same}/{report.predicted_same})"
               f"   recall {_pct(report.recall)} ({report.correct_same}/{report.expected_same})")
    click.echo("\nconfusion (rows expected, columns judged):")
    click.echo("  " + " " * 10 + "".join(f"{c:>11}" for c in _COLUMNS))
    for expected in (v.value for v in Verdict):
        click.echo(f"  {expected:<10}" + "".join(f"{report.confusion[(expected, c)]:>11}" for c in _COLUMNS))
    if show and set(show) & {"wrong", "all"}:
        _print_pairs("Wrong merges", report.wrong_merges())
    if show and set(show) & {"missed", "all"}:
        _print_pairs("Missed merges", report.missed_merges())

    passed = report.passes(min_precision)
    click.echo(f"\n{'PASS' if passed else 'FAIL'}: SAME precision gate {min_precision:.0%}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
