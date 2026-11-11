"""Scores the key judge against the golden slots (`classifier/propositions/key_evaluation.py`).

Asks the judge about every golden slot (paid model calls, unless --judgments names a saved run), then reports join
precision and recall over pairs of keys. Exits 1 when join precision is under --min-precision or a slot got no answer:
a wrong join makes two different entities' contracts comparable.

  AWS_PROFILE=dev ANTHROPIC_API_KEY_SECRET=anthropic-api-key poetry run eval-key-reconcile --save /tmp/keys.json
  poetry run eval-key-reconcile --judgments /tmp/keys.json --show all   # re-score a saved run, free
"""
import logging
import sys
from pathlib import Path

import click

from classifier.client.claude import ClaudeAsker
from classifier.constants import DEFAULT_SUBJECT_MERGE_MODELS
from classifier.propositions.key_evaluation import GOLDEN_PATH, load_golden, load_judgments, save_judgments, score
from classifier.propositions.key_reconcile import KeyJudge
from classifier.workers.config import claude_client

DEFAULT_MIN_PRECISION = 0.99


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def _show(title: str, rows) -> None:
    if not rows:
        return
    click.echo(f"\n{title} ({len(rows)} slots):")
    for golden, judgment, pairs in rows:
        click.echo(f"  {golden.id} [{golden.source}] {golden.slot.subject} {golden.slot.measure}/{golden.slot.scope}")
        for pair in sorted(sorted(p) for p in pairs):
            click.echo(f"    {pair[0]}  <>  {pair[1]}")
        if judgment and judgment.reason:
            click.echo(f"    reason: {judgment.reason}")
        if golden.note:
            click.echo(f"    note: {golden.note}")


@click.command()
@click.option("--golden", type=click.Path(exists=True, path_type=Path), default=GOLDEN_PATH)
@click.option("--show", multiple=True, type=click.Choice(["wrong", "missed", "all"]), default=["wrong"])
@click.option("--min-precision", type=float, default=DEFAULT_MIN_PRECISION, show_default=True)
@click.option("--provider", type=click.Choice(sorted(DEFAULT_SUBJECT_MERGE_MODELS)), default="anthropic", show_default=True)
@click.option("--model", default=None, help="The provider's model to judge with (default: the subject merge judge's).")
@click.option("--judgments", "saved", type=click.Path(exists=True, path_type=Path), default=None,
              help="A saved run to re-score without model calls.")
@click.option("--save", type=click.Path(path_type=Path), default=None)
def main(golden: Path, show: tuple[str, ...], min_precision: float, provider: str, model: str | None,
         saved: Path | None, save: Path | None):
    logging.basicConfig(level=logging.WARNING)
    slots = load_golden(golden)
    if saved:
        judgments = load_judgments(saved)
    else:
        asker = ClaudeAsker(claude_client, [provider], {**DEFAULT_SUBJECT_MERGE_MODELS, **({provider: model} if model else {})})
        judge = KeyJudge(asker)
        judgments = judge.judge([g.slot for g in slots])
        click.echo(f"tokens: {judge.input_tokens:,} in, {judge.output_tokens:,} out")
        if save:
            save_judgments(save, judgments)

    report = score(slots, judgments)
    click.echo(f"{len(slots)} golden slots, {report.expected} key pairs naming one entity")
    click.echo(f"  join precision {_pct(report.precision)} ({report.correct}/{report.predicted})"
               f"   recall {_pct(report.recall)} ({report.correct}/{report.expected})   unanswered {report.missing}")
    if set(show) & {"wrong", "all"}:
        _show("Wrong joins", report.wrong_joins())
    if set(show) & {"missed", "all"}:
        _show("Missed joins", report.missed_joins())
    passed = report.passes(min_precision)
    click.echo(f"\n{'PASS' if passed else 'FAIL'}: join precision gate {min_precision:.0%}, every slot answered")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
