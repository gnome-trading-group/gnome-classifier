"""Scores relationship predictions against the golden pairs (`classifier/relationships/evaluation.py`).

Predictions are a JSONL file of `{"a": {"exchange", "exchange_security_id"}, "b": {...}, "type": "EQUIVALENT" |
"IMPLIES", "flags": [...]}` rows. The default is `baseline_judge.jsonl`: the rows the old pair-wise LLM judge wrote to
prod (2026-08-07/08) between golden securities, captured 2026-10-29. It only ever judged the pairs it was shown, so
its recall on the newer candidate and ladder pairs is not meaningful; its precision is.

With --derive it scores the new system instead: extracts propositions for every golden security (paid model calls
unless --propositions names a saved extraction), derives relationships per subject, keeps the rows the classifier would
store (the fewest that rebuild them), and scores the relation a consumer rebuilds from those rows, composed pairs
included.

Exits 1 when EQUIVALENT or IMPLIES precision is under --min-precision.

  poetry run eval-relationships [--predictions FILE] [--source stored_judge] [--show wrong|missed|flags|review|all]
  AWS_PROFILE=dev poetry run eval-relationships --derive --save-propositions /tmp/props.json
  poetry run eval-relationships --derive --propositions /tmp/props.json   # re-score a saved extraction, free
"""
import json
import logging
import sys
from pathlib import Path

import click
from gnomepy.registry.relations import Edge, close, relationship_type
from gnomepy.registry.types import ContractType

from classifier.constants import DEFAULT_PROPOSITION_MODELS, DEFAULT_VOYAGE_EMBEDDING_MODEL
from classifier.propositions.derive import EQUIVALENT, Contract, derivation_units, derive_unit, minimal_rows
from classifier.propositions.evaluation import load_inputs
from classifier.propositions.extract import extract_propositions
from classifier.propositions.llm import PropositionExtractor
from classifier.propositions.subjects import InMemorySubjectIndex
from classifier.propositions.types import Proposition
from classifier.relationships.evaluation import (
    GOLDEN_DIR,
    LABELS,
    Prediction,
    Report,
    Scored,
    TypeScore,
    load_golden,
    load_predictions,
    score,
)
from classifier.workers.config import claude_client, init_voyage

DEFAULT_MIN_PRECISION = 0.99


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def _type_line(name: str, t: TypeScore) -> str:
    return (f"  {name:<11} precision {_pct(t.precision)} ({t.correct}/{t.predicted})"
            f"   recall {_pct(t.recall)} ({t.found}/{t.expected})")


def _side(side: dict) -> str:
    return f"[{side['exchange']}] {side['title']} || {side['outcome_label']} || {(side.get('expiry') or '')[:16]}"


def _print_pairs(title: str, rows: list[Scored]) -> None:
    if not rows:
        return
    click.echo(f"\n{title} ({len(rows)}):")
    for s in rows:
        p = s.pair
        flags = f" flags={list(s.predicted_flags)} expected={list(p.flags)}" if s.predicted_flags or p.flags else ""
        click.echo(f"  {p.id} [{p.source}] expected {p.expected}, predicted {s.predicted}{flags}"
                   f"{'  (needs review)' if p.review else ''}")
        click.echo(f"    A {_side(p.raw['a'])}")
        click.echo(f"    B {_side(p.raw['b'])}")
        if p.note:
            click.echo(f"    note: {p.note}")


def _print_report(report: Report, show: set[str]) -> None:
    click.echo(f"{len(report.scored)} golden pairs")
    click.echo(_type_line("EQUIVALENT", report.equivalent))
    click.echo(_type_line("IMPLIES", report.implies))
    click.echo("\nconfusion (rows expected, columns predicted):")
    click.echo("  " + " " * 12 + "".join(f"{label:>13}" for label in LABELS))
    for expected in LABELS:
        click.echo(f"  {expected:<12}" + "".join(f"{report.confusion[(expected, p)]:>13}" for p in LABELS))
    if show & {"wrong", "all"}:
        _print_pairs("Wrong", report.wrong())
    if show & {"missed", "all"}:
        _print_pairs("Missed", report.missed())
    if show & {"flags", "all"}:
        _print_pairs("Flag mismatches", report.flag_mismatches())
    if show & {"review", "all"}:
        _print_pairs("Labels marked for review", [s for s in report.scored if s.pair.review])


def _listing(key: str) -> tuple[str, str]:
    exchange, _, security = key.partition("|")
    return exchange, security


def _derived_predictions(provider: str, model: str | None, saved: Path | None, save: Path | None) -> list[Prediction]:
    inputs = load_inputs()
    by_id = {item.security_id: key for key, item in inputs.items()}
    if saved:
        stored = json.loads(saved.read_text())
        propositions = {int(sid): Proposition.from_fields(p["subject"], p["fields"]) for sid, p in stored.items()}
    else:
        voyage = init_voyage()
        index = InMemorySubjectIndex(lambda texts, kind: voyage.embed(texts, model=DEFAULT_VOYAGE_EMBEDDING_MODEL, input_type=kind))
        extractor = PropositionExtractor(claude_client, [provider], {**DEFAULT_PROPOSITION_MODELS, **({provider: model} if model else {})})
        results, _ = extract_propositions(list(inputs.values()), extractor, index=index)
        propositions = {sid: e.proposition for sid, e in results.items()}
        if save:
            save.write_text(json.dumps({sid: {"subject": p.subject, "fields": p.to_fields()} for sid, p in propositions.items()}))
    by_event: dict[int, list] = {}
    for item in inputs.values():
        by_event.setdefault(item.event_id, []).append(item)
    complements = {}
    for items in by_event.values():
        if len(items) == 2 and all(i.contract_type == ContractType.BINARY for i in items):
            complements[items[0].security_id], complements[items[1].security_id] = items[1].security_id, items[0].security_id

    by_subject: dict[str, list[Contract]] = {}
    for sid, p in propositions.items():
        if "subject" not in p.unknown:
            by_subject.setdefault(p.subject, []).append(Contract(sid, inputs[by_id[sid]].exchange, p, complements.get(sid),
                                                                     inputs[by_id[sid]].event_id))
    stored: list[Edge] = []
    for contracts in by_subject.values():
        for unit in derivation_units(contracts):
            rows, _ = minimal_rows(unit, derive_unit(unit))
            stored += [Edge(r.security_id_a, r.security_id_b, r.relationship_type, r.flags) for r in rows]
    relation = close(stored, complements)

    predictions = []
    for x, y in sorted(relation):
        if x > y and (y, x) in relation:
            continue
        kind, a, b, flags = relationship_type(relation, x, y)
        predictions.append(Prediction(_listing(by_id[a]), _listing(by_id[b]), kind, flags))
    click.echo(f"{len(propositions)} propositions, {len(stored)} stored rows, {len(predictions)} rebuilt relationships")
    return predictions


@click.command()
@click.option("--golden", type=click.Path(exists=True, path_type=Path), default=GOLDEN_DIR / "pairs.jsonl")
@click.option("--predictions", type=click.Path(exists=True, path_type=Path), default=GOLDEN_DIR / "baseline_judge.jsonl")
@click.option("--source", multiple=True, help="Only score golden pairs from these sources (stored_judge, stored_manual, knn, ladder).")
@click.option("--show", multiple=True, type=click.Choice(["wrong", "missed", "flags", "review", "all"]), default=["wrong"])
@click.option("--min-precision", type=float, default=DEFAULT_MIN_PRECISION, show_default=True)
@click.option("--derive", is_flag=True, help="Score propositions + derivation instead of a predictions file.")
@click.option("--provider", type=click.Choice(sorted(DEFAULT_PROPOSITION_MODELS)), default="anthropic", show_default=True)
@click.option("--model", default=None, help="With --derive: the provider's model to extract with (default: the configured one).")
@click.option("--propositions", "saved", type=click.Path(exists=True, path_type=Path), default=None,
              help="With --derive: a saved extraction to re-score without model calls.")
@click.option("--save-propositions", "save", type=click.Path(path_type=Path), default=None)
def main(golden: Path, predictions: Path, source: tuple[str, ...], show: tuple[str, ...], min_precision: float,
         derive: bool, provider: str, model: str | None, saved: Path | None, save: Path | None):
    logging.basicConfig(level=logging.WARNING)
    pairs = load_golden(golden)
    if source:
        pairs = [p for p in pairs if p.source in source]
    rows = _derived_predictions(provider, model, saved, save) if derive else load_predictions(predictions)
    report = score(pairs, rows)
    _print_report(report, set(show))
    passed = report.passes(min_precision)
    click.echo(f"\n{'PASS' if passed else 'FAIL'}: precision gate {min_precision:.0%}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
