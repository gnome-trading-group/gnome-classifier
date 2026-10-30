"""Scores proposition extraction against the labelled propositions (`classifier/propositions/evaluation.py`).

  poetry run eval-propositions --venue-only           # code readers only: free, no model calls
  poetry run eval-propositions --provider anthropic   # also asks Claude for the rest (paid; needs the provider's key)
  poetry run eval-propositions --stored --exchange-id 2   # replay stored venue sources through the readers
  poetry run eval-propositions --stored --drift           # today's drift sample: code reads vs Claude (paid)

The provider key is read the way the workers read it (ANTHROPIC_API_KEY_SECRET in Secrets Manager, or Bedrock
credentials), so run it with AWS credentials for the account, e.g. AWS_PROFILE=dev.
"""
import logging
from collections import Counter

import click
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import ContractType

from classifier.constants import DEFAULT_PROPOSITION_MODELS, DEFAULT_VOYAGE_EMBEDDING_MODEL
from classifier.propositions.drift import check_drift, sample
from classifier.propositions.evaluation import EXACT_FIELDS, load_labelled, score
from classifier.propositions.extract import extract_propositions, family
from classifier.propositions.inputs import ExtractionInput
from classifier.propositions.llm import PropositionExtractor
from classifier.propositions.subjects import InMemorySubjectIndex
from classifier.propositions.venue import extract_venue
from classifier.workers.config import claude_client, init_voyage


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def _stored_items(exchange_id: int | None) -> list[ExtractionInput]:
    return [ExtractionInput(row.security_id, "", row.security_id, ContractType.BINARY, "", "", None, None,
                            row.source, row.source_hash)
            for row in RegistryClient().get_contract_sources(exchange_id=exchange_id)]


def _stored(exchange_id: int | None) -> None:
    """Parse rate of the venue readers over every stored source, per family."""
    seen, read = Counter(), Counter()
    for item in _stored_items(exchange_id):
        fam = family(item)
        seen[fam] += 1
        read[fam] += extract_venue(item.source) is not None
    for fam, n in seen.most_common():
        click.echo(f"  {fam:<40} {read[fam]:>6}/{n:<6} read by code ({read[fam] / n:.0%})")


@click.command()
@click.option("--venue-only", is_flag=True, help="Only the code readers; no model calls.")
@click.option("--provider", type=click.Choice(sorted(DEFAULT_PROPOSITION_MODELS)), default="anthropic", show_default=True)
@click.option("--model", default=None, help="The provider's model to extract with (default: the configured one).")
@click.option("--stored", is_flag=True, help="Replay stored venue sources from the registry through the readers.")
@click.option("--exchange-id", type=int, default=None)
@click.option("--drift", is_flag=True, help="With --stored: re-read today's sample of code-read contracts with Claude.")
@click.option("--no-embeddings", is_flag=True, help="Candidate subjects by word overlap only (no Voyage calls).")
@click.option("--show", type=click.Choice(["none", "fields", "subjects", "all"]), default="subjects")
def main(venue_only: bool, provider: str, model: str | None, stored: bool, exchange_id: int | None, drift: bool,
         no_embeddings: bool, show: str):
    logging.basicConfig(level=logging.WARNING)
    models = {**DEFAULT_PROPOSITION_MODELS, **({provider: model} if model else {})}
    if stored and drift:
        extractor = PropositionExtractor(claude_client, [provider], models)
        drifts = check_drift(sample(_stored_items(exchange_id)), extractor)
        click.echo(f"{len(drifts)} disagreements between the code readers and Claude")
        return
    if stored:
        _stored(exchange_id)
        return
    labelled = load_labelled()
    extractor = None if venue_only else PropositionExtractor(claude_client, [provider], models)
    index = None
    if extractor is not None and not no_embeddings:
        voyage = init_voyage()
        index = InMemorySubjectIndex(lambda texts, kind: voyage.embed(texts, model=DEFAULT_VOYAGE_EMBEDDING_MODEL, input_type=kind))
    results, extraction_report = extract_propositions([lp.item for lp in labelled], extractor, index=index)
    report = score(labelled, results)

    click.echo(f"{len(labelled)} labelled propositions, {len(report.missing)} not read")
    for (fam, method), n in sorted(extraction_report.by_family.items()):
        click.echo(f"  read: {fam:<36} {method:<6} {n}")
    click.echo(f"subject agreement {_pct(report.subject_agreement)} ({report.subject_agreed}/{report.subject_pairs} pairs)")
    click.echo(f"entity agreement  {_pct(report.entity_agreement)} ({report.entity_agreed}/{report.entity_pairs} pairs)")
    for field in EXACT_FIELDS + ("window_end",):
        click.echo(f"  {field:<12} {_pct(report.field_accuracy(field))} ({report.field_correct[field]}/{report.field_checked[field]})")
    if show in ("subjects", "all"):
        for group, members in report.split_groups:
            click.echo(f"split subject group {group}:")
            for key, subject in members:
                click.echo(f"    {subject}  <- {key}")
    if show in ("fields", "all"):
        for key, field, expected, got in report.disagreements:
            click.echo(f"  {key}: {field} expected {expected!r}, got {got!r}")


if __name__ == "__main__":
    main()
