"""Reads every security's proposition: the venue readers first, Claude for the rest, and the second side of each
binary pair as the negation of the first. Counts what each family was read by, so a venue rewording its rules
(which sends a family from the readers to the LLM) shows up in the logs instead of only in the bill.
"""
import dataclasses
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping

from classifier.propositions.inputs import ExtractionInput, is_primary_side
from classifier.propositions.llm import PropositionExtractor
from classifier.propositions.subjects import DEFAULT_CANDIDATES, KnownSubject, SubjectIndex, add_known, candidates_for, merge_candidates
from classifier.propositions.types import Extraction, Method, Proposition
from classifier.propositions.venue import claims, extract_venue

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExtractionReport:
    by_family: Counter = dataclasses.field(default_factory=Counter)
    failed: list[int] = dataclasses.field(default_factory=list)
    subject_descriptions: dict[str, str] = dataclasses.field(default_factory=dict)
    known: dict[str, KnownSubject] = dataclasses.field(default_factory=dict)
    fell_back: list[int] = dataclasses.field(default_factory=list)
    # Of `failed`: those only the LLM can read, with it off (not `fell_back` ones, which a reader should have read);
    # and those whose request no provider could take.
    needs_llm: list[int] = dataclasses.field(default_factory=list)
    unavailable: list[int] = dataclasses.field(default_factory=list)

    def log(self):
        for (family, method), n in sorted(self.by_family.items()):
            # Log wording kept stable: a metric filter counts LLM fallbacks per family.
            logger.info("propositions read: family=%s method=%s count=%d", family, method, n)
        if self.failed:
            logger.warning("propositions not read this pass: %d securities", len(self.failed))


def family(item: ExtractionInput) -> str:
    source = item.source
    venue = source.get("venue", item.exchange)
    if venue == "KALSHI":
        return f"KALSHI:{((source.get('market') or {}).get('ticker') or '').split('-')[0]}"
    if venue == "HYPERLIQUID":
        description = (source.get("outcome") or {}).get("description", "")
        return f"HYPERLIQUID:{description.split('|')[0] if description.startswith('class:') else 'free'}"
    return venue


_WAVE_ORDER = ("KALSHI", "POLYMARKET_INTL", "POLYMARKET_US", "HYPERLIQUID")


def _waves(groups: dict[str, tuple[str, list[ExtractionInput]]]) -> list[dict[str, tuple[str, list[ExtractionInput]]]]:
    by_venue: dict[str, dict] = defaultdict(dict)
    for group, value in groups.items():
        by_venue[value[1][0].source.get("venue", value[1][0].exchange)][group] = value
    ordered = [v for v in _WAVE_ORDER if v in by_venue] + sorted(v for v in by_venue if v not in _WAVE_ORDER)
    return [by_venue[v] for v in ordered]


def _record_answers(wave, answers, results: dict[int, Extraction], report: "ExtractionReport") -> None:
    for _, chunk in wave.values():
        for item in chunk:
            if item.security_id not in answers:
                report.failed.append(item.security_id)
                report.unavailable.append(item.security_id)
                continue
            answer = answers[item.security_id]
            if answer is None:
                report.failed.append(item.security_id)
                continue
            proposition, description = answer
            results[item.security_id] = Extraction(proposition, Method.LLM, item.source_hash)
            report.subject_descriptions.setdefault(proposition.subject, description)
            add_known(report.known, proposition, description)
            report.by_family[(family(item), Method.LLM.value)] += 1


def request_group(item: ExtractionInput) -> str:
    """Contracts read in one request. A venue's own event groups markets that the registry splits into separate
    events (each Kalshi strike of a ladder is its own event), so a ladder is read together and gets one subject."""
    source = item.source
    event = source.get("event") or {}
    if source.get("venue") == "KALSHI" and event.get("event_ticker"):
        return f"KALSHI:{event['event_ticker']}"
    if source.get("venue") in ("POLYMARKET_INTL", "POLYMARKET_US") and event.get("slug"):
        return f"{source['venue']}:{event['slug']}"
    return f"event:{item.event_id}"


def extract_propositions(
    items: list[ExtractionInput],
    extractor: PropositionExtractor | None,
    known: dict[str, KnownSubject] | None = None,
    index: SubjectIndex | None = None,
    partners: Mapping[int, tuple[Proposition, Method]] | None = None,
) -> tuple[dict[int, Extraction], ExtractionReport]:
    """{security_id: Extraction} for every security that could be read, and a report of how. `known` holds subjects
    already stored; with those read by the venue readers in this pass they are the LLM's candidate subjects.
    `partners` gives, for a second side whose first side isn't among `items`, the first side's stored proposition and
    how it was read; the second side is its negation."""
    partners = partners or {}
    report = ExtractionReport(known=dict(known or {}))
    results: dict[int, Extraction] = {}
    by_event: dict[int, list[ExtractionInput]] = defaultdict(list)
    for item in items:
        by_event[item.event_id].append(item)

    primaries = [i for i in items if is_primary_side(i)]
    llm_events: dict[str, tuple[str, list[ExtractionInput]]] = {}
    for item in primaries:
        proposition = extract_venue(item.source)
        if proposition is not None:
            results[item.security_id] = Extraction(proposition, Method.VENUE, item.source_hash)
            report.by_family[(family(item), Method.VENUE.value)] += 1
            add_known(report.known, proposition, item.event_title)
        else:
            if claims(item.source):
                # Log wording kept stable: a metric filter counts these and alarms on a family the readers lost.
                logger.warning("venue reader fell back: family=%s security=%d", family(item), item.security_id)
                report.fell_back.append(item.security_id)
            llm_events.setdefault(request_group(item), (item.event_title, []))[1].append(item)

    if llm_events and extractor is not None:
        if index is not None:
            index.add(list(report.known.values()))
        # One venue per wave, so a venue's contracts see the subjects earlier venues' contracts were given.
        for wave in _waves(llm_events):
            texts = {group: " ".join([title] + [i.outcome_label for i in chunk]) for group, (title, chunk) in wave.items()}
            nearest = dict(zip(texts, index.neighbours(list(texts.values()), DEFAULT_CANDIDATES))) if index else {}
            candidates = {
                group: merge_candidates(candidates_for(text, report.known), nearest.get(group, []))
                for group, text in texts.items()
            }
            before = set(report.known)
            answers = extractor.extract(wave, candidates)
            _record_answers(wave, answers, results, report)
            if index is not None:
                index.add([report.known[k] for k in report.known if k not in before])
    else:
        unread = [i.security_id for _, chunk in llm_events.values() for i in chunk]
        report.failed.extend(unread)
        fell_back = set(report.fell_back)
        report.needs_llm.extend(sid for sid in unread if sid not in fell_back)

    for item in items:
        if is_primary_side(item):
            continue
        partner = next((p for p in by_event[item.event_id] if p.security_id != item.security_id and is_primary_side(p)), None)
        # The venue readers already negate a No side they are given, but only primary sides reach them: the second
        # side of a binary is always the first side's negation.
        if partner is None and item.security_id in partners:
            stored, method = partners[item.security_id]
            results[item.security_id] = Extraction(stored.negate(), method, item.source_hash)
            continue
        read = results.get(partner.security_id) if partner else None
        if read is None:
            report.failed.append(item.security_id)
            if partner is not None and partner.security_id in report.needs_llm:
                report.needs_llm.append(item.security_id)
            if partner is not None and partner.security_id in report.unavailable:
                report.unavailable.append(item.security_id)
            if partner is not None and partner.security_id in report.fell_back:
                report.fell_back.append(item.security_id)
            continue
        results[item.security_id] = Extraction(read.proposition.negate(), read.method, item.source_hash)
    return results, report
