import logging
from contextlib import contextmanager
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector.psycopg2 import register_vector

from gnomepy.registry.types import ContractType, Currency, Event, EventContract, Listing, Security

from classifier.propositions.derive import Contract
from classifier.propositions.inputs import ExtractionInput
from classifier.propositions.subjects import KnownSubject
from classifier.propositions.types import Proposition

logger = logging.getLogger(__name__)


class ClassifierDB:
    def __init__(self, dsn: str):
        self._pool = psycopg2.pool.SimpleConnectionPool(1, 2, dsn)

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            register_vector(conn)
            yield conn
        finally:
            self._pool.putconn(conn)

    def get_exchange_event(self, exchange_id: int, native_id: str) -> int | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id FROM sm.event"
                    " WHERE exchange_id = %s AND native_event_id = %s",
                    (exchange_id, native_id),
                )
                row = cur.fetchone()
                return row[0] if row else None

    def get_exchange_events(self, keys: list[tuple[int, str]]) -> dict[tuple[int, str], int]:
        if not keys:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                exchange_ids, native_ids = zip(*keys)
                cur.execute(
                    "SELECT exchange_id, native_event_id, event_id"
                    " FROM sm.event"
                    " WHERE (exchange_id, native_event_id) IN"
                    " (SELECT unnest(%s::int[]), unnest(%s::text[]))",
                    (list(exchange_ids), list(native_ids)),
                )
                return {(row[0], row[1]): row[2] for row in cur.fetchall()}

    def get_events(self, event_ids: list[int]) -> dict[int, dict]:
        """Returns {event_id: {"title", "category", "tags"}} for the given ids."""
        if not event_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, title, category, tags FROM sm.event"
                    " WHERE event_id = ANY(%s)",
                    (event_ids,),
                )
                return {
                    row[0]: {"title": row[1], "category": row[2] or "OTHER", "tags": row[3] or []}
                    for row in cur.fetchall()
                }

    def get_currencies(self) -> dict[str, int]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol, currency_id FROM sm.currency")
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_unlisted_securities(self, symbols: list[str]) -> dict[str, int]:
        """Returns {symbol: security_id} for securities with these symbols that have no listing."""
        if not symbols:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT s.symbol, s.security_id FROM sm.security s"
                    " WHERE s.symbol = ANY(%s)"
                    " AND NOT EXISTS (SELECT 1 FROM sm.listing l WHERE l.security_id = s.security_id)",
                    (symbols,),
                )
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_all_security_ids(self) -> set[int]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT security_id FROM sm.security WHERE active = true")
                return {row[0] for row in cur.fetchall()}

    def get_existing_listings(
        self, keys: list[tuple[int, str]]
    ) -> dict[tuple[int, str], tuple[int, int]]:
        """Returns {(exchange_id, exchange_security_id): (listing_id, security_id)}."""
        if not keys:
            return {}
        exchange_ids = [k[0] for k in keys]
        security_ids = [k[1] for k in keys]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT exchange_id, exchange_security_id, listing_id, security_id FROM sm.listing"
                    " WHERE (exchange_id, exchange_security_id)"
                    " IN (SELECT * FROM unnest(%s::int[], %s::text[]))",
                    (exchange_ids, security_ids),
                )
                return {(row[0], row[1]): (row[2], row[3]) for row in cur.fetchall()}

    def get_unsettled_contracts(
        self, exchange_id: int, max_age: timedelta, after_id: int, limit: int, include_active: bool = False,
    ) -> list[tuple[int, str]]:
        """Returns up to `limit` [(event_contract_id, exchange_security_id)], in id order after `after_id`, for outcomes
        with no settlement value yet whose listing was deactivated within `max_age` (or is active, if asked)."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ec.event_contract_id, l.exchange_security_id"
                    " FROM sm.listing l"
                    " JOIN sm.event_contract ec ON ec.security_id = l.security_id"
                    " WHERE l.exchange_id = %s AND ec.settlement_price IS NULL"
                    " AND ((NOT l.active AND l.date_modified > NOW() - %s) OR (l.active AND %s))"
                    " AND ec.event_contract_id > %s"
                    " ORDER BY ec.event_contract_id LIMIT %s",
                    (exchange_id, max_age, include_active, after_id, limit),
                )
                return [(row[0], row[1]) for row in cur.fetchall()]

    def get_unsettled_event_contracts(
        self, exchange_id: int, exchange_security_ids: list[str],
    ) -> dict[str, tuple[int, int]]:
        """Returns {exchange_security_id: (event_contract_id, event_id)} for these listings' outcomes with no settlement
        value yet."""
        if not exchange_security_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT l.exchange_security_id, ec.event_contract_id, ec.event_id"
                    " FROM sm.listing l"
                    " JOIN sm.event_contract ec ON ec.security_id = l.security_id"
                    " WHERE l.exchange_id = %s AND l.exchange_security_id = ANY(%s) AND ec.settlement_price IS NULL",
                    (exchange_id, exchange_security_ids),
                )
                return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    def get_unresolved_event_settlements(self, event_ids: list[int]) -> dict[int, dict[int, bool]]:
        """Returns {event_id: {event_contract_id: has a settlement value}} for the unresolved ones of these events."""
        if not event_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ec.event_id, ec.event_contract_id, ec.settlement_price IS NOT NULL"
                    " FROM sm.event_contract ec"
                    " JOIN sm.event e ON e.event_id = ec.event_id"
                    " WHERE ec.event_id = ANY(%s) AND NOT e.resolved",
                    (event_ids,),
                )
                result: dict[int, dict[int, bool]] = {}
                for event_id, event_contract_id, is_settled in cur.fetchall():
                    result.setdefault(event_id, {})[event_contract_id] = is_settled
                return result

    def get_retired_unsettled_listings(self, exchange_id: int, max_age: timedelta) -> list[tuple[int, int, str]]:
        """Returns [(listing_id, security_id, exchange_security_id)] for inactive listings deactivated within `max_age`
        whose outcome has no settlement value."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT l.listing_id, l.security_id, l.exchange_security_id"
                    " FROM sm.listing l"
                    " JOIN sm.event_contract ec ON ec.security_id = l.security_id"
                    " WHERE l.exchange_id = %s AND NOT l.active AND ec.settlement_price IS NULL"
                    " AND l.date_modified > NOW() - %s",
                    (exchange_id, max_age),
                )
                return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    def get_resolved_event_ids_for_securities(self, security_ids: list[int]) -> set[int]:
        if not security_ids:
            return set()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT ec.event_id FROM sm.event_contract ec"
                    " JOIN sm.event e ON e.event_id = ec.event_id"
                    " WHERE ec.security_id = ANY(%s) AND e.resolved",
                    (security_ids,),
                )
                return {row[0] for row in cur.fetchall()}

    def get_event_ids_by_security(self, security_ids: list[int]) -> dict[int, int]:
        """Returns {security_id: event_id} for the securities already linked to an event."""
        if not security_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT security_id, event_id FROM sm.event_contract WHERE security_id = ANY(%s)",
                    (security_ids,),
                )
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_existing_listing_specs(self, listing_ids: list[int]) -> dict[int, tuple[int, int, int, int, int]]:
        """Returns mapping of listing_id -> (tick_size, lot_size, min_notional, contract_multiplier, min_size)."""
        if not listing_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT ON (listing_id)"
                    " listing_id, tick_size, lot_size, min_notional, contract_multiplier, min_size"
                    " FROM sm.listing_spec"
                    " WHERE listing_id = ANY(%s) AND effective_from <= NOW()"
                    " ORDER BY listing_id, effective_from DESC, recorded_at DESC",
                    (listing_ids,),
                )
                return {row[0]: (row[1], row[2], row[3], row[4], row[5]) for row in cur.fetchall()}

    def get_listing_fee_keys(self, listing_ids: list[int]) -> dict[int, tuple[str | None, str | None]]:
        """Returns mapping of listing_id -> (fee_group, fee_event)."""
        if not listing_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT listing_id, fee_group, fee_event FROM sm.listing WHERE listing_id = ANY(%s)",
                    (listing_ids,),
                )
                return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    def get_fee_source_refs(self, exchange_id: int) -> set[str]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source_ref FROM sm.fee_schedule WHERE exchange_id = %s AND source_ref IS NOT NULL",
                    (exchange_id,),
                )
                return {row[0] for row in cur.fetchall()}

    def get_unresolved_events(self) -> list[Event]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, title, description, category,"
                    " tags, resolved, resolved_at, expiry, date_modified, date_created,"
                    " exchange_id, native_event_id, native_url"
                    " FROM sm.event WHERE resolved = false"
                )
                return [
                    Event(
                        event_id=row[0], title=row[1], description=row[2],
                        category=row[3], tags=row[4],
                        resolved=row[5], resolved_at=str(row[6]) if row[6] else None,
                        expiry=str(row[7]) if row[7] else None,
                        date_modified=str(row[8]), date_created=str(row[9]),
                        exchange_id=row[10], native_event_id=row[11], native_url=row[12],
                    )
                    for row in cur.fetchall()
                ]

    def get_all_event_contracts(self) -> list[EventContract]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ec.event_contract_id, ec.event_id, ec.security_id, ec.outcome_label, ec.date_created"
                    " FROM sm.event_contract ec"
                    " JOIN sm.event e ON e.event_id = ec.event_id"
                    " WHERE e.resolved = false"
                )
                return [
                    EventContract(
                        event_contract_id=row[0], event_id=row[1],
                        security_id=row[2], outcome_label=row[3],
                        date_created=str(row[4]),
                    )
                    for row in cur.fetchall()
                ]

    def get_event_ids_for_securities(self, security_ids: list[int]) -> set[int]:
        """Returns event_ids linked to these securities via event_contract."""
        if not security_ids:
            return set()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT ec.event_id FROM sm.event_contract ec"
                    " JOIN sm.event e ON e.event_id = ec.event_id"
                    " WHERE ec.security_id = ANY(%s) AND e.resolved = false",
                    (security_ids,),
                )
                return {row[0] for row in cur.fetchall()}

    def get_event_contracts_for_events(self, event_ids: list[int]) -> list[EventContract]:
        """Returns all unresolved event contracts for the given event_ids."""
        if not event_ids:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ec.event_contract_id, ec.event_id, ec.security_id, ec.outcome_label, ec.date_created"
                    " FROM sm.event_contract ec"
                    " JOIN sm.event e ON e.event_id = ec.event_id"
                    " WHERE ec.event_id = ANY(%s) AND e.resolved = false",
                    (event_ids,),
                )
                return [
                    EventContract(
                        event_contract_id=row[0], event_id=row[1],
                        security_id=row[2], outcome_label=row[3],
                        date_created=str(row[4]),
                    )
                    for row in cur.fetchall()
                ]

    def get_all_active_listings(self) -> list[Listing]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT listing_id, security_id, exchange_id, exchange_security_id,"
                    " exchange_security_symbol, date_modified, date_created"
                    " FROM sm.listing WHERE active = true"
                )
                return [
                    Listing(
                        listing_id=row[0], security_id=row[1], exchange_id=row[2],
                        exchange_security_id=row[3], exchange_security_symbol=row[4],
                        active=True, date_modified=str(row[5]), date_created=str(row[6]),
                    )
                    for row in cur.fetchall()
                ]

    def get_all_securities(self) -> list[Security]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT security_id, symbol, type, contract_type, asset_class,"
                    " base_currency_id, quote_currency_id, settle_currency_id,"
                    " inverse, is_quanto, expiry, strike_price, active,"
                    " underlying_security_id, description, date_modified, date_created"
                    " FROM sm.security WHERE active = true"
                )
                return [
                    Security(
                        security_id=row[0], symbol=row[1], type=row[2],
                        contract_type=row[3], asset_class=row[4],
                        base_currency_id=row[5], quote_currency_id=row[6],
                        settle_currency_id=row[7], inverse=row[8], is_quanto=row[9],
                        expiry=str(row[10]) if row[10] else None,
                        strike_price=row[11], active=row[12],
                        underlying_security_id=row[13], description=row[14],
                        date_modified=str(row[15]), date_created=str(row[16]),
                    )
                    for row in cur.fetchall()
                ]

    def get_all_currencies(self) -> list[Currency]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT currency_id, symbol, name, decimals, date_modified, date_created"
                    " FROM sm.currency"
                )
                return [
                    Currency(
                        currency_id=row[0], symbol=row[1], name=row[2],
                        decimals=row[3], date_modified=str(row[4]), date_created=str(row[5]),
                    )
                    for row in cur.fetchall()
                ]

    def get_securities_with_active_listings(self, security_ids: list[int]) -> set[int]:
        """Returns the subset of security_ids that still have at least one active listing."""
        if not security_ids:
            return set()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT security_id FROM sm.listing"
                    " WHERE active = true AND security_id = ANY(%s)",
                    (security_ids,),
                )
                return {row[0] for row in cur.fetchall()}

    def get_active_listings_for_securities(
        self, security_ids: list[int],
    ) -> list[tuple[int, int, int, str]]:
        """Returns [(listing_id, security_id, exchange_id, exchange_security_id)] for active listings."""
        if not security_ids:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT listing_id, security_id, exchange_id, exchange_security_id"
                    " FROM sm.listing"
                    " WHERE active = true AND security_id = ANY(%s)",
                    (security_ids,),
                )
                return [(row[0], row[1], row[2], row[3]) for row in cur.fetchall()]

    def get_active_listings_by_exchange_security(
        self, exchange_id: int, exchange_security_ids: list[str],
    ) -> list[tuple[int, int, str]]:
        """Returns [(listing_id, security_id, exchange_security_id)] for active listings."""
        if not exchange_security_ids:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT listing_id, security_id, exchange_security_id"
                    " FROM sm.listing"
                    " WHERE active = true"
                    " AND exchange_id = %s"
                    " AND exchange_security_id = ANY(%s)",
                    (exchange_id, exchange_security_ids),
                )
                return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    def securities_listed_more_than_once(self, security_ids: list[int]) -> set[int]:
        """Securities with several listings: left from before identity was one security per listing, and on a
        prediction venue each listing is its own market."""
        if not security_ids:
            return set()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT security_id FROM sm.listing WHERE security_id = ANY(%s) GROUP BY security_id"
                    " HAVING count(*) > 1",
                    (list(security_ids),),
                )
                return {row[0] for row in cur.fetchall()}

    def get_event_ids_for_security(self, security_id: int) -> list[int]:
        """Returns event_ids linked to this security via event_contract."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id FROM sm.event_contract WHERE security_id = %s",
                    (security_id,),
                )
                return [row[0] for row in cur.fetchall()]

    def get_active_security_count_for_event(self, event_id: int) -> int:
        """Returns count of active securities linked to this event."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM sm.event_contract ec"
                    " JOIN sm.security s ON s.security_id = ec.security_id"
                    " WHERE ec.event_id = %s AND s.active = true",
                    (event_id,),
                )
                return cur.fetchone()[0]

    # Column order is what _input reads.
    _INPUT_COLUMNS = (
        "SELECT cs.security_id, x.exchange_code, ec.event_id, s.contract_type, e.title, ec.outcome_label,"
        " e.description, e.expiry, cs.source, cs.source_hash"
        " FROM sm.contract_source cs"
        " JOIN sm.exchange x ON x.exchange_id = cs.exchange_id"
        " JOIN sm.security s ON s.security_id = cs.security_id"
        " JOIN sm.event_contract ec ON ec.security_id = cs.security_id"
        " JOIN sm.event e ON e.event_id = ec.event_id"
    )

    @staticmethod
    def _input(row) -> ExtractionInput:
        return ExtractionInput(
            security_id=row[0], exchange=row[1], event_id=row[2], contract_type=ContractType(row[3]), event_title=row[4],
            outcome_label=row[5], description=row[6], expiry=row[7].isoformat() if row[7] else None, source=row[8],
            source_hash=row[9],
        )

    def get_extraction_inputs(self, security_ids: list[int]) -> list[ExtractionInput]:
        """Inputs for the securities that have a stored venue source and an open event."""
        if not security_ids:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(self._INPUT_COLUMNS + " WHERE cs.security_id = ANY(%s) AND e.resolved = false",
                            (list(security_ids),))
                return [self._input(row) for row in cur.fetchall()]

    def sample_extraction_inputs(self, limit: int) -> list[ExtractionInput]:
        """A random sample of open securities' inputs, for the drift check."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(self._INPUT_COLUMNS + " WHERE e.resolved = false ORDER BY random() LIMIT %s", (limit,))
                return [self._input(row) for row in cur.fetchall()]

    def securities_needing_propositions(self, extractor_version: str, limit: int, after: int = 0) -> list[int]:
        """Open securities after `after` with a venue source and no proposition read from that source by this extractor
        version: the first run's backlog, any security whose message never reached the worker, and those only the LLM
        can read while it is off."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT cs.security_id FROM sm.contract_source cs"
                    " JOIN sm.event_contract ec ON ec.security_id = cs.security_id"
                    " JOIN sm.event e ON e.event_id = ec.event_id AND e.resolved = false"
                    " LEFT JOIN sm.contract_proposition p ON p.security_id = cs.security_id"
                    " WHERE cs.security_id > %s"
                    " AND (p.security_id IS NULL OR p.source_hash <> cs.source_hash OR p.extractor_version <> %s)"
                    " ORDER BY cs.security_id LIMIT %s",
                    (after, extractor_version, limit),
                )
                return [row[0] for row in cur.fetchall()]

    def get_proposition_versions(self, security_ids: list[int]) -> dict[int, tuple[str, str]]:
        """{security_id: (source_hash, extractor_version)} of stored propositions."""
        if not security_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT security_id, source_hash, extractor_version FROM sm.contract_proposition"
                    " WHERE security_id = ANY(%s)",
                    (list(security_ids),),
                )
                return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    def dirty_subjects(self, limit: int) -> list[int]:
        """Canonical subjects due a derivation: never derived, or changed since."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT s.subject_id FROM sm.proposition_subject s"
                    " LEFT JOIN sm.proposition_subject_alias a ON a.alias_subject_id = s.subject_id"
                    " WHERE a.alias_subject_id IS NULL AND (s.derived_at IS NULL OR s.date_modified > s.derived_at)"
                    " ORDER BY s.date_modified LIMIT %s",
                    (limit,),
                )
                return [row[0] for row in cur.fetchall()]

    def get_subject_contracts(self, subject_id: int) -> tuple[list[Contract], set[int], datetime]:
        """The subject's derivable contracts, the ones whose propositions changed since its last derivation (all of them
        when it was never derived), and when they were read (the derivation's as-of). A contract is derivable while its
        event is open and its proposition was read from its current source: resolved contracts keep the relationships
        they have, and a proposition the classifier has not re-read since its source changed describes nothing."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT now()")
                read_at = cur.fetchone()[0]
                cur.execute(
                    "SELECT p.security_id, x.exchange_code, s.key, p.fields,"
                    " (s.derived_at IS NULL OR p.date_modified > s.derived_at),"
                    # The other security of a BINARY event with exactly two, as the registry's connected read
                    # reports it, so consumers reverse the same rows.
                    " (SELECT o.security_id FROM sm.event_contract o"
                    "  JOIN sm.security so ON so.security_id = o.security_id AND so.contract_type = 7"
                    "  WHERE o.event_id = ec.event_id AND o.security_id <> p.security_id"
                    "  AND sec.contract_type = 7"
                    "  AND (SELECT count(*) FROM sm.event_contract n WHERE n.event_id = ec.event_id) = 2 LIMIT 1)"
                    " FROM sm.contract_proposition p"
                    " JOIN sm.security sec ON sec.security_id = p.security_id"
                    " JOIN sm.proposition_subject s ON s.subject_id = p.subject_id"
                    " JOIN sm.contract_source cs ON cs.security_id = p.security_id AND cs.source_hash = p.source_hash"
                    " JOIN sm.exchange x ON x.exchange_id = cs.exchange_id"
                    " JOIN sm.event_contract ec ON ec.security_id = p.security_id"
                    " JOIN sm.event e ON e.event_id = ec.event_id AND e.resolved = false"
                    " WHERE p.subject_id = %s",
                    (subject_id,),
                )
                rows = cur.fetchall()
                contracts = [Contract(row[0], row[1], Proposition.from_fields(row[2], row[3]), row[5]) for row in rows]
                return contracts, {row[0] for row in rows if row[4]}, read_at

    def nearest_subjects(self, embedding: list[float], limit: int) -> list[KnownSubject]:
        """The canonical (unmerged) subjects nearest an embedding, with the entities and measures their stored
        propositions use."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT s.key, s.description,"
                    " COALESCE(array_agg(DISTINCT p.fields->>'entity') FILTER (WHERE p.fields->>'entity' IS NOT NULL), '{}'),"
                    " COALESCE(array_agg(DISTINCT p.fields->>'measure') FILTER (WHERE p.fields->>'measure' IS NOT NULL), '{}')"
                    " FROM (SELECT ns.subject_id, ns.key, ns.description, ns.embedding <=> %s::vector AS distance"
                    "       FROM sm.proposition_subject ns"
                    "       LEFT JOIN sm.proposition_subject_alias a ON a.alias_subject_id = ns.subject_id"
                    "       WHERE ns.embedding IS NOT NULL AND a.alias_subject_id IS NULL"
                    "       ORDER BY ns.embedding <=> %s::vector LIMIT %s) s"
                    " LEFT JOIN sm.contract_proposition p ON p.subject_id = s.subject_id"
                    " GROUP BY s.key, s.description, s.distance ORDER BY s.distance",
                    (embedding, embedding, limit),
                )
                return [KnownSubject(row[0], row[1], set(row[2]), set(row[3])) for row in cur.fetchall()]

    def subjects_without_embeddings(self, limit: int = 1000) -> list[KnownSubject]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT s.key, s.description,"
                    " COALESCE(array_agg(DISTINCT p.fields->>'entity') FILTER (WHERE p.fields->>'entity' IS NOT NULL), '{}')"
                    " FROM sm.proposition_subject s LEFT JOIN sm.contract_proposition p ON p.subject_id = s.subject_id"
                    " WHERE s.embedding IS NULL GROUP BY s.subject_id ORDER BY s.subject_id LIMIT %s",
                    (limit,),
                )
                return [KnownSubject(row[0], row[1], set(row[2])) for row in cur.fetchall()]

    def put_subject_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        """Sets subjects' embeddings by key."""
        if not embeddings:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    "UPDATE sm.proposition_subject s SET embedding = v.embedding::vector"
                    " FROM (VALUES %s) AS v(key, embedding) WHERE s.key = v.key",
                    [(key, emb) for key, emb in embeddings.items()],
                )
            conn.commit()

    def get_security_ids_for_events(self, event_ids: list[int]) -> set[int]:
        """Returns security_ids linked to the given events via event_contract."""
        if not event_ids:
            return set()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT security_id FROM sm.event_contract WHERE event_id = ANY(%s)",
                    (event_ids,),
                )
                return {row[0] for row in cur.fetchall()}

    def get_active_exchange_native_ids(self) -> dict[int, set[str]]:
        """Returns {exchange_id: {native_event_id}} for unresolved events."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT exchange_id, native_event_id FROM sm.event"
                    " WHERE resolved = false AND native_event_id IS NOT NULL"
                )
                result: dict[int, set[str]] = {}
                for exchange_id, native_id in cur.fetchall():
                    result.setdefault(exchange_id, set()).add(native_id)
                return result
