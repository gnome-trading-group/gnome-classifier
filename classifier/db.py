import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector.psycopg2 import register_vector

from gnomepy.registry.types import ContractRelationship, Currency, Event, EventContract, Listing, Security

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
                    "SELECT event_id FROM sm.exchange_event"
                    " WHERE exchange_id = %s AND native_event_id = %s",
                    (exchange_id, native_id),
                )
                row = cur.fetchone()
                return row[0] if row else None

    def get_all_exchange_events(self) -> dict[tuple[int, str], int]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ee.exchange_id, ee.native_event_id, ee.event_id"
                    " FROM sm.exchange_event ee"
                    " JOIN sm.event e ON e.event_id = ee.event_id"
                    " WHERE e.resolved = false"
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

    def get_events_for_dedup(self) -> list[tuple[str, str | None, int]]:
        """Returns (title, expiry, event_id) for unresolved events — used for title+expiry dedup."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT title, expiry, event_id FROM sm.event WHERE resolved = false")
                return [(row[0], str(row[1]) if row[1] else None, row[2]) for row in cur.fetchall()]

    def get_currencies(self) -> dict[str, int]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol, currency_id FROM sm.currency")
                return {row[0]: row[1] for row in cur.fetchall()}

    def get_existing_securities(self, symbols: list[str]) -> dict[str, int]:
        if not symbols:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT symbol, security_id FROM sm.security WHERE symbol = ANY(%s)",
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
    ) -> dict[tuple[int, str], int]:
        """Returns {(exchange_id, exchange_security_id): listing_id}."""
        if not keys:
            return {}
        exchange_ids = [k[0] for k in keys]
        security_ids = [k[1] for k in keys]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT exchange_id, exchange_security_id, listing_id FROM sm.listing"
                    " WHERE (exchange_id, exchange_security_id)"
                    " IN (SELECT * FROM unnest(%s::int[], %s::text[]))",
                    (exchange_ids, security_ids),
                )
                return {(row[0], row[1]): row[2] for row in cur.fetchall()}

    def get_existing_event_contracts(
        self, keys: list[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Returns set of (event_id, security_id) that already exist."""
        if not keys:
            return set()
        event_ids = [k[0] for k in keys]
        security_ids = [k[1] for k in keys]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, security_id FROM sm.event_contract"
                    " WHERE (event_id, security_id)"
                    " IN (SELECT * FROM unnest(%s::int[], %s::int[]))",
                    (event_ids, security_ids),
                )
                return {(row[0], row[1]) for row in cur.fetchall()}

    def get_existing_listing_specs(self, listing_ids: list[int]) -> set[int]:
        """Returns set of listing_ids that already have at least one spec."""
        if not listing_ids:
            return set()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT listing_id FROM sm.listing_spec WHERE listing_id = ANY(%s)",
                    (listing_ids,),
                )
                return {row[0] for row in cur.fetchall()}

    def get_unresolved_events(self) -> list[Event]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, title, description, category,"
                    " tags, resolved, resolved_at, expiry, date_modified, date_created"
                    " FROM sm.event WHERE resolved = false"
                )
                return [
                    Event(
                        event_id=row[0], title=row[1], description=row[2],
                        category=row[3], tags=row[4],
                        resolved=row[5], resolved_at=str(row[6]) if row[6] else None,
                        expiry=str(row[7]) if row[7] else None,
                        date_modified=str(row[8]), date_created=str(row[9]),
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

    def get_events_for_ids(self, event_ids: list[int]) -> list[Event]:
        """Returns Event objects for the given event_ids."""
        if not event_ids:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, title, description, category,"
                    " tags, resolved, resolved_at, expiry, date_modified, date_created"
                    " FROM sm.event WHERE event_id = ANY(%s)",
                    (event_ids,),
                )
                return [
                    Event(
                        event_id=row[0], title=row[1], description=row[2],
                        category=row[3], tags=row[4],
                        resolved=row[5], resolved_at=str(row[6]) if row[6] else None,
                        expiry=str(row[7]) if row[7] else None,
                        date_modified=str(row[8]), date_created=str(row[9]),
                    )
                    for row in cur.fetchall()
                ]

    def get_tradeable_securities(self) -> list[Security]:
        """Returns active non-event-contract securities (spot, perp, etc.) for hedgeable pair matching."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT security_id, symbol, type, contract_type, asset_class,"
                    " base_currency_id, quote_currency_id, settle_currency_id,"
                    " inverse, is_quanto, expiry, strike_price, active,"
                    " underlying_security_id, description, date_modified, date_created"
                    " FROM sm.security WHERE active = true AND type != 'EVENT_CONTRACT'"
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

    def get_hedge_keywords(self) -> list[tuple[int, str]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT security_id, keyword FROM sm.hedge_keyword")
                return [(row[0], row[1]) for row in cur.fetchall()]

    def get_contract_relationships_for_securities(self, security_ids: list[int]) -> list[ContractRelationship]:
        if not security_ids:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT relationship_id, security_id_a, security_id_b,"
                    " relationship_type, confidence, method, date_created"
                    " FROM sm.contract_relationship"
                    " WHERE security_id_a = ANY(%s) OR security_id_b = ANY(%s)",
                    (security_ids, security_ids),
                )
                return [
                    ContractRelationship(
                        relationship_id=row[0], security_id_a=row[1], security_id_b=row[2],
                        relationship_type=row[3], confidence=float(row[4]), method=row[5],
                        date_created=str(row[6]),
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

    def find_neighbors(
        self, embedding: list[float], threshold: float, limit: int = 50
    ) -> list[tuple[int, float]]:
        """Returns [(event_id, similarity)] for events above threshold, ordered by similarity desc."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ee.event_id, 1 - (ee.embedding <=> %s::vector) AS similarity"
                    " FROM sm.event_embedding ee"
                    " JOIN sm.event e ON e.event_id = ee.event_id"
                    " WHERE e.resolved = false"
                    " AND 1 - (ee.embedding <=> %s::vector) >= %s"
                    " ORDER BY ee.embedding <=> %s::vector"
                    " LIMIT %s",
                    (embedding, embedding, threshold, embedding, limit),
                )
                return [(row[0], row[1]) for row in cur.fetchall()]

    def get_embeddings(self, event_ids: list[int]) -> dict[int, list[float]]:
        """Returns {event_id: embedding} for the given event_ids."""
        if not event_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, embedding FROM sm.event_embedding WHERE event_id = ANY(%s)",
                    (event_ids,),
                )
                return {row[0]: list(row[1]) for row in cur.fetchall()}

    def put_embeddings(self, embeddings: dict[int, list[float]]) -> None:
        """Upsert embeddings into sm.event_embedding."""
        if not embeddings:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO sm.event_embedding (event_id, embedding) VALUES %s"
                    " ON CONFLICT (event_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                    [(eid, emb) for eid, emb in embeddings.items()],
                )
            conn.commit()

    def get_events_without_embeddings(self) -> list[Event]:
        """Returns unresolved events that have no embedding row — minimal fields only."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT e.event_id, e.title, e.description"
                    " FROM sm.event e"
                    " LEFT JOIN sm.event_embedding ee ON ee.event_id = e.event_id"
                    " WHERE e.resolved = false AND ee.event_id IS NULL"
                )
                return [
                    Event(
                        event_id=row[0], title=row[1], description=row[2],
                        category=None, tags=None,
                        resolved=False, resolved_at=None, expiry=None,
                        date_modified="", date_created="",
                    )
                    for row in cur.fetchall()
                ]

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
