import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector.psycopg2 import register_vector

from gnomepy.registry.types import ContractRelationship, Currency, Event, EventContract, Security

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
        """Returns (title, expiry, event_id) for all events — used for title+expiry dedup."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT title, expiry, event_id FROM sm.event")
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
                cur.execute("SELECT security_id FROM sm.security")
                return {row[0] for row in cur.fetchall()}

    def get_existing_listings(
        self, keys: list[tuple[int, str]]
    ) -> dict[tuple[int, str], int]:
        """Returns {(exchange_id, exchange_security_id): listing_id}."""
        if not keys:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT exchange_id, exchange_security_id, listing_id FROM sm.listing"
                    " WHERE (exchange_id, exchange_security_id) = ANY(%s)",
                    (keys,),
                )
                return {(row[0], row[1]): row[2] for row in cur.fetchall()}

    def get_existing_event_contracts(
        self, keys: list[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """Returns set of (event_id, security_id) that already exist."""
        if not keys:
            return set()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, security_id FROM sm.event_contract"
                    " WHERE (event_id, security_id) = ANY(%s)",
                    (keys,),
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
                    "SELECT event_id, title, description, category, resolution_source,"
                    " tags, resolved, resolved_at, expiry, date_modified, date_created"
                    " FROM sm.event WHERE resolved = false"
                )
                return [
                    Event(
                        event_id=row[0], title=row[1], description=row[2],
                        category=row[3], resolution_source=row[4], tags=row[5],
                        resolved=row[6], resolved_at=str(row[7]) if row[7] else None,
                        expiry=str(row[8]) if row[8] else None,
                        date_modified=str(row[9]), date_created=str(row[10]),
                    )
                    for row in cur.fetchall()
                ]

    def get_all_event_contracts(self) -> list[EventContract]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_contract_id, event_id, security_id, outcome_label, date_created"
                    " FROM sm.event_contract"
                )
                return [
                    EventContract(
                        event_contract_id=row[0], event_id=row[1],
                        security_id=row[2], outcome_label=row[3],
                        date_created=str(row[4]),
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
                    " FROM sm.security"
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

    def get_contract_relationships_for_securities(self, security_ids: list[int]) -> list[ContractRelationship]:
        if not security_ids:
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT relationship_id, security_id_a, security_id_b,"
                    " relationship_type, confidence, method, reviewed, reviewed_at, date_created"
                    " FROM sm.contract_relationship"
                    " WHERE security_id_a = ANY(%s) OR security_id_b = ANY(%s)",
                    (security_ids, security_ids),
                )
                return [
                    ContractRelationship(
                        relationship_id=row[0], security_id_a=row[1], security_id_b=row[2],
                        relationship_type=row[3], confidence=float(row[4]), method=row[5],
                        reviewed=row[6], reviewed_at=str(row[7]) if row[7] else None,
                        date_created=str(row[8]),
                    )
                    for row in cur.fetchall()
                ]

    def find_neighbors(
        self, embedding: list[float], threshold: float, limit: int = 50
    ) -> list[tuple[int, float]]:
        """Returns [(event_id, similarity)] for events above threshold, ordered by similarity desc."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, 1 - (embedding <=> %s::vector) AS similarity"
                    " FROM sm.event_embedding"
                    " WHERE 1 - (embedding <=> %s::vector) >= %s"
                    " ORDER BY embedding <=> %s::vector"
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
