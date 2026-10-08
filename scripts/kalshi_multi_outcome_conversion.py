"""Prints the SQL that converts open single-market mutually exclusive Kalshi events from binary to multi-outcome.

The adapter used to map a mutually exclusive event with one market as a binary (`T:yes` / `T:no`) and switch it to
multi-outcome (`T`) once a second market appeared. It now maps every mutually exclusive event as multi-outcome, so
the ones stored as binaries are converted in place: the YES listing becomes the bare ticker on the same security (its
market data and positions carry over), and the NO side is retired. Run this before registry migration 034, with the
classifier's fetch paused; each block is a no-op where the listing doesn't exist, so the same output fits dev and prod.

    poetry run python -m scripts.kalshi_multi_outcome_conversion > convert.sql
"""
import sys

from classifier.adapters.kalshi import KalshiAdapter
from classifier.utils import format_security_symbol


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def conversion_sql(event: dict) -> str | None:
    markets = event.get("markets", [])
    if not event.get("mutually_exclusive") or len(markets) != 1:
        return None
    market = markets[0]
    ticker = market["ticker"]
    outcome = market.get("yes_sub_title") or ticker
    listing_symbol = f"{event.get('title', '')[:60]} -- {outcome}"[:100]
    security_symbol = format_security_symbol(KalshiAdapter.symbol_prefix, ticker)
    return f"""
-- {event['event_ticker']}: {ticker}
DO $$
DECLARE
    kalshi INTEGER := (SELECT exchange_id FROM sm.exchange WHERE exchange_code = 'KALSHI');
    yes_security INTEGER;
    no_security INTEGER;
BEGIN
    SELECT security_id INTO yes_security FROM sm.listing WHERE exchange_id = kalshi AND exchange_security_id = {_literal(ticker + ':yes')};
    IF yes_security IS NULL THEN RETURN; END IF;
    SELECT security_id INTO no_security FROM sm.listing WHERE exchange_id = kalshi AND exchange_security_id = {_literal(ticker + ':no')};
    IF EXISTS (SELECT 1 FROM ledger.position p JOIN sm.listing l ON l.listing_id = p.listing_id
               WHERE l.security_id = no_security AND p.net_quantity <> 0) THEN
        RAISE EXCEPTION 'A strategy holds the NO side of {ticker}; close it before converting';
    END IF;
    UPDATE sm.listing SET exchange_security_id = {_literal(ticker)}, exchange_security_symbol = {_literal(listing_symbol)},
        date_modified = NOW() WHERE security_id = yes_security AND exchange_id = kalshi;
    UPDATE sm.security SET contract_type = 8, symbol = {_literal(security_symbol)}, date_modified = NOW()
        WHERE security_id = yes_security;
    UPDATE sm.event_contract SET outcome_label = {_literal(outcome)} WHERE security_id = yes_security;
    IF no_security IS NOT NULL THEN
        DELETE FROM sm.event_contract WHERE security_id = no_security;
        UPDATE sm.listing SET active = FALSE, date_modified = NOW() WHERE security_id = no_security;
        UPDATE sm.security SET active = FALSE, date_modified = NOW() WHERE security_id = no_security;
    END IF;
END $$;"""


def main():
    blocks = [sql for event in KalshiAdapter()._fetch_active_events() if (sql := conversion_sql(event))]
    print("BEGIN;" + "".join(blocks) + "\nCOMMIT;")
    print(f"{len(blocks)} single-market mutually exclusive events", file=sys.stderr)


if __name__ == "__main__":
    main()
