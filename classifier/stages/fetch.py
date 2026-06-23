import logging

from classifier.adapters import ADAPTERS
from classifier.adapters.types import AdapterContract

logger = logging.getLogger(__name__)


def fetch_all(
    exchange_by_name: dict,
    max_per_adapter: int | None = None,
) -> list[AdapterContract]:
    all_contracts: list[AdapterContract] = []
    for adapter in ADAPTERS:
        exchange = exchange_by_name.get(adapter.exchange_name)
        if not exchange:
            logger.warning("No exchange record for adapter '%s' — skipping", adapter.exchange_name)
            continue
        try:
            contracts = adapter.fetch(exchange.exchange_id)
            if max_per_adapter is not None:
                contracts = contracts[:max_per_adapter]
            logger.info("Fetched %d contracts from %s", len(contracts), adapter.exchange_name)
            all_contracts.extend(contracts)
        except Exception as e:
            logger.error("Failed to fetch from %s: %s", adapter.exchange_name, e)

    return all_contracts
