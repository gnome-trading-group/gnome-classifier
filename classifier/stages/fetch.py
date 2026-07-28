import logging

from classifier.adapters import ADAPTERS
from classifier.adapters.types import AdapterContract
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import Exchange

logger = logging.getLogger(__name__)


def fetch_exchanges(
    registry: RegistryClient,
    adapter_name: str | None = None,
) -> dict[str, Exchange]:
    exchanges = registry.get_exchange()
    exchange_by_name = {e.exchange_name.lower(): e for e in exchanges}
    if adapter_name:
        key = adapter_name.lower()
        if key not in exchange_by_name:
            raise ValueError(f"Unknown adapter '{adapter_name}'. Choices: {list(exchange_by_name)}")
        return {key: exchange_by_name[key]}
    return exchange_by_name


def fetch_resolved_outcomes(
    exchange_by_name: dict,
    lookback_days: int,
) -> tuple[dict[int, set[str]], list[str]]:
    resolved_by_exchange: dict[int, set[str]] = {}
    failed: list[str] = []
    for adapter in ADAPTERS:
        exchange = exchange_by_name.get(adapter.exchange_name)
        if not exchange:
            logger.warning("No exchange record for adapter '%s' — skipping", adapter.exchange_name)
            continue
        try:
            resolved_ids = adapter.fetch_resolved(exchange.exchange_id, lookback_days)
            if resolved_ids:
                resolved_by_exchange[exchange.exchange_id] = resolved_ids
            logger.info("Fetched %d resolved ids from %s", len(resolved_ids), adapter.exchange_name)
        except Exception as e:
            logger.error("Failed to fetch resolved from %s: %s", adapter.exchange_name, e)
            failed.append(adapter.exchange_name)
    return resolved_by_exchange, failed


def fetch_all(
    exchange_by_name: dict,
    max_per_adapter: int | None = None,
) -> tuple[list[AdapterContract], list[str]]:
    all_contracts: list[AdapterContract] = []
    failed: list[str] = []
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
            failed.append(adapter.exchange_name)

    return all_contracts, failed
