import dataclasses
import hashlib
import logging

from classifier.adapters import ADAPTERS
from classifier.adapters.types import AdapterContract
from gnomepy.registry import RegistryClient
from gnomepy.registry.types import Exchange

logger = logging.getLogger(__name__)


def contract_hash(contract: AdapterContract) -> str:
    content = "\x00".join([
        str(contract.exchange_id),
        contract.exchange_security_id,
        contract.outcome_label,
        contract.event_title,
        contract.exchange_event_native_id,
    ])
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def diff_contracts(
    active_contracts: list[AdapterContract],
    known_hashes: dict[str, str],
    failed_adapters: list[str],
    exchange_by_name: dict[str, Exchange],
    min_event_volume: float | None,
    max_messages: int,
) -> tuple[list[dict], dict[str, str], dict[int, set[str]], set[int]]:
    """Returns (new_messages, updated_hashes, active_by_exchange, successful_exchange_ids)."""
    contracts_by_native: dict[tuple[int, str], list[AdapterContract]] = {}
    current_hashes: dict[str, str] = {}
    active_by_exchange: dict[int, set[str]] = {}

    for contract in active_contracts:
        nk = (contract.exchange_id, contract.exchange_event_native_id)
        contracts_by_native.setdefault(nk, []).append(contract)
        ck = f"{contract.exchange_id}:{contract.exchange_security_id}"
        current_hashes[ck] = contract_hash(contract)
        active_by_exchange.setdefault(contract.exchange_id, set()).add(contract.exchange_event_native_id)

    successful_exchange_ids = {
        ex.exchange_id for name, ex in exchange_by_name.items() if name not in failed_adapters
    }

    if min_event_volume is not None:
        filtered_count = 0
        for nk, group in list(contracts_by_native.items()):
            vol = group[0].event_volume
            if vol is not None and vol < min_event_volume:
                filtered_count += 1
                del contracts_by_native[nk]
                for c in group:
                    current_hashes.pop(f"{c.exchange_id}:{c.exchange_security_id}", None)
        if filtered_count:
            logger.info("Filtered %d low-volume event groups (min_event_volume=%.2f)", filtered_count, min_event_volume)

    new_messages: list[dict] = []
    for nk, group in contracts_by_native.items():
        has_new_or_changed = any(
            known_hashes.get(f"{c.exchange_id}:{c.exchange_security_id}") != current_hashes[f"{c.exchange_id}:{c.exchange_security_id}"]
            for c in group
        )
        if has_new_or_changed:
            new_messages.append({"type": "new", "contracts": [dataclasses.asdict(c) for c in group]})

    if len(new_messages) > max_messages:
        logger.info("Capping fetch from %d to %d groups", len(new_messages), max_messages)
        new_messages = new_messages[:max_messages]

    sent_contract_keys = {
        f"{c['exchange_id']}:{c['exchange_security_id']}"
        for msg in new_messages
        for c in msg["contracts"]
    }

    updated_hashes: dict[str, str] = {}
    for ck in current_hashes:
        if ck in sent_contract_keys:
            updated_hashes[ck] = current_hashes[ck]
        elif ck in known_hashes:
            updated_hashes[ck] = known_hashes[ck]

    return new_messages, updated_hashes, active_by_exchange, successful_exchange_ids


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
            contracts: list[AdapterContract] = []
            for page in adapter.fetch(exchange.exchange_id):
                contracts.extend(page)
                if max_per_adapter is not None and len(contracts) >= max_per_adapter:
                    contracts = contracts[:max_per_adapter]
                    break
            logger.info("Fetched %d contracts from %s", len(contracts), adapter.exchange_name)
            all_contracts.extend(contracts)
        except Exception as e:
            logger.error("Failed to fetch from %s: %s", adapter.exchange_name, e)
            failed.append(adapter.exchange_name)

    return all_contracts, failed
