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
        str(int(contract.tick_size)),
        str(int(contract.lot_size)),
        str(int(contract.min_notional)),
        str(int(contract.contract_multiplier)),
        str(int(contract.min_size)),
        contract.fee_group or "",
        contract.fee_event or "",
    ])
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def event_min_volume(
    contract: AdapterContract,
    default: float | None,
    by_tag: dict[str, float | None],
    by_category: dict[str, float | None],
) -> float | None:
    """The volume an event needs: the smallest matching tag/category override, else the default. None is ungated."""
    matches = [v for tag, v in by_tag.items() if tag.lower() in contract.event_tags]
    if contract.mapped_category:
        matches += [v for cat, v in by_category.items() if cat.upper() == contract.mapped_category]
    if not matches:
        return default
    if any(v is None for v in matches):
        return None
    return min(matches)


def diff_contracts(
    active_contracts: list[AdapterContract],
    known_hashes: dict[str, str],
    failed_adapters: list[str],
    exchange_by_code: dict[str, Exchange],
    min_event_volume: float | None,
    max_messages: int,
    debug: bool = False,
    min_volume_by_tag: dict[str, float | None] | None = None,
    min_volume_by_category: dict[str, float | None] | None = None,
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
        ex.exchange_id for name, ex in exchange_by_code.items() if name not in failed_adapters
    }

    by_tag = min_volume_by_tag or {}
    by_category = min_volume_by_category or {}
    if min_event_volume is not None or by_tag or by_category:
        filtered_count = 0
        for nk, group in list(contracts_by_native.items()):
            # Volume only gates ingesting new events; an event fetch has already sent keeps
            # flowing so quiet markets still pick up added and removed contracts.
            already_sent = any(f"{c.exchange_id}:{c.exchange_security_id}" in known_hashes for c in group)
            vol = group[0].event_volume
            threshold = event_min_volume(group[0], min_event_volume, by_tag, by_category)
            if not already_sent and vol is not None and threshold is not None and vol < threshold:
                filtered_count += 1
                del contracts_by_native[nk]
                for c in group:
                    current_hashes.pop(f"{c.exchange_id}:{c.exchange_security_id}", None)
        if filtered_count and debug:
            logger.info(
                "Filtered %d low-volume event groups (min_event_volume=%s, by_tag=%s, by_category=%s)",
                filtered_count, min_event_volume, by_tag, by_category,
            )

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
    exchange_code: str | None = None,
) -> dict[str, Exchange]:
    exchanges = registry.get_exchange()
    exchange_by_code = {e.exchange_code: e for e in exchanges}
    if exchange_code:
        key = exchange_code.upper()
        if key not in exchange_by_code:
            raise ValueError(f"Unknown exchange code '{exchange_code}'. Choices: {list(exchange_by_code)}")
        return {key: exchange_by_code[key]}
    return exchange_by_code


def fetch_resolved_outcomes(
    exchange_by_code: dict,
    lookback_days: int,
) -> tuple[dict[int, set[str]], list[str]]:
    resolved_by_exchange: dict[int, set[str]] = {}
    failed: list[str] = []
    for adapter in ADAPTERS:
        exchange = exchange_by_code.get(adapter.exchange_code)
        if not exchange:
            logger.warning("No exchange record for adapter '%s' — skipping", adapter.exchange_code)
            continue
        try:
            resolved_ids = adapter.fetch_resolved(exchange.exchange_id, lookback_days)
            if resolved_ids:
                resolved_by_exchange[exchange.exchange_id] = resolved_ids
            logger.info("Fetched %d resolved ids from %s", len(resolved_ids), adapter.exchange_code)
        except Exception as e:
            logger.error("Failed to fetch resolved from %s: %s", adapter.exchange_code, e)
            failed.append(adapter.exchange_code)
    return resolved_by_exchange, failed


def fetch_all(
    exchange_by_code: dict,
    max_per_adapter: int | None = None,
) -> tuple[list[AdapterContract], list[str]]:
    all_contracts: list[AdapterContract] = []
    failed: list[str] = []
    for adapter in ADAPTERS:
        exchange = exchange_by_code.get(adapter.exchange_code)
        if not exchange:
            logger.warning("No exchange record for adapter '%s' — skipping", adapter.exchange_code)
            continue
        try:
            contracts: list[AdapterContract] = []
            for page in adapter.fetch(exchange.exchange_id):
                contracts.extend(page)
                if max_per_adapter is not None and len(contracts) >= max_per_adapter:
                    contracts = contracts[:max_per_adapter]
                    break
            logger.info("Fetched %d contracts from %s", len(contracts), adapter.exchange_code)
            all_contracts.extend(contracts)
        except Exception as e:
            logger.error("Failed to fetch from %s: %s", adapter.exchange_code, e)
            failed.append(adapter.exchange_code)

    return all_contracts, failed
