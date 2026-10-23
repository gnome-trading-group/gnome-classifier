import json
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from classifier.adapters.types import AdapterContract
from classifier.runtime_config import ClassifierConfig, FeatureFlags, Thresholds
from classifier.stages.fetch import diff_contracts, event_min_volume, fetch_all
from classifier.stages.settle import empty_counts
from classifier.workers.fetch import FetchResult, FetchRunner
from gnomepy.registry.types import AssetClass, ContractType, Exchange, SecurityType


def _make_exchange(code: str, exchange_id: int = 1) -> Exchange:
    return Exchange(
        exchange_id=exchange_id, exchange_code=code, exchange_name=code.title(),
        region="", schema_type="", date_modified="", date_created="",
    )


def test_fetch_all_skips_unknown_adapter():
    exchange_by_code = {"unknown": _make_exchange("unknown")}
    contracts, failed = fetch_all(exchange_by_code)
    assert contracts == []
    assert failed == []


def test_fetch_all_limits_per_adapter():
    from classifier.adapters.types import AdapterContract
    from gnomepy.registry.types import SecurityType, ContractType, AssetClass

    def _make_contract(title: str) -> AdapterContract:
        return AdapterContract(
            exchange_id=1,
            exchange_security_id=title,
            exchange_security_symbol=title,
            base_currency="USDC",
            quote_currency="USDC",
            settle_currency="USDC",
            security_type=SecurityType.EVENT_CONTRACT,
            contract_type=ContractType.BINARY,
            asset_class=AssetClass.PREDICTION,
            inverse=False,
            is_quanto=False,
            tick_size=1.0,
            lot_size=1.0,
            min_notional=0.0,
            contract_multiplier=1.0,
            event_title=title,
            outcome_label="Yes",
            exchange_event_native_id=f"native:{title}",
            security_symbol=f"PM_I-{title}",
        )

    mock_adapter = MagicMock()
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.return_value = iter([[_make_contract(f"Event {i}") for i in range(20)]])

    exchange_by_code = {"POLYMARKET_INTL": _make_exchange("POLYMARKET_INTL")}

    with patch("classifier.stages.fetch.ADAPTERS", [mock_adapter]):
        contracts, failed = fetch_all(exchange_by_code, max_per_adapter=5)

    assert len(contracts) == 5
    assert failed == []


def test_fetch_all_handles_adapter_error():
    mock_adapter = MagicMock()
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.side_effect = RuntimeError("API down")

    exchange_by_code = {"POLYMARKET_INTL": _make_exchange("POLYMARKET_INTL")}

    with patch("classifier.stages.fetch.ADAPTERS", [mock_adapter]):
        contracts, failed = fetch_all(exchange_by_code)

    assert contracts == []
    assert failed == ["POLYMARKET_INTL"]


def _make_contract(
    native_id: str,
    security_id: str,
    event_volume: float | None = None,
) -> AdapterContract:
    return AdapterContract(
        exchange_id=1,
        exchange_security_id=security_id,
        exchange_security_symbol=security_id,
        base_currency="USDC",
        quote_currency="USDC",
        settle_currency="USDC",
        security_type=SecurityType.EVENT_CONTRACT,
        contract_type=ContractType.BINARY,
        asset_class=AssetClass.PREDICTION,
        inverse=False,
        is_quanto=False,
        tick_size=1.0,
        lot_size=1.0,
        min_notional=0.0,
        contract_multiplier=1.0,
        event_title="Test Event",
        outcome_label="Yes",
        exchange_event_native_id=native_id,
        security_symbol=f"PM_I-{security_id}",
        event_volume=event_volume,
    )


def _make_fetch_rc(min_event_volume: float | None = None, by_tag=None, by_category=None):
    rc = MagicMock()
    rc.config = ClassifierConfig(
        feature_flags=FeatureFlags(fetch_enabled=True),
        thresholds=Thresholds(
            min_event_volume=min_event_volume,
            min_event_volume_by_tag=by_tag or {},
            min_event_volume_by_category=by_category or {},
        ),
    )
    return rc


def _run_fetch(moto_env, contracts, min_event_volume=None, by_tag=None, by_category=None):
    rc = _make_fetch_rc(min_event_volume, by_tag, by_category)
    r = MagicMock()
    r.get.return_value = None
    runner = FetchRunner()
    mock_adapter = MagicMock()
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.return_value = iter([contracts] if contracts else [])
    with (
        patch("classifier.workers.fetch.fetch_exchanges", return_value={"POLYMARKET_INTL": MagicMock(exchange_id=1)}),
        patch("classifier.workers.fetch.ADAPTERS", [mock_adapter]),
    ):
        runner._run_fetch(rc, r, moto_env["sqs"], MagicMock())
    messages = []
    while True:
        resp = moto_env["sqs"].receive_message(
            QueueUrl=moto_env["contracts_queue"], MaxNumberOfMessages=10, WaitTimeSeconds=0
        )
        batch = resp.get("Messages", [])
        if not batch:
            break
        messages.extend(batch)
    return messages


class TestVolumeFiltering:
    def test_high_volume_event_passes(self, moto_env):
        contract = _make_contract("evt-1", "sec-1", event_volume=5000.0)
        msgs = _run_fetch(moto_env, [contract], min_event_volume=1000.0)
        assert len(msgs) == 1

    def test_low_volume_event_filtered(self, moto_env):
        contract = _make_contract("evt-1", "sec-1", event_volume=50.0)
        msgs = _run_fetch(moto_env, [contract], min_event_volume=1000.0)
        assert len(msgs) == 0

    def test_none_volume_always_passes(self, moto_env):
        contract = _make_contract("evt-1", "sec-1", event_volume=None)
        msgs = _run_fetch(moto_env, [contract], min_event_volume=1000.0)
        assert len(msgs) == 1

    def test_no_threshold_passes_all(self, moto_env):
        contract = _make_contract("evt-1", "sec-1", event_volume=0.01)
        msgs = _run_fetch(moto_env, [contract], min_event_volume=None)
        assert len(msgs) == 1

    def test_mixed_volume_selectively_filters(self, moto_env):
        high = _make_contract("evt-high", "sec-high", event_volume=5000.0)
        low = _make_contract("evt-low", "sec-low", event_volume=100.0)
        no_vol = _make_contract("evt-none", "sec-none", event_volume=None)
        msgs = _run_fetch(moto_env, [high, low, no_vol], min_event_volume=1000.0)
        assert len(msgs) == 2


def _run_fetch_pages(moto_env, pages, known_contracts=None, raise_after_pages=False):
    rc = _make_fetch_rc()
    r = MagicMock()
    r.get.return_value = json.dumps(known_contracts).encode() if known_contracts is not None else None
    runner = FetchRunner()

    def fetch(exchange_id):
        yield from pages
        if raise_after_pages:
            raise RuntimeError("API error at page 2")

    mock_adapter = MagicMock()
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.side_effect = fetch
    with (
        patch("classifier.workers.fetch.fetch_exchanges", return_value={"POLYMARKET_INTL": MagicMock(exchange_id=1)}),
        patch("classifier.workers.fetch.ADAPTERS", [mock_adapter]),
    ):
        result = runner._run_fetch(rc, r, moto_env["sqs"], MagicMock())
    saved_hashes = json.loads(r.set.call_args[0][1])
    return result.active_by_exchange, result.successful_ids, saved_hashes


class TestPaginatedFetch:
    def test_active_events_accumulate_across_pages(self, moto_env):
        pages = [
            [_make_contract("evt-1", "sec-1")],
            [_make_contract("evt-2", "sec-2")],
            [_make_contract("evt-3", "sec-3")],
        ]
        active_by_exchange, successful_ids, _ = _run_fetch_pages(moto_env, pages)
        assert active_by_exchange == {1: {"evt-1", "evt-2", "evt-3"}}
        assert successful_ids == {1}

    def test_failure_mid_pagination_keeps_unfetched_hashes(self, moto_env):
        known = {"1:sec-1": "old-hash-1", "1:sec-2": "old-hash-2"}
        pages = [[_make_contract("evt-1", "sec-1")]]
        active_by_exchange, successful_ids, saved_hashes = _run_fetch_pages(
            moto_env, pages, known_contracts=known, raise_after_pages=True,
        )
        assert saved_hashes["1:sec-2"] == "old-hash-2"
        assert saved_hashes["1:sec-1"] != "old-hash-1"

    def test_failure_mid_pagination_marks_exchange_failed(self, moto_env):
        pages = [[_make_contract("evt-1", "sec-1")]]
        active_by_exchange, successful_ids, _ = _run_fetch_pages(moto_env, pages, raise_after_pages=True)
        assert 1 not in active_by_exchange
        assert successful_ids == set()



def _run_fetch_with_known(moto_env, contracts, known_contracts, min_event_volume):
    rc = _make_fetch_rc(min_event_volume)
    r = MagicMock()
    r.get.return_value = json.dumps(known_contracts).encode()
    mock_adapter = MagicMock()
    mock_adapter.exchange_code = "POLYMARKET_INTL"
    mock_adapter.fetch.return_value = iter([contracts])
    with (
        patch("classifier.workers.fetch.fetch_exchanges", return_value={"POLYMARKET_INTL": MagicMock(exchange_id=1)}),
        patch("classifier.workers.fetch.ADAPTERS", [mock_adapter]),
    ):
        FetchRunner()._run_fetch(rc, r, moto_env["sqs"], MagicMock())
    resp = moto_env["sqs"].receive_message(QueueUrl=moto_env["contracts_queue"], MaxNumberOfMessages=10, WaitTimeSeconds=0)
    return [json.loads(m["Body"]) for m in resp.get("Messages", [])], json.loads(r.set.call_args[0][1])


class TestVolumeGateOnlyForNewEvents:
    def test_quiet_known_event_still_sends_changes(self, moto_env):
        existing = _make_contract("evt-1", "sec-1", event_volume=10.0)
        added = _make_contract("evt-1", "sec-2", event_volume=10.0)
        msgs, saved = _run_fetch_with_known(moto_env, [existing, added], {"1:sec-1": "old-hash"}, min_event_volume=1000.0)
        assert len(msgs) == 1
        assert {c["exchange_security_id"] for c in msgs[0]["contracts"]} == {"sec-1", "sec-2"}
        assert {"1:sec-1", "1:sec-2"} <= set(saved)

    def test_quiet_new_event_is_still_filtered(self, moto_env):
        msgs, saved = _run_fetch_with_known(moto_env, [_make_contract("evt-2", "sec-9", event_volume=10.0)], {}, min_event_volume=1000.0)
        assert msgs == []
        assert "1:sec-9" not in saved


def _tagged(native_id, volume, tags=(), category=None):
    contract = _make_contract(native_id, f"sec-{native_id}", event_volume=volume)
    contract.event_tags = list(tags)
    contract.mapped_category = category
    return contract


class TestEventMinVolume:
    def test_no_match_uses_default(self):
        assert event_min_volume(_tagged("e", 1.0, ["nba"], "SPORTS"), 1000.0, {"cs2": 50.0}, {"CRYPTO": 10.0}) == 1000.0

    def test_tag_match_replaces_default(self):
        assert event_min_volume(_tagged("e", 1.0, ["cs2", "esports"]), 1000.0, {"cs2": 50.0}, {}) == 50.0

    def test_override_can_raise_threshold(self):
        assert event_min_volume(_tagged("e", 1.0, ["crypto"]), 1000.0, {"crypto": 5000.0}, {}) == 5000.0

    def test_category_match(self):
        assert event_min_volume(_tagged("e", 1.0, [], "SPORTS"), 1000.0, {}, {"SPORTS": 200.0}) == 200.0

    def test_smallest_match_wins_across_tags_and_categories(self):
        contract = _tagged("e", 1.0, ["cs2", "esports"], "SPORTS")
        assert event_min_volume(contract, 1000.0, {"cs2": 50.0, "esports": 300.0}, {"SPORTS": 200.0}) == 50.0

    def test_none_override_ungates(self):
        contract = _tagged("e", 1.0, ["cs2"], "SPORTS")
        assert event_min_volume(contract, 1000.0, {"cs2": None}, {"SPORTS": 200.0}) is None

    def test_keys_match_case_insensitively(self):
        contract = _tagged("e", 1.0, ["cs2"], "SPORTS")
        assert event_min_volume(contract, 1000.0, {"CS2": 50.0}, {}) == 50.0
        assert event_min_volume(contract, 1000.0, {}, {"sports": 200.0}) == 200.0


class TestVolumeOverridesInDiff:
    def _sent_events(self, contracts, default, by_tag=None, by_category=None):
        msgs, _, _, _ = diff_contracts(
            contracts, {}, [], {}, default, max_messages=100,
            min_volume_by_tag=by_tag, min_volume_by_category=by_category,
        )
        return {m["contracts"][0]["exchange_event_native_id"] for m in msgs}

    def test_tag_override_lets_small_event_through(self):
        contracts = [_tagged("cs2-match", 80.0, ["cs2"]), _tagged("nba-game", 80.0, ["nba"])]
        assert self._sent_events(contracts, 1000.0, {"cs2": 50.0}) == {"cs2-match"}

    def test_override_applies_with_global_gate_disabled(self):
        contracts = [_tagged("btc", 80.0, ["crypto"]), _tagged("nba-game", 80.0, ["nba"])]
        assert self._sent_events(contracts, None, {"crypto": 500.0}) == {"nba-game"}

    def test_category_override_lets_small_event_through(self):
        contracts = [_tagged("game", 150.0, [], "SPORTS"), _tagged("vote", 150.0, [], "POLITICS")]
        assert self._sent_events(contracts, 1000.0, by_category={"SPORTS": 100.0}) == {"game"}


class TestVolumeOverridesInWorker:
    def test_worker_passes_overrides_from_config(self, moto_env):
        contracts = [_tagged("cs2-match", 80.0, ["cs2"]), _tagged("nba-game", 80.0, ["nba"])]
        msgs = _run_fetch(moto_env, contracts, min_event_volume=1000.0, by_tag={"cs2": 50.0})
        assert [json.loads(m["Body"])["contracts"][0]["exchange_event_native_id"] for m in msgs] == ["cs2-match"]


class TestLeftFeed:
    def _run(self, moto_env, contracts, known_contracts, fail=False):
        rc = _make_fetch_rc()
        r = MagicMock()
        r.get.return_value = json.dumps(known_contracts).encode()

        def fetch(exchange_id):
            yield contracts
            if fail:
                raise RuntimeError("down")

        mock_adapter = MagicMock()
        mock_adapter.exchange_code = "POLYMARKET_INTL"
        mock_adapter.fetch.side_effect = fetch
        with (
            patch("classifier.workers.fetch.fetch_exchanges", return_value={"POLYMARKET_INTL": _make_exchange("POLYMARKET_INTL")}),
            patch("classifier.workers.fetch.ADAPTERS", [mock_adapter]),
        ):
            return FetchRunner()._run_fetch(rc, r, moto_env["sqs"], MagicMock())

    def test_contracts_missing_from_this_fetch_left_the_feed(self, moto_env):
        known = {"1:cond:tok-a": "h", "1:cond:tok-b": "h", "2:other": "h"}
        result = self._run(moto_env, [_make_contract("evt-1", "cond:tok-a")], known)
        # Ids keep their own colons; another exchange's contracts are not this one's to report.
        assert result.left_feed_by_exchange == {1: {"cond:tok-b"}}
        assert "1:cond:tok-a" in result.fetched_keys and "1:cond:tok-b" not in result.fetched_keys
        assert set(result.exchanges) == {"POLYMARKET_INTL"}

    def test_a_failed_fetch_reports_nothing(self, moto_env):
        result = self._run(moto_env, [_make_contract("evt-1", "sec-1")], {"1:sec-2": "h"}, fail=True)
        assert result.left_feed_by_exchange == {} and result.exchanges == {}


class TestAfterFetch:
    def _rc(self, **flags):
        rc = MagicMock()
        rc.config = ClassifierConfig(feature_flags=FeatureFlags(**flags))
        return rc

    def _fetched(self):
        return FetchResult({1: {"evt"}}, {1}, exchanges={"POLYMARKET_INTL": _make_exchange("POLYMARKET_INTL")},
                           fetched_keys={"1:sec": "h"}, left_feed_by_exchange={1: {"gone"}})

    def test_runs_closure_check_recent_settle_and_reactivation_and_announces_resolutions(self, monkeypatch):
        calls = []
        resolved = {**empty_counts(), "events_resolved": 1, "resolved_event_ids": [7], "resolved_event_names": ["E"]}
        monkeypatch.setattr("classifier.workers.fetch.check_closures", lambda *a: calls.append(("closures", a[2])) or resolved)
        monkeypatch.setattr("classifier.workers.fetch.record_settlements", lambda *a: calls.append(("recent", a[4])) or empty_counts())
        monkeypatch.setattr("classifier.workers.fetch.reactivate_listings", lambda *a: calls.append(("reactivate", a[0])))
        runner = FetchRunner()
        runner._db = "db"
        runner._sns = MagicMock()
        monkeypatch.setenv("NOTIFICATIONS_TOPIC_ARN", "arn:topic")

        runner._run_after_fetch(self._rc(settlement_enabled=True, reactivation_enabled=True), MagicMock(), self._fetched())

        assert calls == [("closures", {1: {"gone"}}), ("recent", timedelta(hours=6)), ("reactivate", {1})]
        message = json.loads(runner._sns.publish.call_args.kwargs["Message"])
        assert message["type"] == "resolved" and message["resolved_event_ids"] == [7]
        assert "failed_exchanges" not in message

    def test_does_nothing_with_its_flags_off_or_without_a_successful_fetch(self, monkeypatch):
        calls = []
        for name in ("check_closures", "record_settlements", "reactivate_listings"):
            monkeypatch.setattr(f"classifier.workers.fetch.{name}", lambda *a: calls.append(a))
        runner = FetchRunner()
        runner._sns = MagicMock()
        runner._run_after_fetch(self._rc(), MagicMock(), self._fetched())
        runner._run_after_fetch(self._rc(settlement_enabled=True, reactivation_enabled=True), MagicMock(),
                                FetchResult({}, set()))
        assert calls == [] and not runner._sns.publish.called

    def test_one_failing_step_does_not_stop_the_others(self, monkeypatch):
        calls = []

        def boom(*a):
            raise RuntimeError("venue down")

        monkeypatch.setattr("classifier.workers.fetch.check_closures", boom)
        monkeypatch.setattr("classifier.workers.fetch.record_settlements", lambda *a: calls.append("recent") or empty_counts())
        monkeypatch.setattr("classifier.workers.fetch.reactivate_listings", lambda *a: calls.append("reactivate"))
        runner = FetchRunner()
        runner._db = "db"
        runner._run_after_fetch(self._rc(settlement_enabled=True, reactivation_enabled=True), MagicMock(), self._fetched())
        assert calls == ["recent", "reactivate"]


def test_resolve_cycle_reports_closed_contracts_once(moto_env):
    rc = MagicMock()
    rc.config = ClassifierConfig(feature_flags=FeatureFlags(resolve_enabled=True))
    r = MagicMock()
    r.get.return_value = json.dumps([[1, "already-sent"]]).encode()
    with (
        patch("classifier.workers.fetch.fetch_exchanges", return_value={}),
        patch("classifier.workers.fetch.fetch_resolved_outcomes",
              return_value=({1: {"already-sent", "cond:tok"}}, [])),
    ):
        FetchRunner()._run_resolve(rc, r, moto_env["sqs"], MagicMock())
    resp = moto_env["sqs"].receive_message(QueueUrl=moto_env["contracts_queue"], MaxNumberOfMessages=10, WaitTimeSeconds=0)
    assert [json.loads(m["Body"]) for m in resp.get("Messages", [])] == [
        {"type": "closed", "exchange_id": 1, "native_id": "cond:tok"}]
