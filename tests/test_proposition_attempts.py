import os
from unittest.mock import MagicMock

import pytest

from classifier.adapters import kalshi
from classifier.stages.entities import create_entities

from classifier.propositions.attempts import (
    FAILED,
    GIVE_UP_AFTER,
    NEEDS_LLM,
    WEEK_SECONDS,
    Attempt,
    RedisAttemptStore,
    wait_seconds,
)
from classifier.propositions.types import Kind, Proposition
from classifier.stages import propositions as stage
from classifier.stages.propositions import read_backlog, read_propositions
from scripts.testing import InMemoryAttemptStore
from tests.test_stage_propositions import AnswersEverything, btc_ladder, embed, load, mention_market


class FailsEverything:
    def __init__(self):
        self.asked: list[int] = []

    def extract(self, groups, candidates):
        items = [i for _, (_, chunk) in groups.items() for i in chunk]
        self.asked += [i.security_id for i in items]
        return {i.security_id: None for i in items}


class Unavailable(FailsEverything):
    def extract(self, groups, candidates):
        super().extract(groups, candidates)
        return {}


def mention_event(words=("AI", "Tariff")) -> dict:
    """A mutually exclusive Kalshi event: one multi-outcome security per word, read only by the LLM."""
    event = mention_market()
    event["mutually_exclusive"] = True
    event["markets"] = [{**event["markets"][0], "ticker": f"KXTRUMPMENTION-26OCT29-{w.upper()}", "yes_sub_title": w}
                        for w in words]
    return event


def load_multi(stub_registry, stub_db, event) -> dict[str, int]:
    """{market ticker: security id} for a multi-outcome event, whose securities have no Yes/No side."""
    contracts = kalshi._with_sources(event, kalshi.KalshiAdapter()._map_event(2, event))
    create_entities(stub_registry, MagicMock(), contracts, db=stub_db, canonicalize_enabled=False)
    by_ticker = {src["source"]["market"]["ticker"]: sid for sid, src in stub_registry._contract_sources.items()}
    return {c.exchange_security_id: by_ticker[c.exchange_security_id] for c in contracts}


def reread(stub_registry, sid, **changes):
    source = stub_registry._contract_sources[sid]
    stub_registry.upsert_contract_sources([{**source, **changes}])


class TestOnlyWhatChanged:
    def test_a_new_outcome_reads_only_that_outcome(self, stub_registry, stub_db):
        first = load_multi(stub_registry, stub_db, mention_event())
        llm = AnswersEverything()
        read_propositions(sorted(first.values()), db=stub_db, registry=stub_registry, extractor=llm, embed=embed)
        assert sorted(llm.asked) == sorted(first.values())

        both = load_multi(stub_registry, stub_db, mention_event(("AI", "Tariff", "Crypto")))
        llm.asked.clear()
        read_propositions(sorted(both.values()), db=stub_db, registry=stub_registry, extractor=llm, embed=embed)
        assert llm.asked == sorted(set(both.values()) - set(first.values()))

    def test_a_new_strike_reads_only_its_pair(self, stub_registry, stub_db):
        first = load(stub_registry, stub_db, btc_ladder((70000, 71000)))
        read_propositions(sorted(first.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
        added = load(stub_registry, stub_db, btc_ladder((70000, 71000, 72000)))
        result = read_propositions(sorted(added.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
        assert (result.read, result.stored) == (2, 2)

    def test_a_reread_first_side_rewrites_its_second_side(self, stub_registry, stub_db):
        sids = load(stub_registry, stub_db, btc_ladder((70000,)))
        read_propositions(sorted(sids.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
        yes, no = (v for k, v in sorted(sids.items()) if k.endswith(":yes")), (v for k, v in sids.items() if k.endswith(":no"))
        yes, no = next(yes), next(no)
        reread(stub_registry, yes, source_hash="new")
        result = read_propositions([yes], db=stub_db, registry=stub_registry, extractor=None, embed=embed)
        assert result.read == 2
        assert stub_registry._propositions[yes]["source_hash"] == "new"
        assert stub_registry._propositions[no]["fields"]["negated"] is True

    def test_a_reread_second_side_is_negated_from_the_stored_first_side(self, stub_registry, stub_db):
        sids = load(stub_registry, stub_db, btc_ladder((70000,)))
        read_propositions(sorted(sids.values()), db=stub_db, registry=stub_registry, extractor=None, embed=embed)
        yes = next(v for k, v in sids.items() if k.endswith(":yes"))
        no = next(v for k, v in sids.items() if k.endswith(":no"))
        reread(stub_registry, no, source_hash="new")
        result = read_propositions([no], db=stub_db, registry=stub_registry, extractor=None, embed=embed)
        assert result.read == 1
        assert stub_registry._propositions[no]["source_hash"] == "new"
        assert stub_registry._propositions[no]["fields"] == {**stub_registry._propositions[yes]["fields"], "negated": True}


class TestBackoff:
    def setup_method(self):
        self.store = InMemoryAttemptStore()

    def read(self, stub_registry, stub_db, sids, extractor):
        return read_propositions(sids, db=stub_db, registry=stub_registry, extractor=extractor, embed=embed,
                                 attempts=self.store)

    def test_a_failed_read_waits_longer_each_time_then_weekly(self, stub_registry, stub_db):
        sids = sorted(load_multi(stub_registry, stub_db, mention_event(("AI",))).values())
        llm = FailsEverything()
        gave_up = 0
        for attempt in range(1, GIVE_UP_AFTER + 1):
            llm.asked.clear()
            result = self.read(stub_registry, stub_db, sids, llm)
            assert llm.asked == sids
            gave_up += result.gave_up
            assert self.store.get_many(sids)[sids[0]] == Attempt(
                stub_registry._contract_sources[sids[0]]["source_hash"], "p1", FAILED, attempt,
                self.store.clock + wait_seconds(attempt))
            llm.asked.clear()
            self.store.clock += wait_seconds(attempt) - 1
            assert self.read(stub_registry, stub_db, sids, llm).backing_off == 1
            assert llm.asked == []
            self.store.clock += 1
        assert gave_up == 1
        assert wait_seconds(GIVE_UP_AFTER) == WEEK_SECONDS

    def test_a_changed_source_is_read_straight_away(self, stub_registry, stub_db):
        sids = sorted(load_multi(stub_registry, stub_db, mention_event(("AI",))).values())
        self.read(stub_registry, stub_db, sids, FailsEverything())
        reread(stub_registry, sids[0], source_hash="new")
        llm = AnswersEverything()
        assert self.read(stub_registry, stub_db, sids, llm).read == 1
        assert self.store.get_many(sids) == {}

    def test_an_unavailable_provider_records_nothing(self, stub_registry, stub_db):
        sids = sorted(load_multi(stub_registry, stub_db, mention_event(("AI",))).values())
        llm = Unavailable()
        self.read(stub_registry, stub_db, sids, llm)
        assert self.store.get_many(sids) == {}
        llm.asked.clear()
        self.read(stub_registry, stub_db, sids, llm)
        assert llm.asked == sids

    def test_llm_only_securities_wait_while_it_is_off_and_are_read_when_it_is_on(self, stub_registry, stub_db):
        sids = sorted(load_multi(stub_registry, stub_db, mention_event(("AI",))).values())
        assert self.read(stub_registry, stub_db, sids, None).parked == 1
        assert self.store.get_many(sids)[sids[0]].reason == NEEDS_LLM
        assert self.read(stub_registry, stub_db, sids, None).backing_off == 1
        result, _ = read_backlog(db=stub_db, registry=stub_registry, extractor=AnswersEverything(), embed=embed,
                                 limit=10, attempts=self.store)
        assert result.read == 1


    def test_a_reader_that_falls_back_with_the_llm_off_records_nothing(self, stub_registry, stub_db):
        event = btc_ladder((70000,))
        event["markets"][0]["rules_primary"] = "Reworded: the market resolves to Yes on a BRTI above 69,999.99."
        sids = sorted(load(stub_registry, stub_db, event).values())
        result = self.read(stub_registry, stub_db, sids, None)
        assert (result.read, result.parked) == (0, 0)
        assert self.store.get_many(sids) == {}

    def test_parked_securities_are_looked_at_again_when_the_readers_change(self, stub_registry, stub_db, monkeypatch):
        sids = sorted(load_multi(stub_registry, stub_db, mention_event(("AI",))).values())
        self.read(stub_registry, stub_db, sids, None)
        assert self.read(stub_registry, stub_db, sids, None).backing_off == 1
        monkeypatch.setattr(stage, "READERS_VERSION", "changed")
        result = self.read(stub_registry, stub_db, sids, None)
        assert (result.backing_off, result.parked) == (0, 1)


@pytest.mark.skipif(not os.environ.get("REDIS_TEST_URL"), reason="needs REDIS_TEST_URL (a throwaway Redis)")
def test_the_redis_store_keeps_records_until_they_expire():
    store = RedisAttemptStore(os.environ["REDIS_TEST_URL"])
    store.clear([1, 2])
    failed, parked = Attempt("h", "p1", FAILED, 2, 100.0), Attempt("h", "p1", NEEDS_LLM, 0, 200.0)
    store.put(1, failed, 60)
    store.put(2, parked, 60)
    assert store.get_many([1, 2, 3]) == {1: failed, 2: parked}
    assert 0 < store._redis.ttl("prop:attempt:2") <= 60
    store.clear([1, 2])
    assert store.get_many([1, 2]) == {}
