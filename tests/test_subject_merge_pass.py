import dataclasses
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from classifier.propositions.derive import Contract
from classifier.propositions.merges import Judgment, SubjectProfile, Verdict
from classifier.propositions.quotes import ListingRef, Quote, kalshi_quote, polymarket_intl_quote, polymarket_us_quote
from classifier.propositions.types import Kind, Proposition
from classifier.stages.subject_merges import MERGE_ACTOR, Outcome, fingerprint, is_season_key, run_merge_pass

ELECTION_DAY = datetime(2026, 11, 4, tzinfo=timezone.utc)


def winner(sid, subject, entity, venue, event_id):
    return Contract(sid, venue, Proposition(subject, Kind.WINNER, measure="winner", entity=entity,
                                            window_end=ELECTION_DAY), event_id=event_id)


class FakeDB:
    """Subjects as {key: (subject_id, contracts, 24h volume)}."""

    def __init__(self, subjects):
        self.subjects = subjects
        self.neighbours = {}
        self.clock = datetime(2026, 11, 3, 12, 0)
        self.changed_at = {}
        self.neighbour_lookups = []

    def subject_clock(self):
        return self.clock

    def _id(self, subject_id):
        return next(k for k, (i, _, _) in self.subjects.items() if i == subject_id)

    def subject_summaries(self):
        out = []
        for key, (subject_id, contracts, volume) in self.subjects.items():
            ps = [c.proposition for c in contracts]
            out.append((subject_id, key, key, sorted({c.venue for c in contracts}),
                        sorted({p.entity for p in ps if p.kind == Kind.WINNER}), sorted({p.measure for p in ps}),
                        sorted({p.window_end.date().isoformat() for p in ps}), len(ps), volume,
                        self.changed_at.get(key, datetime(2026, 11, 1))))
        return out

    def subject_neighbours(self, subject_ids, limit, min_similarity):
        self.neighbour_lookups.append(sorted(subject_ids))
        return {i: ns for i, ns in self.neighbours.items() if i in subject_ids}

    def subject_samples(self, subject_ids, per_venue=10):
        return {i: [(c.venue, "title", "rules", c.proposition.entity, self._id(i), c.proposition.to_fields())
                    for c in self.subjects[self._id(i)][1]] for i in subject_ids}

    def get_subject_contracts(self, subject_id):
        return list(self.subjects[self._id(subject_id)][1]), set(), ELECTION_DAY

    def get_active_listings_for_securities(self, security_ids):
        return [(sid, sid, 1, f"m{sid}") for sid in security_ids]

    def propositions_extracted_under(self, subject_id):
        key = self._id(subject_id)
        return [(c.security_id, subject_id, key, c.proposition.to_fields(), "LLM", "p6", "h")
                for c in self.subjects[key][1]]


class FakeRegistry:
    def __init__(self, decided=(), unmerged=()):
        self.merges, self.queued, self.upserted = [], [], []
        self.decided = decided
        self.unmerged = unmerged

    def get_subject_merge_log(self, action=None):
        return [SimpleNamespace(alias_subject_id=a, canonical_subject_id=c, action="UNMERGE") for a, c in self.unmerged]

    def get_subject_merge_candidates(self, status=None):
        return [SimpleNamespace(alias_subject_id=a, canonical_subject_id=c, status="DISMISSED") for a, c in self.decided]

    def upsert_subject_merge_candidates(self, items):
        self.queued.extend(items)

    def merge_proposition_subjects(self, *, from_subject_id, into_subject_id, actor):
        self.merges.append((from_subject_id, into_subject_id, actor))

    def upsert_contract_propositions(self, rows):
        self.upserted.extend(rows)


class FakeJudge:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.asked = []

    def judge(self, pairs):
        self.asked.extend((a.key, b.key) for a, b in pairs)
        return {(a.key, b.key): self.verdicts[(a.key, b.key)] for a, b in pairs}


class Memory:
    def __init__(self):
        self.rows = {}
        self.sweep = None

    def recall(self, pairs):
        return {p: self.rows[p] for p in pairs if p in self.rows}

    def remember(self, pair, remembered):
        self.rows[pair] = remembered

    def last_sweep(self):
        return self.sweep

    def set_last_sweep(self, at):
        self.sweep = at


GA = "georgia-governor-2026"
GA_SHORT = "ga-governor-2026"
SAME = Judgment(Verdict.SAME, "same election", (("rep", "republican"),))


def governor_db(volume=50_000.0):
    return FakeDB({
        GA: (1, [winner(11, GA, "republican", "KALSHI", 10), winner(12, GA, "democratic", "KALSHI", 10)], volume),
        GA_SHORT: (2, [winner(21, GA_SHORT, "rep", "POLYMARKET_US", 20)], volume),
    })


def run(db, registry, judge, memory=None, auto_merge=True, quotes=lambda refs: {}, judge_limit=100):
    return run_merge_pass(db=db, registry=registry, judge=judge, memory=memory or Memory(), auto_merge=auto_merge,
                          min_volume=10_000, judge_limit=judge_limit, max_price_gap=0.15, quotes=quotes)


class TestMergePass:
    def test_merges_into_the_bigger_subject_and_aligns_its_keys(self):
        registry = FakeRegistry()
        result = run(governor_db(), registry, FakeJudge({(GA_SHORT, GA): SAME}))
        assert registry.merges == [(2, 1, MERGE_ACTOR)]
        assert [(r["security_id"], r["fields"]["entity"], r["extracted_subject_id"]) for r in registry.upserted] == [
            (21, "republican", 2)]
        assert (result.merged, result.relationships_added, result.keys_aligned) == (1, 1, 1)

    def test_only_logs_while_auto_merge_is_off(self):
        registry = FakeRegistry()
        result = run(governor_db(), registry, FakeJudge({(GA_SHORT, GA): SAME}), auto_merge=False)
        assert registry.merges == [] and result.would_merge == 1

    def test_remembers_judgments_until_a_subject_changes(self):
        memory, judge = Memory(), FakeJudge({(GA_SHORT, GA): SAME})
        run(governor_db(), FakeRegistry(), judge, memory, auto_merge=False)
        run(governor_db(), FakeRegistry(), judge, memory, auto_merge=False)
        assert judge.asked == [(GA_SHORT, GA)]

        changed = governor_db()
        changed.subjects[GA_SHORT][1].append(winner(22, GA_SHORT, "dem", "POLYMARKET_US", 20))
        run(changed, FakeRegistry(), judge, memory, auto_merge=False)
        assert judge.asked == [(GA_SHORT, GA)] * 2

    def test_handles_a_pair_once_and_again_when_auto_merge_turns_on(self):
        memory, registry = Memory(), FakeRegistry()
        judge = FakeJudge({(GA_SHORT, GA): SAME})
        assert run(governor_db(), registry, judge, memory, auto_merge=False).would_merge == 1
        assert run(governor_db(), registry, judge, memory, auto_merge=False).handled == 0
        assert run(governor_db(), registry, judge, memory, auto_merge=True).merged == 1
        assert memory.rows[(GA_SHORT, GA)].outcome == Outcome.MERGED

    def test_checks_a_pair_that_gained_nothing_again_only_when_its_contracts_change(self):
        memory, judge = Memory(), FakeJudge({(GA_SHORT, GA): Judgment(Verdict.SAME, "same")})
        assert run(governor_db(), FakeRegistry(), judge, memory).no_gain == 1
        assert run(governor_db(), FakeRegistry(), judge, memory).handled == 0
        grown = governor_db()
        grown.subjects[GA][1].append(winner(13, GA, "libertarian", "KALSHI", 10))
        assert run(grown, FakeRegistry(), judge, memory).handled == 1

    def test_checks_a_pair_that_gained_nothing_again_once_when_the_rules_change(self):
        memory, judge = Memory(), FakeJudge({(GA_SHORT, GA): Judgment(Verdict.SAME, "same")})
        run(governor_db(), FakeRegistry(), judge, memory)
        memory.rows[(GA_SHORT, GA)] = dataclasses.replace(memory.rows[(GA_SHORT, GA)], rules=1)
        assert run(governor_db(), FakeRegistry(), judge, memory).handled == 1
        assert run(governor_db(), FakeRegistry(), judge, memory).handled == 0

    def test_leaves_a_held_pair_to_the_person_reviewing_it(self):
        memory, reads = Memory(), []

        def quotes(refs):
            reads.append(refs)
            return {11: Quote(0.70, 0.72), 21: Quote(0.30, 0.32)}

        run(governor_db(), FakeRegistry(), FakeJudge({(GA_SHORT, GA): SAME}), memory, quotes=quotes)
        run(governor_db(), FakeRegistry(), FakeJudge({}), memory, quotes=quotes)
        assert len(reads) == 1 and memory.rows[(GA_SHORT, GA)].outcome == Outcome.QUEUED_PRICE

    def test_looks_up_neighbours_only_for_subjects_changed_since_the_last_sweep(self):
        memory, db = Memory(), governor_db()
        run(db, FakeRegistry(), FakeJudge({(GA_SHORT, GA): SAME}), memory, auto_merge=False)
        assert db.neighbour_lookups == [[1, 2]] and memory.sweep == db.clock

        db.clock += timedelta(hours=3)
        db.changed_at[GA] = db.clock - timedelta(minutes=5)
        run(db, FakeRegistry(), FakeJudge({}), memory, auto_merge=False)
        assert db.neighbour_lookups[-1] == [1]

    def test_a_sweep_stays_open_while_pairs_are_left_to_judge(self):
        memory, db = Memory(), governor_db()
        result = run(db, FakeRegistry(), FakeJudge({}), memory, judge_limit=0)
        assert result.left_to_judge == 1 and memory.sweep is None

    def test_holds_a_merge_whose_prices_disagree(self):
        registry = FakeRegistry()
        quotes = lambda refs: {11: Quote(0.70, 0.72), 21: Quote(0.30, 0.32)}
        result = run(governor_db(), registry, FakeJudge({(GA_SHORT, GA): SAME}), quotes=quotes)
        assert registry.merges == [] and result.queued_price == 1
        assert registry.queued[0]["hold"] == "PRICE" and registry.queued[0]["details"]["prices"][0]["mid_a"] == 0.71

    def test_holds_a_merge_that_makes_two_outcomes_alike(self):
        db = governor_db()
        db.subjects[GA][1].append(winner(13, GA, "libertarian", "KALSHI", 10))
        db.subjects[GA_SHORT] = (2, [winner(21, GA_SHORT, "rep", "POLYMARKET_US", 20),
                                     winner(22, GA_SHORT, "republican-party", "POLYMARKET_US", 20)], 50_000.0)
        judgment = Judgment(Verdict.SAME, "same", (("rep", "republican"), ("republican-party", "republican")))
        registry = FakeRegistry()
        result = run(db, registry, FakeJudge({(GA_SHORT, GA): judgment}))
        assert registry.merges == [] and result.queued_collision == 1 and registry.queued[0]["hold"] == "COLLISION"

    def test_never_merges_related_pairs_or_queues_them(self):
        registry = FakeRegistry()
        result = run(governor_db(), registry, FakeJudge({(GA_SHORT, GA): Judgment(Verdict.RELATED, "other race")}))
        assert registry.merges == [] and registry.queued == [] and result.handled == 0

    def test_skips_a_merge_that_adds_nothing(self):
        registry = FakeRegistry()
        result = run(governor_db(), registry, FakeJudge({(GA_SHORT, GA): Judgment(Verdict.SAME, "same")}))
        assert registry.merges == [] and result.no_gain == 1

    def test_never_merges_a_season_key_into_another_kind_of_subject(self):
        db = FakeDB({
            "mlb:nl-cy-young-2026": (1, [winner(11, "mlb:nl-cy-young-2026", "andrew-abbott", "KALSHI", 10),
                                         winner(12, "mlb:nl-cy-young-2026", "paul-skenes", "KALSHI", 10)], 50_000.0),
            "andrew-abbott:mlb-2026": (2, [winner(21, "andrew-abbott:mlb-2026", "andrew-abbott", "POLYMARKET_US", 20)],
                                       50_000.0),
        })
        db.neighbours = {1: [2]}
        registry = FakeRegistry()
        result = run(db, registry, FakeJudge({("andrew-abbott:mlb-2026", "mlb:nl-cy-young-2026"): SAME}))
        assert registry.merges == [] and result.season_key == 1

    def test_a_pair_a_person_unmerged_is_never_merged_again(self):
        judge, registry = FakeJudge({(GA_SHORT, GA): SAME}), FakeRegistry(unmerged=[(2, 1)])
        run(governor_db(), registry, judge)
        assert judge.asked == [] and registry.merges == []

    def test_a_dismissed_pair_is_not_judged_again(self):
        judge = FakeJudge({})
        run(governor_db(), FakeRegistry(decided=[(2, 1)]), judge)
        assert judge.asked == []

    def test_quiet_subjects_are_left_alone(self):
        judge = FakeJudge({})
        run(governor_db(volume=500.0), FakeRegistry(), judge)
        assert judge.asked == []


def test_season_keys():
    assert is_season_key("josh-allen:nfl-2026-27") and is_season_key("andrew-abbott:mlb-2026")
    assert not is_season_key("mlb:nl-cy-young-2026") and not is_season_key("nba-champion:2026-27")


def test_fingerprint_follows_entities_not_samples():
    from_entities = SubjectProfile(GA, "", frozenset({"KALSHI"}), frozenset({"republican"}), frozenset({"winner"}),
                                   frozenset(), 2, ())
    assert fingerprint(from_entities, from_entities) == fingerprint(
        from_entities, dataclasses.replace(from_entities, samples=({"outcome": "x"},), proposition_count=9))
    assert fingerprint(from_entities, from_entities) != fingerprint(
        from_entities, dataclasses.replace(from_entities, entities=frozenset({"democratic"})))


class TestQuotes:
    def test_kalshi_no_side_is_priced_as_the_complement(self):
        market = {"status": "active", "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.45",
                  "yes_bid_size_fp": "10", "yes_ask_size_fp": "0"}
        assert kalshi_quote(market, "yes") == Quote(0.40, None)
        assert kalshi_quote(market, "no") == Quote(None, 0.6)
        assert kalshi_quote({**market, "status": "closed"}, "yes") is None

    def test_polymarket_intl_takes_the_best_levels(self):
        book = {"bids": [{"price": "0.10", "size": "5"}, {"price": "0.30", "size": "1"}, {"price": "0.35", "size": "0"}],
                "asks": [{"price": "0.90", "size": "1"}, {"price": "0.40", "size": "2"}]}
        assert polymarket_intl_quote(book) == Quote(0.30, 0.40)

    def test_polymarket_us_short_is_the_complement(self):
        market = {"status": "MARKET_STATUS_OPEN", "bestBidQuote": {"value": "0.2"}, "bestAskQuote": {"value": "0.25"}}
        assert polymarket_us_quote(market, "long") == Quote(0.2, 0.25)
        assert polymarket_us_quote(market, "short") == Quote(0.75, 0.8)

    def test_listing_refs_are_hashable(self):
        assert len({ListingRef(1, "KALSHI", "m"), ListingRef(1, "KALSHI", "m")}) == 1
