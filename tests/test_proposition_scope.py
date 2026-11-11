from classifier.propositions.scope import EVERYTHING, LLMScope
from classifier.runtime_config import ClassifierConfig, _from_dict


def test_an_empty_scope_takes_every_event():
    assert EVERYTHING.unlimited and EVERYTHING.includes(None, ()) and EVERYTHING.fingerprint() == "*"


def test_matches_any_category_or_tag_ignoring_case():
    scope = LLMScope.of([" sports "], ["NFL", ""])
    assert scope.includes("SPORTS", ()) and scope.includes("POLITICS", ["nfl"])
    assert not scope.includes("POLITICS", ["nba"]) and not scope.includes(None, ())


def test_fingerprint_ignores_order_and_case():
    assert LLMScope.of(["SPORTS", "politics"], ["nfl"]).fingerprint() == LLMScope.of(["Politics", "sports"], ["NFL"]).fingerprint()
    assert LLMScope.of(["SPORTS"]).fingerprint() != LLMScope.of(tags=["sports"]).fingerprint()


def test_config_reads_the_scope_and_defaults_to_everything():
    assert ClassifierConfig().relationship_scope.categories == []
    cfg = _from_dict(ClassifierConfig, {"relationship_scope": {"categories": ["SPORTS"], "tags": ["nfl"]}})
    assert (cfg.relationship_scope.categories, cfg.relationship_scope.tags) == (["SPORTS"], ["nfl"])
