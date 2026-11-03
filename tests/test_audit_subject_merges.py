from scripts.audit_subject_merges import _Proposition, split_sides


def row(sid, extracted, resolved=False, entity="x"):
    return _Proposition(security_id=sid, subject_id=1, extracted_subject_id=extracted,
                        fields={"kind": "WINNER", "measure": "winner", "entity": entity}, exchange_code="KALSHI",
                        event_title="t", outcome_label=entity, event_resolved=resolved)


def test_each_proposition_goes_back_to_the_subject_it_was_read_onto():
    sides = split_sides([row(1, 1), row(2, 2), row(3, 2, resolved=True)], alias_id=2, alias_key="b", canonical_key="a")
    assert [c.proposition.subject for c in sides["a"]] == ["a"]
    assert len(sides["b"]) == 1 and sides["b"][0].proposition.subject == "b"


def test_a_side_with_nothing_open_shows_its_resolved_contracts():
    sides = split_sides([row(1, 1), row(2, 2, resolved=True)], alias_id=2, alias_key="b", canonical_key="a")
    assert len(sides["b"]) == 1
