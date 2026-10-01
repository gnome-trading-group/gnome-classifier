from scripts.merged_cleanup import _select_targets


def test_merged_event_and_its_securities_are_targeted():
    natives = {1: {(5, "KXWTI15M-A"), (5, "KXWTI15M-B")}, 2: {(5, "KXOTHER")}}
    events_by_security = {10: {1}, 11: {1}, 20: {2}}
    event_ids, security_ids = _select_targets(natives, events_by_security, {}, set())
    assert event_ids == {1}
    assert security_ids == {10, 11}


def test_security_shared_across_events_is_targeted_without_its_events():
    natives = {1: {(5, "a")}, 2: {(5, "b")}}
    events_by_security = {10: {1, 2}, 11: {1}}
    event_ids, security_ids = _select_targets(natives, events_by_security, {}, set())
    assert event_ids == set()
    assert security_ids == {10}


def test_security_listed_on_two_exchanges_is_targeted():
    natives = {1: {(4, "a")}}
    event_ids, security_ids = _select_targets(natives, {10: {1}}, {10: {4, 5}}, set())
    assert security_ids == {10}


def test_excluded_security_keeps_its_whole_merged_event():
    natives = {1: {(5, "a"), (5, "b")}}
    events_by_security = {10: {1}, 11: {1}}
    event_ids, security_ids = _select_targets(natives, events_by_security, {}, {10})
    assert event_ids == set()
    assert security_ids == set()
