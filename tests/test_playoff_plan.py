from mega import playoff_plan as PP


def test_tags_mark_the_extremes_only():
    assert PP._tag(1) == "easy" and PP._tag(PP.EASY) == "easy"
    assert PP._tag(PP.TOUGH) == "tough" and PP._tag(32) == "tough"
    assert PP._tag(16) is None and PP._tag(None) is None
