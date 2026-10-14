from mega import playoff_plan as PP


def test_tags_mark_the_extremes_only():
    assert PP._tag(1) == "easy" and PP._tag(PP.EASY) == "easy"
    assert PP._tag(PP.TOUGH) == "tough" and PP._tag(32) == "tough"
    assert PP._tag(16) is None and PP._tag(None) is None


def test_each_team_starts_its_own_kdef_and_streams_only_on_a_bye(monkeypatch):
    import pandas as pd
    from mega import kdef, title_odds as T
    ros = pd.DataFrame([{"fantasy_team": "A", "pos": "DEF", "key": "KC"}, {"fantasy_team": "A", "pos": "K", "key": "Kick A"},
                        {"fantasy_team": "B", "pos": "DEF", "key": "NYJ"}])            # B has no kicker
    defs = pd.DataFrame([{"team": t, "week": w, "proj": p} for w in (6, 7) for t, p in
                         [("KC", 9.0 if w == 6 else 0.0), ("NYJ", 4.0), ("F1", 8.0), ("F2", 7.0), ("F3", 6.0), ("F4", 5.0)]])
    ks = pd.DataFrame([{"player": n, "week": w, "proj": p} for w in (6, 7) for n, p in
                       [("Kick A", 7.0), ("K1", 10.0), ("K2", 9.0), ("K3", 8.0)]])
    monkeypatch.setattr(kdef, "rostered", lambda r: ros)
    monkeypatch.setattr(kdef, "project", lambda s, weeks: {"DEF": defs, "K": ks})
    m = T.kdef_means(2026, [6, 7], None)
    assert m[("A", 6)] == 9.0 + 7.0          # his own, even when a free agent projects higher
    assert m[("A", 7)] == 6.0 + 7.0          # KC on bye: the 3rd-best free defense
    assert m[("B", 6)] == 4.0 + 8.0          # no kicker: the 3rd-best free kicker
