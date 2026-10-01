"""mega/digest_web.py — the weekly digest page's data."""
from __future__ import annotations

import pandas as pd

from mega import digest_web as D


def _scores():
    # week 1: A 120 beats B 80, C 110 loses to D 115; week 2: A 70 beats C 60, B 130 beats D 100
    rows = [(1, "A", 120, "B", 80), (1, "C", 110, "D", 115),
            (2, "A", 70, "C", 60), (2, "B", 130, "D", 100)]
    out = []
    for wk, a, pa, b, pb in rows:
        out += [dict(week=wk, team=a, points=pa, opponent=b, opp_points=pb),
                dict(week=wk, team=b, points=pb, opponent=a, opp_points=pa)]
    return pd.DataFrame(out)


def test_the_week_just_played_is_told_in_sentences():
    fx = pd.DataFrame([{"week": 3, "home": "A", "away": "B"}, {"week": 3, "home": "C", "away": "D"}])
    league = {"playoff_odds": [{"team": t, "p_playoffs": p, "p_title": p / 4}
                               for t, p in (("A", .9), ("B", .6), ("C", .1), ("D", .4))]}
    hist = pd.DataFrame([{"week": 1, "team": t, "p_playoffs": p, "p_title": p / 4}
                         for t, p in (("A", .7), ("B", .5), ("C", .3), ("D", .5))])
    d = D.build(2026, "A", _scores(), fx, league, {"headline": "do things"}, hist)
    assert d["week"] == 2 and d["next_week"] == 3
    text = " ".join(l["text"] for l in d["headlines"])
    assert "You beat C 70.00–60.00 and sit at 2-0" in text
    assert "all-play 1-2" in text                       # 70 beat only C's 60
    assert "Luckiest: A won with the 3rd-best score" in text
    assert "High score: B, 130.00." in text
    assert d["odds_movers"]["available"] and d["odds_movers"]["rows"][0]["team"] == "A"
    assert d["my_game"]["b"] == "B" and d["game_of_week"]["a"] == "C"
    assert d["table"][0] == {"team": "B", "points": 130.0, "rank": 1, "allplay": "3-0", "won": True,
                             "opponent": "D", "record": "1-1"}


def test_odds_are_kept_once_per_week(tmp_path):
    path = tmp_path / "odds.csv"
    odds = [{"team": "A", "p_playoffs": .5, "p_bye": .1, "p_title": .1}]
    assert D.snapshot_odds(2026, 3, odds, path)
    assert not D.snapshot_odds(2026, 3, [{**odds[0], "p_playoffs": .9}], path)
    assert pd.read_csv(path)["p_playoffs"].tolist() == [.5]
