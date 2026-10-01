"""mega/schedule_web.py: the next-N-games matchup outlook per team x position."""
from __future__ import annotations

import pandas as pd

from mega import schedule_web as S


def _m(rows):
    return pd.DataFrame(rows, columns=["team", "pos", "week", "opp", "pct"])


def test_outlook_averages_the_next_games_in_week_order():
    m = _m([("CHI", "QB", 7, "GB", 6.0), ("CHI", "QB", 4, "NYJ", 2.0), ("CHI", "QB", 5, "LV", -4.0),
            ("CHI", "QB", 6, "MIA", 8.0), ("CHI", "QB", 8, "DEN", 100.0)])
    o = S.outlook(m, n=4)["CHI|QB"]
    assert [g[0] for g in o["g"]] == [4, 5, 6, 7]            # the fifth game is out of the window
    assert o["pct"] == 3.0 and o["g"][0] == [4, "NYJ", 2.0]


def test_a_bye_is_skipped_not_counted_and_unmodelled_games_drop_out():
    m = _m([("CAR", "WR", 4, "DET", 1.0), ("CAR", "WR", 6, "TB", None), ("CAR", "WR", 7, "NO", 3.0)])
    o = S.outlook(m)["CAR|WR"]
    assert [g[0] for g in o["g"]] == [4, 7] and o["pct"] == 2.0     # week 5 bye never existed in the frame


def test_names_map_only_skill_players_with_a_team():
    idx = pd.DataFrame({"name": ["A", "B", "C"], "pos": ["QB", "K", "RB"], "team": ["CHI", "CHI", None]})
    assert S.names(idx) == {"A": "CHI|QB"}


def test_empty_frame_is_unavailable():
    assert S.outlook(pd.DataFrame()) == {}
