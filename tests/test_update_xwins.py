"""tools/update_xwins.py — the xWins workbook filled from data/yahoo_scores.csv."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "update_xwins", Path(__file__).resolve().parents[1] / "tools" / "update_xwins.py")
UX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(UX)


def _week(wk, names):
    rows = []
    for a, b in zip(names[::2], names[1::2]):
        rows += [dict(team=a, week=wk, points=100.0, opponent=b, opp_points=90.0),
                 dict(team=b, week=wk, points=90.0, opponent=a, opp_points=100.0)]
    return rows


def test_every_seat_has_one_row_and_a_partial_week_is_left_out():
    from mega.config import TEAM_BY_SEAT
    assert sorted(UX.ROW_SEATS) == list(range(1, 13))
    names = [TEAM_BY_SEAT[s] for s in range(1, 13)]
    names[6] = "A Place in the Hampton"          # an older file still on the old name
    sc = pd.DataFrame(_week(1, names) + _week(2, names)[:10])
    out = UX.weeks_from_scores(sc)
    assert list(out) == [1]
    assert out[1][7] == (100.0, 1) and out[1][8] == (90.0, 0)
