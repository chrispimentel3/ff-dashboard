"""tools/fit_ml_matchup.py: a feature may only ever see games before the one being predicted."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd


def _tool():
    p = pathlib.Path(__file__).resolve().parent.parent / "tools" / "fit_ml_matchup.py"
    spec = importlib.util.spec_from_file_location("fit_ml_matchup", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_prior_sums_never_see_the_game_being_predicted():
    d = pd.DataFrame({"gsis_id": ["a"] * 5 + ["a", "b", "b"], "season": [2024] * 5 + [2025, 2024, 2024],
                      "week": [1, 2, 3, 4, 5, 1, 1, 2], "x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 100.0, 200.0]})
    P = _tool().prior_sums(d)
    assert P("x").tolist() == [0, 1, 3, 7, 15, 0, 0, 100]          # season to date; a new season starts over
    assert P("x", 3).tolist() == [0, 1, 3, 7, 14, 0, 0, 100]       # his last 3 games
    assert P("x", 1).tolist() == [0, 1, 2, 4, 8, 0, 0, 100]        # his last game
