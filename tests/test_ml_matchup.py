"""mega/ml_matchup.py: a feature may only ever see games before the one being predicted."""
from __future__ import annotations

import pandas as pd

from mega import ml_matchup as M


def test_prior_sums_never_see_the_game_being_predicted():
    d = pd.DataFrame({"gsis_id": ["a"] * 5 + ["a", "b", "b"], "season": [2024] * 5 + [2025, 2024, 2024],
                      "week": [1, 2, 3, 4, 5, 1, 1, 2], "x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 100.0, 200.0]})
    P = M.prior_sums(d)
    assert P("x").tolist() == [0, 1, 3, 7, 15, 0, 0, 100]          # season to date; a new season starts over
    assert P("x", 3).tolist() == [0, 1, 3, 7, 14, 0, 0, 100]       # his last 3 games
    assert P("x", 1).tolist() == [0, 1, 2, 4, 8, 0, 0, 100]        # his last game


def test_a_call_needs_a_full_point_and_reasons_add_up():
    assert [M.call(x) for x in (1.0, 0.99, -0.99, -1.0)] == ["good", "normal", "normal", "bad"]
    row = {"gsis_id": "a", "avg": 10.0, "ml": 12.34, "ml_week": 1.5, **{g: 0.0 for g in M.WEEK_GROUPS}}
    row.update({"Vegas line": 1.2, "Teammates out": 0.5, "Opposing defense": -0.2})
    out = M.for_export(pd.DataFrame([row]))["a"]
    assert out["call"] == "good" and out["proj"] == 12.3
    assert [w["label"] for w in out["why"]] == ["Vegas line", "Teammates out"]     # largest first, small ones left off
