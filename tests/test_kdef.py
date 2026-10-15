"""mega/kdef.py — Yahoo default K/DEF scoring, and the merges into start/sit and stats."""
from __future__ import annotations

import pytest

from mega import kdef, start_sit


def test_points_allowed_tiers_are_yahoos():
    assert [kdef.pa_points(x) for x in (0, 1, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 50)] == \
        [10, 7, 7, 4, 4, 1, 1, 0, 0, -1, -1, -4, -4]


def test_expected_tier_falls_as_the_opponent_scores_more():
    vals = [kdef.pa_expected(mu) for mu in (10, 17, 24, 31)]
    assert vals == sorted(vals, reverse=True)
    assert kdef.pa_expected(0.0, sd=0.01) == pytest.approx(10, abs=1e-6)


def test_kicker_points_by_distance():
    # 2 short FGs (3 each), one 45-yarder (4), one 52 (5), 3 PATs
    row = {"fg_made_20_29": 1, "fg_made_30_39": 1, "fg_made_40_49": 1, "fg_made_50_59": 1, "pat_made": 3}
    assert kdef.k_points(row) == 3 + 3 + 4 + 5 + 3


def test_kicker_and_defense_join_the_start_sit_total():
    payload = {"available": True, "headline": "Your best legal lineup projects 88 points in week 5.",
               "kpis": {"proj_total": 88.0, "starters_n": 7}, "starters": [{"player": "x"}] * 7, "bench": []}
    rows = [{"lineup": "K", "player": "K1", "pos": "K", "proj_adj": 8.3, "start": True},
            {"lineup": "DEF", "player": "D1", "pos": "DEF", "proj_adj": 6.6, "start": True},
            {"lineup": "BENCH", "player": "D2", "pos": "DEF", "proj_adj": 5.0, "start": False}]
    payload["kpis"].update(close_calls=2, differs=[])
    payload["subhead"] = "2 calls within two points; the consensus would set the same lineup."
    out = start_sit.add_kdef(payload, rows)
    assert out["subhead"] == payload["subhead"]          # K/DEF don't change who we part with
    assert out["kpis"]["proj_total"] == pytest.approx(102.9)
    assert out["kpis"]["starters_n"] == 9 and len(out["bench"]) == 1
    assert "103 points" in out["headline"]


def test_current_kicker_is_the_active_one_on_the_roster_not_the_last_to_kick():
    import pandas as pd
    k = pd.DataFrame({"gsis_id": ["a", "a", "b"], "player": ["Old", "Old", "Bee"],
                      "team": ["KC", "KC", "BUF"], "week": [3, 4, 4]})
    roster = pd.DataFrame({"team": ["KC", "KC", "KC", "KC"], "week": [4, 5, 5, 5],
                           "gsis_id": ["a", "a", "n", "x"], "full_name": ["Old", "Old", "New", "Dev"],
                           "status": ["ACT", "RES", "ACT", "DEV"]})
    cur = kdef.current_kickers(k, roster).set_index("team")
    assert cur.loc["KC", "player"] == "New"          # Old went on IR, New signed, no kicks yet
    assert cur.loc["BUF", "player"] == "Bee"         # no roster rows: last to kick
    assert kdef.current_kickers(k, roster.iloc[:0]).set_index("team").loc["KC", "player"] == "Old"
