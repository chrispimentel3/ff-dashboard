"""Handoff v2: close calls on Start/Sit (chance + why), dead starts, per-kickoff grading."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from mega import scorecard as S
from mega.start_sit import close_calls

ET = ZoneInfo("America/New_York")
ONE, FOUR = datetime(2026, 11, 15, 13, 0, tzinfo=ET), datetime(2026, 11, 15, 16, 25, tzinfo=ET)


def _wk(role, vm=1.0, pa=1.0, dm=1.0):
    m = role * vm * dm
    return {"role_pg": role, "vegas_mult": vm, "return_mult": 1.0, "def_mult": dm, "p_active": pa,
            "mean_if_active": m, "median": 0.9 * m, "p10": 0.35 * m, "p90": 1.75 * m}


def test_close_calls_turn_flags_into_chances_with_reasons_that_add_up():
    lu = pd.DataFrame([
        {"player": "Star", "pos": "WR", "gsis_id": "s", "start": True, "proj_adj": 9.0, "close_call": "",
         "consensus": 8.0, "pct_fitted": 2.0, "vegas": None},
        {"player": "Bench", "pos": "RB", "gsis_id": "b", "start": False, "proj_adj": 8.0, "close_call": "≈ Star",
         "consensus": 10.0, "pct_fitted": -12.0, "vegas": None},
        {"player": "Far", "pos": "WR", "gsis_id": "f", "start": False, "proj_adj": 3.0, "close_call": "", "vegas": None},
        {"player": "Bye", "pos": "WR", "gsis_id": "y", "start": False, "proj_adj": 0.0, "close_call": "≈ Star", "vegas": None}])
    weeks = {"s": _wk(10.0, 0.9), "b": _wk(8.0), "f": _wk(3.0), "y": {"bye": True}}
    calls = close_calls(lu, weeks, applied={})
    assert len(calls) == 1                                       # no flag, or on bye: no call
    c = calls[0]
    assert (c["start"], c["sit"], c["gap"]) == ("Star", "Bench", 1.0) and 0.5 < c["p"] < 0.65
    assert abs(sum(x["pts"] for x in c["why"]) - c["gap"]) <= 0.15
    assert any("matchup -12%" in x and "not in the number" in x for x in c["context"])
    assert any(x.startswith("Consensus disagrees") for x in c["context"])
    applied = close_calls(lu, weeks, applied={"RB": True})[0]
    assert not any("matchup" in x for x in applied["context"])   # an applied knob is a reason, not context


def test_dead_starts_count_starters_without_a_stat_line_but_not_byes():
    ss = {"starters": [{"player": "Hurt", "pos": "RB", "lineup": "RB", "proj_adj": 12.0, "nfl_team": "NYJ"},
                       {"player": "Fine", "pos": "WR", "lineup": "WR", "proj_adj": 9.0, "nfl_team": "IND"},
                       {"player": "Idle", "pos": "WR", "lineup": "FLEX", "proj_adj": 0.0, "nfl_team": "TB"}],
          "bench": []}
    g = S.grade_lineup(ss, {"fine": 11.0}, playing={"NYJ", "IND"})
    assert g["dead_starts"] == {"n": 1, "proj": 12.0, "players": ["Hurt"]}


def test_each_player_is_graded_from_the_snapshot_before_his_own_kickoff(monkeypatch):
    early = {"next_week": 10, "starters": [{"player": "Early", "lineup": "WR", "nfl_team": "IND"},
                                           {"player": "Late Starter", "lineup": "FLEX", "nfl_team": "SF"}],
             "bench": [{"player": "Late Sub", "lineup": "BENCH", "nfl_team": "SF"}]}
    late = {"next_week": 10, "starters": [{"player": "Early", "lineup": "WR", "nfl_team": "IND"},
                                          {"player": "Late Sub", "lineup": "FLEX", "nfl_team": "SF"}],
            "bench": [{"player": "Late Starter", "lineup": "BENCH", "nfl_team": "SF"}],
            "calls": [{"start": "Late Sub", "sit": "Late Starter", "p": 0.61}]}
    import json
    monkeypatch.setattr(S, "snapshot", lambda path, k: json.dumps(early if k == ONE else late))
    got = S.lineup_at_kickoffs(10, early, {"IND": ONE, "SF": FOUR})
    assert sorted(r["player"] for r in got["starters"]) == ["Early", "Late Sub"]          # the 3:00 swap counts
    rows = S.week_calls(2026, 10, ONE, got, {"late sub": 9.0, "late starter": 4.0}, {"IND": ONE, "SF": FOUR})
    assert rows == [{"week": 10, "start": "Late Sub", "sit": "Late Starter", "p": 0.61, "won": True, "source": "published"}]
