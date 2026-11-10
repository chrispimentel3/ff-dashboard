"""mega/gameday.py, the Sunday kickoff slots in tools/refresh_watch.py, and the status merge."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pathlib

from mega import gameday, status

ET = gameday.ET
NOON = dt.datetime(2026, 11, 15, 11, 45, tzinfo=ET)
ONE, FOUR = NOON.replace(hour=13, minute=0), NOON.replace(hour=16, minute=25)


def _row(player, pos, team, proj, **kw):
    return {"player": player, "pos": pos, "nfl_team": team, "proj_adj": proj, **kw}


def test_changes_only_for_games_still_to_play():
    ss = {"starters": [_row("Josh Downs", "WR", "IND", 9.1), _row("Late Guy", "WR", "SF", 8.0)],
          "bench": [_row("Denzel Boston", "WR", "CLE", 0.0, report_status="Out (Sleeper, 11:41 AM)"),
                    _row("Played Already", "WR", "NE", 0.0, report_status="Out")]}
    slots = {"Josh Downs": "BN", "Late Guy": "WR", "Denzel Boston": "WR", "Played Already": "WR"}
    kick = {"IND": ONE, "CLE": ONE, "SF": FOUR, "NE": NOON.replace(hour=9, minute=30)}
    out = gameday.changes(ss, slots, kick, NOON)
    assert out == ["Start Josh Downs (WR, 9.1 pts)", "Bench Denzel Boston (WR, Out (Sleeper, 11:41 AM))"]
    assert gameday.changes(ss, slots, kick, ONE + dt.timedelta(minutes=5)) == []   # 1 PM games under way


def test_close_call_the_sunday_lines_read_the_other_way_is_flagged():
    ss = {"starters": [_row("A", "WR", "SF", 10.2, vegas=9.0)],
          "bench": [_row("B", "WR", "SF", 9.5, vegas=11.0, close_call="≈ A"),
                    _row("C", "WR", "SF", 9.0, vegas=8.0, close_call="≈ A")]}
    out = gameday.changes(ss, {"A": "WR", "B": "BN", "C": "BN"}, {"SF": FOUR}, NOON)
    assert len(out) == 1 and out[0].startswith("Close call: Sunday lines have B 11.0 vs A 9.0")


def test_lineup_that_matches_says_nothing():
    ss = {"starters": [_row("A", "WR", "SF", 10.2)], "bench": [_row("B", "WR", "SF", 3.0)]}
    assert gameday.changes(ss, {"A": "W/R", "B": "BN"}, {"SF": FOUR}, NOON) == []


def test_out_for_week_adds_the_game_day_tags_for_that_week_only(monkeypatch, tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"week": 10, "out": {"denzel boston": "Out (Sleeper, 11:41 AM)"}}))
    monkeypatch.setattr(gameday, "GAMEDAY", f)
    monkeypatch.setattr(status, "STATUS_CSV", tmp_path / "none.csv")
    assert status.out_for_week(10) == {"denzel boston": "Out (Sleeper, 11:41 AM)"}
    assert status.out_for_week(11) == {}


def _watch():
    p = pathlib.Path(__file__).resolve().parent.parent / "tools" / "refresh_watch.py"
    spec = importlib.util.spec_from_file_location("refresh_watch", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kickoff_slots_run_once_each_sunday_and_not_late(monkeypatch, tmp_path):
    w = _watch()
    monkeypatch.setattr(w, "KICKOFF_DONE", tmp_path / "done.json")
    at = NOON.replace(minute=41)
    slot = w.kickoff_due(at)
    assert slot == NOON.replace(minute=40).isoformat()
    (tmp_path / "done.json").write_text(json.dumps([slot]))
    assert w.kickoff_due(at) is None                                   # already ran
    assert w.kickoff_due(NOON.replace(hour=12, minute=5)) is None      # Mac woke too late
    assert w.kickoff_due(NOON.replace(hour=15, minute=3)) == NOON.replace(hour=15, minute=0).isoformat()
    assert w.kickoff_due(at - dt.timedelta(days=1)) is None            # Saturday


# ---------------------------------------------------------------- Sunday lines on close calls
def test_close_call_grade_counts_only_disagreements_in_sunday_weeks():
    from mega import scorecard as sc
    r = lambda w, pos, ours, vegas, actual: {"week": w, "pos": pos, "ours": ours, "vegas": vegas, "actual": actual}
    rows = [r(10, "WR", 10.0, 9.0, 4.0), r(10, "WR", 9.0, 11.0, 12.0),     # disagree, Vegas right
            r(10, "RB", 8.0, 7.0, 3.0),                                    # RB vs WRs: flex pairs count
            r(10, "TE", 9.5, 20.0, 30.0),                                  # TE vs WR: never a choice
            r(9, "WR", 10.0, 9.0, 4.0), r(9, "WR", 9.0, 11.0, 12.0)]       # week 9: midweek lines
    g = sc.grade_close(rows, {10})
    # week 10, within 2 points: WR 10 vs WR 9 disagree (Vegas right); both WRs vs the RB agree
    assert g["weeks"] == 1 and g["disagreements"] == 1 and g["vegas_right"] == 1.0
    assert g["active"] is False                                            # one week isn't proof


def test_tiebreak_swaps_a_flagged_bench_player_in_when_vegas_has_him_higher():
    import pandas as pd
    from mega.lineup import vegas_tiebreak
    df = pd.DataFrame([
        {"player": "A", "pos": "WR", "lineup": "FLEX", "start": True, "vegas": 9.0, "close_call": ""},
        {"player": "B", "pos": "RB", "lineup": "BENCH", "start": False, "vegas": 11.0, "close_call": "≈ A"},
        {"player": "C", "pos": "WR", "lineup": "BENCH", "start": False, "vegas": 10.0, "close_call": "≈ A"},
        {"player": "D", "pos": "WR", "lineup": "WR", "start": True, "vegas": 12.0, "close_call": ""},
        {"player": "E", "pos": "WR", "lineup": "BENCH", "start": False, "vegas": None, "close_call": "≈ D"}])
    out = vegas_tiebreak(df).set_index("player")
    assert out.loc["B", "lineup"] == "FLEX" and out.loc["A", "lineup"] == "BENCH"    # best line wins the slot
    assert out.loc["C", "lineup"] == "BENCH"                                        # A already went
    assert out.loc["D", "start"] and not out.loc["E", "start"]                      # no line, no swap
    assert out.loc["A", "close_call"] == "≈ B"
