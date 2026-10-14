import pandas as pd
import pytest

from mega import scorecard as S
from mega.intel import _norm


def _row(player, pos, lineup, close="", team="AAA"):
    return {"player": player, "pos": pos, "lineup": lineup, "close_call": close, "nfl_team": team, "proj_adj": 10}


SS = {"starters": [_row("Quincy One", "QB", "QB"), _row("Ray One", "RB", "RB"), _row("Ray Two", "RB", "RB"),
                   _row("Wes One", "WR", "WR"), _row("Wes Two", "WR", "WR"), _row("Ty One", "TE", "TE"),
                   _row("Wes Three", "WR", "FLEX"), _row("K", "K", "K")],
      "bench": [_row("Wes Four", "WR", "BENCH", "≈ Wes Three"), _row("Quincy Two", "QB", "BENCH", "≈ Quincy One", team="BYE"),
                _row("Ray Three", "RB", "BENCH")]}
PTS = {_norm(n): p for n, p in [("Quincy One", 20), ("Ray One", 15), ("Ray Two", 10), ("Wes One", 12),
                                 ("Wes Two", 9), ("Ty One", 6), ("Wes Three", 4), ("Wes Four", 11),
                                 ("Quincy Two", 0), ("Ray Three", 3)]}


def test_lineup_grade_counts_points_left_and_close_calls():
    g = S.grade_lineup(SS, PTS, playing={"AAA"})
    assert g["ours"] == 76 and g["best"] == 83 and g["left"] == 7     # W4's 11 beat W3's 4 at flex
    assert g["should_have_started"] == ["Wes Four"]
    assert g["calls"] == [{"started": "Wes Three", "sat": "Wes Four", "started_pts": 4, "sat_pts": 11, "right": False}]
    assert g["bye_calls"] == ["Quincy Two"]                                   # no game: not a call at all


def test_best_lineup_is_legal():
    rows = [{"player": p, "pos": pos, "pts": pts} for p, pos, pts in
            [("q", "QB", 20), ("q2", "QB", 30), ("t", "TE", 1), ("t2", "TE", 25),
             ("r1", "RB", 5), ("r2", "RB", 4), ("r3", "RB", 3), ("w1", "WR", 9), ("w2", "WR", 8), ("w3", "WR", 7)]]
    best = S.best_lineup(rows)
    assert sorted(r["player"] for r in best) == ["q2", "r1", "r2", "t2", "w1", "w2", "w3"]


def test_projection_summary_is_head_to_head_on_the_same_players():
    rows = [{"pos": "WR", "ours": 10, "consensus": 12, "vegas": None, "actual": 10},
            {"pos": "RB", "ours": 8, "consensus": 6, "vegas": 7, "actual": 5}]
    s = S.summarize_projections(rows)
    assert s["n"] == 2 and s["mae_ours"] == 1.5 and s["mae_consensus"] == 1.5 and s["ours_closer"] == 0.5
    assert s["vegas"] == {"n": 1, "mae_vegas": 2.0, "mae_ours": 3.0, "mae_consensus": 1.0}


def test_a_player_on_bye_projects_zero(monkeypatch):
    from mega import lineup as L
    monkeypatch.setattr(L, "attach_matchup",
                        lambda df, s, w: df.assign(opp=[None if t == "CAR" else "XXX" for t in df["nfl_team"]], mult=1.0,
                                                          mult_fitted=float("nan")))
    roster = pd.DataFrame([{"player": "Young", "pos": "QB", "nfl_team": "CAR", "proj": 17.6, "proj_source": "nflverse-est"},
                           {"player": "Daniels", "pos": "QB", "nfl_team": "WAS", "proj": 18.5, "proj_source": "ffanalytics"}])
    out = L.optimize_lineup(roster, 2026, 5).set_index("player")
    assert out.loc["Young", "proj_adj"] == 0 and out.loc["Young", "report_status"] == "BYE"
    assert out.loc["Daniels", "proj_adj"] == pytest.approx(18.5) and out.loc["Daniels", "lineup"] == "QB"
