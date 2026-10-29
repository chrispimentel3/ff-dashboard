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


def test_start_sit_names_the_starters_the_consensus_would_bench():
    from mega import start_sit
    rows = [("Daniels", "QB", 18, 18), ("Young", "QB", 15, 19),          # ours starts Daniels, consensus Young
            ("R One", "RB", 14, 14), ("R Two", "RB", 12, 12), ("W One", "WR", 13, 13), ("W Two", "WR", 11, 11),
            ("T One", "TE", 9, 9), ("W Three", "WR", 10, 8), ("Out Guy", "WR", 0, 30)]   # ruled out: 0 on both
    lu = pd.DataFrame([{"player": p, "pos": pos, "proj": o, "proj_adj": o, "consensus": c} for p, pos, o, c in rows])
    from mega.lineup import fill
    lu["lineup"] = lu.index.map(lambda i: fill(lu).get(i, "BENCH"))
    lu["start"] = lu["lineup"] != "BENCH"
    assert start_sit._differs(lu) == [start_sit.short_name("Daniels")]


def test_week_expected_only_for_the_week_it_was_built_for():
    from mega.proj_ros import week_expected
    d = {"meta": {"week": 6}, "players": {"g1": {"weeks": {"6": {"expected": 12.5}, "7": {"expected": 3.0}}},
                                         "g2": {"weeks": {"7": {"expected": 9.0}}}}}
    assert week_expected(2026, 6, d) == {"g1": 12.5}
    assert week_expected(2026, 5, d) == {}


def test_the_live_blend_switches_on_only_after_winning_walk_forward():
    import random
    rnd = random.Random(3)
    rows = []
    for w in range(3, 9):
        for i in range(60):
            truth = rnd.uniform(5, 20)
            rows.append({"week": w, "pos": "QB", "actual": truth,
                         "ours": truth + rnd.gauss(0, 6),          # ours is noisy at QB...
                         "consensus": truth + rnd.gauss(0, 2)})    # ...the consensus is not
    lb = S.live_blend(rows)
    assert lb["tested"] == 5 and lb["active"] and lb["weights"]["QB"] < 0.8
    early = S.live_blend([r for r in rows if r["week"] <= 5])
    assert not early["active"]                                      # two tested weeks: not yet


def test_matchup_tags_are_graded_against_each_players_own_average():
    act = pd.DataFrame([{"gsis_id": g, "week": w, "half_ppr": p} for g, pts in
                        {"good": [10, 10, 15], "tough": [10, 10, 5], "flat": [10, 10, 10]}.items()
                        for w, p in zip((3, 4, 5), pts)])
    snaps = {5: {"rows": [{"gsis_id": "good", "matchup_pct": 8.0}, {"gsis_id": "tough", "matchup_pct": -9.0},
                          {"gsis_id": "flat", "matchup_pct": 1.0}]}}
    m = S.grade_matchups(snaps, act)
    assert m == {"good": {"n": 1, "vs_avg_pct": 50.0}, "tough": {"n": 1, "vs_avg_pct": -50.0},
                 "neutral": {"n": 1, "vs_avg_pct": 0.0}}


def _board(partner, give, get, d=None, thesis=""):
    fmt = lambda xs: " + ".join(f"{n} ({p})" for n, p in xs)
    return {"partner": partner, "give": fmt(give), "get": fmt(get), "d_me": d, "thesis": thesis}


def test_trade_ideas_are_replayed_on_our_best_lineup_each_week_since():
    roster = [{"player": n, "pos": p} for n, p in [("Quincy One", "QB"), ("Ray One", "RB"), ("Ray Two", "RB"),
                                                   ("Wes One", "WR"), ("Wes Two", "WR"), ("Ty One", "TE")]]
    snaps = {5: [_board("A", [("Ray Two", "RB")], [("Ray New", "RB")], 2.0),
                 _board("B", [("Wes One", "WR")], [("Wes Bad", "WR")], thesis="Wes Bad starts: +1.5 pts/wk to your lineup")],
             6: [_board("A", [("Ray Two", "RB")], [("Ray New", "RB")])]}          # shown again: graded once
    pts = {5: {_norm("Ray Two"): 5, _norm("Ray New"): 12, _norm("Wes One"): 10, _norm("Wes Bad"): 4},
           6: {_norm("Ray Two"): 9, _norm("Ray New"): 6, _norm("Wes One"): 10, _norm("Wes Bad"): 4}}
    g = S.grade_trades(snaps, {5: roster, 6: roster}, pts)
    by = {r["partner"]: r for r in g["rows"]}
    assert g["n"] == 2 and set(by) == {"A", "B"}
    assert by["A"]["weeks"] == 2 and by["A"]["per_week"] == 2.0 and by["A"]["right"] is True   # +7, -3
    assert by["A"]["said"] == 2.0 and by["B"]["said"] == 1.5              # older boards: read off the thesis
    assert by["B"]["per_week"] == -6.0 and by["B"]["right"] is False
    assert g["helped"] == 1 and g["hurt"] == 1
    # a week where the player we'd give is no longer ours isn't replayed
    g2 = S.grade_trades({5: snaps[5]}, {5: roster, 6: [r for r in roster if r["player"] != "Ray Two"]}, pts)
    assert {r["partner"]: r["weeks"] for r in g2["rows"]}["A"] == 1


def test_board_pieces_parse_as_printed():
    assert S._pieces("Kenneth Walker III (RB) + D'Andre Swift (RB)") == [("Kenneth Walker III", "RB"), ("D'Andre Swift", "RB")]


def test_win_odds_are_graded_from_the_favorites_side():
    snaps = {6: {"mine": {"opp": "Them", "p_win": 0.3},
                 "games": [{"root_for": "C", "against": "D", "p_root_wins": 0.8},
                           {"root_for": "E", "against": "F", "p_root_wins": 0.55}]}}
    res = pd.DataFrame([{"team": "Me", "week": 6, "points": 90, "opponent": "Them", "opp_points": 100},
                        {"team": "C", "week": 6, "points": 80, "opponent": "D", "opp_points": 95},
                        {"team": "E", "week": 6, "points": 70, "opponent": "F", "opp_points": 70}])     # tie: skipped
    g = S.grade_odds(snaps, res, "Me")
    assert g["n"] == 2 and g["favorites_won"] == 1
    assert {(r["favorite"], r["p"], r["won"]) for r in g["games"]} == {("Them", 0.7, True), ("C", 0.8, False)}
    assert {b["bucket"]: (b["n"], b["happened"]) for b in g["buckets"]} == {"60–75%": (1, 1.0), "75%+": (1, 0.0)}


def test_playoff_odds_wait_for_the_regular_season_to_end():
    hist = pd.DataFrame([{"season": 2026, "week": 6, "team": t, "p_playoffs": p, "model": "player"}
                         for t, p in [("A", 0.9), ("B", 0.1), ("C", 0.5)]])
    st = pd.DataFrame([{"rank": 1, "team": "A"}, {"rank": 7, "team": "B"}, {"rank": 2, "team": "C"}])
    assert S.grade_playoff_odds(hist, st, pd.DataFrame({"week": [13]}))["available"] is False
    g = S.grade_playoff_odds(hist, st, pd.DataFrame({"week": [14]}))
    assert {b["bucket"]: (b["n"], b["happened"]) for b in g["buckets"]} == \
        {"under 25%": (1, 0.0), "25–75%": (1, 1.0), "75%+": (1, 1.0)}
