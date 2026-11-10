"""Acceptance tests for mega/live_calib.py. Fixtures: tests/fixtures/live_calib/ (from the handoff)."""
import json
from pathlib import Path

import pandas as pd

from mega import live_calib as L

FX = Path(__file__).parent / "fixtures" / "live_calib"
B_HIST = {"QB": 0.288, "RB": 0.577, "WR": 0.365, "TE": 0.267}   # data/matchup_constants.json b_def


def test_demo():
    L.demo()


def test_reproduces_2026_wk09():
    rows = pd.read_csv(FX / "graded_2026_wk03-09.csv").to_dict("records")
    games = pd.read_csv(FX / "odds_games_2026_wk07-09.csv").to_dict("records")
    got = L.build(rows, games, B_HIST)
    want = json.loads((FX / "live_calib_2026_wk09.json").read_text())
    assert {p: got["defense"][p]["active"] for p in L.POS} == {"QB": True, "RB": False, "WR": False, "TE": False}
    assert abs(got["defense"]["QB"]["beta"] - want["defense"]["QB"]["beta"]) < 0.005
    assert got["availability"]["active"] and abs(got["availability"]["buckets"][3]["p"] - 0.96) < 0.005
    assert got["odds"]["n"] == 18 and not got["odds"]["active"] and abs(got["odds"]["lam"] - 0.85) < 0.01


def test_tags_graded_against_our_projection_pass0():
    """Pass 0: good / neutral / tough against ours at kickoff, played rows only."""
    from mega import scorecard as sc
    rows = pd.read_csv(FX / "graded_2026_wk03-09.csv")
    rows["tag"] = rows["tag"].astype(object).where(rows["tag"].notna(), None)
    m = sc.grade_matchups(rows.to_dict("records"))
    assert {b: m[b]["n"] for b in ("good", "neutral", "tough")} == {"good": 159, "neutral": 523, "tough": 187}
    assert (m["good"]["vs_ours_pct"], m["neutral"]["vs_ours_pct"], m["tough"]["vs_ours_pct"]) == (6.4, 2.7, -6.9)
    assert m["spread"] == 13.3


def test_knob_file_for_another_season_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "CALIB", tmp_path / "c.json")
    assert L.load(2026) == {}
    (tmp_path / "c.json").write_text(json.dumps({"season": 2025, "odds": {"active": True}}))
    assert L.load(2026) == {}
    assert L.apply_availability(1.0, {"active": True, "buckets": {"3": {"said": 1.0, "p": 0.96}}}) == 0.96
