"""HANDOFF v1.3 Pass 2: the rest-of-season projection (mega/proj_ros.py) on synthetic games."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mega import proj_ros as P


def _games(rows):
    """rows: (week, gid, pos, team, x_rec, t_rec, pts)"""
    out = []
    for w, gid, pos, team, x, t, pts in rows:
        out.append({"season": 2030, "week": w, "gsis_id": gid, "pos": pos, "team": team,
                    "x_rec": x, "t_rec": t, "x_rush": 0.0, "t_rush": 10.0, "x_pass": 0.0, "t_pass": 20.0,
                    "xfp": x, "pts": pts})
    return pd.DataFrame(out)


def test_ewm_weights_the_newest_game_most():
    assert P._ewm(np.array([0.0, 10.0]), h=1) == pytest.approx(10 / 1.5)
    assert P._ewm(np.array([10.0, 0.0]), h=1) == pytest.approx(5 / 1.5)


def test_projection_is_share_times_volume_plus_shrunk_efficiency():
    # one WR, 25% of a 40-pt receiving offense every week, scoring 3 over expected
    g = _games([(w, "a", "WR", "KC", 10.0, 40.0, 13.0) for w in (1, 2, 3)])
    pj = P.project(g, 4, ({}, {}, pd.DataFrame(columns=["gsis_id", "res_sum", "res_n"])),
                   {"h": 3, "k_s": 0, "k_t": 0, "k_eff": 3})
    r = pj.set_index("gsis_id").loc["a"]
    assert r["share_rec"] == pytest.approx(0.25)
    assert r["xfp_proj"] == pytest.approx(10.0)
    assert r["res_shrunk"] == pytest.approx(3.0 * 3 / (3 + 3))     # n/(n+k_eff) of the residual
    assert r["pts_pg"] == pytest.approx(11.5)


def test_future_weeks_are_never_read():
    g = _games([(1, "a", "WR", "KC", 10.0, 40.0, 10.0), (5, "a", "WR", "KC", 40.0, 40.0, 40.0)])
    pj = P.project(g, 4, ({}, {}, pd.DataFrame(columns=["gsis_id", "res_sum", "res_n"])),
                   {"h": 3, "k_s": 0, "k_t": 0, "k_eff": math.inf})
    assert pj.set_index("gsis_id").loc["a", "share_rec"] == pytest.approx(0.25)


def test_last_season_is_a_prior_not_a_replacement():
    prev = _games([(w, "a", "WR", "KC", 4.0, 40.0, 4.0) for w in range(1, 9)])    # 10% share
    cur = _games([(1, "a", "WR", "KC", 12.0, 40.0, 12.0)])                         # 30% share
    pj = P.project(cur, 2, P.priors(prev), {"h": 3, "k_s": 2, "k_t": 0, "k_eff": math.inf})
    s = pj.set_index("gsis_id").loc["a", "share_rec"]
    assert s == pytest.approx((0.30 * 1 + 0.10 * 2) / 3)


def test_ecr_is_mapped_onto_our_scale_by_rank():
    ours = pd.DataFrame({"pos": ["WR"] * 3, "pts_pg": [9.0, 15.0, 12.0]})
    ecr = pd.DataFrame({"pos": ["WR", "WR"], "ecr_pos_rank": [1, 3]})
    assert P.ecr_points(ours, ecr).tolist() == [15.0, 9.0]


def test_bands_and_tiers():
    assert [P.band(w) for w in (1, 6, 7, 10, 11, 17)] == ["3-6", "3-6", "7-10", "7-10", "11-14", "11-14"]
    edges = {"WR": [8.0, 12.0]}
    assert [P.tier(v, "WR", edges) for v in (5, 9, 13)] == ["low", "mid", "high"]


def test_the_fitted_file_carries_the_ship_rules():
    cfg = P.load_params()
    if "blend_w" not in cfg:
        pytest.skip("config/proj_ros_params.json not fitted")
    for pos in P.POS:
        assert set(cfg["blend_w"][pos]) == {"3-6", "7-10", "11-14"}
        assert all(0 <= v <= 1 for v in cfg["blend_w"][pos].values())
        assert set(cfg["disagreement_allowed"][pos]) == {"usage", "efficiency"}
        assert pos in cfg["schedule_lens"]


def _qb(p_active, bye=None):
    wk = {}
    for w, pa in p_active.items():
        wk[w] = ({"median": 0.0, "p10": 0.0, "p90": 0.0, "p_active": 0.0, "expected": 0.0, "bye": True}
                 if w == bye else
                 {"median": 18.0, "p10": 9.0, "p90": 28.0, "p_active": pa, "expected": round(20 * pa, 2),
                  "mean_if_active": 20.0})
    return {"pos": "QB", "weeks": wk, "drivers": []}


def test_a_backup_quarterback_plays_only_when_the_starter_does_not():
    # the starter is 90% to play, so his backup is on the field ~10% of weeks — not 100%
    players = {"starter": _qb({4: 0.9, 5: 0.9, 6: 0.9}), "backup": _qb({4: 1.0, 5: 1.0, 6: 1.0})}
    cuffs = pd.DataFrame([{"gsis_id": "backup", "cuff_of": "starter", "pos": "QB", "team": "CHI", "share": 1.0}])
    P._backup_qbs(players, cuffs)
    b = players["backup"]
    assert b["backup_of"] == "starter"
    assert b["weeks"][4]["p_active"] == pytest.approx(0.1)
    assert b["weeks"][4]["expected"] == pytest.approx(2.0)
    assert b["weeks"][4]["mean_if_active"] == 20.0          # per game played is untouched
    assert b["start_p"] == pytest.approx(0.1)
    assert "starter" in players and "backup_of" not in players["starter"]


def test_a_backup_starts_every_week_the_starter_is_out_and_skips_the_bye():
    players = {"s": _qb({4: 0.0, 5: 0.0, 6: 0.9}, bye=5), "b": _qb({4: 1.0, 5: 1.0, 6: 1.0}, bye=5)}
    cuffs = pd.DataFrame([{"gsis_id": "b", "cuff_of": "s", "pos": "QB", "team": "CHI", "share": 1.0}])
    P._backup_qbs(players, cuffs)
    assert players["b"]["weeks"][4]["expected"] == pytest.approx(20.0)
    assert players["b"]["weeks"][5].get("bye") and players["b"]["weeks"][5]["expected"] == 0.0
    assert players["b"]["start_p"] == pytest.approx((1.0 + 0.1) / 2)


def test_backup_rule_leaves_non_quarterbacks_and_chained_cuffs_alone():
    rb = {"pos": "RB", "weeks": {4: {"p_active": 1.0, "expected": 10.0, "mean_if_active": 10.0}}, "drivers": []}
    players = {"a": rb, "b": _qb({4: 1.0}), "c": _qb({4: 1.0})}
    cuffs = pd.DataFrame([
        {"gsis_id": "a", "cuff_of": "b", "pos": "RB", "team": "X", "share": 1.0},
        {"gsis_id": "b", "cuff_of": "c", "pos": "QB", "team": "X", "share": 1.0},
    ])
    P._backup_qbs(players, cuffs)
    assert rb["weeks"][4]["expected"] == 10.0 and "backup_of" not in rb
    assert players["b"]["backup_of"] == "c"
