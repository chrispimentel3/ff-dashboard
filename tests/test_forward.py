"""§13 forward opportunity and §14 earned adjustment.

Carries the spec's hand-checkable cases: §8 #1 and #2, §19.3 #15, #16 and #21.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mega import forward as fw


# ---------------------------------------------------------------- §8 #1 blend
@pytest.mark.parametrize("games,expect", [(0, 12.0), (1, 12.533), (2, 13.067), (3, 13.600),
                                          (8, 13.600)])
def test_usage_weight_ramps_in_by_the_third_game(games, expect):
    """proj 12, xfp 16 → g=1: 12.533; g=3: 13.600, and it stops climbing after that."""
    got = fw.blend(pd.Series([12.0]), pd.Series([16.0]), pd.Series([games])).iloc[0]
    assert got == pytest.approx(expect, abs=1e-3)


def test_blend_falls_back_when_one_side_is_missing():
    assert fw.blend(pd.Series([np.nan]), pd.Series([16.0]), pd.Series([3])).iloc[0] == pytest.approx(16.0)
    assert fw.blend(pd.Series([12.0]), pd.Series([np.nan]), pd.Series([3])).iloc[0] == pytest.approx(12.0)


# ---------------------------------------------------------------- §8 #2 half-PPR xFP
def test_xfp_is_recomputed_in_half_ppr():
    """rec_yards_exp 60, receptions_exp 5, rec_td_exp 0.4 → 6.0 + 2.5 + 2.4 = 10.9.

    ff_opportunity's own total_fantasy_points_exp is not half-PPR, which is why the
    component columns are re-scored rather than read."""
    ffo = pd.DataFrame([{"rec_yards_gained_exp": 60.0, "receptions_exp": 5.0,
                         "rec_touchdown_exp": 0.4}])
    assert fw.xfp_components(ffo)["xfp_rec"].iloc[0] == pytest.approx(10.9)


def test_passing_xfp_uses_the_league_passing_weights():
    ffo = pd.DataFrame([{"pass_yards_gained_exp": 300.0, "pass_touchdown_exp": 2.0,
                         "pass_interception_exp": 1.0}])
    # 0.04*300 + 4*2 - 1*1 = 12 + 8 - 1
    assert fw.xfp_components(ffo)["xfp_pass"].iloc[0] == pytest.approx(19.0)


# ---------------------------------------------------------------- §19.3 #15
def test_implied_totals_and_the_sign_of_the_spread():
    """total 47, spread_line +3 → home 25.0, away 22.0. A positive spread_line means the
    HOME team is favoured; getting it backwards inverts every game-script adjustment."""
    sched = pd.DataFrame([{"game_type": "REG", "week": 1, "home_team": "LA",
                           "away_team": "SF", "total_line": 47.0, "spread_line": 3.0}])
    imp = fw.implied(sched).set_index("team")
    assert imp.loc["LA", "implied"] == pytest.approx(25.0)
    assert imp.loc["SF", "implied"] == pytest.approx(22.0)
    assert imp.loc["LA", "spread"] == pytest.approx(3.0)     # favoured
    assert imp.loc["SF", "spread"] == pytest.approx(-3.0)


# ---------------------------------------------------------------- §19.3 #16
def _ffo_team(team="LA", weeks=(1, 2), rec_pg=40.0) -> pd.DataFrame:
    """A team whose receiving xFP per game is exactly `rec_pg`."""
    rows = []
    for w in weeks:
        rows.append({"posteam": team, "week": w, "player_id": f"p{w}",
                     "rec_yards_gained_exp_team": rec_pg * 10, "receptions_exp_team": 0.0,
                     "rec_touchdown_exp_team": 0.0, "rec_two_point_conv_exp_team": 0.0})
    return pd.DataFrame(rows)


def test_forward_volume_scales_with_the_implied_total():
    """team rec xFP 40/g, implied 25 against an average of 22.5, γ 0.6, spread 0 → 42.61."""
    ffo = _ffo_team()
    # two posted lines, implied 20 and 25, so the team average is exactly 22.5. A third
    # week would land in that average too and move it off the spec's arithmetic.
    sched = pd.DataFrame([
        {"game_type": "REG", "week": 1, "home_team": "LA", "away_team": "SF",
         "total_line": 40.0, "spread_line": 0.0},          # implied 20
        {"game_type": "REG", "week": 2, "home_team": "LA", "away_team": "SF",
         "total_line": 50.0, "spread_line": 0.0},          # implied 25
    ])
    vol = fw.team_volume(ffo, sched, [2])
    la = vol[vol["team"] == "LA"].iloc[0]
    assert la["xfp_rec"] == pytest.approx(42.61, abs=0.01)


def test_a_week_with_no_posted_line_falls_back_to_the_team_average():
    """Vegas prices a week or two ahead. A missing line must not read as a low-scoring
    game — the ratio has to be 1 and the game-script term 0."""
    ffo = _ffo_team()
    sched = pd.DataFrame([
        {"game_type": "REG", "week": 1, "home_team": "LA", "away_team": "SF",
         "total_line": 45.0, "spread_line": 0.0},
        {"game_type": "REG", "week": 14, "home_team": "LA", "away_team": "SF",
         "total_line": None, "spread_line": None},
    ])
    vol = fw.team_volume(ffo, sched, [14]).set_index("team")
    assert vol.loc["LA", "implied_ratio"] == pytest.approx(1.0)
    assert vol.loc["LA", "xfp_rec"] == pytest.approx(40.0)


def test_game_script_tilts_run_and_pass_in_opposite_directions():
    """Favoured teams run more (β_rush positive) and throw less (β_rec negative)."""
    ffo = pd.DataFrame([{"posteam": "LA", "week": 1, "player_id": "p",
                         "rush_yards_gained_exp_team": 1000.0,
                         "rec_yards_gained_exp_team": 1000.0}])
    flat = pd.DataFrame([{"game_type": "REG", "week": 1, "home_team": "LA",
                          "away_team": "SF", "total_line": 44.0, "spread_line": 0.0},
                         {"game_type": "REG", "week": 2, "home_team": "LA",
                          "away_team": "SF", "total_line": 44.0, "spread_line": 14.0}])
    # positive spread_line = the HOME team is favoured, so LA is the 14-point favourite
    v = fw.team_volume(ffo, flat, [2]).set_index("team")
    assert v.loc["LA", "spread"] == pytest.approx(14.0)
    base_rush = 100.0 * (v.loc["LA", "implied_ratio"] ** fw.GAMMA)
    assert v.loc["LA", "xfp_rush"] > base_rush              # runs more
    base_rec = 100.0 * (v.loc["LA", "implied_ratio"] ** fw.GAMMA)
    assert v.loc["LA", "xfp_rec"] < base_rec                # throws less


# ---------------------------------------------------------------- §19.3 #21
@pytest.mark.parametrize("z,expect", [(3.0, 1.06), (-3.0, 0.94), (1.5, 1.06),
                                      (0.0, 1.00), (0.5, 1.02)])
def test_earned_adjustment_clamps_at_six_percent(z, expect):
    tab = pd.DataFrame([{"gsis_id": "A", "pos": "WR", "tprr_z": z, "fd_rr_z": z,
                         "wopr_z": z, "separation_z": z}])
    assert fw.earned(tab)["earn_adj"].iloc[0] == pytest.approx(expect, abs=1e-9)


def test_nulls_are_skipped_rather_than_scored_as_average():
    """A missing NGS row means nobody measured him. Treating it as z=0 would drag every
    unmeasured player toward the middle (§12.2, §19.3 #24)."""
    tab = pd.DataFrame([{"gsis_id": "A", "pos": "WR", "tprr_z": 2.0, "fd_rr_z": 2.0,
                         "wopr_z": 2.0, "separation_z": np.nan}])
    out = fw.earned(tab).iloc[0]
    assert out["z_earn"] == pytest.approx(2.0)      # not 1.5
    assert out["z_parts"] == 3


def test_a_player_with_nothing_measured_gets_a_neutral_adjustment():
    tab = pd.DataFrame([{"gsis_id": "A", "pos": "WR", "tprr_z": np.nan, "fd_rr_z": np.nan,
                         "wopr_z": np.nan, "separation_z": np.nan}])
    out = fw.earned(tab).iloc[0]
    assert pd.isna(out["z_earn"]) and out["earn_adj"] == pytest.approx(1.0)


def test_opportunity_score_is_100_at_the_role_average():
    tab = pd.DataFrame([{"gsis_id": "A", "pos": "RB", "target_share_z": 0.0,
                         "fd_per_touch_z": 0.0, "ryoe_z": 0.0, "snap_pct_z": 0.0},
                        {"gsis_id": "B", "pos": "RB", "target_share_z": 1.0,
                         "fd_per_touch_z": 1.0, "ryoe_z": 1.0, "snap_pct_z": 1.0}])
    out = fw.earned(tab).set_index("gsis_id")["opp_score"]
    assert out["A"] == pytest.approx(100.0)
    assert out["B"] == pytest.approx(115.0)


def test_each_position_reads_its_own_earning_metrics():
    assert "separation_z" in fw.EARN_Z["WR"]
    assert "ryoe_z" in fw.EARN_Z["RB"] and "separation_z" not in fw.EARN_Z["RB"]
    assert fw.EARN_Z["QB"] == ("carry_share_z", "inside10_share_z")


# ---------------------------------------------------------------- shares
def test_a_player_who_changed_teams_uses_only_his_current_offence():
    """Averaging two offences into one share produces a number describing neither."""
    ffo = pd.DataFrame([
        {"player_id": "A", "posteam": "LA", "week": 1, "rec_yards_gained_exp": 100.0,
         "rec_yards_gained_exp_team": 1000.0},
        {"player_id": "A", "posteam": "SF", "week": 2, "rec_yards_gained_exp": 400.0,
         "rec_yards_gained_exp_team": 1000.0},
    ])
    pw = pd.DataFrame([{"gsis_id": "A", "week": 1, "team": "LA"},
                       {"gsis_id": "A", "week": 2, "team": "SF"}])
    sh = fw.player_shares(None, pw, ffo).set_index("gsis_id")
    assert sh.loc["A", "share_rec"] == pytest.approx(0.40)   # SF only, not 0.25
    assert sh.loc["A", "team"] == "SF"
