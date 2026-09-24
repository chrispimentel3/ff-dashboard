"""§12 role context. Includes the spec's hand-checkable cases (HANDOFF §19.3 #13, #14,
#20, #24 and §8 #11), so a number that drifts fails here rather than on the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mega import roles


def _pw(rows, cols) -> pd.DataFrame:
    d = pd.DataFrame(rows, columns=cols)
    for c in ("targets", "team_targets", "carries", "team_carries", "offense_snaps",
              "team_snaps", "receiving_air_yards", "team_air_yards", "receptions",
              "receiving_first_downs", "rushing_first_downs", "routes", "attempts",
              "i10_carries", "team_i10_carries", "half_ppr_exp", "team_xfp"):
        if c not in d.columns:
            d[c] = 0.0
    return d


# ---------------------------------------------------------------- §19.3 #13
def test_baseline_is_summed_over_summed_not_an_average_of_ratios():
    """Two WR1 games, 10 of 40 team targets and 2 of 20 → 20.0%, not 17.5%."""
    pw = _pw([("A", "WR", "LA", 1, 10, 40), ("A", "WR", "LA", 2, 2, 20)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets"])
    tot = roles.totals(pw, pd.DataFrame(
        [{"gsis_id": "A", "pos": "WR", "team": "LA", "role": "WR1", "role_src": "usage"}]))
    base = roles.baselines(tot)
    assert base["role"]["target_share"]["WR1"] == pytest.approx(0.20)
    assert base["role"]["target_share"]["WR1"] != pytest.approx(0.175)


# ---------------------------------------------------------------- §19.3 #14
def test_shrinkage_pulls_toward_the_role_baseline():
    """12 targets of 30 team targets, μ_role 0.20, k 60 → (12 + 12) ÷ 90 = 0.2667."""
    tot = pd.DataFrame([{"gsis_id": "A", "pos": "WR", "team": "LA", "role": "WR1",
                         "role_src": "usage", "games": 1,
                         "targets": 12.0, "team_targets": 30.0}])
    base = {"role": {"target_share": {"WR1": 0.20}}, "pos": {"target_share": {"WR": 0.20}},
            "spread": {}, "n": {}}
    out = roles.index(tot, base)
    assert out["target_share"].iloc[0] == pytest.approx(24 / 90)
    assert out["target_share"].iloc[0] == pytest.approx(0.26667, abs=1e-5)
    assert out["target_share_raw"].iloc[0] == pytest.approx(0.40)   # unshrunk


def test_index_is_100_at_the_role_baseline_and_at_the_nfl_average():
    tot = pd.DataFrame([{"gsis_id": "A", "pos": "WR", "team": "LA", "role": "WR2",
                         "role_src": "usage", "games": 4,
                         "targets": 0.18 * 1000, "team_targets": 1000.0}])
    base = {"role": {"target_share": {"WR2": 0.18}}, "pos": {"target_share": {"WR": 0.12}},
            "spread": {}, "n": {}}
    out = roles.index(tot, base)
    assert out["target_share_idx"].iloc[0] == pytest.approx(100.0)   # average for a WR2
    assert out["target_share_idxp"].iloc[0] == pytest.approx(150.0)  # 1.5x the NFL WR


# ---------------------------------------------------------------- §19.3 #20
@pytest.mark.parametrize("carries,targets,expect", [
    (55, 0, "LEAD"),          # carry share 55%
    (35, 0, "COMMITTEE"),     # 35%
    (20, 10, "RECEIVING"),    # carry share 20%, target share 10%
    (5, 1, "BACKUP"),
])
def test_rb_role_assignment(carries, targets, expect):
    pw = _pw([("A", "RB", "LA", 1, carries, 100, targets, 100)],
             ["gsis_id", "pos", "team", "week", "carries", "team_carries",
              "targets", "team_targets"])
    assert roles.assign(pw, min_games=1)["role"].iloc[0] == expect


def test_wr_roles_rank_within_the_team():
    pw = _pw([("A", "WR", "LA", 1, 30, 100, 60, 60), ("B", "WR", "LA", 1, 20, 100, 60, 60),
              ("C", "WR", "LA", 1, 10, 100, 40, 60), ("D", "WR", "LA", 1, 5, 100, 10, 60)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets",
              "offense_snaps", "team_snaps"])
    got = dict(zip(roles.assign(pw, min_games=1)["gsis_id"], roles.assign(pw, min_games=1)["role"]))
    assert got == {"A": "WR1", "B": "WR2", "C": "WR3", "D": "WR4+"}


def test_third_wr_below_half_the_snaps_is_not_a_wr3():
    """§12.1 makes WR3 conditional on snap % ≥ 50 — a rotational body is not a WR3."""
    pw = _pw([("A", "WR", "LA", 1, 30, 100, 60, 60), ("B", "WR", "LA", 1, 20, 100, 60, 60),
              ("C", "WR", "LA", 1, 10, 100, 20, 60)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets",
              "offense_snaps", "team_snaps"])
    assert roles.assign(pw, min_games=1).set_index("gsis_id")["role"]["C"] == "WR4+"


def test_tight_ends_split_on_whether_they_are_targeted():
    pw = _pw([("A", "TE", "LA", 1, 20, 100, 50, 60), ("B", "TE", "SF", 1, 5, 100, 50, 60)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets",
              "offense_snaps", "team_snaps"])
    got = roles.assign(pw, min_games=1).set_index("gsis_id")["role"]
    assert got["A"] == "TE1-REC"      # 20% target share
    assert got["B"] == "TE1-BLK"      # 5%, below the 12% line


def test_thin_sample_falls_back_to_the_depth_chart():
    """One game is not enough to call someone a WR1 (§12.1)."""
    pw = _pw([("A", "WR", "LA", 1, 30, 100, 60, 60)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets",
              "offense_snaps", "team_snaps"])
    depth = pd.DataFrame([{"dt": "2026-09-23", "gsis_id": "A", "pos_abb": "WR", "pos_rank": 3}])
    out = roles.assign(pw, depth=depth, min_games=2).iloc[0]
    assert out["role"] == "WR3" and out["role_src"] == "depth chart"


# ---------------------------------------------------------------- §19.3 #24
def test_missing_nextgen_stays_null_and_never_becomes_zero():
    """NGS only publishes qualifying players. A zero would read as 'separates badly'
    when the truth is 'was never measured'."""
    pw = _pw([("A", "WR", "LA", 1, 10, 50), ("B", "WR", "LA", 2, 8, 50)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets"])
    ngs = pd.DataFrame([{"player_gsis_id": "A", "week": 1, "avg_separation": 3.0}])
    fr = roles.frame(pw, None, ngs, None)
    assert fr.loc[fr["gsis_id"] == "B", "separation_num"].isna().all()
    out = roles.build(pw, None, ngs, None)["table"].set_index("gsis_id")
    assert pd.isna(out.loc["B", "separation"])
    assert not (out["separation"].fillna(-1) == 0).any()


def test_a_measured_nextgen_rate_is_weighted_by_volume():
    """3.0 separation on 10 targets and 1.0 on 30 is 1.5, not 2.0."""
    pw = _pw([("A", "WR", "LA", 1, 10, 50), ("A", "WR", "LA", 2, 30, 50)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets"])
    ngs = pd.DataFrame([{"player_gsis_id": "A", "week": 1, "avg_separation": 3.0},
                        {"player_gsis_id": "A", "week": 2, "avg_separation": 1.0}])
    fr = roles.frame(pw, None, ngs, None)
    tot = roles.totals(fr, roles.assign(fr, min_games=1))
    assert tot["separation_num"].iloc[0] / tot["targets"].iloc[0] == pytest.approx(1.5)


def test_nextgen_week_zero_season_totals_are_ignored():
    pw = _pw([("A", "WR", "LA", 1, 10, 50)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets"])
    ngs = pd.DataFrame([{"player_gsis_id": "A", "week": 0, "avg_separation": 99.0},
                        {"player_gsis_id": "A", "week": 1, "avg_separation": 3.0}])
    fr = roles.frame(pw, None, ngs, None)
    assert fr["separation_num"].iloc[0] == pytest.approx(30.0)   # 3.0 x 10 targets


# ---------------------------------------------------------------- §12.5 flags
def _ladder_pw() -> pd.DataFrame:
    """A four-deep WR room over two games, so baselines and neighbours both exist."""
    rows = []
    for wk in (1, 2):
        for gid, tgt, snaps in (("A", 30, 60), ("B", 20, 60), ("C", 10, 40), ("D", 4, 20)):
            rows.append((gid, "WR", "LA", wk, tgt, 100, snaps, 60))
        for gid, tgt, snaps in (("E", 28, 58), ("F", 18, 58), ("G", 9, 38), ("H", 3, 18)):
            rows.append((gid, "WR", "SF", wk, tgt, 100, snaps, 60))
    return _pw(rows, ["gsis_id", "pos", "team", "week", "targets", "team_targets",
                      "offense_snaps", "team_snaps"])


def test_role_plus_fires_when_a_player_beats_the_rung_above_him():
    """ROLE+ is a LEAGUE comparison, not a team one. Handing the WR3 more targets than his
    own WR2 just makes him the WR2 — the flag exists for the WR3 on a concentrated offence
    who out-produces the average WR2 elsewhere, while still being third on his own team.
    """
    rows = []
    for wk in (1, 2):
        # LA throws to everyone; SF is top-heavy, so the league WR2 baseline sits low
        for gid, tgt in (("A", 30), ("B", 26), ("C", 24), ("D", 4)):
            rows.append((gid, "WR", "LA", wk, tgt, 100, 60, 60))
        for gid, tgt in (("E", 34), ("F", 10), ("G", 6), ("H", 3)):
            rows.append((gid, "WR", "SF", wk, tgt, 100, 60, 60))
    pw = _pw(rows, ["gsis_id", "pos", "team", "week", "targets", "team_targets",
                    "offense_snaps", "team_snaps"])
    out = roles.build(pw)["table"].set_index("gsis_id")
    assert out.loc["C", "role"] == "WR3"          # still third on his own team
    assert "ROLE+" in (out.loc["C", "flags"] or [])


def test_a_flag_needs_the_flat_floor_as_well_as_the_role_index():
    """A WR4 at 1.2x a tiny WR4 baseline has still done nothing — §12.5 keeps the old
    flat numbers as a floor precisely so he does not flag."""
    pw = _ladder_pw()
    pw.loc[pw["gsis_id"] == "D", "targets"] = 6      # above the WR4+ baseline, still 6%
    out = roles.build(pw)["table"].set_index("gsis_id")
    assert out.loc["D", "target_share_idx"] > roles.ROLE_FLAG_IDX
    assert "TGT" not in (out.loc["D", "flags"] or [])


def test_wopr_has_a_baseline_so_wr_role_flags_can_fire():
    """WOPR is the metric that ranks WRs; without a baseline every WR ROLE+/ROLE− test
    compared against None and silently did nothing."""
    base = roles.build(_ladder_pw())["baselines"]
    assert "wopr" in base["role"] and "wopr" in base["pos"]
    assert base["role"]["wopr"]["WR1"] > base["role"]["wopr"]["WR2"]


# ---------------------------------------------------------------- §8 #11 persistence
def test_persistence_tags():
    """Flag in 1 of 1 games → spike. In 2 of 3 → sustained."""
    one = _pw([("A", "WR", "LA", 1, 30, 100)],
              ["gsis_id", "pos", "team", "week", "targets", "team_targets"])
    tags = roles.persistence(one).set_index("gsis_id")["tags"]["A"]
    assert tags.get("TGT") == "spike"

    three = _pw([("A", "WR", "LA", 1, 30, 100), ("A", "WR", "LA", 2, 5, 100),
                 ("A", "WR", "LA", 3, 30, 100)],
                ["gsis_id", "pos", "team", "week", "targets", "team_targets"])
    tags = roles.persistence(three).set_index("gsis_id")["tags"]["A"]
    assert tags.get("TGT") == "sustained"


def test_a_single_big_game_inside_a_three_game_window_is_only_a_spike():
    pw = _pw([("A", "WR", "LA", 1, 5, 100), ("A", "WR", "LA", 2, 5, 100),
              ("A", "WR", "LA", 3, 40, 100)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets"])
    assert roles.persistence(pw).set_index("gsis_id")["tags"]["A"].get("TGT") == "spike"


# ---------------------------------------------------------------- baseline table
def test_baseline_table_is_ordered_by_the_role_ladder():
    base = roles.build(_ladder_pw())["baselines"]
    bt = roles.baseline_table(base, "role")
    order = bt["role"].tolist()
    assert order.index("WR1") < order.index("WR2") < order.index("WR4+")
    assert (bt["players"] > 0).all()


def test_nfl_positional_average_is_reported_alongside_the_role():
    base = roles.build(_ladder_pw())["baselines"]
    assert not roles.baseline_table(base, "pos").empty
    wr = base["pos"]["target_share"]["WR"]
    assert base["role"]["target_share"]["WR1"] > wr > base["role"]["target_share"]["WR4+"]


def test_window_counts_games_played_not_weeks_elapsed():
    """A bye or a missed week must not shorten the window — §12.2 says games played."""
    pw = _pw([("A", "WR", "LA", w, 10, 100) for w in (1, 2, 7, 8, 9)],
             ["gsis_id", "pos", "team", "week", "targets", "team_targets"])
    tot = roles.totals(pw, roles.assign(pw, min_games=1), window=4)
    assert tot["games"].iloc[0] == 4
    assert tot["targets"].iloc[0] == 40
