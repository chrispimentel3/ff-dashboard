"""mega/rankings_web.py's pure merge+rank step. Exercises the exact column-collision bug
found while building this (blended_week's own "basis"/"opponent" columns silently
shadowing the matchup-model's columns of the same name in a merge)."""
from __future__ import annotations

import pandas as pd

from mega import rankings_web as rw


def _proj(rows):
    cols = ["gsis_id", "player", "pos", "proj", "proj_source", "basis", "opponent"]
    return pd.DataFrame(rows, columns=cols)


def test_ranks_within_position_by_projection():
    proj = _proj([
        ("A", "Big WR", "WR", 20.0, "ffanalytics", "2026 only", None),
        ("B", "Small WR", "WR", 10.0, "ffanalytics", "2026 only", None),
        ("C", "Top RB", "RB", 15.0, "ffanalytics", "2026 only", None),
    ])
    teams = pd.DataFrame({"gsis_id": ["A", "B", "C"], "team": ["AAA", "BBB", "CCC"]})
    out = rw._shape(proj, teams, pd.DataFrame(), set())
    assert out.set_index("gsis_id").loc["A", "rank"] == 1
    assert out.set_index("gsis_id").loc["B", "rank"] == 2
    assert out.set_index("gsis_id").loc["C", "rank"] == 1   # separate position, own rank-1


def test_matchup_columns_are_not_shadowed_by_projections_own_basis_and_opponent():
    """The real bug: blended_week() already has "basis" (nflverse_estimate's provenance,
    e.g. "2026 + 2025 prior") and "opponent" (FantasyPros', sparse) columns. A naive merge
    of the matchup-model frame (which also has "basis" and needs to supply "opp") must not
    let either of those silently win over the matchup model's own values."""
    proj = _proj([("A", "Player", "WR", 12.0, "ffanalytics", "2026 + 2025 prior", "ZZZ")])
    teams = pd.DataFrame({"gsis_id": ["A"], "team": ["AAA"]})
    matchups = pd.DataFrame([{
        "team": "AAA", "pos": "WR", "opp": "BBB", "pct": 7.3, "basis": "defense+vegas",
    }])
    out = rw._shape(proj, teams, matchups, set())
    row = out.iloc[0]
    assert row["basis"] == "defense+vegas"     # not "2026 + 2025 prior"
    assert row["opp"] == "BBB"                 # not "ZZZ"
    assert row["pct"] == 7.3


def test_no_matchup_data_leaves_matchup_fields_null_not_missing():
    proj = _proj([("A", "Player", "WR", 12.0, "ffanalytics", "2026 only", None)])
    teams = pd.DataFrame({"gsis_id": ["A"], "team": ["AAA"]})
    out = rw._shape(proj, teams, pd.DataFrame(), set())
    row = out.iloc[0]
    assert row["opp"] is None
    assert pd.isna(row["pct"])
    assert row["basis"] is None


def test_mine_flag_reflects_the_gsis_list():
    proj = _proj([("A", "Mine", "WR", 12.0, "ffanalytics", "2026 only", None),
                  ("B", "Not mine", "WR", 11.0, "ffanalytics", "2026 only", None)])
    teams = pd.DataFrame({"gsis_id": ["A", "B"], "team": ["AAA", "BBB"]})
    out = rw._shape(proj, teams, pd.DataFrame(), {"A"}).set_index("gsis_id")
    assert bool(out.loc["A", "mine"]) is True
    assert bool(out.loc["B", "mine"]) is False
