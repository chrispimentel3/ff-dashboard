"""Weekly rankings: every skill-position player ranked within his own position for the
upcoming week, by projected half-PPR points — league-wide, not just the roster
`mega/start_sit.py` scopes to.

Streamlit-free (like `logic_web.py`/`headshots_web.py`) — `blended_week()`,
`player_index()` and `week_matchups()` are all plain functions already, so this needs
no `st.cache_data` context, just calling them directly at export time.
"""
from __future__ import annotations

import pandas as pd

from . import data as _data
from .lookup import player_index
from .matchup_model import week_matchups
from .projections import blended_week

SKILL = ("QB", "RB", "WR", "TE")


def _shape(proj: pd.DataFrame, team_by_gsis: pd.DataFrame, matchups: pd.DataFrame,
           gsis_list: set[str]) -> pd.DataFrame:
    """Pure merge + rank step, split out from `build()` for testing on synthetic frames
    with no network fetch — the same split `mega/matchup_model.py` uses.

    `proj` is `blended_week()`-shaped, `team_by_gsis` is a `gsis_id, team` frame,
    `matchups` is `week_matchups()`-shaped."""
    df = proj.merge(team_by_gsis.drop_duplicates("gsis_id"), on="gsis_id", how="left")
    df = df[df["pos"].isin(SKILL)].dropna(subset=["proj"]).copy()
    # nflverse_estimate's own "basis" (e.g. "2026 + 2025 prior") and FantasyPros' sparse
    # "opponent" would otherwise collide with the matchup-model columns of the same name
    # below — drop them first rather than let a merge suffix silently rename one aside.
    df = df.drop(columns=["basis", "opponent"], errors="ignore")

    m = matchups[["team", "pos", "opp", "pct", "basis"]] if matchups is not None and not matchups.empty else None
    if m is not None and not m.empty:
        df = df.merge(m, on=["team", "pos"], how="left")
    else:
        df["opp"], df["pct"], df["basis"] = None, None, None

    df["rank"] = df.groupby("pos")["proj"].rank(ascending=False, method="min").astype(int)
    df["mine"] = df["gsis_id"].isin(gsis_list)
    return df.sort_values(["pos", "rank"])


def build(season: int, next_week: int, gsis_list: set[str] | None = None) -> dict:
    gsis_list = gsis_list or set()

    proj = blended_week(season, next_week)
    if proj.empty:
        return {"available": False, "season": season, "next_week": next_week, "rows": []}

    idx = player_index(
        {season: _data.load_player_stats(season), season - 1: _data.load_player_stats(season - 1)},
        _data.load_rosters(season),
    )
    try:
        m = week_matchups(season, next_week)
        m = m[m["week"] == next_week] if not m.empty else m
    except Exception:
        m = pd.DataFrame()

    df = _shape(proj, idx[["gsis_id", "team"]], m, gsis_list)

    rows = [
        {
            "gsis_id": r["gsis_id"],
            "player": r["player"],
            "pos": r["pos"],
            "team": r["team"] if pd.notna(r["team"]) else None,
            "rank": int(r["rank"]),
            "proj": round(float(r["proj"]), 1),
            "proj_source": r["proj_source"],
            "opponent": r["opp"] if pd.notna(r.get("opp")) else None,
            "matchup_pct": round(float(r["pct"]), 1) if pd.notna(r.get("pct")) else None,
            "matchup_basis": r.get("basis") if pd.notna(r.get("basis")) else None,
            "mine": bool(r["mine"]),
        }
        for _, r in df.iterrows()
    ]
    return {"available": True, "season": season, "next_week": next_week, "rows": rows}
