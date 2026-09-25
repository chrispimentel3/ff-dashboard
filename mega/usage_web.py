"""Pure-data Usage trends: weekly per-player time series across five metrics.

Ships one combined dataset for the whole roster; the Streamlit tab's metric selector and
player multiselect become client-side filters over this one export, the same way the
Matchups Vegas table and Archetypes position filter already work — nothing here needs
recomputing per selection.
"""
from __future__ import annotations

import json

import pandas as pd

ROW_COLS = ["player", "week", "target_share", "half_ppr", "targets", "carries", "snap_share"]


def build(sw: pd.DataFrame | None, snaps: pd.DataFrame | None, skill: pd.DataFrame,
          name_by_id: dict, season: int) -> dict:
    """`sw`/`snaps`/`skill`/`name_by_id` are the same module-level frames every other tab
    reads — this just reshapes them into one long table instead of app.py's per-metric
    branch, since the metric picker doesn't change what data is needed, only which column
    gets plotted."""
    if skill is None or skill.empty:
        return {"available": False, "season": season, "players": [], "starters": [], "rows": []}

    gsis_list = skill["gsis_id"].dropna().tolist()
    if sw is not None and not sw.empty:
        base = sw[sw["gsis_id"].isin(gsis_list)].copy()
        base["player"] = base["gsis_id"].map(name_by_id)
        for col in ("target_share", "half_ppr", "targets", "carries"):
            if col not in base.columns:
                base[col] = None
        rows = base[["player", "week", "target_share", "half_ppr", "targets", "carries"]]
    else:
        rows = pd.DataFrame(columns=["player", "week", "target_share", "half_ppr", "targets", "carries"])

    snap_col = None
    if snaps is not None and not snaps.empty:
        for c in ("offense_pct", "off_pct"):
            if c in snaps.columns:
                snap_col = c
                break

    if snap_col:
        pfr_map = dict(zip(skill["pfr_id"], skill["name"]))
        s = snaps[snaps["pfr_id"].isin(pfr_map.keys())].copy()
        s["player"] = s["pfr_id"].map(pfr_map)
        s["snap_share"] = pd.to_numeric(s[snap_col], errors="coerce")
        s = s[["player", "week", "snap_share"]].dropna(subset=["player"])
        rows = rows.merge(s, on=["player", "week"], how="outer")
    else:
        rows["snap_share"] = None

    rows = rows.dropna(subset=["player"])
    starters = sorted(skill[skill["slot"] != "BN"]["name"].tolist())

    return {
        "available": not rows.empty,
        "season": season,
        "players": sorted(skill["name"].tolist()),
        "starters": starters,
        "rows": json.loads(rows.reindex(columns=ROW_COLS).sort_values(["player", "week"]).to_json(orient="records")),
    }
