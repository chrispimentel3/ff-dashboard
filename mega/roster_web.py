"""Pure-data My roster: full roster table + headline/KPIs + route-usage scatter data.

Headline/KPI derivation is real logic (moved out of app.py's `_tab_over`, mirroring
action_board.py); the route-usage tables are packaging only — `rz.totals()` already returns
plain DataFrames.
"""
from __future__ import annotations

import json

import pandas as pd

from mega.ui import short_name

ROSTER_COLS = [
    "slot", "player", "pos", "games", "half_ppr_pg", "roll_pg", "last_wk",
    "xfp_tot", "xfp_diff", "tgt_pg", "tgt_pct", "tm_rank", "carry_pg",
    "report_status", "opp", "implied",
]
ROUTE_COLS = ["slot", "player", "pos", "team", "fd_rr", "tprr", "tgt_pct",
              "targets", "routes", "routes_pg", "route_flag"]
SCATTER_COLS = ["player", "pos", "team", "tprr", "fd_rr", "routes", "qualified"]


def _records(df: pd.DataFrame | None, cols: list[str]) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.reindex(columns=[c for c in cols if c in df.columns]).to_json(orient="records"))


def build(agg: pd.DataFrame | None, roll: int) -> dict:
    """`agg` is app.py's roster aggregate (same one action_board.py reads)."""
    if agg is None or agg.empty:
        return {"available": False, "headline": "", "subhead": "", "kpis": None, "roster": []}

    starters = agg[agg["slot"] != "BN"]
    proj = float(starters["roll_pg"].sum()) if not starters.empty else 0.0
    best = starters.loc[starters["roll_pg"].idxmax()] if not starters.empty else None
    worst = starters.loc[starters["roll_pg"].idxmin()] if not starters.empty else None
    xdelta = float(starters["xfp_diff"].sum()) if "xfp_diff" in starters.columns and not starters.empty else None
    n_hurt = (
        int(starters["report_status"].isin(["Out", "Doubtful", "Questionable", "IR"]).sum())
        if "report_status" in starters.columns else 0
    )

    hot = starters.nlargest(1, "xfp_diff") if "xfp_diff" in starters.columns else None
    cold = starters.nsmallest(1, "xfp_diff") if "xfp_diff" in starters.columns else None
    headline = f"Your starters are averaging {proj:.0f} points a week."
    if hot is not None and not hot.empty and float(hot.iloc[0]["xfp_diff"]) > 2:
        headline = (f"{short_name(hot.iloc[0]['player'])} is scoring "
                    f"{float(hot.iloc[0]['xfp_diff']):+.0f} points above what his usage earns "
                    "— the one to shop.")
    subhead = (
        f"{short_name(cold.iloc[0]['player'])} is the other way round at "
        f"{float(cold.iloc[0]['xfp_diff']):+.0f}: hold him, the work is there."
        if cold is not None and not cold.empty and float(cold.iloc[0]["xfp_diff"]) < -1.5 else
        "Red is scoring above the work, navy is below it."
    )

    return {
        "available": True,
        "headline": headline,
        "subhead": subhead,
        "kpis": {
            "roll": roll,
            "starters_proj": round(proj, 0),
            "roster_act_minus_xfp": round(xdelta, 0) if xdelta is not None else None,
            "top_starter": short_name(best["player"]) if best is not None else None,
            "top_starter_ppg": round(float(best["roll_pg"]), 1) if best is not None else None,
            "coldest_starter": short_name(worst["player"]) if worst is not None else None,
            "coldest_starter_ppg": round(float(worst["roll_pg"]), 1) if worst is not None else None,
            "injury_flags": n_hurt,
        },
        "roster": _records(agg, ROSTER_COLS),
    }


def build_routes(rt: pd.DataFrame | None, pool: pd.DataFrame | None, gsis_list: set,
                  threshold: float, min_routes: int) -> dict:
    """`rt` is app.py's `_rt` (my roster's route table); `pool` is `rz.totals()` over every
    qualified WR/TE in the league, used for the scatter (mine vs everyone else)."""
    if rt is None or rt.empty:
        return {"available": False, "table": [], "scatter": [], "threshold": threshold,
                "min_routes": min_routes}

    scatter = pd.DataFrame()
    if pool is not None and not pool.empty:
        qualified = pool[pool["qualified"] & pool["fd_rr"].notna()].copy()
        if len(qualified) >= 8:
            qualified["mine"] = qualified["gsis_id"].isin(gsis_list)
            scatter = qualified

    return {
        "available": True,
        "table": _records(rt, ROUTE_COLS),
        "scatter": _records(scatter, SCATTER_COLS + ["mine"]),
        "threshold": threshold,
        "min_routes": min_routes,
    }
