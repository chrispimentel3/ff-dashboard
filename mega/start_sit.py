"""Pure-data Start/Sit: recommended starters, bench, and the headline call.

Extracted out of app.py's `_tab_start` for the same reason as `mega/action_board.py` —
one source of truth for the Streamlit tab and the mega-bowl-web export. This module does
NOT run `optimize_lineup()` itself; it takes the already-computed lineup DataFrame (app.py
still owns building the roster/projection/Vegas pipeline that feeds it, same as it always
did) and only shapes the final payload, so nothing about how a lineup gets optimized lives
in two places.
"""
from __future__ import annotations

import json

import pandas as pd

from mega.ui import short_name

COLS = [
    "lineup", "player", "pos", "nfl_team", "report_status", "opp", "ease_rank",
    "proj", "proj_adj", "proj_source", "vegas", "vegas_edge",
    "tgt_pct", "tm_rank", "start_sit", "close_call",
    # the matchup verdict the player cards show: the fitted model's % on a normal week,
    # what that's worth in points, and its two parts (defense rank, Vegas team line)
    "matchup_pct", "matchup_pts", "matchup_def_rank", "matchup_vegas_pct", "matchup_basis",
]
SLOT_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLEX": 4, "K": 5, "DEF": 6}


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def build(lu: pd.DataFrame | None, ts: pd.DataFrame, next_week: int) -> dict:
    """Full Start/Sit payload as JSON-safe plain Python — no Streamlit calls.

    `lu` is `mega.lineup.optimize_lineup()`'s output; `ts` is `target_share()`'s output,
    merged in as the volume tiebreaker for close calls.
    """
    if lu is None or lu.empty:
        return {"available": False, "headline": "", "subhead": "", "kpis": None,
                "starters": [], "bench": [], "next_week": next_week}

    lu = lu.merge(ts, on="gsis_id", how="left") if "gsis_id" in lu.columns else lu.copy()
    for c in ("tgt_pct", "tm_rank"):
        if c not in lu.columns:
            lu[c] = None

    if "pct_fitted" in lu.columns:
        lu["matchup_pct"] = pd.to_numeric(lu["pct_fitted"], errors="coerce").round(1)
        mult = pd.to_numeric(lu.get("mult_fitted"), errors="coerce")
        # on the same baseline as the player card (nflverse's estimate of a normal week),
        # so one player reads the same number on both pages
        base = pd.to_numeric(lu["nfl_est"], errors="coerce") if "nfl_est" in lu.columns else pd.Series(float("nan"), index=lu.index)
        base = base.fillna(pd.to_numeric(lu["proj"], errors="coerce"))
        lu["matchup_pts"] = (base * (mult - 1)).round(1)

    starters = lu[lu["start"]]
    bench = lu[~lu["start"]]
    proj_total = float(starters["proj_adj"].sum())
    n_close = int((bench["close_call"] != "").sum())
    fp_cov = int((starters["proj_source"] == "FantasyPros").sum())
    moved = [r for _, r in lu.iterrows()
             if str(r.get("lineup", "")) != "BENCH" and str(r.get("slot", "")) == "BN"]
    est = int((starters["proj_source"] != "FantasyPros").sum())

    headline = (
        f"Your best legal lineup projects {proj_total:.0f} points in week {next_week}."
        if not moved else
        f"Start {', '.join(short_name(r['player']) for r in moved[:2])} — that is "
        f"worth {proj_total:.0f} points in week {next_week}, more than your current nine."
    )
    subhead = (
        f"{n_close} call{'s' if n_close != 1 else ''} within two points"
        + (f", and {est} of {len(starters)} starters run on estimates rather than real "
           "projections — break those on target share." if est else ".")
    )

    cols = [c for c in COLS if c in lu.columns]
    sview = starters.assign(_o=starters["lineup"].map(SLOT_ORDER)).sort_values("_o")[cols]
    bview = bench.sort_values("proj_adj", ascending=False)[cols]

    return {
        "available": True,
        "headline": headline,
        "subhead": subhead,
        "kpis": {
            "proj_total": round(proj_total, 1),
            "close_calls": n_close,
            "fp_backed": fp_cov,
            "starters_n": len(starters),
        },
        "starters": _records(sview),
        "bench": _records(bview),
        "next_week": next_week,
    }


def add_kdef(payload: dict, rows: list[dict]) -> dict:
    """Fold my kicker and defense (mega/kdef.py lineup_rows) into a built payload: the
    starter of each joins the lineup and the projected total, any second one the bench."""
    if not payload.get("available") or not rows:
        return payload
    out = dict(payload)
    old = float((out.get("kpis") or {}).get("proj_total") or 0.0)
    starts = [{k: v for k, v in r.items() if k != "start"} for r in rows if r["start"]]
    out["starters"] = list(out["starters"]) + starts
    out["bench"] = list(out["bench"]) + [{k: v for k, v in r.items() if k != "start"} for r in rows if not r["start"]]
    new = round(old + sum(float(r["proj_adj"]) for r in starts), 1)
    out["kpis"] = {**out["kpis"], "proj_total": new, "starters_n": len(out["starters"])}
    # ponytail: the headline's number is swapped in place; build() owns its wording
    out["headline"] = out["headline"].replace(f"{old:.0f} points", f"{new:.0f} points", 1)
    return out
