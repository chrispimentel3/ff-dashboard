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
]
SLOT_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLEX": 4}


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
