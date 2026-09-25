"""Pure-data Trades: the league-wide "offers the league is set up for" list, grouped by manager.

This is deliberately only the static half of the Streamlit Trades tab. "Trade around one
player" — the interactive search that reruns `mega.trade_engine`/`mega.trade_league` per pick,
including a 1,500-season playoff-odds simulation for the top candidates — needs live
computation per query. mega-bowl-web is a statically-generated site with no backend by design
(see the project's grilling-session decisions), so that search tool isn't portable here without
adding a live API, which is a bigger architectural change than this export. This module only
covers what's already a fixed, precomputed list.
"""
from __future__ import annotations

import json

import pandas as pd

COLS = ["give", "give_val", "get", "get_val", "addresses", "fairness", "edge"]


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def build(trades: pd.DataFrame | None, roster_src: str | None) -> dict:
    """`trades` is IB["trades"] (raw — give_pos/get_pos not yet folded into give/get)."""
    if trades is None or trades.empty:
        return {"available": False, "groups": [], "roster_src": roster_src}

    tt = trades.copy()
    tt["give"] = tt["give"] + " (" + tt["give_pos"] + ")"
    tt["get"] = tt["get"] + " (" + tt["get_pos"] + ")"
    tt["addresses"] = tt["addresses"].str.replace(r"^my (\S+) need$", r"\1", regex=True)

    groups = []
    for partner, grp in tt.groupby("partner", sort=False):
        need = grp["they_need"].iloc[0] if "they_need" in grp.columns else None
        groups.append({
            "partner": partner,
            "they_need": need if need and need != "—" else None,
            "offers": _records(grp.reindex(columns=COLS)),
        })

    return {"available": True, "groups": groups, "roster_src": roster_src}
