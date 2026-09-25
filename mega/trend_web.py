"""Pure-data packaging for the three weekly trend charts (WOPR, archetype fit, trade
value) — reads mega/history.py's snapshot CSVs directly. No Streamlit involved: these
files are plain CSVs on disk, so this can (and does) run outside the AppTest export loop
tools/export_web.py otherwise uses.

Scoped to the current roster, same reasoning as Usage trends (mega/usage_web.py): the
full league's history is more players than a line-chart picker is useful with, and
"how has my own roster's opportunity been trending" is the question this answers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .history import HIST_DIR


def _series(path: Path, value_col: str, norms: set[str]) -> dict:
    if not path.is_file():
        return {"available": False, "players": [], "rows": []}
    df = pd.read_csv(path)
    if df.empty:
        return {"available": False, "players": [], "rows": []}
    df = df[df["norm"].isin(norms)]
    if df.empty:
        return {"available": False, "players": [], "rows": []}
    rows = df[["player", "week", value_col]].rename(columns={value_col: "value"})
    return {
        "available": True,
        "players": sorted(df["player"].unique().tolist()),
        "rows": json.loads(rows.sort_values(["player", "week"]).to_json(orient="records")),
    }


def build(norms: set[str]) -> dict:
    return {
        "wopr": _series(HIST_DIR / "wopr_weekly.csv", "wopr_anchored", norms),
        "archetype": _series(HIST_DIR / "archetype_weekly.csv", "arch_fit", norms),
        "trade_value": _series(HIST_DIR / "trade_value_weekly.csv", "value", norms),
    }
