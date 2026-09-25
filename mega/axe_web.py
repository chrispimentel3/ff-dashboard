"""Pure-data packaging for the "Points vs opportunity" (actual vs expected) tab.

Packaging only — app.py's `_tab_axe` already builds `cmp` as a plain DataFrame before
charting/rendering it.
"""
from __future__ import annotations

import json

import pandas as pd

COLS = ["player", "expected", "actual", "diff"]


def build(cmp: pd.DataFrame | None, week: int) -> dict:
    if cmp is None or cmp.empty:
        return {"available": False, "week": week, "rows": []}
    return {
        "available": True,
        "week": week,
        "rows": json.loads(cmp.reindex(columns=COLS).to_json(orient="records")),
    }
