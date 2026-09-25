"""Pure-data packaging for the Draft value tab.

`IB["draft_delta"]` and `IB["buysell"]` are already pure DataFrames — packaging only, like
matchups.py. The "My picks only" checkbox filters client-side over one export (the `mine`
column ships with every row) rather than shipping two separate boards.
"""
from __future__ import annotations

import json

import pandas as pd

BOARD_COLS = ["player", "pos", "drafted_by", "round", "value", "value_delta", "mine"]
REGRESSION_COLS = ["player", "pos", "gms", "actual", "expected", "diff_pg", "signal"]


def _records(df: pd.DataFrame | None, cols: list[str]) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.reindex(columns=cols).to_json(orient="records"))


def build(dd: pd.DataFrame | None, bs: pd.DataFrame | None, my_norms: set) -> dict:
    """`dd` is IB["draft_delta"] (already sorted by value_delta desc — natural "risers" order);
    `bs` is IB["buysell"]; `my_norms` is the normalized-name set app.py builds from
    `skill["name"]`, same as archetypes_web.build()."""
    if dd is None or dd.empty:
        return {"available": False, "board": [], "regression": []}

    board = dd[dd["value"] > 0]

    mine_bs = pd.DataFrame()
    if bs is not None and not bs.empty:
        mine_bs = bs[bs["norm"].isin(my_norms)].sort_values("diff_pg")

    return {
        "available": True,
        "board": _records(board, BOARD_COLS),
        "regression": _records(mine_bs, REGRESSION_COLS),
    }
