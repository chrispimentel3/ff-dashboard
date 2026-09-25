"""Pure-data Archetypes (Blueprint fit): roster fit + position target board.

Filtering by position (QB/RB/WR/TE) happens client-side over this one export, the same way
the Streamlit radio button just re-filters the already-computed `arch` DataFrame — no
recomputation needed per position, so there's nothing to duplicate by doing it in JS too.
"""
from __future__ import annotations

import json

import pandas as pd

MINE_COLS = ["player", "pos", "team", "arch_fit", "tags", "carries_pg", "tgt_share",
             "tm_rank", "age", "exp_yrs", "why"]
BOARD_COLS = ["status", "player", "pos", "team", "arch_fit", "tags", "half_ppr_pg",
              "proj_ppg", "tgt_share", "tm_rank", "why"]


def _records(df: pd.DataFrame | None, cols: list[str]) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.reindex(columns=cols).to_json(orient="records"))


def build(arch: pd.DataFrame | None, my_norms: set, rostered: set) -> dict:
    """`arch` is app.py's `_archetypes(season)`; `my_norms`/`rostered` are normalized-name
    sets app.py already builds from `skill["name"]` and `IB["draft"]["player"]`."""
    if arch is None or arch.empty:
        return {"available": False, "mine": [], "board": []}

    arch = arch.assign(mine=arch["norm"].isin(my_norms))
    mine = arch[arch["mine"]].sort_values("arch_fit", ascending=False)

    board = arch.copy()
    board["status"] = board.apply(
        lambda r: "mine" if r["mine"] else ("rostered" if r["norm"] in rostered else "available"),
        axis=1,
    )
    board = board.sort_values("arch_fit", ascending=False)

    return {
        "available": True,
        "mine": _records(mine, MINE_COLS),
        "board": _records(board, BOARD_COLS),
    }
