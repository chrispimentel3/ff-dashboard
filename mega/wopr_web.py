"""Pure-data packaging for the WOPR (Receiving opportunity) tab.

`mega.wopr.summary()` is already pure — no Streamlit calls, nothing here to extract — so this
is packaging only, same as `mega/matchups.py`: shape its mine/opp/fa/unknown frames into JSON.
"""
from __future__ import annotations

import json

import pandas as pd

from .wopr import SCHEMA

MCOLS = ["name", "pos", "team_2026_nfl", "wopr_anchored", "wopr_posrank",
         "board_posrank", "rank_delta", "ppg_minus_xppg", "tags"]


def _records(df: pd.DataFrame | None, cols: list[str] | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    if cols is not None:
        df = df.reindex(columns=cols)
    return json.loads(df.to_json(orient="records"))


def build(W: dict | None, season: int) -> dict:
    """`W` is `mega.wopr.summary(season)`'s return value."""
    if not W or W["df"].empty:
        return {
            "available": False, "meta": {}, "mine": [], "opp": [], "fa": [], "unknown": [],
            "full_csv_name": f"wopr_targets_{season}.csv",
        }

    return {
        "available": True,
        "meta": {
            "ownership_source": W["meta"].get("ownership_source", ""),
            "base_season": W["meta"].get("base_season"),
            "percentiles": W["meta"].get("percentiles", {}),
        },
        "mine": _records(W["mine"], MCOLS),
        "opp": _records(W["opp"], ["owner"] + MCOLS),
        "fa": _records(W["fa"], MCOLS),
        "unknown": _records(W["unknown"], ["name", "pos", "tags"]),
        "full_csv_name": f"wopr_targets_{season}.csv",
    }
