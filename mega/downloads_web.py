"""Pure-data packaging for the Downloads tab — the mapped roster and its id-match report.

Unlike "Ask anything" (service/main.py, a live query), this half of the old "Ask the data"
page is a fixed, precomputable view — same export pattern as every other tab. The weekly
CSVs (player_stats.csv, ff_opportunity.csv) are written straight to data/web/ by app.py
itself rather than routed through here, since they're files, not JSON.
"""
from __future__ import annotations

import json

import pandas as pd

COLS = ["name", "slot", "pos", "nfl_team", "yahoo_id", "gsis_id", "pfr_id",
        "matched_name", "match_method", "resolved", "unmapped"]


def build(mapped: pd.DataFrame, match_report: dict) -> dict:
    from . import ids as player_ids

    if mapped is None or mapped.empty:
        return {"available": False, "roster": [], "match_line": ""}
    return {
        "available": True,
        "roster": json.loads(mapped.reindex(columns=COLS).to_json(orient="records")),
        "match_line": player_ids.report_line(match_report),
    }
