"""Pure-data Waivers: worth-bidding-on claims, speculative adds, or the roster-blind board.

Reuses `worth_claiming` from action_board.py rather than defining a third copy of "what counts
as a claim that would actually help" — action_board.py, the Streamlit _tab_wire tab, and this
export must never disagree on that definition.
"""
from __future__ import annotations

import json

import pandas as pd

from mega.action_board import worth_claiming
from mega.ui import short_name

WORTH_COLS = ["player", "pos", "nfl_team", "ppg", "gain", "bid", "max_bid", "drop", "why"]
SPEC_COLS = ["player", "pos", "nfl_team", "ppg", "add_score", "upside", "why"]


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def build(wv: pd.DataFrame | None, rivals: dict | None, market: dict | None) -> dict:
    """`wv` is IB["waivers"]. `rivals`/`market` (from `mega.faab`) are only meaningful on the
    "need-aware" path (when `wv` has a `bid` column) — pass None on the roster-blind fallback."""
    if wv is None or wv.empty:
        return {
            "available": False, "need_aware": False, "headline": "", "subhead": "",
            "faab": None, "market": None, "worth": [], "speculative": [], "blind_board": [],
        }

    need_aware = "bid" in wv.columns
    if not need_aware:
        return {
            "available": True, "need_aware": False, "headline": "", "subhead": "",
            "faab": None, "market": None, "worth": [], "speculative": [],
            "blind_board": _records(wv.drop(columns=["norm"], errors="ignore")),
        }

    worth = worth_claiming(wv)
    spec = wv[~wv.index.isin(worth.index)]
    top = worth.iloc[0] if len(worth) else None
    headline = (
        f"Put ${int(top['bid'])} on {short_name(top['player'])} — he adds "
        f"{float(top['gain']):.1f} points a game to your starting nine."
        if top is not None else
        "Nothing on the wire would start for you. Hold the budget."
    )
    subhead = (
        f"{len(worth)} free agent{'' if len(worth) == 1 else 's'} would change your lineup"
        + (f"; the league has been settling claims around ${market['median']:.0f}."
           if market and market.get("claims") else ".")
    )

    return {
        "available": True,
        "need_aware": True,
        "headline": headline,
        "subhead": subhead,
        "faab": rivals,
        "market": market,
        "worth": _records(worth.reindex(columns=WORTH_COLS)),
        "speculative": _records(spec.reindex(columns=SPEC_COLS).head(15)),
        "blind_board": [],
    }
