"""Pure-data Waivers: the HANDOFF v1.3 lanes (Bid now / Early signal / Stash), or the
roster-blind board when the valuation engine can't build.

Reuses `worth_claiming` from action_board.py rather than defining a third copy of "what counts
as a claim that would actually help" — action_board.py, the Streamlit _tab_wire tab, and this
export must never disagree on that definition. Under v1.3 that definition is the Bid now lane.
"""
from __future__ import annotations

import json

import pandas as pd

from mega.action_board import worth_claiming
from mega.ui import short_name

LANE_COLS = ["player", "pos", "nfl_team", "ppg", "lane", "fit_pts", "start", "cover", "insure",
             "next3", "mechanism", "handcuff", "insures", "drop", "drop_cost", "drop_insure",
             "drop_flip", "signal", "p_expand", "gain_if_expands", "signal_score", "pct_ros",
             "market_on", "flip", "bid", "max_bid", "role", "why", "d_title", "se_title",
             "title_noise", "out_status", "out_back", "rival_top", "rival_team", "rivals_n", "bid_note"]
CHIP_COLS = ["player", "pos", "nfl_team", "flip", "flip_buyers"]
LANES = ("bid_now", "early_signal", "stash")


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def build(wv: pd.DataFrame | None, rivals: dict | None, market: dict | None) -> dict:
    """`wv` is IB["waivers"]. `rivals`/`market` (from `mega.faab`) are only meaningful on the
    "need-aware" path (when `wv` has a `lane` column) — pass None on the roster-blind fallback."""
    empty_lanes = {k: [] for k in LANES}
    if wv is None or wv.empty:
        return {
            "available": False, "need_aware": False, "headline": "", "subhead": "",
            "faab": None, "market": None, "lanes": empty_lanes, "trade_chips": [],
            "roster_notes": [], "meta": {}, "blind_board": [],
        }

    if "lane" not in wv.columns:
        return {
            "available": True, "need_aware": False, "headline": "", "subhead": "",
            "faab": None, "market": None, "lanes": empty_lanes, "trade_chips": [],
            "roster_notes": [], "meta": {},
            "blind_board": _records(wv.drop(columns=["norm"], errors="ignore")),
        }

    worth = worth_claiming(wv)
    top = worth.iloc[0] if len(worth) else None
    n_other = int(wv["lane"].isin(["early_signal", "stash"]).sum())
    headline = (
        f"Put ${int(top['bid'])} on {short_name(top['player'])} — he adds "
        f"{float(top['next3']):.1f} points a week over the next three."
        if top is not None else
        "Nothing on the wire improves your lineup over the next three weeks. Hold the budget."
    )
    subhead = (
        f"{len(worth)} to bid on now, {n_other} worth a stash or a watch"
        + (f"; the league has spent ${market['league_spend']:.0f} in FAAB so far."
           if market and market.get("league_spend") else ".")
    )

    return {
        "available": True,
        "need_aware": True,
        "headline": headline,
        "subhead": subhead,
        "faab": rivals,
        "market": market,
        "lanes": {k: _records(wv[wv["lane"] == k].reindex(columns=LANE_COLS)) for k in LANES},
        "trade_chips": _records(wv[wv["lane"] == "trade_chip"].reindex(columns=CHIP_COLS)),
        "roster_notes": list(wv.attrs.get("roster_notes") or []),
        "meta": dict(wv.attrs.get("v13") or {}),
        "blind_board": [],
    }
