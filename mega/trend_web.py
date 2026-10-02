"""Pure-data packaging for the three weekly trend charts (WOPR, archetype fit, trade
value) — reads mega/history.py's snapshot CSVs directly. No Streamlit involved: these
files are plain CSVs on disk, so this can (and does) run outside the AppTest export loop
tools/export_web.py otherwise uses.

Every player in the snapshot is exported, not just the roster: with two snapshots a week
the interesting question is "who moved", and the biggest movers are mostly on other teams.
`mine` marks the roster (the chart opens on those), `movers` is the week-over-week change
between each player's last two snapshots, biggest first.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .history import HIST_DIR

N_MOVERS = 8
MIN_VALUE = {"value": 500.0}      # a trade value under this is noise, not a mover


def _empty() -> dict:
    return {"available": False, "players": [], "rows": [], "info": {}, "movers": {"up": [], "down": []}}


def movers(df: pd.DataFrame, value_col: str, norms: set[str], n: int = N_MOVERS,
           only: set[str] | None = None) -> dict:
    """Change between each player's two most recent snapshots. A player with only one
    snapshot hasn't moved; one missing from the latest snapshot has dropped off the board.
    `only` limits who can be listed (the chart still carries everyone)."""
    if only is not None:
        df = df[df["norm"].isin(only | norms)]
    weeks = sorted(df["week"].unique())
    if len(weeks) < 2:
        return {"weeks": [], "up": [], "down": []}
    prev_w, last_w = int(weeks[-2]), int(weeks[-1])
    a = df[df["week"] == prev_w].set_index("norm")
    b = df[df["week"] == last_w].set_index("norm")
    both = a.index.intersection(b.index)
    floor = MIN_VALUE.get(value_col, 0.0)
    rows = []
    for k in both:
        pv, lv = float(a.at[k, value_col]), float(b.at[k, value_col])
        if max(pv, lv) < floor:
            continue
        rows.append({"player": b.at[k, "player"], "pos": b.at[k, "pos"], "mine": k in norms,
                     "prev": round(pv, 4), "last": round(lv, 4), "delta": round(lv - pv, 4)})
    rows.sort(key=lambda r: r["delta"])
    down = [r for r in rows if r["delta"] < 0][:n]
    up = [r for r in reversed(rows) if r["delta"] > 0][:n]
    return {"weeks": [prev_w, last_w], "up": up, "down": down}


def _series(path: Path, value_col: str, norms: set[str], only: set[str] | None = None) -> dict:
    if not path.is_file():
        return _empty()
    df = pd.read_csv(path)
    if df.empty:
        return _empty()
    df = df.dropna(subset=[value_col])
    last = df.sort_values("week").groupby("norm").tail(1).set_index("player")
    info = {p: {"pos": r["pos"], "mine": r["norm"] in norms} for p, r in last.iterrows()}
    rows = df[["player", "week", value_col]].rename(columns={value_col: "value"})
    return {
        "available": True,
        "players": sorted(info),
        "info": info,
        "rows": json.loads(rows.sort_values(["player", "week"]).round(4).to_json(orient="records")),
        "movers": movers(df, value_col, norms, only=only),
    }


def _valued(path: Path) -> set[str]:
    """Players the trade market prices — the ones a mover list is worth reading for. Without
    this the biggest swings in WOPR and archetype fit belong to depth players with a game or
    two of data."""
    if not path.is_file():
        return set()
    df = pd.read_csv(path)
    return set(df[df["week"] == df["week"].max()]["norm"]) if not df.empty else set()


def build(norms: set[str]) -> dict:
    valued = _valued(HIST_DIR / "trade_value_weekly.csv") or None
    return {
        "wopr": _series(HIST_DIR / "wopr_weekly.csv", "wopr_anchored", norms, valued),
        "archetype": _series(HIST_DIR / "archetype_weekly.csv", "arch_fit", norms, valued),
        "trade_value": _series(HIST_DIR / "trade_value_weekly.csv", "value", norms),
    }
