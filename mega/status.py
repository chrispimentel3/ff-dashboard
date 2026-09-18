"""Player availability the data doesn't know yet.

data/player_status.csv — Chris edits this by hand:

    player,status,through_week,note
    Kyler Murray,OUT,2,Out for the Week 2 games
    Tyreek Hill,OUT,,Not playing in 2026

`through_week` is the last NFL week the override covers; blank means the rest of the
season. The file is read on every call (it's a handful of rows), so an edit shows up on
the next rerun without restarting anything.

Why it exists: the official injury report lags (Kyler Murray was ruled out before the
Week 2 report carried a status), and a player who simply isn't on a team has no report
at all. Tyreek Hill kept surfacing as a WOPR waiver add off his 2025 volume.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA
from .ids import norm

STATUS_CSV = DATA / "player_status.csv"


def overrides() -> pd.DataFrame:
    cols = ["player", "status", "through_week", "note", "norm"]
    if not STATUS_CSV.is_file():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(STATUS_CSV, dtype=str).fillna("")
    for c in cols[:-1]:
        if c not in df.columns:
            df[c] = ""
    df["status"] = df["status"].str.strip().str.upper()
    df["through_week"] = pd.to_numeric(df["through_week"], errors="coerce")
    df["norm"] = df["player"].map(norm)
    return df[df["norm"] != ""][cols]


def out_for_week(week: int) -> dict[str, str]:
    """norm -> label for everyone ruled out of `week`."""
    o = overrides()
    o = o[(o["status"] == "OUT") & (o["through_week"].isna() | (o["through_week"] >= week))]
    return {r.norm: _label(r) for r in o.itertuples()}


def out_for_season() -> dict[str, str]:
    """norm -> label for players who won't play again this season."""
    o = overrides()
    o = o[(o["status"] == "OUT") & o["through_week"].isna()]
    return {r.norm: _label(r) for r in o.itertuples()}


def _label(r) -> str:
    return "OUT (season)" if pd.isna(r.through_week) else f"OUT (thru wk {int(r.through_week)})"


def note_for(name: object) -> str:
    o = overrides()
    hit = o[o["norm"] == norm(name)]
    return "" if hit.empty else str(hit["note"].iloc[0])


if __name__ == "__main__":
    print("overrides:"); print(overrides().to_string(index=False))
    print("\nout week 2:", out_for_week(2)); print("out week 3:", out_for_week(3)); print("season:", out_for_season())
