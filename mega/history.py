"""Weekly snapshots for metrics that are point-in-time today but should trend over time.

WOPR (receiving opportunity), archetype fit, and FantasyCalc trade value are each computed
fresh every time the app asks for them — there was never a reason to keep last week's
numbers around until the trend charts needed them. The scheduled refreshes call
`snapshot_new` every morning (via tools/publish_web.py), which takes each week's snapshot
once — on the first run after the NFL week turns over, when the week just played is
complete — and leaves it alone after. `snapshot_all` (tuesday.py) always rewrites the
current week; each snapshot is idempotent per (season, week), so a rerun replaces that
week's rows instead of duplicating them.

Files live in data/history/ — committed and public, same as everything under data/, so
mega-bowl-web can read them the same way it reads every other export.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

HIST_DIR = Path(__file__).resolve().parent.parent / "data" / "history"


def _append(path: Path, season: int, week: int, rows: pd.DataFrame) -> None:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows.copy()
    rows.insert(0, "week", week)
    rows.insert(0, "season", season)
    rows["snapshot_date"] = dt.date.today().isoformat()
    if path.is_file():
        existing = pd.read_csv(path)
        existing = existing[~((existing["season"] == season) & (existing["week"] == week))]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows.sort_values(["season", "week"]).to_csv(path, index=False)


def snapshot_wopr(season: int, week: int) -> None:
    from .wopr import summary

    df = summary(season)["df"]
    if df is None or df.empty:
        return
    _append(HIST_DIR / "wopr_weekly.csv", season, week,
            df[["gsis_id", "norm", "name", "pos", "wopr_anchored"]].rename(columns={"name": "player"}))


def snapshot_archetypes(season: int, week: int) -> None:
    from .archetypes import score

    df = score(season)
    if df is None or df.empty:
        return
    _append(HIST_DIR / "archetype_weekly.csv", season, week, df[["norm", "player", "pos", "arch_fit"]])


def snapshot_trade_value(season: int, week: int) -> None:
    from .ids import norm as _norm
    from .intel import draft_value_delta

    df = draft_value_delta(season)
    if df is None or df.empty:
        return
    df = df.assign(norm=df["player"].map(_norm))
    _append(HIST_DIR / "trade_value_weekly.csv", season, week, df[["norm", "player", "pos", "value"]])


def snapshot_all(season: int, week: int) -> None:
    """Best-effort: one metric's source data being unavailable shouldn't block the others."""
    for fn in (snapshot_wopr, snapshot_archetypes, snapshot_trade_value):
        try:
            fn(season, week)
        except Exception as e:
            print(f"[history] {fn.__name__} skipped: {e}")


SNAPSHOTS = ((snapshot_wopr, "wopr_weekly.csv"), (snapshot_archetypes, "archetype_weekly.csv"),
             (snapshot_trade_value, "trade_value_weekly.csv"))


def _has(path: Path, season: int, week: int) -> bool:
    if not path.is_file():
        return False
    d = pd.read_csv(path, usecols=["season", "week"])
    return bool(((d["season"] == season) & (d["week"] == week)).any())


def snapshot_new(season: int, week: int) -> list[str]:
    """Take this week's snapshot of each metric that doesn't have one yet; returns the
    files written. Running it daily is safe: a week already captured is never retaken, so
    the snapshot stays the Tuesday-morning one (the prior week's games all final) rather
    than drifting to whatever the numbers were by the next Monday."""
    wrote = []
    for fn, name in SNAPSHOTS:
        if _has(HIST_DIR / name, season, week):
            continue
        try:
            fn(season, week)
            if _has(HIST_DIR / name, season, week):
                wrote.append(name)
        except Exception as e:
            print(f"[history] {fn.__name__} skipped: {e}")
    return wrote

