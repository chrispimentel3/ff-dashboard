"""§20.2 The Vegas weekly projection, assembled.

Props → half-PPR points for every player a book priced this week. The maths lives in
`mega.props` and the fetching in `mega.odds`; this is the join, and the one judgment it
makes is what to do about the players a book priced only partly.

Books post a receiving-yards line on far more players than they post a touchdown market
for. Scoring the missing piece as zero would dock a WR two or three points while still
presenting the number as "Vegas", which is the worst of both worlds — wrong and
authoritative. So any component the market did not price is filled from the player's own
expected-points rate in ff_opportunity, and `vegas_filled` says how many were, so the app
can mark a projection that is only partly the market's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import odds as O
from . import props as P

# ff_opportunity's expected-value columns, mapped onto this module's components.
_FILL = {
    "rec_yds": ("rec_yards_gained_exp",),
    "rush_yds": ("rush_yards_gained_exp",),
    "pass_yds": ("pass_yards_gained_exp",),
    "rec": ("receptions_exp",),
    "pass_tds": ("pass_touchdown_exp",),
    "anytime_td": ("rec_touchdown_exp", "rush_touchdown_exp"),
}


def fill_frame(season: int) -> pd.DataFrame:
    """Per-game expected components for every player, from ff_opportunity.

    A season average, not a projection — it only ever stands in for a market the book did
    not post, and a player with no history at all simply gets nothing rather than a
    positional average he may have no claim to.
    """
    from . import season as S
    try:
        ffo = S.ff_opportunity(season)
    except Exception:
        return pd.DataFrame()
    if ffo is None or ffo.empty or "player_id" not in ffo.columns:
        return pd.DataFrame()
    d = ffo.copy()
    games = d.groupby("player_id")["week"].nunique().rename("g")
    out = pd.DataFrame(index=games.index)
    for comp, cols in _FILL.items():
        tot = None
        for c in cols:
            if c in d.columns:
                s = pd.to_numeric(d[c], errors="coerce").fillna(0.0).groupby(d["player_id"]).sum()
                tot = s if tot is None else tot.add(s, fill_value=0.0)
        if tot is not None:
            out[comp] = tot / games
    return out.reset_index().rename(columns={"player_id": "gsis_id"})


COLS = ["gsis_id", "player", "team", "vegas", "vegas_parts", "vegas_filled",
        "vegas_complete"]


def build_csv(season: int, wk: int):
    """The published, derived projection — raw book prices stay local (see .gitignore)."""
    from .config import ROOT
    d = ROOT / "data" / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"vegas_{season}_wk{int(wk):02d}.csv"


def publish(season: int, wk: int) -> pd.DataFrame:
    """Score the swept props and write the file the hosted app reads.

    Streamlit Cloud has no Odds API key and no R, the same reason `data/build` exists for
    the ffanalytics projections. Only the derived points are published: the per-book prices
    behind them belong to the feed they came from.
    """
    v = week(season, wk)
    if not v.empty:
        v.to_csv(build_csv(season, wk), index=False)
    return v


def week(season: int, wk: int, fill: bool = True) -> pd.DataFrame:
    """gsis_id, player, team, vegas, vegas_parts, vegas_filled, vegas_complete.

    Reads whatever `mega.odds` last wrote for that week, and falls back to the published
    build file when the raw lines are not on this machine — which is the normal case on
    Streamlit Cloud. Never fetches, so the app and the tests can call it freely.
    """
    lines = O.cached(season, wk)
    if lines is None or lines.empty:
        p = build_csv(season, wk)
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
        return pd.DataFrame(columns=COLS)
    lines = lines[lines.get("gsis_id").notna()] if "gsis_id" in lines.columns else lines
    means = P.to_means(lines)
    f = fill_frame(season) if fill else None
    return P.project(means, fill=f)


def coverage(season: int, wk: int) -> dict:
    """How much of this week is really the market's — for the status line."""
    v = week(season, wk)
    if v.empty:
        return {"players": 0, "complete": 0, "filled": 0}
    return {"players": int(len(v)),
            "complete": int(v["vegas_complete"].sum()),
            "filled": int((~v["vegas_complete"]).sum())}


def attach(df: pd.DataFrame, season: int, wk: int, on: str = "gsis_id") -> pd.DataFrame:
    """Left-join the week's Vegas number onto any frame keyed by gsis_id."""
    v = week(season, wk)
    if df is None or df.empty:
        return df
    out = df.copy()
    if v.empty or on not in out.columns:
        for c in ("vegas", "vegas_parts", "vegas_filled", "vegas_complete"):
            if c not in out.columns:
                out[c] = pd.NA
        return out
    keep = ["gsis_id", "vegas", "vegas_parts", "vegas_filled", "vegas_complete"]
    return out.merge(v[keep], left_on=on, right_on="gsis_id", how="left",
                     suffixes=("", "_v")).drop(columns=[c for c in ["gsis_id_v"] if c in out.columns])


def edge(df: pd.DataFrame, proj_col: str = "proj") -> pd.DataFrame:
    """vegas − model projection. Positive = the market likes him more than we do.

    This is the column worth reading. The two numbers are built from entirely different
    evidence — one from usage and role, one from money — so where they disagree by a wide
    margin, one of them knows something.
    """
    out = df.copy()
    if "vegas" not in out.columns or proj_col not in out.columns:
        out["vegas_edge"] = pd.NA
        return out
    v = pd.to_numeric(out["vegas"], errors="coerce")
    p = pd.to_numeric(out[proj_col], errors="coerce")
    out["vegas_edge"] = (v - p).round(1)
    return out
