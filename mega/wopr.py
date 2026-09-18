"""WOPR targeting module — Weighted Opportunity Rating for WR/TE.

Implements the validated core from WOPR_HANDOFF.md:

  TS   = player targets / team targets         (summed over the window, then divided)
  AYS  = player air yards / team air yards
  WOPR = 1.5 * TS + 0.7 * AYS
  xPPG = per-position linear fit of half-PPR PPG on WOPR (refit each season, targets>=30)
  residual = ppg - xppg          (efficiency / TD luck)
  anchored = (g26*wopr26 + 3*wopr25) / (g26 + 3)     (D7, 3-game prior; rookies use pos median)

then an ownership split into MINE / OPP / FA / UNKNOWN views with percentile tags (sec 6).

Adapted to this dashboard (vs the handoff's standalone /wopr/ layout):
  * nflverse stats come through nflreadpy, not raw release CSVs (same underlying data).
  * ownership uses the Yahoo API pull when present (mega.yahoo_api), else the draft
    board as a stand-in for initial rosters — labeled in the UI either way.
  * board VOR / posrank come from the War Room projections (mega.archetypes),
    age from ff_playerids; fdrr isn't in this project's board, so it's carried as NA.

Defaults follow the bold recommendations in handoff sec 1 (both windows, games-played
denominator, WR+TE, percentiles, 3-game anchor, board VOR as the value axis).
"""
from __future__ import annotations

import functools

import nflreadpy as nfl
import numpy as np
import pandas as pd

from .config import MY_TEAM
from .intel import _norm

POS = ("WR", "TE")
PRIOR_GAMES = 3          # D7: the 2025 prior is worth 3 games
MIN_G_SEASON = 8         # sec 4: season rows need >=8 games to be tagged
MIN_TGT_FIT = 30         # sec 3: xPPG fit uses players with >=30 targets
LATE_WEEK = 10           # "late window" for the RISER signal (weeks >= 10)


# ------------------------------------------------------------------ nflverse core
def _load_reg(season: int) -> pd.DataFrame:
    try:
        df = nfl.load_player_stats(seasons=[season]).to_pandas()
    except Exception:
        return pd.DataFrame()
    if df.empty or "season_type" not in df.columns:
        return pd.DataFrame()
    return df[df["season_type"] == "REG"].copy()


def _team_totals(w: pd.DataFrame) -> pd.DataFrame:
    """Team targets + intended air yards per team-week (all positions)."""
    return (
        w.groupby(["team", "week"])[["targets", "receiving_air_yards"]]
        .sum()
        .add_prefix("team_")
        .reset_index()
    )


def _hppr(w: pd.DataFrame) -> pd.Series:
    z = lambda c: pd.to_numeric(w.get(c), errors="coerce").fillna(0)
    return (
        0.1 * (z("receiving_yards") + z("rushing_yards"))
        + 6 * (z("receiving_tds") + z("rushing_tds"))
        + 0.5 * z("receptions")
        - 2 * (z("receiving_fumbles_lost") + z("rushing_fumbles_lost"))
    )


def _aggregate(w: pd.DataFrame) -> pd.DataFrame:
    """Sum numerators and denominators across the window, THEN divide (never avg ratios)."""
    g = w.groupby(["player_id", "player_display_name", "position"], as_index=False).agg(
        games=("week", "nunique"),
        tgt=("targets", "sum"),
        ay=("receiving_air_yards", "sum"),
        ttgt=("team_targets", "sum"),
        tay=("team_receiving_air_yards", "sum"),
        pts=("hppr", "sum"),
        last_team=("team", "last"),
    )
    g["ts"] = g["tgt"] / g["ttgt"].replace(0, np.nan)
    g["ays"] = g["ay"] / g["tay"].replace(0, np.nan)
    g["wopr"] = 1.5 * g["ts"] + 0.7 * g["ays"]
    g["ppg"] = g["pts"] / g["games"].replace(0, np.nan)
    return g


def _fit_resid(o: pd.DataFrame) -> pd.DataFrame:
    """Per-position linear fit of ppg on WOPR (targets>=30); residual = efficiency/TD luck."""
    o = o.copy()
    o["wopr_xppg"] = np.nan
    for pos in POS:
        m = (o["position"] == pos) & (o["tgt"] >= MIN_TGT_FIT) & o["wopr"].notna() & o["ppg"].notna()
        if m.sum() >= 5:
            slope, intercept = np.polyfit(o.loc[m, "wopr"], o.loc[m, "ppg"], 1)
            pm = o["position"] == pos
            o.loc[pm, "wopr_xppg"] = slope * o.loc[pm, "wopr"] + intercept
    o["ppg_minus_xppg"] = o["ppg"] - o["wopr_xppg"]
    return o


@functools.lru_cache(maxsize=6)
def _agg_season(season: int) -> pd.DataFrame:
    """Full-season WR/TE aggregate for one season, with late-window WOPR and the xPPG residual."""
    w_all = _load_reg(season)
    if w_all.empty:
        return pd.DataFrame()
    tt = _team_totals(w_all)
    w = w_all.merge(tt, on=["team", "week"], how="left")
    w["hppr"] = _hppr(w)
    w = w[w["position"].isin(POS)].copy()
    if w.empty:
        return pd.DataFrame()

    base = _aggregate(w)
    late = w[w["week"] >= LATE_WEEK]
    if not late.empty:
        lo = _aggregate(late)[["player_id", "wopr"]].rename(columns={"wopr": "wopr_late"})
        base = base.merge(lo, on="player_id", how="left")
    else:
        base["wopr_late"] = np.nan
    base = _fit_resid(base)
    base["season"] = season
    return base


# ------------------------------------------------------------------ crosswalk / meta
@functools.lru_cache(maxsize=1)
def _crosswalk() -> pd.DataFrame:
    ids = nfl.load_ff_playerids().to_pandas()
    ids["yahoo_id"] = pd.to_numeric(ids.get("yahoo_id"), errors="coerce")
    ids["age"] = pd.to_numeric(ids.get("age"), errors="coerce")
    keep = [c for c in ["gsis_id", "yahoo_id", "age", "name"] if c in ids.columns]
    x = ids[keep].rename(columns={"gsis_id": "player_id"}).dropna(subset=["player_id"])
    return x.drop_duplicates("player_id")


@functools.lru_cache(maxsize=4)
def _roster_meta(season: int) -> pd.DataFrame:
    """Current NFL team + status from the nflverse roster file (authoritative per handoff)."""
    try:
        r = nfl.load_rosters(seasons=[season]).to_pandas()
    except Exception:
        return pd.DataFrame(columns=["player_id", "team_2026_nfl", "nfl_status"])
    gid = "gsis_id" if "gsis_id" in r.columns else ("player_id" if "player_id" in r.columns else None)
    if gid is None:
        return pd.DataFrame(columns=["player_id", "team_2026_nfl", "nfl_status"])
    team_c = "team" if "team" in r.columns else None
    stat_c = "status" if "status" in r.columns else None
    out = pd.DataFrame({"player_id": r[gid]})
    out["team_2026_nfl"] = r[team_c] if team_c else ""
    out["nfl_status"] = r[stat_c] if stat_c else ""
    return out.dropna(subset=["player_id"]).drop_duplicates("player_id")


# ------------------------------------------------------------------ ownership
def _ownership() -> tuple[pd.DataFrame, str]:
    """Return (norm -> owner, yahoo_id) and a human label for the source used."""
    try:
        from . import yahoo_api

        if yahoo_api.available():
            r = yahoo_api.rosters_df()
            if not r.empty:
                own = r[["norm", "team"]].rename(columns={"team": "owner"}).drop_duplicates("norm")
                return own, "Yahoo API roster pull"
    except Exception:
        pass
    try:  # browser scrape — the only live path while the Yahoo API is scope-blocked
        from .yahoo import cached_rosters

        r = cached_rosters()
        if not r.empty:
            own = r[["norm", "team"]].rename(columns={"team": "owner"}).drop_duplicates("norm")
            return own, "scraped Yahoo rosters"
    except Exception:
        pass
    from .draft_board import load_draft

    d = load_draft()
    own = d[d["pos"].isin(POS)][["player", "drafted_by"]].rename(columns={"drafted_by": "owner"})
    own["norm"] = own["player"].map(_norm)
    return own[["norm", "owner"]].drop_duplicates("norm"), "draft-board approximation (no live Yahoo pull yet)"


# ------------------------------------------------------------------ board (vor / posrank)
def _board() -> pd.DataFrame:
    try:
        from .archetypes import warroom_projections

        b = warroom_projections()
        b = b[["norm", "posrank", "vor"]].rename(columns={"posrank": "board_posrank", "vor": "board_vor"})
        # posrank is "WR 1" / "TE 5" — pull out the integer
        b["board_posrank"] = pd.to_numeric(
            b["board_posrank"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
        b["board_vor"] = pd.to_numeric(b["board_vor"], errors="coerce")
        return b
    except Exception:
        return pd.DataFrame(columns=["norm", "board_posrank", "board_vor"])


# ------------------------------------------------------------------ percentiles + tags
def _percentiles(df: pd.DataFrame) -> dict:
    """WOPR percentiles among board-matched, >=6-game players (handoff sec 6 population)."""
    pool = df[df["board_posrank"].notna() & (df["g_2025"] >= 6)]
    if pool.empty:  # pre-board fallback: use everyone with enough games
        pool = df[df["g_2025"] >= 6]
    out = {}
    for pos in POS:
        s = pool[pool["pos"] == pos]["wopr_2025"].dropna()
        if len(s):
            out[pos] = {"p50": float(s.quantile(0.50)), "p75": float(s.quantile(0.75)), "p90": float(s.quantile(0.90))}
    return out


def _tag_row(r: pd.Series, pct: dict) -> str:
    pos = r["pos"]
    p = pct.get(pos, {})
    p50, p75 = p.get("p50", np.nan), p.get("p75", np.nan)
    w, late = r.get("wopr_2025"), r.get("wopr_wk10plus")
    w26 = r.get("wopr_2026")
    resid = r.get("ppg_minus_xppg")
    g25 = r.get("g_2025") or 0
    g26 = r.get("g_2026") or 0
    rdelta = r.get("rank_delta")
    bpr = r.get("board_posrank")
    tags = []

    season_ok = g25 >= MIN_G_SEASON and pd.notna(w)
    if season_ok:
        if pd.notna(p75) and w >= p75 and pd.notna(rdelta) and rdelta >= 6:
            tags.append("UNDERPRICED")
        if pd.notna(p50) and w >= p50 and pd.notna(resid) and resid <= -1.5:
            tags.append("BUY_LOW")
        if pd.notna(resid) and resid >= 2.0 and pd.notna(p75) and w < p75:
            tags.append("SELL_HIGH")
        floor = 0.50 if pos == "WR" else 0.40
        if pd.notna(late) and (late - w) >= 0.08 and late >= floor:
            tags.append("RISER")
    # in-season role jump (>=2 games of the current season)
    if g26 >= 2 and pd.notna(w26) and pd.notna(p75) and w26 >= p75 and (pd.isna(w) or w26 >= w + 0.15):
        tags.append("ROLE_JUMP")
    # board overvalue fade
    if pd.notna(bpr) and pd.notna(w):
        if pos == "WR" and bpr <= 24 and w < 0.52:
            tags.append("FADE")
        if pos == "TE" and bpr <= 8 and w < 0.37:
            tags.append("FADE")
    return ",".join(tags)


# ------------------------------------------------------------------ build
def build(season: int) -> pd.DataFrame:
    """One row per WR/TE with WOPR windows, anchored blend, tags, and an ownership view."""
    base = _agg_season(season - 1)      # baseline (e.g. 2025)
    cur = _agg_season(season)           # current (e.g. 2026) — may be empty/thin early
    if base.empty and not cur.empty:
        base = cur                      # first season of data: baseline == current

    if base.empty:
        return pd.DataFrame()

    df = base.rename(columns={
        "player_display_name": "name", "position": "pos",
        "wopr": "wopr_2025", "games": "g_2025", "wopr_late": "wopr_wk10plus",
        "ppg": "hppr_ppg",
    })[["player_id", "name", "pos", "wopr_2025", "g_2025", "wopr_wk10plus",
        "hppr_ppg", "wopr_xppg", "ppg_minus_xppg", "tgt", "last_team"]].copy()

    # current-season window
    if not cur.empty:
        c = cur.rename(columns={"wopr": "wopr_2026", "games": "g_2026"})[["player_id", "wopr_2026", "g_2026"]]
        df = df.merge(c, on="player_id", how="outer")
        # fill identity for current-only players (e.g. rookies) from cur
        ident = cur.rename(columns={"player_display_name": "name", "position": "pos"})[["player_id", "name", "pos"]]
        df = df.merge(ident, on="player_id", how="left", suffixes=("", "_c"))
        df["name"] = df["name"].fillna(df.pop("name_c"))
        df["pos"] = df["pos"].fillna(df.pop("pos_c"))
    else:
        df["wopr_2026"] = np.nan
        df["g_2026"] = 0

    df = df[df["pos"].isin(POS)].copy()
    df["g_2026"] = pd.to_numeric(df["g_2026"], errors="coerce").fillna(0)
    df["g_2025"] = pd.to_numeric(df["g_2025"], errors="coerce").fillna(0)

    # D7 anchored: rookies (no 2025) use the position-median prior
    pos_prior = {pos: base[(base["position"] == pos) & (base["games"] >= 6)]["wopr"].median() for pos in POS}
    w25_for_anchor = df.apply(
        lambda r: r["wopr_2025"] if pd.notna(r["wopr_2025"]) else pos_prior.get(r["pos"], np.nan), axis=1
    )
    has_cur = df["g_2026"] > 0
    anchored = (df["g_2026"] * df["wopr_2026"].fillna(0) + PRIOR_GAMES * w25_for_anchor) / (df["g_2026"] + PRIOR_GAMES)
    df["wopr_anchored"] = np.where(has_cur, anchored, df["wopr_2025"])

    # positional rank by opportunity, and the board-vs-opportunity delta
    df["wopr_posrank"] = df.groupby("pos")["wopr_anchored"].rank(ascending=False, method="min")

    # crosswalk, age, board, roster meta
    xw = _crosswalk()
    df = df.merge(xw[["player_id", "yahoo_id", "age"]], on="player_id", how="left")
    df["gsis_id"] = df["player_id"]

    board = _board()
    df["norm"] = df["name"].map(_norm)
    df = df.merge(board, on="norm", how="left")
    df["rank_delta"] = df["board_posrank"] - df["wopr_posrank"]

    meta = _roster_meta(season)
    have_meta = not meta.empty
    if have_meta:
        df = df.merge(meta, on="player_id", how="left")
    for c in ("team_2026_nfl", "nfl_status"):
        if c not in df.columns:
            df[c] = ""
    if have_meta:
        # Absent from the current roster file = on no NFL team (retired, unsigned). His
        # last team is history, not where he plays — Tyreek Hill showed as a MIA waiver
        # add off 2025 volume while on no 2026 roster.
        absent = df["nfl_status"].isna()
        df.loc[absent, "nfl_status"] = "NO TEAM"
        df.loc[absent, "team_2026_nfl"] = ""
    df["team_2026_nfl"] = df["team_2026_nfl"].fillna(df["last_team"])
    df["moved_team"] = (df["last_team"].notna() & df["team_2026_nfl"].notna()
                        & (df["last_team"] != df["team_2026_nfl"]))

    # ownership -> view
    own, own_src = _ownership()
    df = df.merge(own, on="norm", how="left")
    df["owner"] = df["owner"].where(df["owner"].notna(), None)
    def _view(owner):
        if owner is None or (isinstance(owner, float) and pd.isna(owner)):
            return "FA"
        return "MINE" if str(owner) == MY_TEAM else "OPP"
    df["view"] = df["owner"].map(_view)

    # tags + value axis label (percentiles over board-matched players)
    pct = _percentiles(df)
    df["fdrr"] = np.nan  # not in this project's board; carried as NA per handoff sec 9
    df["tags"] = df.apply(lambda r: _tag_row(r, pct), axis=1)

    df.attrs["ownership_source"] = own_src
    df.attrs["roster_file"] = have_meta
    df.attrs["percentiles"] = pct
    df.attrs["base_season"] = int(base["season"].iloc[0]) if "season" in base.columns and len(base) else season - 1
    df.attrs["cur_season"] = season
    df.attrs["has_current"] = bool((df["g_2026"] > 0).any())
    return df.sort_values("wopr_anchored", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ views
SCHEMA = [
    "view", "owner", "name", "pos", "team_2026_nfl", "nfl_status", "yahoo_id", "gsis_id",
    "age", "board_posrank", "board_vor", "g_2025", "wopr_2025", "wopr_wk10plus",
    "g_2026", "wopr_2026", "wopr_anchored", "hppr_ppg", "wopr_xppg", "ppg_minus_xppg",
    "fdrr", "wopr_posrank", "rank_delta", "tags",
]


def _eligible(df: pd.DataFrame) -> pd.Series:
    """sec 4: only ACT players reach the trade or waiver lists, and never anyone Chris has
    ruled out for the season (data/player_status.csv).

    Blank status used to pass, which let players on no NFL team through. It still passes
    when the roster file failed to load at all — then nothing can be verified, and an
    empty tab would be worse than an unfiltered one.
    """
    from .status import out_for_season

    s = df["nfl_status"].fillna("").str.upper()
    ok = s.isin(["ACT", "A"]) if df.attrs.get("roster_file", True) else s.isin(["", "ACT", "A"])
    return ok & ~df["norm"].isin(out_for_season())


def summary(season: int) -> dict:
    df = build(season)
    if df.empty:
        return {"df": df, "mine": df, "opp": df, "fa": df, "unknown": df, "meta": {}}

    def _has(tagset):
        return df["tags"].apply(lambda t: bool(set(t.split(",")) & tagset) if t else False)

    mine = df[df["view"] == "MINE"].copy()
    mine["sell_signal"] = _has({"SELL_HIGH", "FADE"})[mine.index]
    mine = mine.sort_values(["sell_signal", "ppg_minus_xppg"], ascending=[False, False])

    elig = _eligible(df)
    opp = df[(df["view"] == "OPP") & elig & (df["tags"] != "")].copy()
    opp = opp.sort_values(["owner", "wopr_anchored"], ascending=[True, False])

    pct = df.attrs.get("percentiles", {})
    p50 = df["pos"].map(lambda p: pct.get(p, {}).get("p50", np.inf))
    fa_keep = _has({"ROLE_JUMP", "RISER"}) | (df["wopr_anchored"] >= p50)
    fa = df[(df["view"] == "FA") & elig & fa_keep].copy()
    fa = fa.sort_values("wopr_anchored", ascending=False)

    unknown = df[df["view"] == "UNKNOWN"].copy()
    return {"df": df, "mine": mine, "opp": opp, "fa": fa, "unknown": unknown, "meta": dict(df.attrs)}


if __name__ == "__main__":
    import sys

    s = int(sys.argv[1]) if len(sys.argv) > 1 else int(nfl.get_current_season())
    out = summary(s)
    df, meta = out["df"], out["meta"]
    print(f"season {s} | ownership: {meta.get('ownership_source')} | baseline {meta.get('base_season')} "
          f"| current data: {meta.get('has_current')}")
    print("percentiles:", meta.get("percentiles"))
    print(f"\nrows: {len(df)}  MINE {len(out['mine'])}  OPP {len(out['opp'])}  FA {len(out['fa'])}\n")
    cols = ["view", "owner", "name", "pos", "wopr_2025", "wopr_anchored", "wopr_posrank",
            "board_posrank", "rank_delta", "ppg_minus_xppg", "tags"]
    print("== MINE ==");  print(out["mine"][cols].to_string(index=False))
    print("\n== top OPP trade targets ==");  print(out["opp"][cols].head(12).to_string(index=False))
    print("\n== top FA waiver adds ==");  print(out["fa"][cols].head(12).to_string(index=False))
