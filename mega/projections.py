"""Forward-looking projections.

Two sources, blended:
  • FantasyPros API (authoritative, but free tier caps at the top ~10 per position):
    weekly projected points (`points_half`), ECR, start/sit grade, opponent.
  • nflverse estimate (covers EVERY player): recent form blended with season rate,
    so deep-roster and waiver names still get a number.

`blended_week(season, week)` returns one row per player with a projected half-PPR
value and a `proj_source` flag. Joins are keyed on gsis_id via ff_playerids
(mfl_id / fantasypros_id / yahoo_id all map cleanly).
"""
from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

import httpx
import nflreadpy as nfl
import numpy as np
import pandas as pd

from .config import DATA
from .intel import _norm, ff_opportunity, weekly

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

FP_KEY = os.environ.get("FANTASYPROS_API_KEY", "")
FP_BASE = "https://api.fantasypros.com/public/v2/json/nfl"
CACHE = DATA / "fp_cache"
CACHE.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 6 * 3600  # seconds

SKILL = ("QB", "RB", "WR", "TE")


# ---------------------------------------------------------------- id map
@functools.lru_cache(maxsize=1)
def _idmap() -> pd.DataFrame:
    ids = nfl.load_ff_playerids().to_pandas()
    for c in ("mfl_id", "fantasypros_id", "yahoo_id"):
        if c in ids.columns:
            ids[c] = pd.to_numeric(ids[c], errors="coerce")
    keep = [c for c in ["mfl_id", "fantasypros_id", "yahoo_id", "gsis_id", "name", "position"] if c in ids.columns]
    return ids[keep].copy()


def _to_gsis_via(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    ids = _idmap()[[right, "gsis_id"]].dropna().drop_duplicates(right)
    df = df.copy()
    df[left] = pd.to_numeric(df[left], errors="coerce")
    return df.merge(ids, left_on=left, right_on=right, how="left").drop(columns=[right])


# ---------------------------------------------------------------- FantasyPros
def _fp_get(path: str, params: dict) -> dict | None:
    if not FP_KEY:
        return None
    key = path.replace("/", "_") + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    cache_file = CACHE / f"{key}.json"
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_TTL:
        return json.loads(cache_file.read_text())
    try:
        r = httpx.get(f"{FP_BASE}/{path}", headers={"x-api-key": FP_KEY}, params=params, timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        cache_file.write_text(json.dumps(j))
        return j
    except Exception:
        return None


def fp_projections(season: int, week: int, positions=SKILL) -> pd.DataFrame:
    rows = []
    for pos in positions:
        j = _fp_get(f"{season}/projections", {"position": pos, "scoring": "HALF", "week": str(week)})
        for p in (j or {}).get("players", []):
            s = p.get("stats", {}) or {}
            rows.append(dict(
                mflid=p.get("mflid"), fpid=p.get("fpid"), name=p.get("name"),
                pos=p.get("position_id"), team=p.get("team_id"),
                fp_proj=s.get("points_half"),
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = _to_gsis_via(df, "mflid", "mfl_id")
    df["norm"] = df["name"].map(_norm)
    return df


def fp_ros(season: int, positions=SKILL) -> pd.DataFrame:
    """FantasyPros rest-of-season projections, half-PPR **points per game**.

    `week="ros"` is the forward-looking number: the weeks already played are excluded, which
    is exactly what a trade is decided on — points already banked are sunk. Completed weeks
    are nflverse's job, and they reach this module through `nflverse_estimate`, which is what
    prices everyone FantasyPros doesn't cover.

    The free API tier caps each position at its top 10, so expect about 40 players. The
    result carries `fp_count` on .attrs: how many FantasyPros actually has, against how many
    it returned, so a caller can tell a thin answer from a complete one.
    """
    rows, avail = [], {}
    for pos in positions:
        j = _fp_get(f"{season}/projections", {"position": pos, "scoring": "HALF", "week": "ros"})
        if not j:
            continue
        avail[pos] = (len(j.get("players") or []), j.get("count"))
        for pl in j.get("players", []):
            pts = (pl.get("stats") or {}).get("points_half")
            if pts is None:
                continue
            rows.append(dict(mflid=pl.get("mflid"), name=pl.get("name"),
                             pos=pl.get("position_id"), team=pl.get("team_id"),
                             fp_ros_pg=float(pts)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = _to_gsis_via(df, "mflid", "mfl_id")
    df["norm"] = df["name"].map(_norm)
    df.attrs["fp_count"] = avail
    return df


def fp_rankings(season: int, week: int, positions=SKILL, rank_type: str = "weekly") -> pd.DataFrame:
    rows = []
    for pos in positions:
        params = {"position": pos, "scoring": "HALF", "type": rank_type}
        if rank_type == "weekly":
            params["week"] = str(week)
        j = _fp_get(f"{season}/consensus-rankings", params)
        for p in (j or {}).get("players", []):
            rows.append(dict(
                yahoo_id=p.get("player_yahoo_id"), name=p.get("player_name"),
                pos=p.get("player_position_id"), team=p.get("player_team_id"),
                ecr=p.get("rank_ecr"), pos_rank=p.get("pos_rank"),
                start_sit=p.get("start_sit_grade"), opponent=p.get("player_opponent"),
                fp_pts=pd.to_numeric(p.get("r2p_pts"), errors="coerce"),
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = _to_gsis_via(df, "yahoo_id", "yahoo_id")
    df["norm"] = df["name"].map(_norm)
    return df


# ---------------------------------------------------------------- nflverse estimate
# How many games last season is worth when anchoring this season's rate. One flat number
# (this was 3, from WOPR handoff D7) is wrong in both directions, because positions differ
# enormously in how noisy a week is relative to how far apart the players actually are.
#
# Measured three ways on 2025 — split-half reliability, a direct within/between variance
# decomposition, and a holdout that predicts each player's second-half mean from his first
# half shrunk toward a prior. The holdout is the one that matters, since it scores
# prediction rather than fit, and these are its minima:
#
#     pos   split-half   direct   holdout
#     QB         10.7       7.2      5-8      weekly QB scores swing 5-40 but every QB
#     RB          1.2       1.2        1      lives in a 15-22 band, so a hot two-game
#     TE          3.6       1.9        3      start says almost nothing
#     WR          1.2       1.7        3
#
# A flat 3 let two games of Bryce Young (28.3/g against a 14.8/g 2025) price him at 20.0 —
# above every rostered QB in the league, which then set replacement level for the whole
# valuation. Rerun tools/calibrate_shrinkage.py against a new season to refresh these.
PRIOR_GAMES_BY_POS = {"QB": 6.0, "RB": 1.5, "WR": 2.5, "TE": 3.0}
PRIOR_GAMES = 3   # fallback for K/DEF and anything unmapped


def _season_rates(season: int) -> pd.DataFrame:
    """Per-player half-PPR rates for one regular season: season, last 3 games played, xFP."""
    w = weekly(season)
    if w.empty:
        return pd.DataFrame(columns=["gsis_id", "player", "pos", "gms", "season_pg", "recent_pg", "xfp_pg", "est"])
    if "season_type" in w.columns:
        w = w[w["season_type"].astype(str).str.upper() == "REG"]
    w = w[w["pos"].isin(SKILL)].sort_values("week")
    r = w.groupby("gsis_id", as_index=False).agg(
        player=("player", "last"), pos=("pos", "last"),
        season_pg=("half_ppr", "mean"), gms=("week", "nunique"))
    # last 3 games he actually played, not the last 3 calendar weeks (byes aren't zeros)
    rec = w.groupby("gsis_id")["half_ppr"].apply(lambda x: x.tail(3).mean()).rename("recent_pg")
    r = r.merge(rec, on="gsis_id", how="left")
    ffo = ff_opportunity(season)
    if not ffo.empty and "half_ppr_exp" in ffo.columns:
        xp = ffo.groupby("gsis_id")["half_ppr_exp"].mean().rename("xfp_pg")
        r = r.merge(xp, on="gsis_id", how="left")
    else:
        r["xfp_pg"] = pd.NA
    r["xfp_pg"] = pd.to_numeric(r["xfp_pg"], errors="coerce").fillna(r["season_pg"])
    r["est"] = 0.45 * r["recent_pg"] + 0.30 * r["season_pg"] + 0.25 * r["xfp_pg"]
    return r


PRIOR_MIN_GAMES = 6   # who counts toward a position's median prior (WOPR handoff pool)


def _prior(season: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Last season as a prior: per player, and the position medians rookies borrow.

    Whole-season rates, half points and half xFP. Not the recency-weighted `est`: a
    prior built from the last three games of last season carried Emeka Egbuka's
    3.1-pts/g finish (on a 9.7 season, 11.5 xFP/g) into this year.
    """
    p = _season_rates(season)
    p["prior"] = 0.5 * p["season_pg"] + 0.5 * p["xfp_pg"]
    pool = p[p["gms"] >= PRIOR_MIN_GAMES]
    return p, pool.groupby("pos")["prior"].median().to_dict()


def nflverse_estimate(season: int) -> pd.DataFrame:
    """A projected half-PPR/game for every player who has played in either season.

    This season's rate anchored to a prior worth PRIOR_GAMES games (WOPR handoff D7):
    the player's own last season, or for a rookie the position median. So a one-game
    sample moves the number without owning it — Denzel Boston's one-TD debut (12.9 pts on
    6.4 xFP) had projected him over Egbuka when rookies went unanchored.
    """
    cur = _season_rates(season)
    prev, pos_median = _prior(season - 1)
    est = cur.merge(prev[["gsis_id", "player", "pos", "prior", "gms"]], on="gsis_id", how="outer",
                    suffixes=("", "_prev"))
    if est.empty:
        return pd.DataFrame(columns=["gsis_id", "norm", "player", "pos", "nfl_est", "gms", "recent_pg",
                                     "season_pg", "xfp_pg", "basis"])
    for c in ("player", "pos"):
        est[c] = est[c].fillna(est[f"{c}_prev"])
    has_cur, has_prev = est["est"].notna(), est["prior"].notna()
    prior = est["prior"].where(has_prev, est["pos"].map(pos_median))
    g = pd.to_numeric(est["gms"], errors="coerce").fillna(0)
    k = est["pos"].map(PRIOR_GAMES_BY_POS).fillna(PRIOR_GAMES)
    anchored = (g * est["est"].fillna(0) + k * prior.fillna(0)) / (g + k)
    est["nfl_est"] = anchored.where(has_cur & prior.notna(), est["est"].where(has_cur, est["prior"])).round(2)
    est["gms"] = g.where(has_cur, est["gms_prev"])
    est["basis"] = f"{season - 1} only"
    est.loc[has_cur & has_prev, "basis"] = f"{season} + {season - 1} prior"
    est.loc[has_cur & ~has_prev, "basis"] = f"{season} + position median"
    est["norm"] = est["player"].map(_norm)
    return est[["gsis_id", "norm", "player", "pos", "nfl_est", "gms", "recent_pg", "season_pg", "xfp_pg", "basis"]]


# ---------------------------------------------------------------- blend
def blended_week(season: int, week: int) -> pd.DataFrame:
    est = nflverse_estimate(season)
    fp = fp_projections(season, week)
    fr = fp_rankings(season, week)

    base = est.copy()
    if not fp.empty:
        base = base.merge(fp[["gsis_id", "fp_proj"]].dropna(subset=["gsis_id"]), on="gsis_id", how="outer")
    if not fr.empty:
        base = base.merge(
            fr[["gsis_id", "ecr", "pos_rank", "start_sit", "opponent"]].dropna(subset=["gsis_id"]),
            on="gsis_id", how="left",
        )
    # guarantee the overlay columns exist even when a source returned nothing this week
    for c in ("fp_proj", "ecr", "pos_rank", "start_sit", "opponent"):
        if c not in base.columns:
            base[c] = pd.NA
    if "nfl_est" not in base.columns:
        base["nfl_est"] = pd.NA
    # HANDOFF §3.1 — ffanalytics' robust average across six sources is the preferred
    # projection: a consensus is harder to fool than any one site, and `sd_pts` says how
    # much the sources disagree. It only exists where R has run, so the order below is a
    # preference, not a requirement.
    try:
        from . import ffa

        ffp = ffa.projections(season, week)
    except Exception:
        ffp = pd.DataFrame()
    if not ffp.empty:
        cols = [c for c in ("gsis_id", "proj", "sd_pts", "floor", "ceiling") if c in ffp.columns]
        base = base.merge(ffp[cols].rename(columns={"proj": "ffa_proj"}).dropna(subset=["gsis_id"]),
                          on="gsis_id", how="outer")
    for c in ("ffa_proj", "sd_pts", "floor", "ceiling"):
        if c not in base.columns:
            base[c] = pd.NA

    # ffanalytics -> FantasyPros -> the nflverse usage estimate, first one present wins
    base["proj"] = base["ffa_proj"].where(
        base["ffa_proj"].notna(),
        base["fp_proj"].where(base["fp_proj"].notna(), base["nfl_est"]))
    base["proj_source"] = np.where(
        base["ffa_proj"].notna(), "ffanalytics",
        np.where(base["fp_proj"].notna(), "FantasyPros", "nflverse-est"))

    # §6.2 — rest-of-season ECR is what a league-mate actually sees. Where ffanalytics
    # supplied one it beats the weekly rank, which measures a different question.
    try:
        ros = ffa.ecr_ros(season)
        if not ros.empty:
            base = base.merge(ros, on="gsis_id", how="left")
            base["ecr"] = base["ecr_ros"].where(base["ecr_ros"].notna(), base["ecr"])
    except Exception:
        pass

    base = base.dropna(subset=["proj"]).sort_values("proj", ascending=False).reset_index(drop=True)
    return base


if __name__ == "__main__":
    import sys

    s, wk = int(sys.argv[1]) if len(sys.argv) > 1 else int(nfl.get_current_season()), \
        int(sys.argv[2]) if len(sys.argv) > 2 else int(nfl.get_current_week())
    print(f"season {s} week {wk} | FP key: {'set' if FP_KEY else 'MISSING'}")
    b = blended_week(s, wk)
    print("rows:", len(b), "| with FP:", (b["proj_source"] == "FantasyPros").sum())
    print(b.head(20)[["player", "pos", "proj", "proj_source", "ecr", "start_sit", "opponent"]].to_string())
