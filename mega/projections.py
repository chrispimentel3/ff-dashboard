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
import pandas as pd

from .config import DATA
from .intel import _norm, form_season, ff_opportunity, weekly

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
def nflverse_estimate(season: int) -> pd.DataFrame:
    """A projected half-PPR/game for every player: blend recent form, season rate, xFP/g."""
    fs = form_season(season)
    w = weekly(fs)
    if w.empty:
        return pd.DataFrame(columns=["gsis_id", "norm", "player", "pos", "nfl_est"])
    w = w[w["pos"].isin(SKILL)]
    maxwk = int(w["week"].max())
    recent = w[w["week"] > maxwk - 3]

    est = (
        w.sort_values("week")
        .groupby("gsis_id", as_index=False)
        .agg(player=("player", "last"), pos=("pos", "last"),
             season_pg=("half_ppr", "mean"), gms=("week", "nunique"))
    )
    rp = recent.groupby("gsis_id", as_index=False)["half_ppr"].mean().rename(columns={"half_ppr": "recent_pg"})
    est = est.merge(rp, on="gsis_id", how="left")

    ffo = ff_opportunity(fs)
    if not ffo.empty and "half_ppr_exp" in ffo.columns:
        xp = ffo.groupby("gsis_id", as_index=False)["half_ppr_exp"].mean().rename(columns={"half_ppr_exp": "xfp_pg"})
        est = est.merge(xp, on="gsis_id", how="left")
    else:
        est["xfp_pg"] = pd.NA

    est["recent_pg"] = est["recent_pg"].fillna(est["season_pg"])
    est["xfp_pg"] = pd.to_numeric(est["xfp_pg"], errors="coerce").fillna(est["season_pg"])
    # weight recency + expected volume, anchor to season
    est["nfl_est"] = (0.45 * est["recent_pg"] + 0.30 * est["season_pg"] + 0.25 * est["xfp_pg"]).round(2)
    est["norm"] = est["player"].map(_norm)
    return est[["gsis_id", "norm", "player", "pos", "nfl_est", "gms", "recent_pg", "season_pg", "xfp_pg"]]


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
    # projection: prefer FantasyPros where present, else the nflverse estimate
    base["proj"] = base["fp_proj"].where(base["fp_proj"].notna(), base["nfl_est"])
    base["proj_source"] = base["fp_proj"].notna().map({True: "FantasyPros", False: "nflverse-est"})
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
