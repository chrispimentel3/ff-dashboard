"""Opponent-adjusted, Vegas-combined matchup multiplier per team x position x week.

Sept 2026: this is the model behind "how much does this matchup help or hurt him,"
replacing the plain points-allowed ratio in `mega.lineup.defense_vs_position` with
one fit on ten years of nflverse outcomes (2016-2025, ~26k player-games) via a
Poisson GLM — see data/matchup_constants.json for the fitted k/rho/exponents. The
old ratio over/under-reacted: a defense that happened to face a run of banged-up
WR2s "allowed" fewer points and looked tougher than it actually was, and every
point of that noise moved the rating 1:1. This version:

  - judges a defense against WHO it faced (points allowed vs those players' own
    matchup-neutral baseline, not raw totals) — `_eligible_actuals()`,
  - shrinks the in-season sample toward last season's rating, fading last year's
    weight in as this year's own sample grows. How fast differs by position: a
    run defense regresses hard year to year (rho 0.30, so last year barely
    matters), while pass defense carries over a lot (rho 0.85) but the in-season
    WR sample is close to pure noise until deep in the year (k=45, vs k=6 for TE)
    — `defense_ratings()`,
  - layers in the Vegas implied team total for the week in question, relative to
    that team's own season-average line — forward-looking, and prices things
    (injuries, weather, a backup QB) a defense rating can't see,
  - and combines the two with fitted exponents rather than a straight multiply,
    because a defense that "allowed" 30% more than expected doesn't actually move
    a QB's expected points by 30% — most of that 30% was noise, and the exponent
    (0.288 for QB, `b_def`) is the fraction that's real signal — `week_matchups()`.

Treat anything inside about +-3% as neutral: the out-of-sample check in the
handoff this model shipped with only sorted reliably at the extremes.
"""
from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd

from .config import DATA
from .intel import weekly
from .projections import nflverse_estimate

POS = ["QB", "RB", "WR", "TE"]
MINB = {"QB": 12, "RB": 6, "WR": 6, "TE": 4}
CONSTANTS_PATH = DATA / "matchup_constants.json"


@functools.lru_cache(maxsize=1)
def constants() -> dict:
    return json.loads(CONSTANTS_PATH.read_text())


@functools.lru_cache(maxsize=1)
def _teams() -> list[str]:
    return sorted({k.split("|")[0] for k in constants()["_prior_def"]})


@functools.lru_cache(maxsize=8)
def team_games(season: int) -> pd.DataFrame:
    """One row per team per regular-season game: Vegas implied total, relative to
    that team's own season-average implied total (`impl_rel`), and whether the
    game's been played. `impl_rel` is what matters, not the raw number — a
    player's baseline already reflects his usual environment, so the question is
    only whether this week is better or worse than normal for his offense."""
    import nflreadpy as nfl

    g = nfl.load_schedules(seasons=[season]).to_pandas()
    g = g[g["game_type"] == "REG"]
    home = g.assign(team=g["home_team"], opp=g["away_team"],
                     implied=(g["total_line"] + g["spread_line"]) / 2)
    away = g.assign(team=g["away_team"], opp=g["home_team"],
                     implied=(g["total_line"] - g["spread_line"]) / 2)
    tg = pd.concat([home, away])[["game_id", "week", "team", "opp", "implied", "away_score"]]
    tg["impl_rel"] = tg["implied"] / tg.groupby("team")["implied"].transform("mean")
    tg["played"] = tg["away_score"].notna()
    return tg


def _eligible_actuals(season: int, through_week: int) -> pd.DataFrame:
    """Actual half-PPR scored against each defense by position, weeks strictly
    before `through_week` only (no look-ahead) — the A (actual) / B (baseline)
    inputs to the shrinkage formula. Baseline reuses `nflverse_estimate` rather
    than a separate leave-one-out calc of its own: one baseline number for a
    player everywhere in the dashboard beats a second, slightly different one
    that only this model sees."""
    w = weekly(season)
    if w.empty or "opponent_team" not in w.columns:
        return pd.DataFrame(columns=["opponent_team", "pos", "pts", "base"])
    w = w[(w["week"] < through_week) & w["pos"].isin(POS)].copy()
    est = nflverse_estimate(season)[["gsis_id", "nfl_est"]]
    w = w.merge(est, on="gsis_id", how="left").dropna(subset=["nfl_est"])
    w = w[w["nfl_est"] >= w["pos"].map(MINB)]
    return w.rename(columns={"half_ppr": "pts", "nfl_est": "base"})[["opponent_team", "pos", "pts", "base"]]


def _defense_ratings_core(agg: pd.DataFrame, C: dict, teams: list[str]) -> pd.DataFrame:
    """Pure shrinkage math, split out from `defense_ratings()` so it's testable on
    synthetic data with no network fetch. `agg` is indexed by (opponent_team, pos)
    with columns A (actual pts allowed, summed) and B (those players' baselines,
    summed) — see `_eligible_actuals`."""
    rows = []
    for pos in POS:
        c = C[pos]
        prior = pd.Series({t: C["_prior_def"].get(f"{t}|{pos}", 1.0) for t in teams})
        prior = prior / prior.mean()
        a = agg.xs(pos, level="pos") if pos in agg.index.get_level_values("pos") else pd.DataFrame()
        a = a.reindex(teams).fillna(0.0) if not a.empty else pd.DataFrame({"A": 0.0, "B": 0.0}, index=teams)
        m = 1 + c["rho"] * (prior - 1)
        D = (a["A"] + c["k"] * c["gbar"] * m) / (a["B"] + c["k"] * c["gbar"])
        D = D / D.mean()
        w_season = a["B"] / (a["B"] + c["k"] * c["gbar"])
        for t in teams:
            rows.append({
                "defense": t, "pos": pos, "rating": round(float(D[t]), 3),
                "pct_vs_avg": round((float(D[t]) - 1) * 100, 1),
                "weight_this_season": round(float(w_season[t]), 2),
            })
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("pos")["rating"].rank(ascending=False, method="min").astype(int)
    return out


def defense_ratings(season: int, through_week: int) -> pd.DataFrame:
    """defense, pos, rating (1.00 = league average, higher = allows more / tougher
    to defend against), pct_vs_avg, rank (1 = toughest matchup), weight_this_season
    (how much of `rating` is this season's own sample vs. last year's prior —
    small early, so a hot week-2 defense doesn't look elite yet)."""
    e = _eligible_actuals(season, through_week)
    agg = e.groupby(["opponent_team", "pos"]).agg(A=("pts", "sum"), B=("base", "sum"))
    return _defense_ratings_core(agg, constants(), _teams())


def _week_matchups_core(D: pd.DataFrame, tg: pd.DataFrame, C: dict, from_week: int) -> pd.DataFrame:
    """Pure combine step, split out from `week_matchups()` for the same reason as
    `_defense_ratings_core`. `D` is a `defense_ratings()`-shaped frame, `tg` a
    `team_games()`-shaped frame."""
    fut = tg[(tg["week"] >= from_week) & ~tg["played"]]
    rows = []
    for r in fut.itertuples():
        for pos in POS:
            c = C[pos]
            dm = D[(D["defense"] == r.opp) & (D["pos"] == pos)]
            if dm.empty:
                continue
            rating = float(dm["rating"].iloc[0])
            def_component = rating ** c["b_def"]
            if pd.notna(r.impl_rel):
                vegas_component = float(r.impl_rel) ** c["b_impl"]
                mult, basis = def_component * vegas_component, "defense+vegas"
                vegas_pct = round((vegas_component - 1) * 100, 1)
            else:
                mult, basis = rating ** c["b_def_only"], "defense_only"
                vegas_pct = None
            rows.append({
                "week": int(r.week), "team": r.team, "opp": r.opp, "pos": pos,
                "mult": round(mult, 3), "pct": round((mult - 1) * 100, 1),
                "def_rank": int(dm["rank"].iloc[0]), "def_pct": round((rating - 1) * 100, 1),
                "vegas_pct": vegas_pct, "basis": basis,
            })
    return pd.DataFrame(rows)


def week_matchups(season: int, from_week: int) -> pd.DataFrame:
    """team, opp, pos, week, mult, pct, def_pct, def_rank, vegas_pct, basis — every
    remaining scheduled game x eligible position from `from_week` on.

    `mult` multiplies a matchup-neutral baseline (nflverse_estimate); `pct` is
    `(mult - 1) * 100`, the number the UI shows. `def_pct`/`vegas_pct` are the two
    components broken back out (both also as a %-change-to-baseline) so a caller
    can say *why* — "defense +6%, trending toward a shootout on the Vegas line
    +9%" reads as a reason, "+15%" alone doesn't. `basis` is "defense+vegas" once
    a line exists for that game, else "defense_only" (a week Vegas hasn't priced
    yet — no line is worse than reaching for a stale one)."""
    D = defense_ratings(season, from_week)
    tg = team_games(season)
    return _week_matchups_core(D, tg, constants(), from_week)
