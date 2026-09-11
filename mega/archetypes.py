"""League-winner archetype scoring, prorated per game.

Blueprint (from the Mega Bowl draft War Room):
  QB : 55+ rush att/season (aim 100+) — the rushing/"Konami" QB.   RUSH @55, RUSH+ @100.
  RB : early ADP (R1-3), young — target years 1-3, avg winner age 25, avoid 27+.
  WR : sticky first-down producer (FD/route 12%+), breakout window (exp yrs 3-6), avoid age 32+.
  TE : alpha on a WR-thin, high-scoring offense — no WR in top-60 ADP, 20+ team ppg, big target share.

Everything is expressed PER GAME. Season thresholds are divided by a 17-game slate
(55 rush att -> 3.24/g, 100 -> 5.88/g), and War Room season projections (pts, carries)
are prorated to per-game. Where the exact stat isn't in free data (routes run), a clearly
labeled proxy is used (first-down rate + first downs per game).
"""
from __future__ import annotations

import ast
import functools
import json
import re
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from .config import ROOT
from .draft_board import load_draft
from .intel import _norm, form_season, weekly

GAMES = 17
WARROOM_HTML = ROOT / "draftwarroom2026.html"

# per-game thresholds derived from the blueprint
QB_RUSH_PG = 55 / GAMES      # 3.24  -> RUSH tag
QB_RUSHPLUS_PG = 100 / GAMES  # 5.88  -> RUSH+ tag
RB_WINNER_AGE = 25
RB_AVOID_AGE = 27
WR_AGE_CAP = 32
WR_WINDOW = (3, 6)            # breakout window, years of experience
WR_AGE_EXCEPTIONS = {"mike evans", "davante adams"}
TE_TOP_ADP = 60              # a team WR inside top-60 ADP breaks the TE archetype
TE_OFFENSE_PPG = 20
TE_KELCE = {"travis kelce"}


# ---------------------------------------------------------------- War Room projections
@functools.lru_cache(maxsize=1)
def warroom_projections() -> pd.DataFrame:
    """Parse the `const PLAYERS = [...]` projection set from the War Room HTML."""
    if not WARROOM_HTML.exists():
        return pd.DataFrame()
    html = WARROOM_HTML.read_text(encoding="utf-8")
    m = re.search(r"const\s+PLAYERS\s*=\s*(\[.*?\]);", html, re.S)
    if not m:
        return pd.DataFrame()
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        data = ast.literal_eval(m.group(1))
    df = pd.DataFrame(data)
    df["norm"] = df["name"].map(_norm)
    df["proj_ppg"] = pd.to_numeric(df["pts"], errors="coerce") / GAMES
    df["proj_carries_pg"] = pd.to_numeric(df["carries"], errors="coerce") / GAMES
    df["vor"] = pd.to_numeric(df["vor"], errors="coerce")
    df["vor_pg"] = df["vor"] / GAMES
    return df[["norm", "name", "pos", "team", "posrank", "bye", "proj_ppg", "proj_carries_pg", "vor", "vor_pg"]]


# ---------------------------------------------------------------- player context
@functools.lru_cache(maxsize=2)
def _ages() -> pd.DataFrame:
    ids = nfl.load_ff_playerids().to_pandas()
    ids["age"] = pd.to_numeric(ids.get("age"), errors="coerce")
    ids["draft_year"] = pd.to_numeric(ids.get("draft_year"), errors="coerce")
    ids["norm"] = ids["name"].map(_norm)
    return ids[["norm", "age", "draft_year", "position"]].drop_duplicates("norm")


@functools.lru_cache(maxsize=2)
def _team_ppg(season: int) -> pd.Series:
    """Points scored per game by each NFL team (season-to-date, prior-year fallback)."""
    fs = form_season(season)
    sched = nfl.load_schedules(seasons=[fs]).to_pandas()
    sched = sched.dropna(subset=["home_score", "away_score"])
    rows = []
    for _, g in sched.iterrows():
        rows.append((g["home_team"], g["home_score"]))
        rows.append((g["away_team"], g["away_score"]))
    pf = pd.DataFrame(rows, columns=["team", "pts"])
    return pf.groupby("team")["pts"].mean()


def _per_game(season: int) -> pd.DataFrame:
    """Per-game production for every skill player (season-to-date / fallback)."""
    fs = form_season(season)
    w = weekly(fs)
    if w.empty:
        return pd.DataFrame()
    w = w[w["pos"].isin(["QB", "RB", "WR", "TE"])]
    g = w.groupby("gsis_id", as_index=False).agg(
        player=("player", "last"), pos=("pos", "last"), team=("team", "last"),
        gms=("week", "nunique"),
        half_ppr_pg=("half_ppr", "mean"),
        carries_pg=("carries", "mean") if "carries" in w.columns else ("half_ppr", "size"),
        targets_pg=("targets", "mean") if "targets" in w.columns else ("half_ppr", "size"),
        tgt_share=("target_share", "mean") if "target_share" in w.columns else ("half_ppr", "size"),
        rec_fd_pg=("receiving_first_downs", "mean") if "receiving_first_downs" in w.columns else ("half_ppr", "size"),
        rush_fd_pg=("rushing_first_downs", "mean") if "rushing_first_downs" in w.columns else ("half_ppr", "size"),
    )
    # first-down rate proxy (FD per target) for the WR "sticky" signal
    if "receiving_first_downs" in w.columns and "targets" in w.columns:
        tot = w.groupby("gsis_id").agg(fd=("receiving_first_downs", "sum"), tg=("targets", "sum"))
        tot["fd_rate"] = tot["fd"] / tot["tg"].replace(0, np.nan)
        g = g.merge(tot["fd_rate"].reset_index(), on="gsis_id", how="left")
    else:
        g["fd_rate"] = np.nan
    g["norm"] = g["player"].map(_norm)
    return g


# ---------------------------------------------------------------- scoring
def _clip01(x) -> float:
    try:
        return float(min(max(x, 0.0), 1.0))
    except Exception:
        return 0.0


def score(season: int) -> pd.DataFrame:
    pg = _per_game(season)
    if pg.empty:
        return pd.DataFrame()
    wr = warroom_projections()
    ages = _ages()
    ppg_team = _team_ppg(season)
    draft = load_draft().assign(norm=lambda d: d["player"].map(_norm))
    # earliest WR ADP on each NFL team (from draft board overall pick) — for TE archetype
    wr_adp = draft[draft["pos"] == "WR"].groupby("nfl_team")["overall"].min().rename("team_wr_top_adp")

    df = pg.merge(ages[["norm", "age", "draft_year"]], on="norm", how="left")
    if not wr.empty:
        df = df.merge(wr[["norm", "proj_ppg", "proj_carries_pg", "vor"]], on="norm", how="left")
    for c in ("proj_ppg", "proj_carries_pg", "vor"):
        if c not in df.columns:
            df[c] = np.nan
    df["exp_yrs"] = (season - df["draft_year"]).clip(lower=0)
    df["team_ppg"] = df["team"].map(ppg_team)
    df["team_wr_top_adp"] = df["team"].map(wr_adp)

    # carries per game: prefer live pace, fall back to War Room projection
    df["carries_pg_use"] = df["carries_pg"].where(df["carries_pg"] > 0, df["proj_carries_pg"])

    rows = []
    for _, r in df.iterrows():
        pos = r["pos"]
        fit, tags, why = 0.0, [], []

        if pos == "QB":
            rush = r["carries_pg_use"] or 0
            rush_score = _clip01(rush / QB_RUSHPLUS_PG)
            proj_score = _clip01((r.get("proj_ppg") or r["half_ppr_pg"]) / 24)
            fit = 100 * (0.8 * rush_score + 0.2 * proj_score)
            if rush >= QB_RUSHPLUS_PG:
                tags.append("RUSH+")
            elif rush >= QB_RUSH_PG:
                tags.append("RUSH")
            why.append(f"{rush:.1f} rush att/g (blueprint 3.2+/g, elite 5.9+/g)")

        elif pos == "RB":
            age = r["age"]
            if pd.isna(age):
                age_score = 0.55
            elif age <= RB_WINNER_AGE:
                age_score = 1.0
            elif age < RB_AVOID_AGE:
                age_score = 0.7
            elif age < 28:
                age_score = 0.4
            else:
                age_score = 0.12
            vol = r["carries_pg_use"] or 0
            vol_score = _clip01(vol / 18)  # bellcow pace ~18+/g
            adp = draft[draft["norm"] == r["norm"]]["overall"]
            adp_v = int(adp.iloc[0]) if len(adp) else 999
            cap_score = 1.0 if adp_v <= 36 else 0.6 if adp_v <= 72 else 0.3
            fit = 100 * (0.40 * vol_score + 0.35 * age_score + 0.25 * cap_score)
            why.append(f"{vol:.1f} carries/g" + (f", age {age:.0f}" if pd.notna(age) else ""))
            if pd.notna(age) and age >= RB_AVOID_AGE:
                tags.append("AGE⚠")
            if adp_v <= 36:
                tags.append("R1-3 ADP")

        elif pos == "WR":
            fd_rate = r["fd_rate"]
            fd_pg = r["rec_fd_pg"] or 0
            sticky = 0.5 * _clip01((fd_rate or 0) / 0.45) + 0.5 * _clip01(fd_pg / 5.0)
            exp = r["exp_yrs"]
            if pd.isna(exp):
                win = 0.5
            elif WR_WINDOW[0] <= exp <= WR_WINDOW[1]:
                win = 1.0
            elif exp == 2:
                win = 0.72
            elif exp <= 1:
                win = 0.55
            elif exp <= 8:
                win = 0.6
            else:
                win = 0.4
            age = r["age"]
            age_gate = 1.0
            if pd.notna(age) and age >= WR_AGE_CAP and r["norm"] not in WR_AGE_EXCEPTIONS:
                age_gate = 0.3
                tags.append("AGE⚠")
            fit = 100 * (0.45 * sticky + 0.30 * win + 0.25 * age_gate)
            fr_txt = f"{fd_rate:.0%} FD/tgt" if pd.notna(fd_rate) else "FD n/a"
            why.append(f"{fr_txt}, {fd_pg:.1f} FD/g (proxy), exp {exp:.0f}y" if pd.notna(exp) else fr_txt)

        elif pos == "TE":
            ts = r["tgt_share"] or 0
            ts_score = _clip01(ts / 0.25)
            off = r["team_ppg"]
            off_score = _clip01((off if pd.notna(off) else 20) / 26)
            top_adp = r["team_wr_top_adp"]
            wr_thin = 1.0 if (pd.isna(top_adp) or top_adp > TE_TOP_ADP) else 0.4
            fit = 100 * (0.40 * ts_score + 0.30 * off_score + 0.30 * wr_thin)
            if r["norm"] in TE_KELCE:
                fit = max(fit, 70.0)
                tags.append("KELCE-exc")
            why.append(f"{ts:.0%} tgt share, team {off:.0f} ppg" if pd.notna(off) else f"{ts:.0%} tgt share")
            if wr_thin == 1.0:
                tags.append("WR-thin O")

        rows.append(dict(
            player=r["player"], pos=pos, team=r["team"], gms=int(r["gms"]),
            arch_fit=round(fit, 1), tags=" ".join(tags),
            half_ppr_pg=round(r["half_ppr_pg"], 1),
            proj_ppg=round(r["proj_ppg"], 1) if pd.notna(r["proj_ppg"]) else np.nan,
            vor=r["vor"], age=r["age"], exp_yrs=r["exp_yrs"],
            carries_pg=round(r["carries_pg_use"], 1) if pd.notna(r["carries_pg_use"]) else np.nan,
            tgt_share=round(r["tgt_share"], 3) if pd.notna(r["tgt_share"]) else np.nan,
            why="; ".join(why), norm=r["norm"],
        ))
    return pd.DataFrame(rows).sort_values("arch_fit", ascending=False).reset_index(drop=True)


def target_board(season: int, rostered_norms: set[str] | None = None, top: int = 15) -> dict[str, pd.DataFrame]:
    """Best archetype fits per position, split into 'available' vs 'rostered'."""
    s = score(season)
    out = {}
    for pos in ("QB", "RB", "WR", "TE"):
        p = s[s["pos"] == pos].copy()
        if rostered_norms is not None:
            p["status"] = p["norm"].apply(lambda n: "rostered" if n in rostered_norms else "available")
        out[pos] = p.head(top)
    return out


if __name__ == "__main__":
    import sys
    s = int(sys.argv[1]) if len(sys.argv) > 1 else int(nfl.get_current_season())
    print(f"season {s} (form {form_season(s)})  |  War Room rows: {len(warroom_projections())}\n")
    board = score(s)
    for pos in ("QB", "RB", "WR", "TE"):
        print(f"== {pos} — top archetype fits ==")
        cols = ["player", "team", "arch_fit", "tags", "carries_pg", "tgt_share", "age", "exp_yrs", "proj_ppg", "why"]
        print(board[board["pos"] == pos].head(8)[cols].to_string(), "\n")
