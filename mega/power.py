"""Market power ratings: every unplayed game priced from the lines the market has posted.

Sportsbooks post lines a week or two ahead, so past that (weeks 8-17 in mid-October, the
fantasy playoffs included) every forward number used to see a league-average game: no
shootouts, no blowouts, no tough defenses. But the lines already posted say what the market
thinks of every team. Each posted game gives two implied team totals,

    home: (total + spread) / 2 = base + hfa/2 + off[home] - def[away]
    away: (total - spread) / 2 = base - hfa/2 + off[away] - def[home]

(nflverse's spread_line is positive when the home team is favored). Solving for each team's
offense and defense — recent games weighted more (HALF_LIFE weeks), every rating pulled
toward average (LAMBDA) — prices any future pairing. tools/fit_power.py picks both
constants and backtests the result against the closing lines that were later posted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = Path(__file__).resolve().parents[1] / "config" / "power_params.json"


def params() -> dict:
    try:
        return json.loads(PARAMS.read_text())
    except Exception:
        return {"half_life": 4.0, "lambda": 2.0}


def fit(games: pd.DataFrame, half_life: float | None = None, lam: float | None = None) -> dict:
    """Ratings from games with both lines. `games`: home_team, away_team, week, total_line,
    spread_line. Returns {"base", "hfa", "off": {team: x}, "def": {team: x}}."""
    p = params()
    half_life = p["half_life"] if half_life is None else half_life
    lam = p["lambda"] if lam is None else lam
    g = games.dropna(subset=["total_line", "spread_line"])
    if g.empty:
        return {}
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    # unknowns: base, hfa, off[0..n), def[0..n)
    rows, y, w = [], [], []
    last = float(g["week"].max())
    for h, a, wk, tot, sp in zip(g["home_team"], g["away_team"], g["week"], g["total_line"], g["spread_line"]):
        wt = 0.5 ** ((last - float(wk)) / half_life) if half_life and np.isfinite(half_life) else 1.0
        for side, other, sign, imp in ((h, a, 1.0, (tot + sp) / 2), (a, h, -1.0, (tot - sp) / 2)):
            r = np.zeros(2 + 2 * n)
            r[0], r[1] = 1.0, sign / 2
            r[2 + ix[side]] = 1.0
            r[2 + n + ix[other]] = -1.0
            rows.append(r)
            y.append(imp)
            w.append(wt)
    X, y, sw = np.array(rows), np.array(y), np.sqrt(np.array(w))
    # ridge on the team terms only: base and home field are fitted free
    reg = np.sqrt(lam) * np.eye(2 + 2 * n)[2:]
    A = np.vstack([X * sw[:, None], reg])
    b = np.concatenate([y * sw, np.zeros(2 * n)])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    return {"base": float(sol[0]), "hfa": float(sol[1]),
            "off": {t: float(sol[2 + i]) for t, i in ix.items()},
            "def": {t: float(sol[2 + n + i]) for t, i in ix.items()}}


def predict(m: dict, home: str, away: str) -> tuple[float, float] | None:
    """(total_line, spread_line) for a game, home perspective like nflverse's."""
    if not m or home not in m["off"] or away not in m["off"]:
        return None
    hi = m["base"] + m["hfa"] / 2 + m["off"][home] - m["def"][away]
    ai = m["base"] - m["hfa"] / 2 + m["off"][away] - m["def"][home]
    return hi + ai, hi - ai


def fill(sched: pd.DataFrame) -> pd.DataFrame:
    """The schedule with every regular-season game lacking a line priced from the ratings the
    posted ones imply; `line_source` says which ("posted" / "model")."""
    if sched is None or sched.empty or "total_line" not in sched.columns:
        return sched
    s = sched.copy()
    reg = s["game_type"].astype(str).str.upper().eq("REG") if "game_type" in s.columns else pd.Series(True, index=s.index)
    posted = s["total_line"].notna() & s["spread_line"].notna()
    s["line_source"] = np.where(posted, "posted", None)
    m = fit(s[reg & posted])
    for i in s.index[reg & ~posted]:
        p = predict(m, s.at[i, "home_team"], s.at[i, "away_team"])
        if p:
            s.at[i, "total_line"], s.at[i, "spread_line"] = round(p[0], 1), round(p[1], 1)
            s.at[i, "line_source"] = "model"
    return s
