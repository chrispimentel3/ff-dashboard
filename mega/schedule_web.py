"""The next few games' matchups for every team x position — "what does the schedule do for
him from here" on the waiver and trade pages, where the question is rest of season rather
than this week's start/sit (mega/start_sit.py already answers that one).

Same fitted model as the lineup page (mega/matchup_model.week_matchups): each game's `pct`
is the percent change to a matchup-neutral baseline, defense and Vegas together. Averaging
the next `N_GAMES` games they actually play skips a bye rather than counting it as a game.

Two small tables so the site doesn't repeat the schedule for every player:

    table  "CHI|QB" -> {"pct": avg %, "g": [[week, "NYJ", pct], ...]}
    names  "Case Keenum" -> "CHI|QB"
"""
from __future__ import annotations

import pandas as pd

N_GAMES = 4
SKILL = ("QB", "RB", "WR", "TE")


def outlook(m: pd.DataFrame, n: int = N_GAMES) -> dict:
    """`m` is `week_matchups()`-shaped (team, pos, week, opp, pct)."""
    table: dict = {}
    if m is None or m.empty:
        return table
    for (team, pos), g in m.sort_values("week").groupby(["team", "pos"]):
        g = g.dropna(subset=["pct"]).head(n)
        if g.empty or pos not in SKILL:
            continue
        table[f"{team}|{pos}"] = {
            "pct": round(float(g["pct"].mean()), 1),
            "g": [[int(r.week), str(r.opp), round(float(r.pct), 1)] for r in g.itertuples()],
        }
    return table


def names(index: pd.DataFrame) -> dict:
    """`index` has name, pos, team — one entry per player the site can show."""
    out = {}
    for r in index.itertuples():
        if r.pos in SKILL and isinstance(r.team, str) and r.team:
            out.setdefault(str(r.name), f"{r.team}|{r.pos}")
    return out


def build(season: int, from_week: int, index: pd.DataFrame) -> dict:
    from .matchup_model import week_matchups

    try:
        table = outlook(week_matchups(season, from_week))
    except Exception:
        table = {}
    return {"available": bool(table), "season": season, "from_week": from_week,
            "games": N_GAMES, "table": table, "names": names(index) if table else {}}
