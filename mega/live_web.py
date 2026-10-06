"""What the site's live scoreboard needs to score a week as it's played: every team's
starters, which NFL team each plays for, a projection for each, and who plays whom.

There is no Yahoo API to read live scores from (and no Yahoo session on Vercel), so
mega-bowl-web scores the week itself: its /api/live route reads ESPN's public NFL box
scores and scores each starter below with the league's half-PPR rules
(src/lib/liveScoring.ts). This file is the half that needs the Python side — current team
names (mega/teams.py), Yahoo's lineup slots, and player-to-NFL-team matching — written once
each morning to data/web/live_setup.json. Lineups are as of that morning's Yahoo scrape;
a change made after it doesn't show until the next one.

`key` is how a player is matched to an ESPN box-score name, and must stay identical to
liveKey() in src/lib/liveScoring.ts.
"""
from __future__ import annotations

import re

import pandas as pd

from . import teams
from .config import DATA

BENCH = {"BN", "IR", "IR+", "NA"}
# league-average weeks, only for a K or DEF the rankings rows (mega/kdef.py) don't project
K_PROJ, DEF_PROJ = 8.0, 7.0
# nflverse -> ESPN team abbreviations, where they differ
ESPN_ABBR = {"WAS": "WSH", "LA": "LAR"}
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def key(name: str) -> str:
    s = str(name).lower().replace(".", "").replace("'", "").replace("’", "").replace("-", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _nicknames() -> dict[str, str]:
    """'Vikings', 'Minnesota' or 'Minnesota Vikings' -> 'MIN' (ESPN's abbreviation), for
    Yahoo's DEF rows, which have used both the nickname and the city. A city two teams
    share (New York, Los Angeles) isn't mapped."""
    t = pd.read_csv(DATA / "nfl_teamlogos.csv")
    out, cities = {}, {}
    for full, code in zip(t["team"], t["team_code"]):
        abbr = ESPN_ABBR.get(code, code)
        words = str(full).split()
        out[key(words[-1])] = abbr
        out[key(full)] = abbr
        cities.setdefault(key(" ".join(words[:-1])), set()).add(abbr)
    for city, abbrs in cities.items():
        if len(abbrs) == 1:
            out.setdefault(city, next(iter(abbrs)))
    return out


def build(season: int, week: int, rosters: pd.DataFrame, fixtures: pd.DataFrame,
          rankings_rows: list[dict], nfl_rosters: pd.DataFrame | None = None,
          my_team: str = "") -> dict:
    if rosters is None or rosters.empty:
        return {"available": False, "reason": "No Yahoo rosters yet."}
    r = rosters.copy()
    r["team"] = teams.canonical(r["team"])

    proj, nfl_team, def_proj = {}, {}, {}
    for row in rankings_rows or []:
        if row.get("pos") == "DEF":
            def_proj[ESPN_ABBR.get(row["team"], row["team"])] = row.get("proj")
            continue
        k = key(row["player"])
        proj.setdefault(k, row.get("proj"))
        if row.get("team"):
            nfl_team.setdefault(k, ESPN_ABBR.get(row["team"], row["team"]))
    if nfl_rosters is not None and not nfl_rosters.empty and {"full_name", "team"} <= set(nfl_rosters.columns):
        for name, tm in zip(nfl_rosters["full_name"], nfl_rosters["team"]):
            if isinstance(name, str) and isinstance(tm, str):
                nfl_team.setdefault(key(name), ESPN_ABBR.get(tm, tm))
    nick = _nicknames()

    out_teams: dict[str, list[dict]] = {}
    for team, g in r.groupby("team"):
        starters = []
        for _, p in g.iterrows():
            slot = str(p.get("slot") or "")
            if slot in BENCH or not slot:
                continue
            name = str(p.get("player") or "")
            if not name or name == "(Empty)":
                starters.append({"slot": slot, "player": None, "key": None, "nfl_team": None, "proj": 0.0})
                continue
            if slot == "DEF":
                abbr = nick.get(key(name)) or (p.get("nfl_team") if isinstance(p.get("nfl_team"), str) else None)
                dp = def_proj.get(abbr)
                starters.append({"slot": slot, "player": name, "key": f"DEF:{abbr}" if abbr else None,
                                 "nfl_team": abbr, "proj": float(dp) if dp is not None else DEF_PROJ,
                                 **({} if dp is not None else {"proj_avg": True})})
                continue
            k = key(name)
            roster_tm = p.get("nfl_team") if isinstance(p.get("nfl_team"), str) else None
            tm = nfl_team.get(k) or (ESPN_ABBR.get(roster_tm, roster_tm) if roster_tm else None)
            pr = proj.get(k)
            row = {"slot": slot, "player": name, "key": k, "nfl_team": tm,
                   "proj": float(pr) if pr is not None else (K_PROJ if slot == "K" else 0.0)}
            if pr is None:
                row["proj_avg"] = slot == "K"
            starters.append(row)
        out_teams[str(team)] = starters

    fx = fixtures[fixtures["week"] == week] if fixtures is not None and not fixtures.empty else pd.DataFrame()
    matchups = ([[str(a), str(b)] for a, b in zip(teams.canonical(fx["home"]), teams.canonical(fx["away"]))]
                if not fx.empty else [])
    return {"available": True, "season": season, "week": week, "my_team": my_team,
            "matchups": matchups, "teams": out_teams}
