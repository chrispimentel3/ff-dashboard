"""Who a team is, whatever it's called this week.

Managers rename teams mid-season, and every Yahoo file is keyed by the name as it was on
the day it was scraped: on 2026-10-01 two teams were renamed between the week-3 scores
(old names) and that morning's standings and rosters (new names), so joining them by name
split each of those teams in two. The draft seat is the stable identity, and the
standings page carries the manager's name next to the team's, which pins a renamed team
to its seat without guessing — "Saquon Enjoyer" turned out to be the old "A Place in the
Hampton", not "Barkley's Balls Deep", which is what the name suggests.

`seat_for` resolves any name a team has gone by; `current_name` maps it to the name the
latest standings use, so older files (scores, fixtures) line up with newer ones.
"""
from __future__ import annotations

import difflib

import pandas as pd

from .config import DATA, FORMER_NAMES, MANAGER_BY_SEAT, SEAT_BY_TEAM, TEAM_BY_SEAT

SEAT_BY_MANAGER = {m.lower(): s for s, m in MANAGER_BY_SEAT.items()}
# names learned at runtime from a standings page (manager -> seat), so a rename that
# hasn't reached config yet still resolves for the rest of that scrape
_learned: dict[str, int] = {}


def learn(team: str, seat: int | None) -> None:
    if team and seat is not None:
        _learned[team] = int(seat)


def seat_for(team: str, manager: str | None = None) -> int | None:
    """Draft seat for a team name, from its manager when the page gives one."""
    if manager and manager.strip().lower() in SEAT_BY_MANAGER:
        return SEAT_BY_MANAGER[manager.strip().lower()]
    if not isinstance(team, str) or not team:
        return None
    for known in (SEAT_BY_TEAM, FORMER_NAMES, _learned, _standings_seats()):
        if team in known:
            return known[team]
    # Yahoo spells some apostrophes curly and some files straight
    flat = team.replace("’", "'")
    for known in (SEAT_BY_TEAM, FORMER_NAMES):
        for name, seat in known.items():
            if name.replace("’", "'") == flat:
                return seat
    hit = difflib.get_close_matches(team, list(SEAT_BY_TEAM), n=1, cutoff=0.6)
    return SEAT_BY_TEAM[hit[0]] if hit else None


def _standings_seats() -> dict[str, int]:
    """team -> seat from the latest standings file, by manager."""
    path = DATA / "yahoo_standings.csv"
    if not path.is_file():
        return {}
    try:
        st = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return {}
    out = {}
    for _, r in st.iterrows():
        seat = SEAT_BY_MANAGER.get(str(r.get("manager", "")).strip().lower())
        if seat is not None:
            out[r["team"]] = seat
    return out


def current_names() -> dict[int, str]:
    """seat -> the name the latest standings use (config's name where they don't say)."""
    out = dict(TEAM_BY_SEAT)
    for team, seat in _standings_seats().items():
        out[seat] = team
    return out


def canonical(series: pd.Series) -> pd.Series:
    """Rename every team in `series` to its current name; unknown names pass through."""
    now = current_names()
    names = {n: now.get(seat_for(n), n) for n in series.dropna().unique()}
    return series.map(lambda n: names.get(n, n))
