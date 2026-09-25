"""Historical (pre-2026) league data — one numeric Yahoo league ID per season rather than
a single ID that carries forward, unlike what the current league's Settings page
("Auto-renew Enabled: Yes") might suggest. Read-only: reuses mega.yahoo.LeagueSession (the
saved browser session, not a fresh login) and mega.yahoo.parse_roster/team_name_from
unmodified — this module only adds parsing for a page shape parse_standings never needed
to handle (a *completed* season's final standings table, which renders differently from
the live in-season standings widget parse_standings targets — a real <table>, not flex
divs, and the "1." rank on the flex-div page is a "0.-0.-0." record streak vs a real "1"
here reflecting the playoff result, not just regular-season seed).

The league ID for each season was found by hand, once: the current league's home page
links to "Final Rosters for Last Season", and the Record Book's per-superlative rows each
link to the team-season page it happened on, which carries that season's own league ID in
its URL. These are frozen historical facts, not living config — Yahoo has no endpoint that
lists them, so there is nothing to derive this from later if a season is ever added.
"""
from __future__ import annotations

import re
import time

import pandas as pd
from lxml import html as lx

from .yahoo import AuthExpired, LeagueSession, parse_roster, team_name_from

# Seconds to rest between page loads. Pulling all 6 seasons back-to-back with no pause
# tripped Yahoo's rate limiting mid-run (every page, including the current, live league,
# started coming back "Request denied") — this showed up as empty DataFrames, not an
# exception, so it went unnoticed for several calls. A deliberate pause between requests
# is cheaper than losing the saved session to a longer block.
REQUEST_DELAY = 4.0


class RateLimited(RuntimeError):
    pass


def _check_denied(html: str) -> None:
    if html.strip().startswith("Request denied") or "Request denied" in html[:200]:
        raise RateLimited(
            "Yahoo returned \"Request denied\" — likely rate-limited from too many "
            "requests too fast. Wait several minutes before retrying, and keep "
            "REQUEST_DELAY generous."
        )

SEASON_LEAGUE_IDS: dict[int, str] = {
    2020: "841377",
    2021: "841610",
    2022: "84535",
    2023: "597033",
    2024: "170189",
    2025: "907",
}

SEASON_LEAGUE_NAMES: dict[int, str] = {
    2020: "Mega Bowl 2.0",
    2021: "Mega Bowl 2.0",
    2022: "Mega Bowl 2.0",
    2023: "Mega Bowl",
    2024: "Mega Bowl",
    2025: "Mega Bowl",
}

# A team name in the standings table carries a trailing icon-font glyph (playoff medal,
# streak arrow) as a literal private-use-area character in the same text node.
_ICON_CHARS = re.compile(r"[-]+$")


def _clean_team(s: str) -> str:
    return _ICON_CHARS.sub("", s or "").strip()


def standings(session: LeagueSession, year: int) -> pd.DataFrame:
    """Final standings for a completed season. Rank already reflects the playoff result,
    not just regular-season seed — Yahoo reorders the table once the bracket finishes.

    The standings module only renders after the in-page "Standings" tab is actually
    clicked — the URL query params that look like they should select it (`?lhst=stand`,
    `?module=standings&lhst=stand`, even the tab's own href) load the page fine but leave
    the tab on its default view; only a real click event swaps the module in. Its table
    has no distinguishing class (`Table Table-mid Table-interactive Table-underline` — the
    same generic classes every table on the page uses), so it's found by its header text
    ("Rank") instead.
    """
    league_id = SEASON_LEAGUE_IDS[year]
    url = f"https://football.fantasysports.yahoo.com/{year}/f1/{league_id}"
    page = session._ctx.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    page.wait_for_timeout(1500)
    page.click('a.Navtarget:has-text("Standings")', timeout=10_000)
    page.wait_for_timeout(2000)
    html = page.content()
    page.close()
    if "Sign in to Yahoo" in html[:5000]:
        raise AuthExpired("Yahoo session expired. Run:  python -m mega.yahoo login")
    _check_denied(html)
    # lx.fromstring on the already-decoded string, never lx.parse on a re-read file —
    # the latter lets lxml guess the encoding and mangle apostrophes/em-dashes.
    doc = lx.fromstring(html)
    tables = doc.xpath('//table[.//thead[contains(., "Rank")]]')
    if not tables:
        return pd.DataFrame()

    rows = []
    for tr in tables[0].xpath(".//tbody/tr"):
        cells = [td.text_content().strip() for td in tr.xpath("./td")]
        if len(cells) < 6:
            continue
        rank_raw, team, rec, pf, pa, streak = cells[:6]
        try:
            w, l, t = (int(x) for x in rec.split("-"))
        except ValueError:
            continue
        rows.append(dict(
            year=year, rank=int(rank_raw.lstrip("*")), made_playoffs=rank_raw.startswith("*"),
            team=_clean_team(team), wins=w, losses=l, ties=t,
            points_for=float(pf), points_against=float(pa), streak=streak,
        ))
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def champion(standings_df: pd.DataFrame) -> str | None:
    top = standings_df[standings_df["rank"] == 1]
    return top.iloc[0]["team"] if not top.empty else None


def roster(session: LeagueSession, year: int, team_num: int) -> pd.DataFrame:
    """A team's final roster for a completed season. `team_num` is the Yahoo team_id
    (1-12, the URL segment) — NOT the current league's draft seat, which renamed/rotating
    teams across years don't map onto."""
    league_id = SEASON_LEAGUE_IDS[year]
    html = session.get(f"https://football.fantasysports.yahoo.com/{year}/f1/{league_id}/{team_num}")
    _check_denied(html)
    df = parse_roster(html)
    if not df.empty:
        df["year"] = year
    return df


def pull_season(session: LeagueSession, year: int) -> dict:
    st = standings(session, year)
    rosters: dict[str, pd.DataFrame] = {}
    for team_num in range(1, 13):
        time.sleep(REQUEST_DELAY)
        r = roster(session, year, team_num)
        if not r.empty:
            rosters[r["team"].iloc[0]] = r
    return {
        "year": year,
        "league_id": SEASON_LEAGUE_IDS[year],
        "league_name": SEASON_LEAGUE_NAMES[year],
        "standings": st,
        "champion": champion(st),
        "rosters": rosters,
    }


def pull_all(seasons: list[int] | None = None) -> dict[int, dict]:
    """One browser session for every season — LeagueSession launches a real browser, so
    this reuses it across seasons rather than paying that cost per year."""
    seasons = seasons or sorted(SEASON_LEAGUE_IDS)
    out = {}
    with LeagueSession() as s:
        for i, year in enumerate(seasons):
            if i:
                time.sleep(REQUEST_DELAY)
            out[year] = pull_season(s, year)
    return out
