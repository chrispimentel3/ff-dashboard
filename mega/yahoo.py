"""Authenticated Yahoo league scraper for a PRIVATE league.

Yahoo has no public read access for private leagues, so we reuse a real logged-in
browser session:

  1.  python -m mega.yahoo login       # opens a browser, you sign in once
  2.  python -m mega.yahoo pull        # headless fetch + parse of every league page

The session (cookies) is saved to data/yahoo_state.json and lasts weeks. When it
expires, `pull` tells you to re-run `login`.

MANUAL FALLBACK (no browser automation): in Yahoo, use File > Save Page As on
  - the Players page (status = Available)        -> data/manual/freeagents*.html
  - each team's roster page                      -> data/manual/roster_<seat>.html
  - the Standings page                           -> data/manual/standings.html
  - the Transactions page                        -> data/manual/transactions.html
then:  python -m mega.yahoo pull --manual
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd
from lxml import html as lx

from .config import DATA, LEAGUE_ID, SEAT_BY_TEAM, TEAM_BY_SEAT

STATE = DATA / "yahoo_state.json"
MANUAL = DATA / "manual"
BASE = f"https://football.fantasysports.yahoo.com/f1/{LEAGUE_ID}"

# Players page: status=A (available), pos filter, stat1 selects the stat block.
#   S_S_2025  -> 2025 full-season totals   |   S_PSP -> preseason proj   |   S_PW_<n> -> week n proj
FA_URL = BASE + "/players?status=A&pos={pos}&sort=PTS&sdir=1&stat1={stat}&count={start}"


class AuthExpired(RuntimeError):
    pass


# ────────────────────────────────────────────────────────── browser session
def login() -> None:
    from playwright.sync_api import sync_playwright

    MANUAL.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        for launch in (dict(channel="chrome"), {}):  # prefer real Chrome, fall back to bundled
            try:
                browser = p.chromium.launch(headless=False, **launch)
                break
            except Exception:
                browser = None
        if browser is None:
            raise RuntimeError("Could not launch a browser for login")
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto("https://login.yahoo.com/")
        print("\n>>> Sign in to Yahoo in the browser window.")
        print(">>> When you can see your Fantasy league, come back here and press Enter.")
        try:
            input()
        except EOFError:
            page.wait_for_timeout(90_000)
        ctx.storage_state(path=str(STATE))
        browser.close()
    print(f"saved session -> {STATE}")


class LeagueSession:
    """Headless page fetcher that reuses the saved login."""

    def __init__(self) -> None:
        if not STATE.exists():
            raise AuthExpired("No saved session. Run:  python -m mega.yahoo login")
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx = self._browser.new_context(storage_state=str(STATE))

    def get(self, url: str) -> str:
        page = self._ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        page.wait_for_timeout(600)
        htmltext = page.content()
        page.close()
        if "login.yahoo.com" in page.url or "Sign in to Yahoo" in htmltext[:5000]:
            raise AuthExpired("Yahoo session expired. Run:  python -m mega.yahoo login")
        return htmltext

    def get_text(self, url: str) -> str:
        """Rendered innerText. Needed where layout newlines carry the structure —
        the standings page has no <table>, so only the visual line breaks separate
        team from manager from record."""
        page = self._ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        page.wait_for_timeout(2500)
        text = page.evaluate("document.body.innerText")
        page.close()
        if "Sign in to Yahoo" in text[:5000]:
            raise AuthExpired("Yahoo session expired. Run:  python -m mega.yahoo login")
        return text

    def close(self) -> None:
        self._ctx.close()
        self._browser.close()
        self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ────────────────────────────────────────────────────────── parsers (pure)
_YID = re.compile(r"/players/(\d+)")
_TEAMPOS = re.compile(r"\b([A-Z]{2,3})\s*[-–]\s*(QB|RB|WR|TE|K|DEF)\b")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def parse_player_table(html_text: str) -> pd.DataFrame:
    """Parse a Yahoo 'Player List' style table (free agents or search results)."""
    doc = lx.fromstring(html_text)
    tables = doc.xpath('//table[contains(@class,"Table") and .//*[contains(@class,"ysf-player-name")]]')
    if not tables:
        return pd.DataFrame()
    tbl = tables[0]

    # header titles (use the most descriptive header row)
    header_rows = tbl.xpath(".//thead//tr")
    titles: list[str] = []
    for hr in header_rows:
        cand = [_clean(th.get("title") or th.text_content()) for th in hr.xpath("./th")]
        if len([c for c in cand if c]) > len([c for c in titles if c]):
            titles = cand

    rows = []
    for tr in tbl.xpath(".//tbody/tr"):
        tds = tr.xpath("./td")
        if not tds:
            continue
        namecell = tr.xpath('.//*[contains(@class,"ysf-player-name")]')
        if not namecell:
            continue
        nc = namecell[0]
        a = nc.xpath(".//a")
        name = _clean(a[0].text_content()) if a else _clean(nc.text_content())
        href = a[0].get("href", "") if a else ""
        yid = (_YID.search(href) or [None, None])[1]
        m = _TEAMPOS.search(_clean(nc.text_content()))
        nfl_team, pos = (m.group(1), m.group(2)) if m else (None, None)

        cell_txt = [_clean(td.text_content()) for td in tds]
        rec = dict(zip(titles, cell_txt)) if titles else {}
        rec.update(yahoo_id=yid, player=name, nfl_team=nfl_team, pos=pos)
        rec["_raw_cells"] = cell_txt
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ren = {
        "Roster Status": "roster_status", "Games Played": "gp", "Bye Week": "bye",
        "Fantasy Points": "fan_pts", "Pre-Season": "proj_preseason", "Actual": "actual_pts",
        "Percent player is rostered in Yahoo leagues": "pct_ros",
        "Passing Yards": "pass_yds", "Passing Touchdowns": "pass_td", "Interceptions": "int",
        "Rushing Attempts": "rush_att", "Rushing Yards": "rush_yds", "Rushing Touchdowns": "rush_td",
        "Targets": "tgt", "Receptions": "rec", "Receiving Yards": "rec_yds",
        "Receiving Touchdowns": "rec_td", "Return Touchdowns": "ret_td",
        "2-Point Conversions": "two_pt", "Fumbles Lost": "fum_lost",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for c in ["gp", "bye", "fan_pts", "proj_preseason", "actual_pts", "pass_yds", "pass_td",
              "int", "rush_att", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td",
              "ret_td", "two_pt", "fum_lost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("-", "0"), errors="coerce")
    if "pct_ros" in df.columns:
        df["pct_ros"] = pd.to_numeric(df["pct_ros"].str.rstrip("%"), errors="coerce")
    keep = ["yahoo_id", "player", "nfl_team", "pos", "roster_status", "gp", "bye", "fan_pts",
            "proj_preseason", "actual_pts", "pct_ros", "pass_yds", "pass_td", "int", "rush_att",
            "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td", "ret_td", "two_pt", "fum_lost"]
    return df[[c for c in keep if c in df.columns]]


_STANDING = re.compile(
    r"^(?P<team>.+)\n(?P<manager>.+)\n(?P<w>\d+) - (?P<l>\d+) - (?P<t>\d+) \| (?P<rank>\d+)(?:st|nd|rd|th)\s*$",
    re.M,
)


def parse_standings(page_text: str) -> pd.DataFrame:
    """Standings from the rendered page text.

    Yahoo renders this page as flex divs, not a <table>, and with "Live Standings"
    on it leads with the current matchup — so pd.read_html finds nothing. Each team
    still reads as three lines: name, manager, then "W - L - T | Nth". The featured
    matchup repeats its two teams, hence the de-dupe on first sighting.
    """
    if "<html" in page_text[:2000].lower():  # --manual path hands us saved HTML
        page_text = "\n".join(
            t.strip() for t in lx.fromstring(page_text).itertext() if t.strip()
        )
    seen: dict[str, dict] = {}
    for m in _STANDING.finditer(page_text):
        seen.setdefault(m.group("team").strip(), m.groupdict())
    if not seen:
        return pd.DataFrame()
    rows = [
        dict(
            rank=int(d["rank"]), team=team, manager=d["manager"].strip(),
            wins=int(d["w"]), losses=int(d["l"]), ties=int(d["t"]),
            seat=_seat_for(team),
        )
        for team, d in seen.items()
    ]
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def _seat_for(team_name: str) -> int | None:
    """Draft seat for a team, tolerating in-season renames."""
    if team_name in SEAT_BY_TEAM:
        return SEAT_BY_TEAM[team_name]
    import difflib

    hit = difflib.get_close_matches(team_name, list(SEAT_BY_TEAM), n=1, cutoff=0.6)
    return SEAT_BY_TEAM[hit[0]] if hit else None


def parse_transactions(html_text: str) -> pd.DataFrame:
    doc = lx.fromstring(html_text)
    rows = []
    for tr in doc.xpath('//table[contains(@class,"Table")]//tbody/tr'):
        txt = _clean(tr.text_content())
        if not txt:
            continue
        rows.append({"text": txt})
    return pd.DataFrame(rows)


def parse_roster(html_text: str, seat: int | None = None) -> pd.DataFrame:
    """Best-effort parse of a team roster page."""
    doc = lx.fromstring(html_text)
    rows = []
    for tr in doc.xpath('//table[contains(@class,"Table")]//tbody/tr'):
        nc = tr.xpath('.//*[contains(@class,"ysf-player-name")]')
        if not nc:
            continue
        a = nc[0].xpath(".//a")
        name = _clean(a[0].text_content()) if a else _clean(nc[0].text_content())
        href = a[0].get("href", "") if a else ""
        yid = (_YID.search(href) or [None, None])[1]
        m = _TEAMPOS.search(_clean(nc[0].text_content()))
        tds = tr.xpath("./td")
        slot = _clean(tds[0].text_content()) if tds else ""
        rows.append(dict(
            seat=seat, team=TEAM_BY_SEAT.get(seat), slot=slot, player=name,
            yahoo_id=yid, nfl_team=m.group(1) if m else None, pos=m.group(2) if m else None,
        ))
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────── orchestration
FA_STAT_BLOCKS = {"season2025": "S_S_2025", "actual": "S_S_2026", "proj_ros": "S_PR"}


def pull(manual: bool = False, fa_pages: int = 4) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    if manual:
        files = sorted(MANUAL.glob("freeagents*.html"))
        fa = pd.concat([parse_player_table(f.read_text()) for f in files], ignore_index=True) if files else pd.DataFrame()
        out["free_agents"] = fa.drop_duplicates("yahoo_id") if not fa.empty else fa
        st = MANUAL / "standings.html"
        out["standings"] = parse_standings(st.read_text()) if st.exists() else pd.DataFrame()
        tx = MANUAL / "transactions.html"
        out["transactions"] = parse_transactions(tx.read_text()) if tx.exists() else pd.DataFrame()
        rosters = [parse_roster(f.read_text(), int(re.search(r"(\d+)", f.stem).group(1)))
                   for f in sorted(MANUAL.glob("roster_*.html"))]
        out["rosters"] = pd.concat(rosters, ignore_index=True) if rosters else pd.DataFrame()
    else:
        with LeagueSession() as s:
            fa_frames = []
            for start in range(0, fa_pages * 25, 25):
                url = FA_URL.format(pos="O", stat="S_S_2025", start=start)
                fa_frames.append(parse_player_table(s.get(url)))
            fa = pd.concat(fa_frames, ignore_index=True)
            out["free_agents"] = fa.drop_duplicates("yahoo_id")
            out["standings"] = parse_standings(s.get_text(BASE + "/standings"))
            out["transactions"] = parse_transactions(s.get(BASE + "/transactions"))
            rosters = [parse_roster(s.get(f"{BASE}/{seat}"), seat) for seat in TEAM_BY_SEAT]
            out["rosters"] = pd.concat(rosters, ignore_index=True)

    for name, df in out.items():
        path = DATA / f"yahoo_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  yahoo_{name:12s} {len(df):>4} rows -> {path}")
    return out


def cached_rosters() -> pd.DataFrame:
    """Last `pull` result from disk, shaped like yahoo_api.rosters_df()."""
    path = DATA / "yahoo_rosters.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    from .intel import _norm

    df["norm"] = df["player"].map(_norm)
    df["pos"] = df["pos"].astype(str).str.upper().str.replace("W/R/T", "W/R", regex=False)
    return df


def cached_standings() -> pd.DataFrame:
    """Last `pull` standings from disk."""
    path = DATA / "yahoo_standings.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if df.empty else df.sort_values("rank").reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mega.yahoo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login")
    pp = sub.add_parser("pull")
    pp.add_argument("--manual", action="store_true")
    pp.add_argument("--fa-pages", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "login":
        login()
    else:
        try:
            pull(manual=args.manual, fa_pages=args.fa_pages)
        except AuthExpired as e:
            print(f"!! {e}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
