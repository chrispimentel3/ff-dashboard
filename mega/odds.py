"""§20.1 The Odds API client — player props, on a budget that has to be respected.

The free plan gives 500 credits a month. The event list is free, but props are priced per
event: `cost = [unique markets returned] x [regions]`. A full NFL slate is 16 games, and
the seven markets that matter in half-PPR make that **112 credits for one sweep of one
week** — about four sweeps a month. So this cannot run on the daily refresh the way the
Yahoo scrape does, and pretending otherwise would blow the month's allowance by Tuesday
and leave the rest of the season with nothing.

What it does instead: one sweep per week, on **Sunday morning** (the daily 7 AM run in
tools/refresh_watch.py). Sunday lines price Friday's designations and Saturday's
downgrades (not the 11:30 inactives: one daily run was chosen over a pre-kickoff one,
2026-11-10); a Tuesday line knows none of it, and close
start/sit calls swing on exactly that (2026 weeks 3-8: midweek props were a coin flip on
calls within two points). A midweek sweep still runs when a week has no file yet and the
month can pay for every Sunday left. It refuses to start a sweep it cannot finish and
never spends below `RESERVE`. Every response carries `x-requests-remaining`, so the ledger
is the account's own number rather than this module's guess about what it spent.

A paid tier only changes `remaining`: the midweek check then passes on its own.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from .config import DATA, ROOT
from .ids import canon_team

# Read at call time, not import time, so the key can be added without a restart — and
# loaded here rather than relying on whichever module happened to be imported first.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
LEDGER = DATA / "odds_budget.json"
_UA = {"User-Agent": "mega-bowl-dashboard/0.1 (personal use)"}
_TIMEOUT = httpx.Timeout(30.0)

# Seven markets, in the order they earn their credit. Half-PPR scores a rushing yard and a
# receiving yard identically, but they are still fetched apart: a book will price one and
# not the other, and a combined line cannot be split back out per player.
MARKETS = ("player_pass_yds", "player_pass_tds", "player_pass_interceptions",
           "player_rush_yds", "player_reception_yds", "player_receptions",
           "player_anytime_td")
REGIONS = "us"
RESERVE = 40                       # credits never spent, so a bad week can still be priced
ET = ZoneInfo("America/New_York")
SUNDAY_WINDOW = (7, 13)            # ET hours of the Sunday sweep: the 7 AM daily run, before 1 PM
SWEEP_COST = 100                   # one Sunday-and-Monday slate, 13-14 games x 7 markets (~92 seen)


def key() -> str:
    return os.environ.get("ODDS_API_KEY", "").strip()


def available() -> bool:
    return bool(key())


def props_csv(season: int, week: int) -> "os.PathLike":
    return DATA / f"props_{season}_wk{int(week):02d}.csv"


# ---------------------------------------------------------------- ledger
def ledger() -> dict:
    try:
        return json.loads(LEDGER.read_text())
    except Exception:
        return {}


def _save_ledger(d: dict) -> None:
    try:
        LEDGER.write_text(json.dumps(d, indent=2, sort_keys=True))
    except Exception:
        pass


def _note(resp: httpx.Response) -> dict:
    """Record the account's own credit numbers from the response headers."""
    d = ledger()
    for h, k in (("x-requests-remaining", "remaining"), ("x-requests-used", "used"),
                 ("x-requests-last", "last")):
        v = resp.headers.get(h)
        if v is not None:
            try:
                d[k] = int(float(v))
            except ValueError:
                pass
    d["checked"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _save_ledger(d)
    return d


def remaining() -> int | None:
    v = ledger().get("remaining")
    return int(v) if v is not None else None


# ---------------------------------------------------------------- fetch
def _get(path: str, **params) -> httpx.Response:
    params["apiKey"] = key()
    r = httpx.get(f"{BASE}{path}", params=params, headers=_UA, timeout=_TIMEOUT)
    _note(r)
    r.raise_for_status()
    return r


def events() -> pd.DataFrame:
    """Upcoming NFL events. This endpoint is free — it does not touch the quota."""
    if not available():
        return pd.DataFrame()
    r = _get(f"/sports/{SPORT}/events")
    rows = [{"event_id": e.get("id"), "commence": e.get("commence_time"),
             "home": e.get("home_team"), "away": e.get("away_team")} for e in r.json()]
    return pd.DataFrame(rows)


def _team_map() -> dict:
    """'Minnesota Vikings' -> 'MIN', from the logo table already in the repo."""
    try:
        t = pd.read_csv(DATA / "nfl_teamlogos.csv")
        return {str(a).strip().lower(): canon_team(b) for a, b in zip(t["team"], t["team_code"])}
    except Exception:
        return {}


def event_props(event_id: str, markets=MARKETS, regions: str = REGIONS) -> pd.DataFrame:
    """Every posted prop for one game, long: one row per player/market/book.

    Over and Under arrive as separate outcomes and are folded into one row here, because a
    price is only de-viggable next to its other side.
    """
    r = _get(f"/sports/{SPORT}/events/{event_id}/odds",
             regions=regions, markets=",".join(markets), oddsFormat="american")
    js = r.json()
    tm = _team_map()
    home = tm.get(str(js.get("home_team", "")).strip().lower(), "")
    away = tm.get(str(js.get("away_team", "")).strip().lower(), "")
    acc: dict = {}
    for bk in js.get("bookmakers", []) or []:
        book = bk.get("key")
        for mk in bk.get("markets", []) or []:
            market = mk.get("key")
            for o in mk.get("outcomes", []) or []:
                who = str(o.get("description") or "").strip()
                if not who:
                    continue
                side = str(o.get("name") or "").strip().lower()
                k = (who, market, book)
                rec = acc.setdefault(k, {"player": who, "market": market, "book": book,
                                         "line": o.get("point"), "price_over": None,
                                         "price_under": None, "home": home, "away": away,
                                         "event_id": event_id})
                if rec["line"] is None:
                    rec["line"] = o.get("point")
                if side in ("over", "yes"):
                    rec["price_over"] = o.get("price")
                elif side in ("under", "no"):
                    rec["price_under"] = o.get("price")
    return pd.DataFrame(list(acc.values()))


# ---------------------------------------------------------------- weekly sweep
def _week_events(ev: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Keep only the events that are this week's games, matched on the team pair.

    Matching on the pair rather than on kickoff time keeps a flexed Sunday-night game in
    the right week; commence_time alone would put a Thursday game and the Monday game that
    closes the previous week on the wrong side of any cut.
    """
    if ev.empty:
        return ev
    tm = _team_map()
    ev = ev.copy()
    ev["h"] = ev["home"].astype(str).str.strip().str.lower().map(tm)
    ev["a"] = ev["away"].astype(str).str.strip().str.lower().map(tm)
    try:
        from . import season as S
        sched = S.schedules(season)
    except Exception:
        return ev
    if sched is None or sched.empty:
        return ev
    s = sched[pd.to_numeric(sched["week"], errors="coerce") == int(week)]
    want = {frozenset((canon_team(h), canon_team(a)))
            for h, a in zip(s["home_team"], s["away_team"])}
    if not want:
        return ev.iloc[0:0]
    keep = [frozenset((h, a)) in want for h, a in zip(ev["h"], ev["a"])]
    return ev[keep]


def in_sunday_window(now: dt.datetime | None = None) -> bool:
    now = (now or dt.datetime.now(ET)).astimezone(ET)
    return now.weekday() == 6 and SUNDAY_WINDOW[0] <= now.hour < SUNDAY_WINDOW[1]


def sunday_weeks(season: int) -> list[int]:
    """Weeks whose props were swept in the Sunday window (mega/scorecard.py grades these)."""
    return [int(w) for w in (ledger().get("sunday_sweeps") or {}).get(str(season), [])]


def _sundays_left(now: dt.datetime) -> int:
    """Sunday sweeps still to pay for this month (the quota resets on the 1st), counting
    today's only while its window hasn't passed."""
    d, n = now.date(), 0
    while d.month == now.month:
        n += d.weekday() == 6
        d += dt.timedelta(days=1)
    return n - (now.weekday() == 6 and now.hour >= SUNDAY_WINDOW[1])


def due(season: int, week: int, now: dt.datetime | None = None) -> bool:
    """Sunday morning: once per week. Midweek: only for a week with no file yet, and only
    when the month can still pay for every Sunday left."""
    now = (now or dt.datetime.now(ET)).astimezone(ET)
    if in_sunday_window(now):
        return int(week) not in sunday_weeks(season)
    if props_csv(season, week).exists():
        return False
    have = remaining()
    return have is None or have - RESERVE - _sundays_left(now) * SWEEP_COST >= SWEEP_COST


def refresh(season: int, week: int, markets=MARKETS, force: bool = False,
            now: dt.datetime | None = None) -> tuple[pd.DataFrame, str]:
    """Sweep one week's props and write them. Returns (frame, human-readable status).

    Refuses to start a sweep it cannot pay for in full. A half-swept week is worse than no
    sweep: the games that got priced would quietly outrank the ones that did not, and every
    comparison in the app runs across players from different games. Games already under way
    are skipped (their lines are in-play, not pregame); their rows from an earlier sweep stay.
    """
    now = (now or dt.datetime.now(ET)).astimezone(ET)
    if not available():
        return pd.DataFrame(), "no ODDS_API_KEY set — Vegas projections are off"
    if not force and not due(season, week, now):
        return cached(season, week), "props are current (next sweep: Sunday morning)"

    ev = _week_events(events(), season, week)
    if "commence" in ev.columns:
        start = pd.to_datetime(ev["commence"], utc=True, errors="coerce")
        ev = ev[start.isna() | (start > now.astimezone(dt.timezone.utc))]
    if ev.empty:
        return cached(season, week), f"no priced games found for week {week}"

    need = len(ev) * len(markets)
    have = remaining()
    if have is not None and have - need < RESERVE:
        return (cached(season, week),
                f"skipped: {len(ev)} games x {len(markets)} markets = {need} credits, "
                f"only {have} left (reserve {RESERVE})")

    frames, failed = [], []
    for _, e in ev.iterrows():
        try:
            df = event_props(e["event_id"], markets)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            failed.append(f'{e.get("away")}@{e.get("home")}: {type(exc).__name__}')
    if not frames:
        return cached(season, week), f"no props returned ({len(failed)} errors)"

    out = pd.concat(frames, ignore_index=True)
    out = _attach_ids(out)
    out["season"], out["week"] = int(season), int(week)
    old = cached(season, week)
    if not old.empty and "event_id" in old.columns:      # Thursday's game, priced midweek
        out = pd.concat([out, old[~old["event_id"].isin(out["event_id"])]], ignore_index=True)
    out.to_csv(props_csv(season, week), index=False)

    d = ledger()
    d["swept"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    d["swept_week"] = int(week)
    if in_sunday_window(now):
        sw = d.setdefault("sunday_sweeps", {}).setdefault(str(season), [])
        if int(week) not in sw:
            sw.append(int(week))
    _save_ledger(d)

    msg = (f"{len(out):,} prop lines for {len(frames)}/{len(ev)} games, "
           f"{out['gsis_id'].notna().sum():,} matched to players")
    if failed:
        msg += f" ({len(failed)} games failed: {', '.join(failed[:3])})"
    if remaining() is not None:
        msg += f" — {remaining()} credits left"
    return out, msg


def _attach_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Book player names -> gsis_id, and the player's own side of the matchup.

    A name shared by two real players (there is a Ravens QB and, separately, a Panthers
    CB both named "Lamar Jackson") is genuinely ambiguous from the name alone — resolve()
    correctly refuses to guess between them. But every props row already carries the two
    teams actually playing in that game, and only one of the two same-named real players
    can belong to either side of it — trying each event team as the candidate's team
    narrows a same-name tie for free, no separate lookup needed. A player who isn't part
    of any tie resolves identically either way, so this is never worse than the old
    name-only lookup, only better for the rare collision."""
    from .ids import canon_team, crosswalk, resolve

    uniq = df[["player", "home", "away"]].drop_duplicates("player")

    def _resolve_as(team_col: str) -> pd.DataFrame:
        cand = uniq[["player"]].copy()
        cand["nfl_team"] = uniq[team_col]
        res, _ = resolve(cand, name_col="player")
        return res.set_index("player")

    try:
        # resolve()'s own `nfl_team`/`pos` output echoes back whatever team hint it was
        # given once a candidate is unambiguous (it trusts the caller's team as
        # authoritative — right for a roster row, wrong here, since "home"/"away" is
        # only ever a disambiguating guess, not the player's real team). Only the
        # gsis_id this settles on is trustworthy; team/pos are looked up fresh from the
        # crosswalk by that id instead of taken from either resolve() call's output.
        home_res, away_res = _resolve_as("home"), _resolve_as("away")
        by_gsis = crosswalk().drop_duplicates("gsis_id").set_index("gsis_id")
        m, t, pos_map = {}, {}, {}
        for p in uniq["player"]:
            hr, ar = home_res.loc[p], away_res.loc[p]
            best = hr if hr.get("resolved") else ar
            gid = best.get("gsis_id")
            m[p] = gid
            if gid in by_gsis.index:
                row = by_gsis.loc[gid]
                t[p] = canon_team(row.get("team"))
                pos_map[p] = row.get("pos_key")
    except Exception:
        m, t, pos_map = {}, {}, {}
    df = df.copy()
    df["gsis_id"] = df["player"].map(m)
    df["pos"] = df["player"].map(pos_map).fillna("")
    df["team"] = df["player"].map(t).fillna("")
    # a player we could not place still belongs to one of the two teams in his own game
    df["team"] = [tt if tt else "" for tt in df["team"]]
    return df


def cached(season: int, week: int) -> pd.DataFrame:
    p = props_csv(season, week)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
