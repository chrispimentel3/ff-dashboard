"""§20.1 The Odds API client — player props, on a budget that has to be respected.

The free plan gives 500 credits a month. The event list is free, but props are priced per
event: `cost = [unique markets returned] x [regions]`. A full NFL slate is 16 games, and
the seven markets that matter in half-PPR make that **112 credits for one sweep of one
week** — about four sweeps a month. So this cannot run on the daily refresh the way the
Yahoo scrape does, and pretending otherwise would blow the month's allowance by Tuesday
and leave the rest of the season with nothing.

What it does instead: one sweep per week by default, refusing to start a sweep it cannot
finish, and never spending below `RESERVE` so a mistake here can't strand the account at
zero. Every response carries `x-requests-remaining`, so the ledger is the account's own
number rather than this module's guess about what it spent.

Upgrading to the $30 tier (20,000 credits) makes a daily sweep comfortable — that is the
only thing standing between this and hourly props, and it is the user's call, not a code
change: set ODDS_REFRESH_HOURS.
"""
from __future__ import annotations

import datetime as dt
import json
import os

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
DEFAULT_REFRESH_HOURS = float(os.environ.get("ODDS_REFRESH_HOURS", "150"))  # ~weekly


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


def due(hours: float = DEFAULT_REFRESH_HOURS, season: int | None = None,
        week: int | None = None) -> bool:
    """Has enough time passed — and is there a file for this week at all?"""
    if season is not None and week is not None and not props_csv(season, week).exists():
        return True
    ts = ledger().get("swept")
    if not ts:
        return True
    try:
        last = dt.datetime.fromisoformat(ts)
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - last).total_seconds() >= hours * 3600.0


def refresh(season: int, week: int, markets=MARKETS, force: bool = False,
            hours: float = DEFAULT_REFRESH_HOURS) -> tuple[pd.DataFrame, str]:
    """Sweep one week's props and write them. Returns (frame, human-readable status).

    Refuses to start a sweep it cannot pay for in full. A half-swept week is worse than no
    sweep: the games that got priced would quietly outrank the ones that did not, and every
    comparison in the app runs across players from different games.
    """
    if not available():
        return pd.DataFrame(), "no ODDS_API_KEY set — Vegas projections are off"
    if not force and not due(hours, season, week):
        return cached(season, week), f"props are current (swept within {hours:.0f}h)"

    ev = _week_events(events(), season, week)
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
    out.to_csv(props_csv(season, week), index=False)

    d = ledger()
    d["swept"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    d["swept_week"] = int(week)
    _save_ledger(d)

    msg = (f"{len(out):,} prop lines for {len(frames)}/{len(ev)} games, "
           f"{out['gsis_id'].notna().sum():,} matched to players")
    if failed:
        msg += f" ({len(failed)} games failed: {', '.join(failed[:3])})"
    if remaining() is not None:
        msg += f" — {remaining()} credits left"
    return out, msg


def _attach_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Book player names -> gsis_id, and the player's own side of the matchup."""
    from .ids import resolve
    uniq = df[["player"]].drop_duplicates()
    try:
        res, _ = resolve(uniq, name_col="player")
        m = dict(zip(res["player"], res["gsis_id"]))
        t = dict(zip(res["player"], res.get("nfl_team", pd.Series(dtype=str))))
        # position decides which fitted skew curve a rushing line goes through (§20)
        pos_map = dict(zip(res["player"], res.get("pos", pd.Series(dtype=str))))
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
