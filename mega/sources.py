"""Free, no-auth external data: Sleeper (trending adds/drops + player map),
FantasyCalc (redraft trade values), and news RSS."""
from __future__ import annotations

import datetime as dt
import functools

import httpx
import pandas as pd

from .config import NEWS_FEEDS

_UA = {"User-Agent": "mega-bowl-dashboard/0.1 (personal use)"}
_TIMEOUT = httpx.Timeout(20.0)


def _norm(s: object) -> str:
    import re

    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"[^a-z ]", " ", s).strip()


# ---------------------------------------------------------------- Sleeper
@functools.lru_cache(maxsize=1)
def sleeper_players() -> pd.DataFrame:
    """Full NFL player dictionary (~5MB). Cached per process."""
    r = httpx.get("https://api.sleeper.app/v1/players/nfl", headers=_UA, timeout=60.0)
    r.raise_for_status()
    recs = []
    for pid, p in r.json().items():
        if not isinstance(p, dict):
            continue
        recs.append(
            dict(
                sleeper_id=pid,
                name=p.get("full_name") or f'{p.get("first_name","")} {p.get("last_name","")}'.strip(),
                pos=p.get("position"),
                team=p.get("team"),
                status=p.get("status"),
                injury_status=p.get("injury_status"),
                years_exp=p.get("years_exp"),
                gsis_id=p.get("gsis_id"),
                espn_id=p.get("espn_id"),
                yahoo_id=p.get("yahoo_id"),
                search_rank=p.get("search_rank"),
            )
        )
    df = pd.DataFrame(recs)
    df["norm"] = df["name"].map(_norm)
    return df


def sleeper_trending(kind: str = "add", hours: int = 48, limit: int = 40) -> pd.DataFrame:
    """Most added/dropped players across all Sleeper leagues."""
    url = f"https://api.sleeper.app/v1/players/nfl/trending/{kind}?lookback_hours={hours}&limit={limit}"
    r = httpx.get(url, headers=_UA, timeout=_TIMEOUT)
    r.raise_for_status()
    tr = pd.DataFrame(r.json())  # columns: player_id, count
    if tr.empty:
        return tr
    tr = tr.rename(columns={"player_id": "sleeper_id", "count": f"sleeper_{kind}s"})
    pl = sleeper_players()[["sleeper_id", "name", "pos", "team", "injury_status", "norm"]]
    out = tr.merge(pl, on="sleeper_id", how="left")
    out[f"{kind}_rank"] = range(1, len(out) + 1)
    return out


# ---------------------------------------------------------------- FantasyCalc
def fantasycalc_values(ppr: float = 0.5, n_teams: int = 12, n_qbs: int = 1) -> pd.DataFrame:
    """Community redraft trade values (0 = worthless, ~10000 = elite)."""
    url = (
        "https://api.fantasycalc.com/values/current"
        f"?isDynasty=false&numQbs={n_qbs}&numTeams={n_teams}&ppr={ppr}"
    )
    r = httpx.get(url, headers=_UA, timeout=_TIMEOUT)
    r.raise_for_status()
    recs = []
    for row in r.json():
        p = row.get("player", {})
        recs.append(
            dict(
                name=p.get("name"),
                pos=p.get("position"),
                team=p.get("maybeTeam"),
                age=p.get("maybeAge"),
                value=row.get("value"),
                overall_rank=row.get("overallRank"),
                pos_rank=row.get("positionRank"),
                trend_30d=row.get("trend30Day"),
            )
        )
    df = pd.DataFrame(recs)
    df["norm"] = df["name"].map(_norm)
    return df


# ---------------------------------------------------------------- News
def news_items(feeds: list[str] | None = None, per_feed: int = 25) -> pd.DataFrame:
    import feedparser

    feeds = feeds or NEWS_FEEDS
    rows = []
    for url in feeds:
        try:
            f = feedparser.parse(url)
        except Exception:
            continue
        src = (f.feed.get("title") or url).strip()
        for e in f.entries[:per_feed]:
            ts = e.get("published_parsed") or e.get("updated_parsed")
            rows.append(
                dict(
                    source=src,
                    title=e.get("title", "").strip(),
                    link=e.get("link", ""),
                    summary=(e.get("summary", "") or "").strip()[:400],
                    published=dt.datetime(*ts[:6]) if ts else pd.NaT,
                )
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("published", ascending=False, na_position="last").reset_index(drop=True)
        df["norm_title"] = df["title"].map(_norm)
    return df


def news_for_players(players: list[str], feeds: list[str] | None = None) -> pd.DataFrame:
    """Filter the news feed to headlines mentioning any of `players`."""
    df = news_items(feeds)
    if df.empty:
        return df
    keys = [_norm(p) for p in players if p]
    lastnames = {k.split()[-1] for k in keys if k}
    mask = df["title"].str.lower().apply(
        lambda t: any(k in _norm(t) for k in keys) or any(ln in t.lower().split() for ln in lastnames)
    )
    return df[mask].reset_index(drop=True)


if __name__ == "__main__":
    print("== trending adds ==")
    print(sleeper_trending("add", limit=15)[["add_rank", "name", "pos", "team", "sleeper_adds", "injury_status"]].to_string())
    print("\n== fantasycalc top 10 ==")
    print(fantasycalc_values().head(10)[["overall_rank", "name", "pos", "team", "value", "trend_30d"]].to_string())
    print("\n== news (5) ==")
    n = news_items()
    print(n.head(5)[["source", "published", "title"]].to_string())
