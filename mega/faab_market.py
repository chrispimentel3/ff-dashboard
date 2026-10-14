"""What this league actually pays: the price to win a waiver claim, from five seasons of bids.

A claim is won by beating the best OTHER bid, so that is the price — the winner's own bid
only says how much he overpaid (the league's winners overpaid by $8 on average, $5 median).
Every 2021-2025 claim gives one: Yahoo's settled-claims list carries every losing bid, and a
paid claim nobody contested had a price of $0.

Bidders see a player's last game, his position and the calendar, so those are the features:
kickers and defenses priced as one group; everyone else by how many half-PPR points he scored
in the game before the claim (a breakout is what draws bids). Each group keeps the sorted
prices it has seen; a group with fewer than MIN_GROUP claims borrows its neighbours'.

    p_win(bid)  = share of comparable claims whose price was below the bid
    price(p)    = the smallest bid that wins with probability p

tools/fit_faab_market.py builds config/faab_market.json and backtests it season by season.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "config" / "faab_market.json"
TIERS = (0.0, 6.0, 12.0, 18.0)       # last-game half-PPR points: [0,6) [6,12) [12,18) [18,∞)
MIN_GROUP = 25
TARGET_P = 0.80                      # (J) the win chance a bid aims for: higher pays for certainty.
                                     #     Backtest: aiming at 80% won 82.5% of 513 claims for $3.81 a claim,
                                     #     where the league's winners paid $8.65
_POS_TAIL = re.compile(r"\s+[A-Za-z]{2,3}\s+-\s+(QB|RB|WR|TE|K|DEF)(\s+\S+)*\s*$")


def group(pos: str, last_pts: float | None) -> str:
    if pos in ("K", "DEF"):
        return "KDEF"
    pts = max(0.0, last_pts or 0.0)            # a negative game is still the bottom tier
    i = sum(pts >= t for t in TIERS) - 1
    return f"{pos if pos == 'QB' else 'SKILL'}:{i}"


def claims(years=range(2021, 2026)) -> pd.DataFrame:
    """year, player, pos, date, bid (the winner's), price (best other bid), contested."""
    from . import league_memory as LM
    from .intel import _norm
    rows = []
    for y in years:
        d = json.loads((LM.HIST / "transactions" / f"{y}.json").read_text())
        seen = set()
        for c in d.get("faab_bids") or []:
            m = _POS_TAIL.search(LM._clean(c["player"]))
            name = LM._clean(c["player"])[: m.start()] if m else LM._clean(c["player"])
            date = re.sub(r",\s*", ", ", LM._clean(c["date"]))
            rows.append({"year": y, "player": name, "pos": m.group(1) if m else None, "date": date,
                         "bid": int(c["winning_bid"]), "price": int(c["runner_up"] or 0), "contested": True})
            seen.add((_norm(name), date))
        for t in d.get("transactions") or []:
            r = LM.parse_row(t)
            if r and r["kind"] == "claim" and (_norm(r["player"]), r["date"]) not in seen:
                rows.append({"year": y, "player": r["player"], "pos": r["pos"], "date": r["date"],
                             "bid": r["bid"], "price": 0, "contested": False})
    return pd.DataFrame(rows).dropna(subset=["pos"])


def attach_last_game(df: pd.DataFrame) -> pd.DataFrame:
    """The NFL week each claim followed and the player's points in it (0 if he didn't play)."""
    import nflreadpy as nfl
    from .intel import _norm, weekly
    out = []
    for y, g in df.groupby("year"):
        s = nfl.load_schedules(seasons=[int(y)]).to_pandas()
        s = s[s["game_type"] == "REG"]
        last_day = s.groupby("week")["gameday"].max()
        w = weekly(int(y))
        pts = {(_norm(n), int(wk)): float(p) for n, wk, p in zip(w["player"], w["week"], w["half_ppr"])}
        for r in g.itertuples():
            when = datetime.strptime(f"{r.date} {y}", "%b %d, %I:%M %p %Y")
            if when.month <= 2:
                when = when.replace(year=int(y) + 1)
            done = [int(wk) for wk, day in last_day.items() if datetime.fromisoformat(day) < when]
            wk = max(done) if done else 0
            out.append({**r._asdict(), "week": wk, "last_pts": pts.get((_norm(r.player), wk), 0.0)})
    return pd.DataFrame(out).drop(columns=["Index"], errors="ignore")


def build_table(df: pd.DataFrame) -> dict:
    """group -> sorted prices; thin groups pool with the next tier down (then up)."""
    df = df.assign(g=[group(p, x) for p, x in zip(df["pos"], df["last_pts"])])
    prices = {g: sorted(int(v) for v in sub["price"]) for g, sub in df.groupby("g")}
    out = {}
    for g in prices:
        pool, kind = list(prices[g]), g.split(":")[0]
        if kind != "KDEF":
            i, step = int(g.split(":")[1]), 1
            while len(pool) < MIN_GROUP and step < len(TIERS):
                for j in (i - step, i + step):
                    pool += prices.get(f"{kind}:{j}", [])
                step += 1
        out[g] = sorted(pool)
    return out


def load_table() -> dict:
    try:
        return json.loads(TABLE.read_text())["groups"]
    except Exception:
        return {}


def p_win(prices: list[int], bid: int) -> float:
    """Ties go to the worse record, so a tie counts as half a win."""
    if not prices:
        return float("nan")
    a = np.asarray(prices)
    return float(((a < bid).sum() + 0.5 * (a == bid).sum()) / len(a))


def price(prices: list[int], target: float = TARGET_P, cap: int = 100) -> int:
    """The smallest whole-dollar bid that wins with probability >= target."""
    for b in range(0, cap + 1):
        if p_win(prices, b) >= target:
            return b
    return cap


def last_games(season: int, week: int) -> dict:
    """gsis_id -> half-PPR points in `week` (the game bidders have just seen)."""
    from .intel import weekly
    w = weekly(season)
    w = w[w["week"] == week]
    return dict(zip(w["gsis_id"], w["half_ppr"]))


def lookup(pos: str, last_pts: float | None, table: dict | None = None) -> list[int]:
    t = load_table() if table is None else table
    g = group(pos, last_pts)
    return t.get(g) or t.get(f"{g.split(':')[0]}:0") or []
