"""League memory: each manager fitted from six seasons of Mega Bowl transactions.

Team names change every season and the people behind them don't, so every historical row is
pinned to a person by the hashed Yahoo profile key in data/history/managers/<year>.json
(tools/pull_history.py --managers), and from that person to his team today (2026). Someone
who has left the league maps to no one and is dropped.

Three things come out, each replacing a hand-set number:

  * **FAAB aggression** — how a manager bids against the rest of the league *on the same
    player*: on every contested claim (all bids are listed, losers too), his bid against the
    mean bid on that claim, ln((b+1)/(mean+1)), averaged and shrunk toward 1 by K_AGG claims.
    Comparing within a claim takes the player's value out of it, which the history can't
    otherwise price. Read by waiver_value.rival_bids through faab.suggest's `aggression`.
  * **Trade habit** — trades per season, shrunk toward the league rate by TRADE_PRIOR seasons;
    the acceptance model adds ln(his rate / league rate) to its logit. A manager who has made
    one trade in five seasons is not the same audience as one who makes three a year.
  * **Positional lean** — his share of FAAB dollars by position against the league's
    (market.faab_bias, which existed and was never fed), replacing the draft-night seeds
    wherever he has spent enough to say something.
"""
from __future__ import annotations

import functools
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"
LIVE_YEAR = 2026
K_AGG = 4.0           # (J) contested bids worth as much as the league average, before his own speak
TRADE_PRIOR = 2.0     # (J) seasons at the league's trade rate added to every manager's own
AGG_CLAMP = (0.5, 2.0)
SKILL = ("QB", "RB", "WR", "TE")

_PUA = re.compile(r"[-]")
_DATE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, ?\d{1,2}:\d{2} (?:am|pm)"
_TRADE = re.compile(rf"Traded to (?P<team>.+?) (?P<date>{_DATE})$")
_MOVE = re.compile(rf".*(?:To Waivers|To Free Agent|Free Agent|Waiver) (?P<team>.+?) (?P<date>{_DATE})$")
_PAID = re.compile(r"^(?P<player>.+?) [A-Za-z]{2,3} - (?P<pos>QB|RB|WR|TE|K|DEF)\b(?:\s+\S+)*?\s+\$(?P<bid>\d+) Waiver")
_BID = re.compile(r"(?P<team>.+?) \$(?P<amt>\d+)$")


_COMMISH = re.compile(r"^(?:\w+ )?\( by Commissioner \) |\s*\( [^)]+ \)$")


def _clean(s: str) -> str:
    return " ".join(_PUA.sub("", s or "").split())


def _name(team: str) -> str:
    """A team name as a join key: commissioner-move annotations off ("Added ( by
    Commissioner ) PHPLTE ( Christopher )"), apostrophe style, case and spacing folded."""
    t = _COMMISH.sub("", _COMMISH.sub("", _clean(team)))
    return "".join(t.replace("’", "'").casefold().split())


def parse_row(text: str) -> dict | None:
    """One transaction row -> {kind: trade|claim|move, team, date, [pos, bid]}."""
    t = _clean(text)
    m = _TRADE.search(t)
    if m:
        return {"kind": "trade", "team": m["team"], "date": m["date"]}
    m = _MOVE.match(t)
    if not m:
        return None
    out = {"kind": "move", "team": m["team"], "date": m["date"]}
    p = _PAID.match(t)
    if p:
        out.update(kind="claim", pos=p["pos"], bid=int(p["bid"]))
    return out


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def owners() -> tuple[dict, dict]:
    """({(year, team name): manager_key}, {manager_key: team name today})."""
    by_team, now = {}, {}
    for f in sorted((HIST / "managers").glob("*.json")):
        d = _json(f) or {}
        for r in d.get("teams") or []:
            by_team[(int(d["year"]), _name(r["team"]))] = r["manager_key"]
            if int(d["year"]) == LIVE_YEAR:
                now[r["manager_key"]] = r["team"]
    return by_team, now


def history(years=range(2020, LIVE_YEAR)) -> dict:
    """Parsed rows for every season on disk, plus this season's contested claims."""
    rows, contested = [], []
    for y in years:
        d = _json(HIST / "transactions" / f"{y}.json")
        if not d:
            continue
        for t in d.get("transactions") or []:
            r = parse_row(t)
            if r:
                rows.append({"year": y, **r})
        for i, c in enumerate(d.get("faab_bids") or []):
            contested.append({"year": y, "claim": f"{y}-{i}", "team": c["winner"], "bid": int(c["winning_bid"])})
            for part in str(c.get("losers") or "").split(";"):
                m = _BID.match(part.strip())
                if m:
                    contested.append({"year": y, "claim": f"{y}-{i}", "team": m["team"].strip(), "bid": int(m["amt"])})
    live = ROOT / "data" / "yahoo_faab_bids.csv"
    if live.is_file():
        for i, c in pd.read_csv(live).iterrows():
            contested.append({"year": LIVE_YEAR, "claim": f"{LIVE_YEAR}-{i}", "team": c["winner"], "bid": int(c["winning_bid"])})
            for part in str(c.get("losers") if pd.notna(c.get("losers")) else "").split(";"):
                m = _BID.match(part.strip())
                if m:
                    contested.append({"year": LIVE_YEAR, "claim": f"{LIVE_YEAR}-{i}", "team": m["team"].strip(),
                                      "bid": int(m["amt"])})
    return {"rows": pd.DataFrame(rows), "contested": pd.DataFrame(contested)}


def _pin(df: pd.DataFrame, by_team: dict, now: dict) -> pd.DataFrame:
    """Each row's team -> its manager -> his team today; rows of people who left are dropped."""
    if df.empty:
        return df
    key = [by_team.get((int(y), _name(t))) for y, t in zip(df["year"], df["team"])]
    return df.assign(manager_key=key, current=[now.get(k) for k in key]).dropna(subset=["current"])


def aggression(contested: pd.DataFrame) -> dict:
    """team today -> (multiplier, contested bids seen)."""
    if contested.empty:
        return {}
    c = contested.assign(mean=contested.groupby("claim")["bid"].transform("mean"))
    c = c[c.groupby("claim")["bid"].transform("size") >= 2]
    c = c.assign(lr=[math.log((b + 1) / (m + 1)) for b, m in zip(c["bid"], c["mean"])])
    out = {}
    for team, g in c.groupby("current"):
        mult = math.exp(g["lr"].sum() / (len(g) + K_AGG))
        out[team] = (round(min(max(mult, AGG_CLAMP[0]), AGG_CLAMP[1]), 3), int(len(g)))
    return out


def trade_habits(rows: pd.DataFrame, seasons: dict) -> dict:
    """team today -> (logit offset, trades, seasons). A trade is every row stamped with the
    same season and minute; the teams it lists as receivers are the ones in it."""
    t = rows[rows["kind"] == "trade"]
    count: dict = {}
    for _, g in t.groupby(["year", "date"]):
        for team in set(g["current"]):
            count[team] = count.get(team, 0) + 1
    league = sum(count.values()) / max(sum(seasons.values()), 1)
    out = {}
    for team, s in seasons.items():
        n = count.get(team, 0)
        rate = (n + TRADE_PRIOR * league) / (s + TRADE_PRIOR)
        out[team] = (round(math.log(rate / league), 3) if league > 0 else 0.0, n, s)
    return out


@functools.lru_cache(maxsize=1)
def fit() -> dict:
    """team today -> {manager, seasons, faab_aggression, contested_bids, trade_offset, trades}
    plus "_bias": the positional multipliers (market.faab_bias on every paid claim)."""
    from .market import faab_bias
    by_team, now = owners()
    if not now:
        return {}
    h = history()
    rows = _pin(h["rows"], by_team, now) if not h["rows"].empty else h["rows"]
    contested = _pin(h["contested"], by_team, now)
    tx_years = set(rows["year"]) if not rows.empty else set()
    seasons = {team: sum((y, t) in by_team and by_team[(y, t)] == k for (y, t) in by_team if y in tx_years)
               for k, team in now.items()}
    agg = aggression(contested)
    habit = trade_habits(rows, seasons) if not rows.empty else {}
    paid = rows[(rows["kind"] == "claim") & rows["pos"].isin(SKILL)] if not rows.empty else rows
    names = {k: None for k in now}
    for f in sorted((HIST / "managers").glob("*.json")):
        for r in (_json(f) or {}).get("teams") or []:
            names[r["manager_key"]] = r["manager"]
    out = {}
    for k, team in now.items():
        a, n_bids = agg.get(team, (1.0, 0))
        off, trades, s = habit.get(team, (0.0, 0, seasons.get(team, 0)))
        out[team] = {"manager": names.get(k), "seasons": s, "faab_aggression": a, "contested_bids": n_bids,
                     "trade_offset": off, "trades": trades}
    out["_bias"] = faab_bias(paid[["current", "pos", "bid"]].rename(columns={"current": "team"})) if not paid.empty else {}
    return out


def faab_aggression(team: str) -> float:
    return float((fit().get(team) or {}).get("faab_aggression", 1.0))


def trade_offset(team: str) -> float:
    return float((fit().get(team) or {}).get("trade_offset", 0.0))


def bias() -> dict:
    """Positional multipliers by team: fitted where a manager has spent enough to say
    something, else the draft-night seed (market.bias's rule: evidence replaces a seed)."""
    from .market import seed_bias
    fitted = fit().get("_bias") or {}
    return {**seed_bias(), **fitted}
