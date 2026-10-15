"""Role changes the betting market has priced before the fantasy rankings move.

A sportsbook re-prices a player's workload the moment his role changes — a backup who becomes
the lead back gets a rushing-yards line he never had, a receiver moved into the slot sees his
receptions line jump — days before consensus rankings catch up. The saved weekly prop sweeps
(data/props_<season>_wk<NN>.csv) are compared market by market, on the volume markets only
(an anytime-touchdown price says little about role):

  * NEW: volume lines now where the earlier sweeps priced him (an anytime-TD price, say) with
         none at all — a book that only posted receiving lines for some backs that week is
         coverage, not a role, so a player with any earlier volume line is never NEW
  * UP:  a volume line up JUMP_PCT or more and by a real step over his latest earlier line
Either way the line now has to be a real role (2.5+ catches, 25+ receiving or 30+ rushing yards).

No history of props exists from past seasons, so this can't be backtested; the scorecard
grades each week's movers against what they then scored.
"""
from __future__ import annotations

import pandas as pd

# market -> (label, smallest rise that counts, smallest line that means a real role)
VOLUME = {"player_receptions": ("receptions", 1.0, 2.5), "player_reception_yds": ("receiving yards", 12.0, 25.0),
          "player_rush_yds": ("rushing yards", 15.0, 30.0)}
JUMP_PCT = 0.35        # (J) a line has to rise 35%+ ...
LOOKBACK = 4           # earlier weeks searched for his previous line


def _raw(season: int, week: int) -> pd.DataFrame:
    from .odds import props_csv
    try:
        return pd.read_csv(props_csv(season, week)).dropna(subset=["gsis_id"])
    except Exception:
        return pd.DataFrame()


def _lines(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw["market"].isin(VOLUME) & raw["line"].notna()] if not raw.empty else raw
    if d.empty:
        return pd.DataFrame(columns=["gsis_id", "market", "line", "player", "pos", "team"])
    # one line per player and market: the median across books
    return d.groupby(["gsis_id", "market"], as_index=False).agg(
        line=("line", "median"), player=("player", "first"), pos=("pos", "first"), team=("team", "first"))


def movers(season: int, week: int) -> list[dict]:
    cur = _lines(_raw(season, week))
    if cur.empty:
        return []
    prior, priced, volume_before = {}, set(), set()
    for w in range(week - 1, max(0, week - 1 - LOOKBACK), -1):
        raw = _raw(season, w)
        if raw.empty:
            continue
        priced |= set(raw["gsis_id"])
        for r in _lines(raw).itertuples():
            volume_before.add(r.gsis_id)
            prior.setdefault((r.gsis_id, r.market), (w, float(r.line)))     # the latest earlier line wins
    out = {}
    for r in cur.itertuples():
        label, step, real = VOLUME[r.market]
        if r.line < real:
            continue
        before = prior.get((r.gsis_id, r.market))
        if before is None:
            if r.gsis_id not in priced or r.gsis_id in volume_before:
                continue        # never covered (no sweep saw him), or coverage, not a role change
            kind, why = "NEW", f"first {label} line ({r.line:g})"
        else:
            w0, l0 = before
            if not (r.line >= l0 * (1 + JUMP_PCT) and r.line - l0 >= step):
                continue
            kind, why = "UP", f"{label} line {l0:g} → {r.line:g} (wk {w0} → {week})"
        row = out.setdefault(r.gsis_id, {"gsis_id": r.gsis_id, "player": r.player, "pos": r.pos, "team": r.team,
                                         "week": week, "kind": kind, "why": []})
        row["why"].append(why)
        if kind == "UP":
            row["kind"] = "UP"
    return sorted(out.values(), key=lambda m: (m["kind"] != "UP", m["player"]))
