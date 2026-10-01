"""The weekly digest for mega-bowl-web: what happened in the week just played, and what's
next — written out as sentences, the way the Tuesday Report (mega/digest.py) was, but
built from the league files rather than re-running the waiver and trade engines. Those
already have their own pages on the site; the digest links to them instead of copying them.

Everything here is arithmetic on data/yahoo_scores.csv, the standings, and the playoff odds
league.json already carries. Week-over-week odds movement needs last week's odds, which
nothing kept until this did: `snapshot_odds` stores each week's odds once, in
data/history/playoff_odds_weekly.csv.

"All-play" is the week's score against every other team's: 6-5 means it would have beaten
six of the eleven. It's the plain version of the weekly xWins in the spreadsheet.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import teams

ODDS_CSV = Path(__file__).resolve().parent.parent / "data" / "history" / "playoff_odds_weekly.csv"


# ───────────────────────────────────────────────────────────── odds history
def snapshot_odds(season: int, week: int, playoff_odds: list[dict], path: Path = ODDS_CSV) -> bool:
    """Keep the odds as they stood after `week` — once: the first export after the week is
    final writes it and later ones leave it (the sim reruns daily with new rosters, but
    "after week 3" should mean one number). Returns whether it wrote."""
    if not playoff_odds or week < 1:
        return False
    have = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["season", "week"])
    if ((have["season"] == season) & (have["week"] == week)).any():
        return False
    rows = pd.DataFrame(playoff_odds)[["team", "p_playoffs", "p_bye", "p_title"]]
    rows.insert(0, "week", week)
    rows.insert(0, "season", season)
    rows["snapshot_date"] = dt.date.today().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([have, rows], ignore_index=True).sort_values(["season", "week"]).to_csv(path, index=False)
    return True


def odds_history(season: int, path: Path = ODDS_CSV) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["week", "team", "p_playoffs", "p_title"])
    h = pd.read_csv(path)
    h = h[h["season"] == season].copy()
    h["team"] = teams.canonical(h["team"])          # snapshots keep the names of their day
    return h


# ───────────────────────────────────────────────────────────── the digest
def _records(sc: pd.DataFrame, through: int) -> dict[str, str]:
    played = sc[sc["week"] <= through]
    out = {}
    for team, g in played.groupby("team"):
        w = int((g["points"] > g["opp_points"]).sum())
        l = int((g["points"] < g["opp_points"]).sum())
        t = len(g) - w - l
        out[team] = f"{w}-{l}" + (f"-{t}" if t else "")
    return out


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _pts(x: float) -> str:
    return f"{x:.2f}"


def build(season: int, my_team: str, scores: pd.DataFrame, fixtures: pd.DataFrame,
          league: dict, action_board: dict | None = None,
          odds_hist: pd.DataFrame | None = None) -> dict:
    sc = scores.copy()
    if sc.empty:
        return {"available": False, "reason": "No finished weeks yet."}
    week = int(sc["week"].max())
    wk = sc[sc["week"] == week].copy()
    if wk["team"].nunique() < 4:
        return {"available": False, "reason": f"Week {week} scores are incomplete."}
    n = wk["team"].nunique()

    # this week's table: score rank, all-play record, result
    wk = wk.sort_values("points", ascending=False).reset_index(drop=True)
    wk["rank"] = wk["points"].rank(ascending=False, method="min").astype(int)
    wk["allplay_w"] = [int((wk["points"] < p).sum()) for p in wk["points"]]
    wk["won"] = wk["points"] > wk["opp_points"]
    rec = _records(sc, week)
    rec_before = _records(sc, week - 1) if week > 1 else {}
    table = [{"team": r.team, "points": round(float(r.points), 2), "rank": int(r.rank),
              "allplay": f"{r.allplay_w}-{n - 1 - r.allplay_w}", "won": bool(r.won),
              "opponent": r.opponent, "record": rec.get(r.team, "")}
             for r in wk.itertuples()]

    games = []
    seen = set()
    for r in wk.itertuples():
        if r.team in seen:
            continue
        seen.update({r.team, r.opponent})
        win, lose = (r, None) if r.points >= r.opp_points else (None, r)
        if win is not None:
            games.append({"winner": r.team, "w_pts": round(float(r.points), 2),
                          "loser": r.opponent, "l_pts": round(float(r.opp_points), 2)})
        else:
            games.append({"winner": r.opponent, "w_pts": round(float(r.opp_points), 2),
                          "loser": r.team, "l_pts": round(float(r.points), 2)})
    for g in games:
        g["margin"] = round(g["w_pts"] - g["l_pts"], 2)
    games.sort(key=lambda g: -g["margin"])

    by_team = {t["team"]: t for t in table}
    lines: list[dict] = []

    me = by_team.get(my_team)
    my_result = None
    if me:
        verb = "beat" if me["won"] else "lost to"
        my_result = {**me, "opp_points": float(wk.loc[wk["team"] == my_team, "opp_points"].iloc[0])}
        lines.append({"kind": "you", "text":
                      f"You {verb} {me['opponent']} {_pts(me['points'])}–{_pts(my_result['opp_points'])} "
                      f"and sit at {me['record']}. Your score was the {_ordinal(me['rank'])}-best of the week — "
                      f"all-play {me['allplay']}."})

    hi, lo = table[0], table[-1]
    lines.append({"kind": "high", "text": f"High score: {hi['team']}, {_pts(hi['points'])}"
                  + ("" if hi["won"] else " — and still lost") + "."})
    lines.append({"kind": "low", "text": f"Low score: {lo['team']}, {_pts(lo['points'])}"
                  + (" — and still won" if lo["won"] else "") + "."})
    close, blow = games[-1], games[0]
    lines.append({"kind": "close", "text":
                  f"Closest game: {close['winner']} over {close['loser']} by {close['margin']:.2f}."})
    lines.append({"kind": "blowout", "text":
                  f"Biggest blowout: {blow['winner']} over {blow['loser']} by {blow['margin']:.2f}."})

    losers = [t for t in table if not t["won"]]
    winners = [t for t in table if t["won"]]
    unlucky = min(losers, key=lambda t: t["rank"]) if losers else None
    if unlucky and unlucky["rank"] <= n // 2:
        lines.append({"kind": "unlucky", "text":
                      f"Unluckiest: {unlucky['team']} scored the {_ordinal(unlucky['rank'])}-most "
                      f"(all-play {unlucky['allplay']}) and lost."})
    lucky = max(winners, key=lambda t: t["rank"]) if winners else None
    if lucky and lucky["rank"] > n // 2:
        lines.append({"kind": "lucky", "text":
                      f"Luckiest: {lucky['team']} won with the {_ordinal(lucky['rank'])}-best score "
                      f"(all-play {lucky['allplay']})."})

    unbeaten = [t for t, r in rec.items() if r.split("-")[1] == "0" and len(r.split("-")) == 2]
    winless = [t for t, r in rec.items() if r.startswith("0-")]
    if week >= 2 and len(unbeaten) == 1:
        lines.append({"kind": "unbeaten", "text": f"{unbeaten[0]} is the last unbeaten team ({rec[unbeaten[0]]})."})
    if week >= 2 and len(winless) == 1:
        lines.append({"kind": "winless", "text": f"{winless[0]} is the only team still winless ({rec[winless[0]]})."})

    # playoff odds, and how they moved since the week before
    odds_now = {r["team"]: r for r in league.get("playoff_odds") or []}
    movers = {"available": False, "rows": []}
    if odds_hist is not None and not odds_hist.empty and odds_now:
        prev = odds_hist[odds_hist["week"] == week - 1]
        if not prev.empty:
            p = prev.set_index("team")
            rows = []
            for team, r in odds_now.items():
                if team not in p.index:
                    continue
                rows.append({"team": team,
                             "p_before": round(float(p.loc[team, "p_playoffs"]), 4),
                             "p_after": round(float(r["p_playoffs"]), 4),
                             "title_before": round(float(p.loc[team, "p_title"]), 4),
                             "title_after": round(float(r["p_title"]), 4)})
            for r in rows:
                r["delta"] = round(r["p_after"] - r["p_before"], 4)
            rows.sort(key=lambda r: -r["delta"])
            if rows:
                movers = {"available": True, "from_week": week - 1, "rows": rows}
                up, down = rows[0], rows[-1]
                if up["delta"] >= 0.05:
                    lines.append({"kind": "riser", "text":
                                  f"Biggest playoff-odds riser: {up['team']}, {up['p_before']:.0%} → {up['p_after']:.0%}."})
                if down["delta"] <= -0.05:
                    lines.append({"kind": "faller", "text":
                                  f"Biggest faller: {down['team']}, {down['p_before']:.0%} → {down['p_after']:.0%}."})
                mine = next((r for r in rows if r["team"] == my_team), None)
                if mine and mine not in (up, down):
                    d = mine["delta"]
                    lines.append({"kind": "you_odds", "text":
                                  f"Your playoff odds: {mine['p_before']:.0%} → {mine['p_after']:.0%} "
                                  f"({'+' if d >= 0 else '−'}{abs(d) * 100:.0f} pts)."})

    # season luck so far, from the xWins power table league.json already carries
    px = (league.get("power_xwins") or {}).get("rows") or []
    if px and week >= 2:
        luck = sorted(px, key=lambda r: r.get("luck_w") or 0)
        bad, good = luck[0], luck[-1]
        if (bad.get("luck_w") or 0) <= -0.75:
            lines.append({"kind": "season_unlucky", "text":
                          f"Season's unluckiest: {bad['team']} has {bad['wins']:.0f} wins on "
                          f"{bad['xwins']:.1f} expected."})
        if (good.get("luck_w") or 0) >= 0.75:
            lines.append({"kind": "season_lucky", "text":
                          f"Season's luckiest: {good['team']} has {good['wins']:.0f} wins on "
                          f"{good['xwins']:.1f} expected."})

    # next week
    nxt = week + 1
    power = {r["team"]: r for r in px}
    ppg = sc.groupby("team")["points"].mean().to_dict()
    fx = fixtures[fixtures["week"] == nxt] if fixtures is not None and not fixtures.empty else pd.DataFrame()
    slate = []
    for r in fx.itertuples():
        a, b = r.home, r.away
        slate.append({"a": a, "b": b, "a_record": rec.get(a, ""), "b_record": rec.get(b, ""),
                      "a_ppg": round(ppg.get(a, 0.0), 1), "b_ppg": round(ppg.get(b, 0.0), 1),
                      "a_power": power.get(a, {}).get("power_rank"), "b_power": power.get(b, {}).get("power_rank"),
                      "a_playoffs": (odds_now.get(a) or {}).get("p_playoffs"),
                      "b_playoffs": (odds_now.get(b) or {}).get("p_playoffs"),
                      "mine": my_team in (a, b)})
    for g in slate:     # the game with the most riding on it: both teams near the cut line
        pa, pb = g["a_playoffs"], g["b_playoffs"]
        g["stakes"] = (pa * (1 - pa) + pb * (1 - pb)) if pa is not None and pb is not None else 0.0
    gotw = max((g for g in slate if not g["mine"]), key=lambda g: g["stakes"], default=None)
    slate.sort(key=lambda g: (not g["mine"], -g["stakes"]))
    my_game = next((g for g in slate if g["mine"]), None)

    ab = action_board or {}
    todo = {"headline": ab.get("headline"),
            "waiver": (ab.get("waivers") or [None])[0],
            "trade": (ab.get("trades") or [None])[0]}

    return {
        "available": True, "season": season, "week": week, "next_week": nxt, "my_team": my_team,
        "my_result": my_result, "headlines": lines, "table": table, "games": games,
        "odds_movers": movers, "slate": slate, "my_game": my_game, "game_of_week": gotw,
        "todo": todo,
    }
