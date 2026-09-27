"""Fit the weekly miss hazard behind waiver INSURE value (HANDOFF v1.3 §3.1) from nflverse.

    python -m tools.fit_injury_hazard [--seasons 2021 2022 2023 2024 2025]

Writes config/injury_hazard.json, which mega/contingency.py reads in place of its old
judgment constants (ONSET x WEEKS_MISSED). Rerun each preseason.

**What is measured.** For every player-week where a QB/RB/WR/TE took at least half his
team's offensive snaps (a starter, `START_PCT`), look k = 1..14 weeks ahead and ask: in
that week, did his team play and he did not take an offensive snap? That empirical rate is
P(miss week w+k | starting in week w), split by position and age band.

**What counts as a miss.** Any game his team played without him — injury, but also being
benched or scratched. For INSURE that is the right event: what the backup is buying is the
chance the job opens, whatever the reason. It is why QBs read higher than an injury table
alone would suggest (benchings are real).

**What is excluded.** His own team's bye weeks (no game to miss) and weeks past the end of
the regular season. A traded player is followed on his new team: the snap feed is keyed by
player, so a trade is not a miss.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEASONS = [2021, 2022, 2023, 2024, 2025]
# A QB's next man up only counts when he took snaps the week before, which backup QBs
# rarely do — five seasons gave 25 cases. Ten give 48 (2012–2025: 61, fraction .65 vs .67,
# so the older years agree). The other positions have 100+ cases on the default window.
QB_INHERIT_SEASONS = list(range(2016, 2026))
POS = ("QB", "RB", "WR", "TE")
START_PCT = 0.50             # "starter": at least half the team's offensive snaps that week
MAX_AHEAD = 14               # a waiver horizon never runs longer than this
LAST_FANTASY_WEEK = 17       # Mega Bowl's final is week 17
AGE_BANDS = ((0, 25, "u25"), (25, 29, "25-28"), (29, 99, "29+"))
OUT = Path(__file__).resolve().parents[1] / "config" / "injury_hazard.json"


def band(age: float) -> str:
    for lo, hi, name in AGE_BANDS:
        if lo <= age < hi:
            return name
    return "25-28"


def load(seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl
    sc = nfl.load_snap_counts(seasons=list(seasons)).to_pandas()
    sc = sc[(sc["game_type"] == "REG") & sc["position"].isin(POS)].copy()
    sched = nfl.load_schedules(seasons=list(seasons)).to_pandas()
    sched = sched[sched["game_type"] == "REG"]
    games = pd.concat([sched[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
                       sched[["season", "week", "away_team"]].rename(columns={"away_team": "team"})])
    players = nfl.load_players().to_pandas()[["pfr_id", "birth_date"]].dropna()
    sc = sc.merge(players.drop_duplicates("pfr_id"), left_on="pfr_player_id", right_on="pfr_id", how="left")
    return sc, games


def fit(sc: pd.DataFrame, games: pd.DataFrame) -> dict:
    sc = sc.copy()
    kick = pd.to_datetime(sc["season"].astype(str) + "-09-01")
    sc["age"] = (kick - pd.to_datetime(sc["birth_date"], errors="coerce")).dt.days / 365.25
    sc = sc.dropna(subset=["age"])
    sc["band"] = sc["age"].map(band)

    played = {(r.pfr_player_id, r.season, r.week) for r in sc[sc["offense_snaps"] > 0].itertuples()}
    team_played = {(r.season, r.team, r.week) for r in games.itertuples()}
    # Capped at the fantasy season's last week: week-18 rest days for playoff-bound starters
    # are not injuries and never matter to a fantasy roster.
    last_week = {s: min(int(w), LAST_FANTASY_WEEK)
                 for s, w in games.groupby("season")["week"].max().items()}
    # A player's team in week w+k: whichever team he last appeared for on or before it. A
    # player who never appears again has no later team row, so we carry the week-w team.
    team_at = sc.sort_values("week").groupby(["pfr_player_id", "season"])

    starts = sc[sc["offense_pct"] >= START_PCT]
    rows = []
    teams_by_player = {k: dict(zip(v["week"], v["team"])) for k, v in team_at}
    for r in starts.itertuples():
        tb = teams_by_player.get((r.pfr_player_id, r.season), {})
        for k in range(1, MAX_AHEAD + 1):
            w = r.week + k
            if w > last_week.get(r.season, 18):
                break
            later = [wk for wk in tb if wk <= w]
            team = tb[max(later)] if later else r.team
            if (r.season, team, w) not in team_played:
                continue                  # his bye
            rows.append((r.position, r.band, k, (r.pfr_player_id, r.season, w) not in played))
    df = pd.DataFrame(rows, columns=["pos", "band", "k", "miss"])

    hazard, n = {}, {}
    for pos in POS:
        hazard[pos], n[pos] = {}, {}
        for _, _, b in AGE_BANDS:
            sub = df[(df["pos"] == pos) & (df["band"] == b)]
            by_k = sub.groupby("k")["miss"].agg(["mean", "size"]).reindex(range(1, MAX_AHEAD + 1))
            hazard[pos][b] = [round(float(x), 4) if pd.notna(x) else None for x in by_k["mean"]]
            n[pos][b] = int(by_k["size"].fillna(0).iloc[0])
    return {"hazard": hazard, "n_at_k1": n}


def fit_inherit(seasons) -> dict:
    """How much of a missing starter's expected points his next man up actually picks up.

    For every team-week (week 4 on) at each position: the starter is the player with the
    most expected points per game over his last three games, the next man up the second
    most — both must have played the week before. When the starter then has no opportunity
    row in a week his team played and the next man up does, measure the next man's gain in
    expected points over his own recent average, as a fraction of the starter's average.
    The same fraction measured in weeks the starter DID play is the regression-to-the-mean
    control, subtracted so a backup's ordinary bounce isn't read as inheritance.

    Expected points (usage), not actual: a backup inherits the touches, not the touchdowns.
    """
    import nflreadpy as nfl
    f = nfl.load_ff_opportunity(seasons=list(seasons), stat_type="weekly", model_version="latest").to_pandas()
    f = f[f["position"].isin(POS)].copy()
    f["week"] = pd.to_numeric(f["week"], errors="coerce").astype(int)
    f["season"] = pd.to_numeric(f["season"], errors="coerce").astype(int)
    f["xfp"] = pd.to_numeric(f["total_fantasy_points_exp"], errors="coerce").fillna(0.0)
    f = f[f["week"] <= LAST_FANTASY_WEEK]
    played_team = set(zip(f["season"], f["posteam"], f["week"]))
    rows = []
    for (season, team, pos), g in f.groupby(["season", "posteam", "position"]):
        by_week = {w: dict(zip(x["player_id"], x["xfp"])) for w, x in g.groupby("week")}
        weeks = sorted(by_week)
        for w in weeks:
            if w < 4 or (season, team, w) not in played_team:
                continue
            prior = [x for x in weeks if x < w][-3:]
            if len(prior) < 2 or prior[-1] != w - 1 and (season, team, w - 1) in played_team:
                continue
            hist: dict[str, list] = {}
            for x in prior:
                for pid, v in by_week[x].items():
                    hist.setdefault(pid, []).append(v)
            active = [pid for pid in hist if pid in by_week[prior[-1]] and len(hist[pid]) >= 2]
            if len(active) < 2:
                continue
            ranked = sorted(active, key=lambda p: -np.mean(hist[p]))
            s, b = ranked[0], ranked[1]
            s_mu, b_mu = float(np.mean(hist[s])), float(np.mean(hist[b]))
            if s_mu <= 0 or b not in by_week[w]:
                continue
            rows.append((pos, s not in by_week[w], (by_week[w][b] - b_mu) / s_mu))
    df = pd.DataFrame(rows, columns=["pos", "starter_out", "frac"])
    out = {}
    for pos in POS:
        d = df[df["pos"] == pos]
        ev, ctl = d[d["starter_out"]]["frac"], d[~d["starter_out"]]["frac"]
        out[pos] = {"inherit": round(float(max(0.0, ev.mean() - ctl.mean())), 4),
                    "raw": round(float(ev.mean()), 4), "control": round(float(ctl.mean()), 4),
                    "n": int(len(ev))}
    return out


def fit_status(seasons) -> dict:
    """P(takes an offensive snap | final injury-report status that week), 2021–2025.

    Replaces the judgment table in mega/contingency.py (AVAIL_NOW: Q .85, D .25, O 0)
    wherever the fitted file is present (handoff §4.1 P_active)."""
    import nflreadpy as nfl
    inj = nfl.load_injuries(seasons=list(seasons)).to_pandas()
    inj = inj[(inj["game_type"] == "REG") & inj["position"].isin(POS) & inj["report_status"].notna()]
    sc = nfl.load_snap_counts(seasons=list(seasons)).to_pandas()
    sc = sc[(sc["game_type"] == "REG") & (sc["offense_snaps"] > 0)]
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
    inj = inj.merge(ids, on="gsis_id", how="inner")
    played = set(zip(sc["pfr_player_id"], sc["season"], sc["week"]))
    inj["played"] = [(p, s, w) in played for p, s, w in zip(inj["pfr_id"], inj["season"], inj["week"])]
    # Only players who played the week before: a backup listed Questionable wasn't going
    # to take a snap healthy either, and counting him drags the rate down for everyone.
    inj["was_active"] = [(p, s, w - 1) in played for p, s, w in zip(inj["pfr_id"], inj["season"], inj["week"])]
    inj = inj[inj["was_active"]]
    out = {}
    for st, g in inj.groupby("report_status"):
        out[str(st)] = {"p_active": round(float(g["played"].mean()), 4), "n": int(len(g))}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    ap.add_argument("--qb-inherit-seasons", nargs="+", type=int, default=QB_INHERIT_SEASONS)
    a = ap.parse_args(argv)
    sc, games = load(a.seasons)
    out = fit(sc, games)
    out = {
        "_doc": "P(miss week w+k | took >= 50% of offensive snaps in week w), k = 1..14, by "
                "position and age band (age on Sep 1). A miss is any game his team played "
                "without him taking an offensive snap. Fitted by tools/fit_injury_hazard.py; "
                "read by mega/contingency.py.",
        "fitted": dt.date.today().isoformat(), "seasons": a.seasons,
        "start_pct": START_PCT, "age_bands": [b for _, _, b in AGE_BANDS], **out,
        "_doc_inherit": "Fraction of a missing starter's expected points per game his next "
                        "man up gains, net of the same measure in weeks the starter played "
                        "(regression-to-the-mean control). See fit_inherit().",
        "inherit": {**fit_inherit(a.seasons),
                    "QB": {**fit_inherit(a.qb_inherit_seasons)["QB"], "seasons": a.qb_inherit_seasons}},
        "_doc_status": "P(takes an offensive snap | final injury-report status). See fit_status().",
        "status": fit_status(a.seasons),
    }
    OUT.write_text(json.dumps(out, indent=1))
    for pos in POS:
        for b in out["age_bands"]:
            h = out["hazard"][pos][b]
            print(f"{pos:3s} {b:6s} n={out['n_at_k1'][pos][b]:>5}  k1={h[0]:.3f}  k3={h[2]:.3f}  k8={h[7]:.3f}")
    print("inherit:", out["inherit"])
    print("status:", out["status"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
