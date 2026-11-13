"""Our own matchup model: what each player scores this week, and whether this week sets up
better or worse than his usual one.

A ridge regression per position, trained on every fantasy-relevant player-game since 2015
(data/ask_history, about 30,000) and refitted before each week on everything played so far.
Every input is something known before kickoff:

  * his production and role: scoring and expected points this season, last 3 games, last game,
    last season; snap, target, carry and air-yard shares; red-zone work;
  * this week's situation: the betting line, what the opposing defense has allowed, which
    teammates are out, who is at quarterback, his injury tag, the game setting.

tools/fit_ml_matchup.py screens those inputs (which carry signal the others don't) and tests
the model walk-forward. Two numbers come out per player:

  * `ml`       projected half-PPR points if he plays;
  * `ml_week`  the "this week's situation" part of that, against the same part averaged over
               his own games this season — so a player on a bad offense isn't marked down
               every week for it. +1 or better is a good week, -1 or worse a bad one, and
               `why` lists the pieces (they add up to `ml_week`).

It is a second opinion beside mega/proj_ros.py until the scorecard has graded it: only then
does mega/scorecard.py's `ml_blend` let it into Start/Sit.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POS = ["QB", "RB", "WR", "TE"]
FIRST = 2015                   # the oldest season in data/ask_history
MIN_PRIOR = 3                  # games he has played this season before the one being predicted
RELEVANT = {"QB": 10.0, "RB": 5.0, "WR": 5.0, "TE": 3.5}   # prior points or xFP per game
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
RIDGE = 10.0
K_DEF = 3                      # pseudo-games of "average defense" in the opponent rating
OUT_BELOW = 0.5                # (J) chance to play under this: not expected to suit up
CALL = 1.0                     # (J) points this week's situation moves him before it's a call
WHY_MIN = 0.3                  # a reason smaller than this isn't shown
COUNTS = ["half_ppr", "targets", "carries", "attempts", "passing_yards", "receiving_air_yards",
          "offense_snaps", "team_snaps", "rz_touches", "gl_touches"]

FAMILIES = {
    "Expected points from his usage (xFP)": ["xfp_std"],
    "Playing time: snaps and routes": ["snap_std", "routes_std"],
    "Share of targets, carries, air yards": ["tgt_std", "car_std", "air_std"],
    "Red-zone and goal-line work": ["rz_std", "gl_std"],
    "Last 3 games: scoring, usage, snaps": ["pts_l3", "xfp_l3", "snap_l3", "tgt_l3", "car_l3"],
    "Last game only": ["pts_l1", "xfp_l1", "snap_l1"],
    "Last season": ["pts_prev", "xfp_prev"],
    "Vegas: his team's expected points": ["implied", "impl_rel", "spread", "total"],
    "Opponent defense vs his position": ["opp_adj", "opp_pts", "opp_xfp", "opp_pa"],
    "Teammates out, QB change": ["vac_tgt", "vac_car", "vac_pos", "new_qb", "qb_ypa"],
    "Game setting: home, dome, rest": ["home", "dome", "thu", "off_bye", "div", "rest_diff"],
    "His health: injury tag, back from absence": ["questionable", "listed", "gap"],
    "Boom-bust history": ["pts_sd"],
    "Team style: pace, pass rate, scoring": ["pace", "pass_rate", "pf_prior"],
    "Week of the season": ["week"],
}
BASE = ["pts_std"]
ALL = BASE + [f for fs in FAMILIES.values() for f in fs]
# this week's situation, in the words the site uses; the rest of ALL is who he is
WEEK_GROUPS = {
    "Vegas line": FAMILIES["Vegas: his team's expected points"],
    "Opposing defense": FAMILIES["Opponent defense vs his position"],
    "Teammates out": ["vac_tgt", "vac_car", "vac_pos"],
    "Quarterback": ["new_qb", "qb_ypa"],
    "Injury report": FAMILIES["His health: injury tag, back from absence"],
    "Game setting": FAMILIES["Game setting: home, dome, rest"],
}


# ------------------------------------------------------------------ data
def player_weeks(seasons) -> pd.DataFrame:
    from . import data
    out = []
    for s in seasons:
        try:
            out.append(data.ask_player_week(s).assign(season=s))
        except Exception as e:
            print(f"[ml] {s} skipped: {type(e).__name__}: {e}")
    d = pd.concat(out, ignore_index=True)
    d = d[d["pos"].isin(POS) & d["team"].notna() & d["gsis_id"].notna()].copy()
    d["team"] = d["team"].replace(TEAM_FIX)
    for c in COUNTS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d = d[(d["offense_snaps"] > 0) | (d[["attempts", "targets", "carries"]].sum(axis=1) > 0)]
    return d.drop_duplicates(["gsis_id", "season", "week"]).assign(played=True)


def upcoming(d: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """One empty row per player expected to play a week that isn't played yet: everyone
    mega/proj_ros.py projects whose team has a game and who is likelier than not to suit up,
    less anyone ruled out by hand or on game day (mega/status.py). Anyone left off counts as
    out for his teammates' "teammates out"."""
    # ponytail: in-or-out at OUT_BELOW; weigh both cases by his chance to play if a
    # game-time decision ever swings a call that matters
    from . import proj_ros
    from .ids import norm
    from .status import out_for_week
    ruled_out = out_for_week(week)
    names = d.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["player"]
    done = d[(d["season"] == season) & (d["week"] == week)]
    have, played_teams = set(done["gsis_id"]), set(done["team"])    # a game already played: he sat it out
    rows = []
    for g, p in ((proj_ros.cached(season) or {}).get("players") or {}).items():
        w = (p.get("weeks") or {}).get(str(week)) or {}
        name = names.get(g)
        if (g in have or p.get("team") in played_teams or not w or w.get("bye") or name is None or p.get("pos") not in POS
                or float(w.get("p_active", 0.0)) < OUT_BELOW or norm(name) in ruled_out):
            continue
        rows.append({"gsis_id": g, "player": name, "pos": p["pos"], "team": p.get("team"),
                     "season": season, "week": week, "played": False})
    return pd.DataFrame(rows)


def games(seasons) -> pd.DataFrame:
    """One row per team per regular-season game, with what was known before kickoff."""
    import nflreadpy as nfl
    s = nfl.load_schedules(seasons=list(seasons)).to_pandas()
    s = s[s["game_type"] == "REG"].copy()
    rows = []
    for me, opp, sign in (("home", "away", 1), ("away", "home", -1)):
        rows.append(pd.DataFrame({
            "season": s["season"].astype(int), "week": s["week"].astype(int),
            "team": s[f"{me}_team"].replace(TEAM_FIX), "opp": s[f"{opp}_team"].replace(TEAM_FIX),
            "home": float(sign == 1), "implied": s["total_line"] / 2 + sign * s["spread_line"] / 2,
            "spread": sign * s["spread_line"], "total": s["total_line"],
            "rest_diff": (s[f"{me}_rest"] - s[f"{opp}_rest"]).clip(-7, 7),
            "off_bye": (s[f"{me}_rest"] >= 13).astype(float), "thu": (s[f"{me}_rest"] <= 5).astype(float),
            "dome": s["roof"].isin(["dome", "closed"]).astype(float), "div": s["div_game"].astype(float),
            "pf": s[f"{me}_score"], "pa": s[f"{opp}_score"], "qb_id": s[f"{me}_qb_id"]}))
    g = pd.concat(rows, ignore_index=True).sort_values(["season", "team", "week"]).reset_index(drop=True)
    k = g.groupby(["season", "team"])
    g["gno"] = k.cumcount()
    for src, dst in (("implied", "impl_avg"), ("pf", "pf_prior"), ("pa", "pa_prior")):
        g[dst] = (k[src].cumsum() - g[src]) / g["gno"].where(g["gno"] > 0)
    g["impl_rel"] = g["implied"] / g["impl_avg"]
    opp = g[["season", "week", "team", "pa_prior"]].rename(columns={"team": "opp", "pa_prior": "opp_pa"})
    return g.merge(opp, on=["season", "week", "opp"], how="left").drop(columns=["pf", "pa", "impl_avg"])


def prior_sums(d: pd.DataFrame):
    """P(col, k): his total of `col` over his previous k games this season (all of them if k is
    None) — never the game on the row itself. `d` is sorted by player, season, week."""
    key = [d["gsis_id"], d["season"]]

    def P(col: str, k: int | None = None) -> pd.Series:
        cs = d.groupby(["gsis_id", "season"])[col].cumsum()
        a = cs.groupby(key).shift(1).fillna(0.0)
        return a if k is None else a - cs.groupby(key).shift(k + 1).fillna(0.0)
    return P


def defense(d: pd.DataFrame) -> pd.DataFrame:
    """What each defense had allowed to a position before each game: points and xFP per game
    against the league's, and points against those players' own averages (opponent-adjusted,
    shrunk toward 1 by K_DEF games)."""
    b = d["pts_std"].where(d["n_prior"] >= 2)
    g = (d.assign(A=d["pts"], X=d["xfp"], Aa=d["pts"].where(b.notna(), 0.0), B=b.fillna(0.0))
          .groupby(["season", "opp", "pos", "week"], as_index=False)[["A", "X", "Aa", "B"]].sum()
          .sort_values(["season", "opp", "pos", "week"]).reset_index(drop=True))
    k = g.groupby(["season", "opp", "pos"])
    prior = k[["A", "X", "Aa", "B"]].cumsum() - g[["A", "X", "Aa", "B"]]
    n = k.cumcount()
    gbar = g.groupby("pos")["B"].transform("mean")
    g["opp_pts"], g["opp_xfp"] = prior["A"] / n.where(n > 0), prior["X"] / n.where(n > 0)
    for c in ("opp_pts", "opp_xfp"):         # against the league at that point of that season
        g[c] = g[c] / g.groupby(["season", "week", "pos"])[c].transform("mean")
    g["opp_adj"] = (prior["Aa"] + K_DEF * gbar) / (prior["B"] + K_DEF * gbar)
    return g[["season", "opp", "pos", "week", "opp_pts", "opp_xfp", "opp_adj"]]


def vacated(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per team-game: the share of the team's targets and carries over its last 3 games that
    belonged to players who don't play this one (no row for it), for the team and by position."""
    k = ["season", "team"]
    tg = d[k + ["week"]].drop_duplicates().sort_values(k + ["week"])
    nxt = pd.concat([tg.assign(tw=tg.groupby(k)["week"].shift(-j)) for j in (1, 2, 3)]).dropna()
    r = d[k + ["week", "gsis_id", "pos", "targets", "carries"]].merge(nxt, on=k + ["week"])
    r["tw"] = r["tw"].astype(int)
    u = r.groupby(k + ["tw", "gsis_id", "pos"], as_index=False)[["targets", "carries"]].sum()
    here = d[k + ["week", "gsis_id"]].rename(columns={"week": "tw"}).assign(plays=1.0)
    u = u.merge(here, on=k + ["tw", "gsis_id"], how="left")
    out = u["plays"].isna()
    u["vt"], u["vc"] = u["targets"].where(out, 0.0), u["carries"].where(out, 0.0)
    team = u.groupby(k + ["tw"], as_index=False)[["targets", "carries", "vt", "vc"]].sum()
    team["vac_tgt"] = team["vt"] / team["targets"].where(team["targets"] > 0)
    team["vac_car"] = team["vc"] / team["carries"].where(team["carries"] > 0)
    pos = u.groupby(k + ["tw", "pos"], as_index=False)[["vt", "vc"]].sum().merge(
        team[k + ["tw", "targets", "carries"]], on=k + ["tw"])
    # his own position's vacancy: carries for a back, targets for a receiver
    pos["vac_pos"] = np.where(pos["pos"] == "RB", pos["vc"] / pos["carries"].where(pos["carries"] > 0),
                              pos["vt"] / pos["targets"].where(pos["targets"] > 0))
    return (team[k + ["tw", "vac_tgt", "vac_car"]].rename(columns={"tw": "week"}),
            pos[k + ["tw", "pos", "vac_pos"]].rename(columns={"tw": "week"}))


def quarterback(d: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: how new the starting QB is (1 = hasn't thrown a pass for them this season)
    and his yards per attempt so far. The starter is the one the schedule lists (never "whoever
    threw the most": that reads a mid-game injury back into a pre-game input). Where it lists
    nobody we have a row for: whoever threw the most, or before the game, whoever of those
    expected to play threw the most last time out."""
    q = d[d["pos"] == "QB"]
    k = ["season", "team", "week"]
    listed = q["gsis_id"] == q["qb_id"]
    thrown = q["attempts"].where(q["played"], q["att_l1"] + 1e-6 * q["att_prior"])
    q = q.assign(_k=thrown.where(~listed, 1e9))
    start = q[listed | (q["attempts"] >= 10) | ~q["played"]].sort_values("_k").groupby(k, as_index=False).tail(1)
    t = q.groupby(k, as_index=False)["attempts"].sum().sort_values(k)
    t["team_att_prior"] = t.groupby(["season", "team"])["attempts"].cumsum() - t["attempts"]
    s = start[k + ["att_prior", "yds_prior"]].merge(t[k + ["team_att_prior"]], on=k)
    s["new_qb"] = 1 - (s["att_prior"] / s["team_att_prior"].where(s["team_att_prior"] > 0)).clip(0, 1)
    s["qb_ypa"] = (s["yds_prior"] / s["att_prior"]).where(s["att_prior"] >= 20)
    return s[k + ["new_qb", "qb_ypa"]]


def injuries(seasons) -> pd.DataFrame:
    import nflreadpy as nfl
    i = nfl.load_injuries(seasons=list(seasons)).to_pandas()
    i = i[i["game_type"] == "REG"] if "game_type" in i.columns else i
    i = i.dropna(subset=["gsis_id"]).drop_duplicates(["season", "week", "gsis_id"])
    return pd.DataFrame({"season": i["season"].astype(int), "week": i["week"].astype(int), "gsis_id": i["gsis_id"],
                         "listed": 1.0, "questionable": (i["report_status"] == "Questionable").astype(float)})


@functools.lru_cache(maxsize=4)
def dataset(season: int, week: int | None = None) -> pd.DataFrame:
    """One row per player-game from FIRST through `season` with the target (`pts`) and every
    feature in ALL; with `week`, also a row per player expected to play that week if it hasn't
    been played. `played` marks a game that happened; `keep` marks a row the model is trained
    and scored on (MIN_PRIOR games behind him, fantasy-relevant, a betting line)."""
    seasons = list(range(FIRST, season + 1))
    d = player_weeks(seasons)
    if week is not None:
        d = pd.concat([d, upcoming(d, season, week)], ignore_index=True)
        d[COUNTS] = d[COUNTS].fillna(0.0)
    d = d.merge(games(seasons), on=["season", "week", "team"], how="inner")
    d = d.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    d["pts"], d["one"] = d["half_ppr"], 1.0
    for src, val, ok in (("half_ppr_exp", "xfp", "xfp_ok"), ("routes", "rt", "rt_ok")):
        x = pd.to_numeric(d[src], errors="coerce")
        d[val], d[ok] = x.fillna(0.0), x.notna().astype(float)
    for col, tot in (("receiving_air_yards", "team_air"), ("targets", "team_tgt"), ("carries", "team_car")):
        d[tot] = d.groupby(["season", "week", "team"])[col].transform("sum")
    d["pts2"] = d["pts"] ** 2
    P = prior_sums(d)
    div = lambda a, b: a / b.where(b > 0)
    d["n_prior"] = d.groupby(["gsis_id", "season"]).cumcount()
    for tag, k in (("std", None), ("l3", 3), ("l1", 1)):
        n = P("one", k)
        d[f"pts_{tag}"] = div(P("pts", k), n)
        d[f"xfp_{tag}"] = div(P("xfp", k), P("xfp_ok", k))
        d[f"snap_{tag}"] = div(P("offense_snaps", k), P("team_snaps", k))
        d[f"tgt_{tag}"] = div(P("targets", k), P("team_tgt", k))
        d[f"car_{tag}"] = div(P("carries", k), P("team_car", k))
    n = P("one")
    d["air_std"] = div(P("receiving_air_yards"), P("team_air")).clip(-0.2, 1.0)
    d["routes_std"] = div(P("rt"), P("rt_ok"))
    d["rz_std"], d["gl_std"] = div(P("rz_touches"), n), div(P("gl_touches"), n)
    d["pts_sd"] = (div(P("pts2"), n) - d["pts_std"] ** 2).clip(lower=0) ** 0.5
    d["pace"] = div(P("team_snaps"), n)
    d["pass_rate"] = div(P("team_tgt"), P("team_tgt") + P("team_car"))
    d["att_prior"], d["yds_prior"], d["att_l1"] = P("attempts"), P("passing_yards"), P("attempts", 1)
    # games his team played without him just before this one (0 if he played its last game)
    prev = d.groupby(["gsis_id", "season"])[["gno", "team"]].shift(1)
    d["gap"] = (d["gno"] - prev["gno"] - 1).where(prev["team"] == d["team"]).clip(0, 4).fillna(0.0)

    last = d[d["played"]].groupby(["gsis_id", "season"], as_index=False).agg(
        pts_prev=("pts", "mean"), xs=("xfp", "sum"), xn=("xfp_ok", "sum"))
    last["xfp_prev"] = div(last["xs"], last["xn"])
    last["season"] += 1
    d = d.merge(last[["gsis_id", "season", "pts_prev", "xfp_prev"]], on=["gsis_id", "season"], how="left")
    d = d.merge(defense(d), on=["season", "opp", "pos", "week"], how="left")
    vt, vp = vacated(d)
    d = d.merge(vt, on=["season", "team", "week"], how="left").merge(vp, on=["season", "team", "week", "pos"], how="left")
    d = d.merge(quarterback(d), on=["season", "team", "week"], how="left")
    d = d.merge(injuries(seasons), on=["season", "week", "gsis_id"], how="left")
    for c in ("listed", "questionable", "vac_tgt", "vac_car", "vac_pos"):
        d[c] = d[c].fillna(0.0)
    d["keep"] = ((d["n_prior"] >= MIN_PRIOR) & (d[["pts_std", "xfp_std"]].max(axis=1) >= d["pos"].map(RELEVANT))
                 & (d["implied"].notna() | ~d["played"]))
    cols = list(dict.fromkeys(["season", "week", "gsis_id", "player", "pos", "team", "opp", "pts", "played", "keep"] + ALL))
    out = d[cols].reset_index(drop=True)
    out[ALL] = out[ALL].apply(pd.to_numeric, errors="coerce").astype(float)
    return out


# ------------------------------------------------------------------ the model
def fit(tr: pd.DataFrame, cols: list[str] = ALL) -> dict:
    """Ridge on standardized columns; a missing value takes the training mean."""
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1.0).fillna(1.0)
    X = ((tr[cols] - mu) / sd).fillna(0.0).to_numpy()
    y = tr["pts"].to_numpy()
    beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(len(cols)), X.T @ (y - y.mean()))
    return {"cols": list(cols), "mu": mu, "sd": sd, "beta": beta, "y0": float(y.mean())}


def terms(m: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Each input's part of the prediction in points; a row's terms add up to prediction - y0."""
    return ((rows[m["cols"]] - m["mu"]) / m["sd"]).fillna(0.0) * m["beta"]


def before(d: pd.DataFrame, season: int, week: int) -> pd.Series:
    """The rows a model for (season, week) may learn from: scored games played before it."""
    return d["keep"] & d["played"] & ((d["season"] < season) | ((d["season"] == season) & (d["week"] < week)))


def score(season: int, week: int, d: pd.DataFrame | None = None) -> pd.DataFrame:
    """The model's read of every scored player in one week, from models fitted on everything
    before it: `ml`, `ml_week`, one column per WEEK_GROUPS piece (they add to ml_week) and
    `avg`, his scoring average going in."""
    d = dataset(season, week) if d is None else d
    learn, out = before(d, season, week), []
    for pos in POS:
        tr = d[learn & (d["pos"] == pos)]
        s = d[(d["season"] == season) & (d["pos"] == pos) & (d["week"] <= week)]
        cur = s[(s["week"] == week) & s["keep"]]
        if len(tr) < 200 or cur.empty:
            continue
        m = fit(tr)
        t = terms(m, s)
        grp = pd.DataFrame({g: t[cols].sum(axis=1) for g, cols in WEEK_GROUPS.items()})
        past = s[(s["week"] < week) & s["played"]]
        usual = grp.loc[past.index].groupby(past["gsis_id"]).mean()        # his games so far this season
        rel = grp.loc[cur.index] - usual.reindex(cur["gsis_id"]).fillna(0.0).to_numpy()
        res = cur[["gsis_id", "player", "pos", "team", "opp"]].assign(
            avg=cur["pts_std"], ml=(m["y0"] + t.loc[cur.index].sum(axis=1)).clip(lower=0.0), ml_week=rel.sum(axis=1))
        out.append(pd.concat([res, rel], axis=1))
    if not out:
        return pd.DataFrame(columns=["gsis_id", "player", "pos", "team", "opp", "avg", "ml", "ml_week", *WEEK_GROUPS])
    return pd.concat(out, ignore_index=True)


def call(ml_week: float) -> str:
    return "good" if ml_week >= CALL else "bad" if ml_week <= -CALL else "normal"


def path(season: int, week: int) -> Path:
    return ROOT / "data" / "build" / f"ml_{season}_wk{int(week):02d}.csv"


def publish(season: int, week: int) -> pd.DataFrame:
    """Score the week and write data/build/ml_<season>_wk<NN>.csv — the file the scorecard
    reads back out of git as it stood at kickoff, to grade what was actually published."""
    s = score(season, week)
    if not s.empty:
        s.assign(call=s["ml_week"].map(call)).round(3).to_csv(path(season, week), index=False)
    return s


def for_export(s: pd.DataFrame) -> dict:
    """gsis_id -> what the site shows: the projection, this week against his usual, the call
    and its reasons (largest first; the ones under WHY_MIN are left off)."""
    out = {}
    for r in s.to_dict("records"):
        why = sorted(({"label": g, "pts": round(float(r[g]), 1)} for g in WEEK_GROUPS if abs(r[g]) >= WHY_MIN),
                     key=lambda x: -abs(x["pts"]))
        out[r["gsis_id"]] = {"proj": round(float(r["ml"]), 1), "week": round(float(r["ml_week"]), 1),
                             "call": call(float(r["ml_week"])), "avg": round(float(r["avg"]), 1), "why": why}
    return out
