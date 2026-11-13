"""Which data points predict a player's next game? The feature screen behind an ML matchup model.

    PYTHONPATH=. .venv/bin/python tools/fit_ml_matchup.py

Player-games 2015-2026 (data/ask_history, regular season, QB/RB/WR/TE). Every feature is built
only from what was known before kickoff: his games earlier that season (season to date, last 3,
last game), last season, the betting line, what the opposing defense had allowed so far, which
teammates are out, the game setting and his injury tag. The target is his half-PPR points in a
game he played; whether he plays at all is a separate question (mega/contingency.py).

Scored walk-forward: each season from FIRST_TEST on is predicted by a ridge regression fitted
on every earlier season, per position. For each family of data points:

  * alone   error cut when the family is added to his scoring average and nothing else;
  * unique  error added back when the family is removed from the model that has everything.

"alone" says whether a data point carries signal; "unique" says whether it carries signal the
others don't. Both are % of squared error, with a standard error from season-week clusters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

POS = ["QB", "RB", "WR", "TE"]
SEASONS = list(range(2015, 2027))
FIRST_TEST = 2019
MIN_PRIOR = 3                  # games he has played this season before the one being predicted
RELEVANT = {"QB": 10.0, "RB": 5.0, "WR": 5.0, "TE": 3.5}   # prior points or xFP per game
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
RIDGE = 10.0
K_DEF = 3                      # pseudo-games of "average defense" in the opponent rating
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
    "Game setting: home, dome, weather, rest": ["home", "dome", "wind_out", "cold", "thu", "off_bye", "div", "rest_diff"],
    "His health: injury tag, back from absence": ["questionable", "listed", "gap"],
    "Boom-bust history": ["pts_sd"],
    "Team style: pace, pass rate, scoring": ["pace", "pass_rate", "pf_prior"],
    "Week of the season": ["week"],
}
BASE = ["pts_std"]
ALL = BASE + [f for fs in FAMILIES.values() for f in fs]


# ------------------------------------------------------------------ data
def player_weeks(seasons) -> pd.DataFrame:
    from mega import data
    out = []
    for s in seasons:
        try:
            out.append(data.ask_player_week(s).assign(season=s))
        except Exception as e:
            print(f"[data] {s} skipped: {type(e).__name__}: {e}")
    d = pd.concat(out, ignore_index=True)
    d = d[d["pos"].isin(POS) & d["team"].notna() & d["gsis_id"].notna()].copy()
    d["team"] = d["team"].replace(TEAM_FIX)
    for c in COUNTS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d = d[(d["offense_snaps"] > 0) | (d[["attempts", "targets", "carries"]].sum(axis=1) > 0)]
    return d.drop_duplicates(["gsis_id", "season", "week"])


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
            "wind": s["wind"], "temp": s["temp"], "pf": s[f"{me}_score"], "pa": s[f"{opp}_score"]}))
    g = pd.concat(rows, ignore_index=True).sort_values(["season", "team", "week"]).reset_index(drop=True)
    g["wind_out"] = g["wind"].fillna(0.0).where(g["dome"] == 0, 0.0)
    g["cold"] = ((g["temp"] <= 32) & (g["dome"] == 0)).astype(float)
    k = g.groupby(["season", "team"])
    g["gno"] = k.cumcount()
    for src, dst in (("implied", "impl_avg"), ("pf", "pf_prior"), ("pa", "pa_prior")):
        g[dst] = (k[src].cumsum() - g[src]) / g["gno"].where(g["gno"] > 0)
    g["impl_rel"] = g["implied"] / g["impl_avg"]
    opp = g[["season", "week", "team", "pa_prior"]].rename(columns={"team": "opp", "pa_prior": "opp_pa"})
    return g.merge(opp, on=["season", "week", "opp"], how="left").drop(columns=["wind", "temp", "pf", "pa", "impl_avg"])


def prior_sums(d: pd.DataFrame):
    """P(col, k): his total of `col` over his previous k games this season (all of them if k is
    None). `d` is sorted by player, season, week."""
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


def vacated(d: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: the share of the team's targets and carries over its last 3 games that
    belonged to players who don't play this one."""
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
    """Per team-game: how new this week's starting QB is (1 = hasn't thrown a pass for them this
    season) and his yards per attempt so far."""
    q = d[d["pos"] == "QB"]
    k = ["season", "team", "week"]
    start = q[q["attempts"] >= 10].sort_values("attempts").groupby(k, as_index=False).tail(1)
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


def build(seasons=SEASONS) -> pd.DataFrame:
    """One row per player-game with the target (`pts`) and every candidate feature."""
    d = player_weeks(seasons).merge(games(seasons), on=["season", "week", "team"], how="inner")
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
    d["att_prior"], d["yds_prior"] = P("attempts"), P("passing_yards")
    # games his team played without him just before this one (0 if he played its last game)
    prev = d.groupby(["gsis_id", "season"])[["gno", "team"]].shift(1)
    d["gap"] = (d["gno"] - prev["gno"] - 1).where(prev["team"] == d["team"]).clip(0, 4).fillna(0.0)

    last = d.groupby(["gsis_id", "season"], as_index=False).agg(pts_prev=("pts", "mean"), xs=("xfp", "sum"), xn=("xfp_ok", "sum"))
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
    thr = d["pos"].map(RELEVANT)
    keep = (d["n_prior"] >= MIN_PRIOR) & (d[["pts_std", "xfp_std"]].max(axis=1) >= thr) & d["implied"].notna()
    cols = list(dict.fromkeys(["season", "week", "gsis_id", "player", "pos", "team", "opp", "pts"] + ALL))
    out = d.loc[keep, cols].reset_index(drop=True)
    out[ALL] = out[ALL].apply(pd.to_numeric, errors="coerce").astype(float)
    return out


# ------------------------------------------------------------------ the screen
def ridge_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Ridge on standardized columns; a missing value takes the training mean."""
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1.0).fillna(1.0)
    X = ((tr[cols] - mu) / sd).fillna(0.0).to_numpy()
    Z = ((te[cols] - mu) / sd).fillna(0.0).to_numpy()
    y = tr["pts"].to_numpy()
    beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(len(cols)), X.T @ (y - y.mean()))
    return y.mean() + Z @ beta


def walk_forward(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Out-of-sample predictions: each test season from a model fitted on earlier seasons, per position."""
    out = pd.Series(np.nan, index=d.index)
    for pos in POS:
        p = d[d["pos"] == pos]
        for s in sorted(x for x in p["season"].unique() if x >= FIRST_TEST):
            te = p[p["season"] == s]
            out.loc[te.index] = ridge_predict(p[p["season"] < s], te, cols)
    return out


def gain(d: pd.DataFrame, a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """% of squared error model `b` removes relative to model `a`, with its standard error
    from season-week clusters (games in one week share weather, news and slate)."""
    ea, eb = (d["pts"] - a) ** 2, (d["pts"] - b) ** 2
    g = pd.DataFrame({"k": d["season"] * 100 + d["week"], "ea": ea, "diff": ea - eb}).groupby("k").sum()
    tot = g["ea"].sum()
    return 100 * g["diff"].sum() / tot, 100 * g["diff"].std(ddof=1) * len(g) ** 0.5 / tot


def screen(d: pd.DataFrame) -> dict:
    t = d[d["season"] >= FIRST_TEST]
    cut = lambda df, pos: df if pos is None else df[df["pos"] == pos]
    preds = {"base": walk_forward(d, BASE), "full": walk_forward(d, ALL)}
    for name, fs in FAMILIES.items():
        preds[("alone", name)] = walk_forward(d, BASE + fs)
        preds[("without", name)] = walk_forward(d, [c for c in ALL if c not in fs])
    rows = []
    for name in FAMILIES:
        r = {"family": name}
        for pos in [None] + POS:
            x = cut(t, pos)
            tag = pos or "all"
            r[f"alone_{tag}"], r[f"alone_se_{tag}"] = gain(x, preds["base"][x.index], preds[("alone", name)][x.index])
            r[f"unique_{tag}"], r[f"unique_se_{tag}"] = gain(x, preds[("without", name)][x.index], preds["full"][x.index])
        rows.append(r)
    summary = {}
    for pos in [None] + POS:
        x = cut(t, pos)
        mae = lambda p: float((x["pts"] - p[x.index]).abs().mean())
        summary[pos or "all"] = {"n": len(x), "mae_avg": float((x["pts"] - x["pts_std"]).abs().mean()),
                                 "mae_base": mae(preds["base"]), "mae_full": mae(preds["full"]),
                                 "gain_full": gain(x, preds["base"][x.index], preds["full"][x.index])}
    return {"families": pd.DataFrame(rows), "summary": summary, "preds": preds}


def fifths(d: pd.DataFrame, resid: pd.Series, cols: list[str]) -> pd.DataFrame:
    """Points scored above what his average predicted, top fifth of each data point against the
    bottom fifth (within position and season), on the test seasons."""
    t = d[d["season"] >= FIRST_TEST].assign(r=resid)
    rows = []
    for c in cols:
        rk = t.groupby(["pos", "season"])[c].rank(pct=True)
        low = t[c] == t[c].min()
        if t[c].dropna().nunique() <= 2:                 # a yes/no data point: yes against no
            top, bot = t[c] == t[c].max(), low
        elif low.mean() > 0.5:                           # mostly zero (games missed): any against none
            top, bot = t[c] > t[c].min(), low
        else:
            top, bot = rk >= 0.8, rk <= 0.2
        for pos in POS:
            m = t["pos"] == pos
            hi, lo = t.loc[m & top, "r"], t.loc[m & bot, "r"]
            if len(hi) > 50 and len(lo) > 50:
                rows.append({"feature": c, "pos": pos, "top": hi.mean(), "bottom": lo.mean(), "gap": hi.mean() - lo.mean(),
                             "se": (hi.var() / len(hi) + lo.var() / len(lo)) ** 0.5})
    return pd.DataFrame(rows)


def main() -> None:
    d = build()
    print(f"player-games: {len(d):,} ({d['season'].min()}-{d['season'].max()}); tested on {FIRST_TEST}+: "
          f"{int((d['season'] >= FIRST_TEST).sum()):,}")
    s = screen(d)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.2f}")
    for k, v in s["summary"].items():
        print(f"{k:>4}: n {v['n']:>6,}  miss if you use his average {v['mae_avg']:.2f}  "
              f"regressed average {v['mae_base']:.2f}  everything {v['mae_full']:.2f}  "
              f"squared error cut {v['gain_full'][0]:.1f}% (+/-{v['gain_full'][1]:.1f})")
    f = s["families"]
    print("\nALONE: % of squared error removed when added to his scoring average")
    print(f[["family"] + [f"alone_{p}" for p in ["all"] + POS] + ["alone_se_all"]].sort_values("alone_all", ascending=False).to_string(index=False))
    print("\nUNIQUE: % of squared error added back when removed from the full model")
    print(f[["family"] + [f"unique_{p}" for p in ["all"] + POS] + ["unique_se_all"]].sort_values("unique_all", ascending=False).to_string(index=False))
    q = fifths(d, d["pts"] - s["preds"]["base"], [c for c in ALL if c not in BASE])
    print("\nTOP FIFTH vs BOTTOM FIFTH: points above what his average predicted")
    order = [c for c in ALL if c not in BASE]
    print(q.pivot(index="feature", columns="pos", values="gap").reindex(order).to_string())
    print("\n...and how many standard errors each gap is (2+ is unlikely to be chance)")
    print(q.assign(t=q["gap"] / q["se"]).pivot(index="feature", columns="pos", values="t").reindex(order).to_string())


if __name__ == "__main__":
    main()
