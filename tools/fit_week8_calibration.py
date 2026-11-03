"""Fit the §19.4 judgment defaults on 2026 weeks 1-8 (HANDOFF v1.3 §12.4, §13, §14).

    PYTHONPATH=. python -m tools.fit_week8_calibration [--through 8] [--pool 2021 2022 2023 2024 2025]

Prints old value, fitted value and the evidence for each. Changes nothing on disk: the
constants live in mega/roles.py and mega/forward.py and are edited by hand.

**roles.K** — split-half reliability. Each player's games split into odd and even weeks;
the two halves' rates are correlated across players with at least MIN_HALF_GAMES games in
each half. Rates are demeaned by position first, so "RBs catch fewer passes than WRs"
does not pass for reliability. With r for the median per-half denominator n,
K = n(1 - r) / r is the denominator at which a player's own rate and the baseline weigh
the same. Left unchanged under MIN_PLAYERS qualifiers or r < MIN_R.

**forward.LAMBDA** — for each cut week W, earn z from roles.build on weeks <= W, against
the relative change in xFP share from the last 4 games through W to weeks W+1..W+3.
The slope on clamped z is LAMBDA (earn_adj = 1 + LAMBDA z is exactly that relative change).

**forward.GAMMA / BETA** — team-week regression:
    log(teamXFP_c,w / mean_c) = GAMMA log(implied_w / implied_avg) + BETA_c spread_w / 7
one GAMMA across components, one BETA per component, on games with a posted line.

**holdout** — project weeks W+1..W+3 from data through W with fw.project, old constants vs
fitted, MAE against each player's actual half-PPR expected points. This is the gate for
keeping a change: tools/backtest_ros.py scores proj_ros.project on 2021-25, which none of
these constants reach.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mega import forward as fw
from mega import roles as rl
from mega import season as S

SEASON = 2026
MIN_HALF_GAMES = 3
MIN_PLAYERS = 40
MIN_R = 0.10
LAMBDA_CUTS = (3, 4, 5)
MIN_SHARE = 0.04             # a 1% share doubling is noise, not opportunity
HOLDOUT_CUTS = (4, 5)
MIN_PROJ = 3.0               # holdout scores players projected for at least this many xFP


# ---------------------------------------------------------------- data
def load(season: int, through: int):
    pw = S.player_week(season)
    pw = pw[pd.to_numeric(pw["week"], errors="coerce") <= through]
    ffo = S.ff_opportunity(season)
    ffo = ffo[pd.to_numeric(ffo["week"], errors="coerce") <= through]
    ngs = {k: S.nextgen(season, k) for k in ("receiving", "rushing")}
    return pw, ffo, ngs, S.schedules(season)


def _upto(df, w):
    if df is None or getattr(df, "empty", True):
        return df
    wk = pd.to_numeric(df["week"], errors="coerce")
    return df[(wk <= w) & (wk > 0)]


# ---------------------------------------------------------------- roles.K
def split_half(fr: pd.DataFrame, num: str, den: str, pos: tuple, wopr: bool = False) -> dict:
    d = fr[fr["pos"].isin(pos)].copy()
    d["half"] = np.where(d["week"].astype(int) % 2 == 1, "odd", "even")
    if wopr:
        # WOPR = 1.5 target share + 0.7 air share, each a summed ratio
        cols = ["targets", "team_targets", "receiving_air_yards", "team_air_yards"]
        g = d.groupby(["gsis_id", "pos", "half"]).agg(
            **{c: (c, "sum") for c in cols}, games=("week", "size")).reset_index()
        g = g[(g["team_targets"] > 0) & (g["team_air_yards"] > 0)]
        g["rate"] = (rl.WOPR_W[0] * g["targets"] / g["team_targets"]
                     + rl.WOPR_W[1] * g["receiving_air_yards"] / g["team_air_yards"])
        g["den"] = g["team_targets"]
    else:
        d = d[d[num].notna()]
        g = d.groupby(["gsis_id", "pos", "half"]).agg(
            n=(num, "sum"), den=(den, "sum"), games=("week", "size")).reset_index()
        g = g[g["den"] > 0]
        g["rate"] = g["n"] / g["den"]
    g = g[g["games"] >= MIN_HALF_GAMES]
    g["rate"] = g["rate"] - g.groupby("pos")["rate"].transform("mean")
    w = g.pivot_table(index="gsis_id", columns="half", values=["rate", "den"])
    w = w.dropna()
    if len(w) < 3:
        return {"players": len(w), "r": np.nan, "n": np.nan, "K": np.nan}
    r = float(np.corrcoef(w[("rate", "odd")], w[("rate", "even")])[0, 1])
    n = float(np.median(np.r_[w[("den", "odd")], w[("den", "even")]]))
    k = n * (1 - r) / r if r > 0 else np.nan
    return {"players": int(len(w)), "r": r, "n": n, "K": k}


def fit_k(fr: pd.DataFrame) -> dict:
    out = {}
    for m in rl.METRICS:
        if m.num in fr.columns and m.den in fr.columns:
            out[m.key] = split_half(fr, m.num, m.den, m.pos)
    out["wopr"] = split_half(fr, "", "", ("WR", "TE"), wopr=True)
    return out


# ---------------------------------------------------------------- LAMBDA
def _shares(pw, ffo, w_lo, w_hi, window=None):
    """Total xFP share per player over weeks (w_lo, w_hi], last `window` games if given."""
    f = ffo.rename(columns={"player_id": "gsis_id", "posteam": "team"})
    f = f.assign(week=pd.to_numeric(f["week"], errors="coerce"))
    f = f[(f["week"] > w_lo) & (f["week"] <= w_hi)]
    me = pd.concat([f[["gsis_id", "team", "week"]], fw.xfp_components(f)["xfp_total"]], axis=1)
    tm = pd.concat([f[["team", "week"]], fw.xfp_components(f, team=True)["xfp_total"].rename("tm")], axis=1)
    tm = tm.groupby(["team", "week"]).max().reset_index()
    me = me.merge(tm, on=["team", "week"], how="left").sort_values(["gsis_id", "week"])
    if window:
        me = me[me.groupby("gsis_id").cumcount(ascending=False) < window]
    g = me.groupby("gsis_id")
    return pd.DataFrame({"share": g["xfp_total"].sum() / g["tm"].sum().replace(0, np.nan),
                         "games": g.size(), "team": g["team"].last()})


def lambda_rows(pw, ffo, ngs, cuts=LAMBDA_CUTS) -> pd.DataFrame:
    rows = []
    for W in cuts:
        tab = rl.build(_upto(pw, W), _upto(ffo, W), _upto(ngs["receiving"], W),
                       _upto(ngs["rushing"], W), None)["table"]
        z = fw.earned(tab)[["gsis_id", "z_earn"]].dropna()
        before = _shares(pw, ffo, 0, W, window=4)
        after = _shares(pw, ffo, W, W + 3)
        d = z.merge(before, left_on="gsis_id", right_index=True) \
             .merge(after, left_on="gsis_id", right_index=True, suffixes=("", "_next"))
        d = d[(d["team"] == d["team_next"]) & (d["share"] >= MIN_SHARE) & (d["games"] >= 2)
              & (d["games_next"] >= 2)]
        d["rel"] = d["share_next"] / d["share"] - 1
        d["zc"] = d["z_earn"].clip(-fw.Z_CLAMP, fw.Z_CLAMP)
        rows.append(d.assign(cut=W))
    return pd.concat(rows, ignore_index=True)


def ols(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return b, se


def fit_lambda(d: pd.DataFrame) -> dict:
    # trim the share-change tail: a backup who gets one start reads as +400%
    d = d[d["rel"].between(d["rel"].quantile(0.02), d["rel"].quantile(0.98))]
    X = np.c_[np.ones(len(d)), d["zc"]]
    b, se = ols(X, d["rel"])
    per_cut = {}
    for W, g in d.groupby("cut"):
        bb, ss = ols(np.c_[np.ones(len(g)), g["zc"]], g["rel"])
        per_cut[int(W)] = (float(bb[1]), float(ss[1]), len(g))
    return {"lambda": float(b[1]), "se": float(se[1]), "n": len(d), "per_cut": per_cut}


# ---------------------------------------------------------------- GAMMA / BETA
def team_weeks(season: int, through: int = 18) -> pd.DataFrame:
    ffo = S.ff_opportunity(season)
    if ffo is None or ffo.empty:
        return pd.DataFrame()
    f = ffo.rename(columns={"posteam": "team"})
    f = f.assign(week=pd.to_numeric(f["week"], errors="coerce"))
    f = f[f["week"] <= through]
    comp = fw.xfp_components(f, team=True)
    tw = pd.concat([f[["team", "week"]], comp], axis=1).groupby(["team", "week"]).max().reset_index()
    tw["team"] = tw["team"].map(fw._canon)
    s = S.schedules(season)
    s = s[(s["game_type"] == "REG") & s["spread_line"].notna() & s["total_line"].notna()
          & (pd.to_numeric(s["week"]) <= through)]
    imp = pd.concat([pd.DataFrame({"team": s[side].map(fw._canon), "week": pd.to_numeric(s["week"]),
                                   "implied": s["total_line"] / 2 + sg * s["spread_line"] / 2,
                                   "spread": sg * s["spread_line"]})
                     for side, sg in (("home_team", 1.0), ("away_team", -1.0))])
    d = tw.merge(imp, on=["team", "week"], how="inner")
    d["ratio"] = d["implied"] / d.groupby("team")["implied"].transform("mean")
    for c in fw.COMPONENTS:
        d[f"y_{c}"] = d[f"xfp_{c}"] / d.groupby("team")[f"xfp_{c}"].transform("mean")
    return d.assign(season=season)


def fit_gamma_beta(d: pd.DataFrame) -> dict:
    rows = []
    for i, c in enumerate(fw.COMPONENTS):
        x = d[d[f"y_{c}"] > 0]
        dummies = np.zeros((len(x), len(fw.COMPONENTS)))
        dummies[:, i] = x["spread"] / fw.SPREAD_SCALE
        rows.append(np.c_[np.log(x[f"y_{c}"]), np.log(x["ratio"]), dummies])
    a = np.vstack(rows)
    b, se = ols(a[:, 1:], a[:, 0])
    out = {"gamma": (float(b[0]), float(se[0])), "team_weeks": len(d)}
    for i, c in enumerate(fw.COMPONENTS):
        out[c] = (float(b[1 + i]), float(se[1 + i]))
    return out


# ---------------------------------------------------------------- holdout
def holdout(pw, ffo, ngs, sched, consts: dict) -> dict:
    saved = {"K": dict(rl.K), "GAMMA": fw.GAMMA, "BETA": dict(fw.BETA), "LAMBDA": fw.LAMBDA}
    try:
        rl.K.update(consts.get("K", {}))
        fw.GAMMA = consts.get("GAMMA", fw.GAMMA)
        fw.BETA = consts.get("BETA", fw.BETA)
        fw.LAMBDA = consts.get("LAMBDA", fw.LAMBDA)
        f = ffo.rename(columns={"player_id": "gsis_id"})
        act = pd.concat([f[["gsis_id"]], pd.to_numeric(f["week"], errors="coerce").rename("week"),
                         fw.xfp_components(f)["xfp_total"].rename("act")], axis=1)
        errs = []
        for W in HOLDOUT_CUTS:
            tab = rl.build(_upto(pw, W), _upto(ffo, W), _upto(ngs["receiving"], W),
                           _upto(ngs["rushing"], W), None)["table"]
            pj = fw.project(tab, _upto(pw, W), _upto(ffo, W), sched, range(W + 1, W + 4))
            m = pj.merge(act, on=["gsis_id", "week"], how="inner")
            m = m[m["xfp_fwd"] >= MIN_PROJ]
            errs.append((m["xfp_fwd"] - m["act"]).abs())
        e = pd.concat(errs)
        return {"mae": float(e.mean()), "n": int(len(e))}
    finally:
        rl.K.clear()
        rl.K.update(saved["K"])
        fw.GAMMA, fw.BETA, fw.LAMBDA = saved["GAMMA"], saved["BETA"], saved["LAMBDA"]


# ---------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", type=int, default=8)
    ap.add_argument("--pool", type=int, nargs="*", default=[2021, 2022, 2023, 2024, 2025])
    a = ap.parse_args(argv)
    pw, ffo, ngs, sched = load(SEASON, a.through)
    print(f"2026 weeks in player stats: {sorted(pw['week'].astype(int).unique())}")
    print(f"2026 weeks in ff_opportunity: {sorted(pd.to_numeric(ffo['week']).astype(int).unique())}")

    fitted: dict = {"K": {}}
    print("\n== roles.K — split-half reliability (odd vs even weeks)")
    print(f"{'metric':<15}{'old K':>8}{'r':>8}{'n/half':>9}{'players':>9}{'fit K':>9}  verdict")
    fr = rl.frame(pw, ffo, ngs["receiving"], ngs["rushing"])
    for key, v in fit_k(fr).items():
        ok = v["players"] >= MIN_PLAYERS and pd.notna(v["r"]) and v["r"] >= MIN_R
        if key == "wopr":
            verdict = "not used: WOPR is composed after shrinkage"
        elif ok:
            verdict = "fit"
            fitted["K"][key] = float(round(v["K"], 1))
        else:
            verdict = "unchanged (too few players or r < 0.1)"
        print(f"{key:<15}{rl.K.get(key, np.nan):>8.1f}{v['r']:>8.3f}{v['n']:>9.1f}"
              f"{v['players']:>9}{v['K']:>9.1f}  {verdict}")

    print("\n== forward.LAMBDA — next-3-week share change on earn z")
    lam = fit_lambda(lambda_rows(pw, ffo, ngs))
    for W, (b, se, n) in lam["per_cut"].items():
        print(f"  z through wk {W}, share wks {W + 1}-{W + 3}: slope {b:+.3f} ± {se:.3f} (n={n})")
    print(f"  pooled: {lam['lambda']:+.3f} ± {lam['se']:.3f} (n={lam['n']}), old {fw.LAMBDA}")
    fitted["LAMBDA"] = float(round(np.clip(lam["lambda"], 0.0, 0.10), 3))
    print(f"  -> {fitted['LAMBDA']} (kept within 0-0.10)")

    print("\n== forward.GAMMA / BETA — team-week volume vs implied total and spread")
    d26 = team_weeks(SEASON, a.through)
    gb26 = fit_gamma_beta(d26)
    pool = pd.concat([team_weeks(s) for s in a.pool] + [d26], ignore_index=True)
    gbp = fit_gamma_beta(pool)
    for label, g in (("2026 only", gb26), (f"pooled {a.pool[0]}-2026", gbp)):
        print(f"  {label} ({g['team_weeks']} team-weeks): GAMMA {g['gamma'][0]:.3f} ± {g['gamma'][1]:.3f}; "
              + "; ".join(f"BETA_{c} {g[c][0]:+.3f} ± {g[c][1]:.3f}" for c in fw.COMPONENTS))
    print(f"  old: GAMMA {fw.GAMMA}; BETA {fw.BETA}")
    # 2026 alone is ~230 team-weeks; use the pooled fit, which has the same lines source
    fitted["GAMMA"] = float(round(gbp["gamma"][0], 3))
    fitted["BETA"] = {c: float(round(gbp[c][0], 3)) for c in fw.COMPONENTS}
    print(f"  -> pooled: GAMMA {fitted['GAMMA']}; BETA {fitted['BETA']}")

    print("\n== 2026 holdout: project weeks W+1..W+3 from data through W, W in", HOLDOUT_CUTS)
    base = holdout(pw, ffo, ngs, sched, {})
    print(f"  current constants: MAE {base['mae']:.4f} xFP (n={base['n']})")
    for name, c in (("K", {"K": fitted["K"]}), ("LAMBDA", {"LAMBDA": fitted["LAMBDA"]}),
                    ("GAMMA", {"GAMMA": fitted["GAMMA"]}), ("BETA", {"BETA": fitted["BETA"]}),
                    ("all", fitted)):
        h = holdout(pw, ffo, ngs, sched, c)
        print(f"  with fitted {name:<7}: MAE {h['mae']:.4f} ({h['mae'] - base['mae']:+.4f})")
    return fitted


if __name__ == "__main__":
    main()
