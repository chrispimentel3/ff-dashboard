"""Which data points predict a player's next game, and does the model built on them hold up?
The research behind mega/ml_matchup.py.

    PYTHONPATH=. .venv/bin/python tools/fit_ml_matchup.py

Player-games since 2015 (mega.ml_matchup.dataset: every feature from before kickoff; the target
is his half-PPR points in a game he played). Scored walk-forward: each season from FIRST_TEST on
is predicted by a ridge regression fitted on every earlier season, per position.

  * The screen. For each family of data points: `alone` is the error cut when the family is
    added to his scoring average and nothing else; `unique` is the error added back when it is
    removed from the model that has everything. Both are % of squared error, with a standard
    error from season-week clusters.
  * The calls. mega.ml_matchup's good-week / bad-week call (this week's situation against his
    usual one), graded on what those players then scored against the rest of the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mega import ml_matchup as M  # noqa: E402

FIRST_TEST = 2019
POS, FAMILIES, BASE, ALL = M.POS, M.FAMILIES, M.BASE, M.ALL


def build(season: int | None = None) -> pd.DataFrame:
    """The scored games: played, with enough behind him to have features."""
    if season is None:
        import nflreadpy as nfl
        season = int(nfl.get_current_season())
    d = M.dataset(season)
    return d[d["keep"] & d["played"]].reset_index(drop=True)


def walk_forward(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Out-of-sample predictions: each test season from a model fitted on earlier seasons, per position."""
    out = pd.Series(np.nan, index=d.index)
    for pos in POS:
        p = d[d["pos"] == pos]
        for s in sorted(x for x in p["season"].unique() if x >= FIRST_TEST):
            te, m = p[p["season"] == s], M.fit(p[p["season"] < s], cols)
            out.loc[te.index] = m["y0"] + M.terms(m, te).sum(axis=1)
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


def calls(season: int) -> pd.DataFrame:
    """Every good / normal / bad call the site's rule would have made on the test seasons (models
    fitted on earlier seasons; "his usual" from his own earlier games that season), with what he
    scored against the rest of the model (`over` = points above the projection less this week's part)."""
    d = M.dataset(season)
    out = []
    for pos in POS:
        p = d[d["pos"] == pos]
        for s in sorted(x for x in p["season"].unique() if x >= FIRST_TEST):
            m = M.fit(p[p["keep"] & p["played"] & (p["season"] < s)])
            g = p[(p["season"] == s) & p["played"]].sort_values(["gsis_id", "week"])
            t = M.terms(m, g)
            grp = pd.DataFrame({k: t[c].sum(axis=1) for k, c in M.WEEK_GROUPS.items()})
            by = grp.groupby(g["gsis_id"])
            n = by.cumcount()
            usual = ((by.cumsum() - grp).div(n.where(n > 0), axis=0)).fillna(0.0)      # his earlier games
            wk = (grp - usual).sum(axis=1)
            k = g["keep"]
            out.append(pd.DataFrame({"season": s, "week": g.loc[k, "week"], "pos": pos, "ml_week": wk[k],
                                     "over": g.loc[k, "pts"] - (m["y0"] + t.loc[k].sum(axis=1) - wk[k])}))
    c = pd.concat(out, ignore_index=True)
    c["call"] = c["ml_week"].map(M.call)
    return c


def main() -> None:
    d = build()
    season = int(d["season"].max())
    print(f"player-games: {len(d):,} ({d['season'].min()}-{season}); tested on {FIRST_TEST}+: "
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
    c = calls(season)
    print(f"\nTHE CALLS (this week's situation moves him {M.CALL:g}+ points against his usual), {FIRST_TEST}+")
    g = c.groupby("call")
    print(pd.DataFrame({"n": g.size(), "share": g.size() / len(c), "said": g["ml_week"].mean(), "happened": g["over"].mean(),
                        "se": g["over"].std() / g.size() ** 0.5}).reindex(["good", "normal", "bad"]).to_string())
    gp = c.groupby(["pos", "call"])
    print(pd.DataFrame({"n": gp.size(), "said": gp["ml_week"].mean(), "happened": gp["over"].mean()}).unstack("call").to_string())


if __name__ == "__main__":
    main()
