"""Refit and backtest mega/matchup_model.py — the "how much does this matchup move him" model.

    PYTHONPATH=. .venv/bin/python tools/fit_matchup.py

The constants in data/matchup_constants.json were fit once, offline, with nothing in the
repo to reproduce them. This rebuilds the fit so every change is measured the same way:

  * player-games 2016-2025 (2015 seeds the first prior), skill positions, fantasy-relevant
    (a baseline of MINB+ half-PPR); a player's baseline is his mean over his OTHER games that
    season, so the game being scored never informs its own baseline;
  * a defense is rated only from games BEFORE the week being scored: points it allowed to a
    position over those players' baselines, shrunk toward last season's rating (k, rho per
    position), recent games weighted by half-life `hl`;
  * expected points = baseline x rating^b_def x (implied total / team average)^b_impl
    x exp(b_spread * spread / 7), the exponents fit by Poisson regression (IRLS, offset
    log baseline);
  * trained on 2016-2022, scored on 2023-2025 by Poisson deviance against baseline-only.

Variants are compared on the same held-out games; the winner is written to
data/matchup_constants.json with its backtest.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

POS = ["QB", "RB", "WR", "TE"]
MINB = {"QB": 12, "RB": 6, "WR": 6, "TE": 4}
SEASONS = list(range(2015, 2026))
TRAIN = list(range(2016, 2023))
TEST = [2023, 2024, 2025]
FIT_SEL, VAL_SEL = list(range(2016, 2020)), [2020, 2021, 2022]      # settings are chosen here, never on TEST


# ------------------------------------------------------------------ data
def load_rows() -> pd.DataFrame:
    import nflreadpy as nfl
    from mega.intel import weekly
    out = []
    for s in SEASONS:
        w = weekly(s)
        w = w[w["season_type"].astype(str).str.upper().eq("REG") & w["pos"].isin(POS)].copy()
        n = lambda c: pd.to_numeric(w.get(c), errors="coerce").fillna(0.0) if c in w.columns else 0.0
        w["rush_pts"] = 0.1 * n("rushing_yards") + 6 * n("rushing_tds")
        w["rec_pts"] = 0.5 * n("receptions") + 0.1 * n("receiving_yards") + 6 * n("receiving_tds")
        w["pass_pts"] = 0.04 * n("passing_yards") + 4 * n("passing_tds") - n("passing_interceptions")
        g = w.groupby("gsis_id")
        cnt, tot = g["half_ppr"].transform("size"), g["half_ppr"].transform("sum")
        w["base"] = (tot - w["half_ppr"]) / (cnt - 1).replace(0, np.nan)
        for c in ("rush_pts", "rec_pts", "pass_pts"):
            ct = g[c].transform("sum")
            w[f"base_{c}"] = (ct - w[c]) / (cnt - 1).replace(0, np.nan)
        w = w[(cnt >= 5) & (w["base"] >= w["pos"].map(MINB))]
        sch = nfl.load_schedules(seasons=[s]).to_pandas()
        sch = sch[sch["game_type"] == "REG"]
        lines = []
        for side, sign in (("home_team", 1), ("away_team", -1)):
            lines.append(pd.DataFrame({"team": sch[side], "week": sch["week"].astype(int),
                                       "implied": sch["total_line"] / 2 + sign * sch["spread_line"] / 2,
                                       "spread": sign * sch["spread_line"]}))
        L = pd.concat(lines)
        L["impl_rel"] = L["implied"] / L.groupby("team")["implied"].transform("mean")
        w = w.merge(L, on=["team", "week"], how="left")
        w["season"] = s
        out.append(w[["season", "week", "gsis_id", "pos", "team", "opponent_team", "half_ppr", "base",
                      "rush_pts", "rec_pts", "pass_pts", "base_rush_pts", "base_rec_pts", "base_pass_pts",
                      "implied", "spread", "impl_rel"]])
    return pd.concat(out, ignore_index=True).dropna(subset=["base", "impl_rel"])


# ------------------------------------------------------------------ defense ratings
def ratings(df: pd.DataFrame, pos: str, k: float, rho: float, hl: float | None,
            actual: str = "half_ppr", base: str = "base") -> pd.Series:
    """Per row of `df` at `pos`: the opponent defense's rating from games before that week."""
    d = df[df["pos"] == pos]
    gbar = float(d[base].mean())
    out = pd.Series(np.nan, index=d.index)
    prior_by_season = {}
    for s in sorted(d["season"].unique()):
        ds = d[d["season"] == s]
        # last season's full-season rating, shrunk, is the prior
        prev = prior_by_season.get(s - 1, {})
        agg = ds.groupby(["opponent_team", "week"])[[actual, base]].sum().rename(columns={actual: "A", base: "B"})
        teams = sorted(set(ds["opponent_team"]))
        weeks = sorted(ds["week"].unique())
        A = agg["A"].unstack(fill_value=0.0).reindex(index=teams, columns=weeks, fill_value=0.0)
        B = agg["B"].unstack(fill_value=0.0).reindex(index=teams, columns=weeks, fill_value=0.0)
        pri = pd.Series({t: prev.get(t, 1.0) for t in teams})
        pri = pri / pri.mean()
        m = 1 + rho * (pri - 1)
        for wk in weeks:
            past = [x for x in weeks if x < wk]
            if past:
                wt = np.array([0.5 ** ((wk - x) / hl) if hl else 1.0 for x in past])
                a, b = (A[past] * wt).sum(axis=1), (B[past] * wt).sum(axis=1)
            else:
                a = b = pd.Series(0.0, index=teams)
            r = (a + k * gbar * m) / (b + k * gbar)
            r = r / r.mean()
            rows = ds.index[ds["week"] == wk]
            out.loc[rows] = ds.loc[rows, "opponent_team"].map(r).values
        full = (A.sum(axis=1) + k * gbar) / (B.sum(axis=1) + k * gbar)
        prior_by_season[s] = (full / full.mean()).to_dict()
    return out


# ------------------------------------------------------------------ Poisson GLM
def poisson_fit(y, X, offset, iters: int = 30) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = offset + X @ beta
        mu = np.exp(eta)
        z = (eta - offset) + (y - mu) / mu
        W = mu
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X + 1e-9 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            return beta_new
        beta = beta_new
    return beta


def deviance(y, mu) -> float:
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(t - (y - mu)))


def design(d: pd.DataFrame, feats: list[str]) -> np.ndarray:
    cols = {"def": np.log(d["rating"].clip(lower=1e-3)), "impl": np.log(d["impl_rel"].clip(lower=1e-3)),
            "spread": d["spread"] / 7.0}
    return np.column_stack([cols[f] for f in feats])


def evaluate(df: pd.DataFrame, pos: str, k: float, rho: float, hl: float | None, feats: list[str],
             rating_col: pd.Series | None = None, train=None, test=None) -> dict:
    train, test = train or TRAIN, test or TEST
    d = df[df["pos"] == pos].copy()
    d["rating"] = ratings(df, pos, k, rho, hl) if rating_col is None else rating_col
    d = d.dropna(subset=["rating"])
    y = np.maximum(d["half_ppr"].to_numpy(), 0.0)          # Poisson needs y >= 0; negative weeks are rare
    off = np.log(d["base"].to_numpy())
    tr, te = d["season"].isin(train).to_numpy(), d["season"].isin(test).to_numpy()
    X = design(d, feats)
    beta = poisson_fit(y[tr], X[tr], off[tr])
    mu = np.exp(off + X @ beta)
    base_dev = deviance(y[te], np.exp(off[te]))
    dev = deviance(y[te], mu[te])
    return {"pos": pos, "k": k, "rho": rho, "hl": hl, "feats": feats, "beta": dict(zip(feats, map(float, beta))),
            "n_test": int(te.sum()), "dev_gain_pct": round(100 * (base_dev - dev) / base_dev, 3),
            "mae_base": round(float(np.mean(np.abs(y[te] - np.exp(off[te])))), 4),
            "mae": round(float(np.mean(np.abs(y[te] - mu[te]))), 4), "_mu": mu, "_te": te, "_d": d}


def choose(df, pos, feats_options=(("def", "impl"),), hls=(None,)) -> dict:
    """Shrinkage, recency and features picked on 2020-22 (fit on 2016-19); TEST never seen."""
    best = None
    for k, rho, hl, feats in itertools.product((4, 6, 12, 20, 30, 45, 60), (0.3, 0.5, 0.7, 0.85), hls, feats_options):
        r = evaluate(df, pos, k, rho, hl, list(feats), train=FIT_SEL, test=VAL_SEL)
        if best is None or r["dev_gain_pct"] > best["dev_gain_pct"]:
            best = r
    return {"k": best["k"], "rho": best["rho"], "hl": best["hl"], "feats": best["feats"]}


def future_weeks(df: pd.DataFrame, pos: str, k: float, rho: float) -> dict:
    """Weeks past the posted lines: today's defense-only formula vs the full formula fed
    market-priced lines (mega/power.py, fit on the lines posted by "now"). Every "now" from
    week 4 to 12 of the TEST seasons; exponents fit on TRAIN, as production uses them."""
    import nflreadpy as nfl
    from mega import power
    d = df[df["pos"] == pos].copy()
    d["rating"] = ratings(df, pos, k, rho, None)
    d = d.dropna(subset=["rating"])
    y, off = np.maximum(d["half_ppr"].to_numpy(), 0.0), np.log(d["base"].to_numpy())
    tr = d["season"].isin(TRAIN).to_numpy()
    b_full = poisson_fit(y[tr], design(d, ["def", "impl"])[tr], off[tr])
    b_def = poisson_fit(y[tr], design(d, ["def"])[tr], off[tr])
    dev_def = dev_mod = dev_base = 0.0
    n = 0
    for s in TEST:
        sch = nfl.load_schedules(seasons=[s]).to_pandas()
        sch = sch[sch["game_type"] == "REG"]
        ds = d[d["season"] == s]
        for now in range(4, 13):
            m = power.fit(sch[sch["week"] <= now + 1])
            fut = ds[ds["week"] >= now + 2]
            if fut.empty or not m:
                continue
            imp = {}
            for g in sch.itertuples():
                if g.week <= now + 1:
                    tot, sp = g.total_line, g.spread_line
                else:
                    p_ = power.predict(m, g.home_team, g.away_team)
                    if p_ is None:
                        continue
                    tot, sp = p_
                imp[(g.home_team, int(g.week))] = tot / 2 + sp / 2
                imp[(g.away_team, int(g.week))] = tot / 2 - sp / 2
            mean = pd.Series(imp).groupby(level=0).mean()
            rel = np.array([imp.get((t, int(w)), np.nan) / mean.get(t, np.nan) for t, w in zip(fut["team"], fut["week"])])
            ok = np.isfinite(rel)
            fy, fo = y[d.index.get_indexer(fut.index)][ok], off[d.index.get_indexer(fut.index)][ok]
            lr = np.log(fut["rating"].to_numpy()[ok])
            mu_def = np.exp(fo + b_def[0] * lr)
            mu_mod = np.exp(fo + b_full[0] * lr + b_full[1] * np.log(rel[ok]))
            dev_def += deviance(fy, mu_def)
            dev_mod += deviance(fy, mu_mod)
            dev_base += deviance(fy, np.exp(fo))
            n += int(ok.sum())
    return {"n": n, "defense_only_gain_pct": round(100 * (dev_base - dev_def) / dev_base, 3),
            "with_modeled_lines_gain_pct": round(100 * (dev_base - dev_mod) / dev_base, 3)}


def main() -> None:
    df = load_rows()
    print(f"[matchup] {len(df):,} player-games, {df['season'].min()}-{df['season'].max()}")
    C = json.loads((ROOT / "data" / "matchup_constants.json").read_text())
    line = lambda r: f"dev -{r['dev_gain_pct']:.3f}%  MAE {r['mae']:.4f} (base {r['mae_base']:.4f})"
    for pos in POS:
        c = C[pos]
        cur = evaluate(df, pos, c["k"], c["rho"], None, ["def", "impl"])
        print(f"[{pos}] now (k={c['k']}, rho={c['rho']}):  {line(cur)}")
        if "--quick" not in sys.argv:
            ch = choose(df, pos, feats_options=(("def", "impl"), ("def", "impl", "spread")), hls=(None, 2.0, 4.0, 8.0))
            r = evaluate(df, pos, ch["k"], ch["rho"], ch["hl"], ch["feats"])
            print(f"[{pos}] chosen on 2020-22 {ch}:  {line(r)}  {({k: round(v, 3) for k, v in r['beta'].items()})}")
        fw = future_weeks(df, pos, c["k"], c["rho"])
        print(f"[{pos}] weeks past the lines (n={fw['n']}): defense only -{fw['defense_only_gain_pct']}%, "
              f"with modeled lines -{fw['with_modeled_lines_gain_pct']}%")


if __name__ == "__main__":
    main()
