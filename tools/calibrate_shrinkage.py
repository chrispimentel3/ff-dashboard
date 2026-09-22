"""How many games is last season worth? Re-derives projections.PRIOR_GAMES_BY_POS.

    PYTHONPATH=. .venv/bin/python tools/calibrate_shrinkage.py [season]

Anchoring this season's rate to a prior is a shrinkage problem, and the right weight is
the ratio of within-player weekly variance to the between-player spread of true talent,
expressed in games. Three estimates, because any one of them can mislead:

  split-half   cov(odd-week mean, even-week mean) estimates true-talent variance; whatever
               is left in each half's variance is noise. Unbiased but jumpy on small n.
  direct       average within-player weekly variance over the variance of player means,
               with the sampling noise in those means removed.
  holdout      predict each player's second-half mean from his first half shrunk toward a
               prior, and pick the k with the lowest RMSE. This is the one to trust: it
               scores prediction rather than fit.

The holdout's prior here is the position median, while production anchors to the player's
own prior season — a sharper prior, which if anything justifies a little more weight than
these numbers suggest. Take the holdout minimum and round.
"""
from __future__ import annotations

import sys

import numpy as np

from mega import intel

POS = ("QB", "RB", "WR", "TE")
GRID = (0, 1, 2, 3, 5, 8, 12, 20)


def split_half(w) -> dict:
    out = {}
    for pos, g in w.groupby("pos"):
        a = g[g["week"] % 2 == 1].groupby("norm")["half_ppr"].agg(["mean", "size"])
        b = g[g["week"] % 2 == 0].groupby("norm")["half_ppr"].agg(["mean", "size"])
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["size_a"] >= 4) & (j["size_b"] >= 4)]
        if len(j) < 30:
            continue
        between = np.cov(j["mean_a"], j["mean_b"])[0, 1]
        noise = max(1e-9, j["mean_a"].var() - between) * j["size_a"].median()
        out[pos] = noise / max(1e-9, between)
    return out


def direct(w) -> dict:
    out = {}
    for pos, g in w.groupby("pos"):
        c = g.groupby("norm")["half_ppr"].agg(["mean", "var", "size"])
        c = c[c["size"] >= 8]
        if len(c) < 30:
            continue
        within = c["var"].mean()
        true_between = max(1e-9, c["mean"].var() - within / c["size"].mean())
        out[pos] = within / true_between
    return out


def holdout(w, split_week: int = 9) -> dict:
    out = {}
    for pos, g in w.groupby("pos"):
        a = g[g["week"] <= split_week].groupby("norm")["half_ppr"].agg(["mean", "size"])
        b = g[g["week"] > split_week].groupby("norm")["half_ppr"].agg(["mean", "size"])
        j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
        j = j[(j["size_1"] >= 4) & (j["size_2"] >= 4)]
        if len(j) < 25:
            continue
        prior = j["mean_1"].median()
        scored = []
        for k in GRID:
            pred = (j["size_1"] * j["mean_1"] + k * prior) / (j["size_1"] + k)
            scored.append((np.sqrt(((pred - j["mean_2"]) ** 2).mean()), k))
        out[pos] = min(scored)[1]
    return out


def main(season: int = 2025) -> None:
    w = intel.weekly(season)
    if w.empty:
        sys.exit(f"no weekly data for {season}")
    w = w[w["pos"].isin(POS)]
    sh, di, ho = split_half(w), direct(w), holdout(w)
    print(f"{season}: {len(w)} weekly rows\n")
    print(f"{'pos':<5}{'split-half':>12}{'direct':>10}{'holdout':>10}{'in use':>9}")
    from mega.projections import PRIOR_GAMES, PRIOR_GAMES_BY_POS

    for pos in POS:
        live = PRIOR_GAMES_BY_POS.get(pos, PRIOR_GAMES)
        f = lambda d: f"{d[pos]:.1f}" if pos in d else "-"
        print(f"{pos:<5}{f(sh):>12}{f(di):>10}{f(ho):>10}{live:>9}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2025)
