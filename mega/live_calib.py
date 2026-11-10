"""Weekly self-calibration: three knobs, each refit from this season's graded weeks and each
switched on only once it has beaten "leave it alone" walk-forward. This is the same pattern as
mega/scorecard.live_blend (shrink toward the prior, gate on a walk-forward test), applied to
the three places the 2026 scorecard shows we aren't learning:

  1. defense   proj x rating(opp, pos) ** beta[pos]. The fitted defense rating
               (mega/matchup_model.defense_ratings) never reaches the number Start/Sit sets the
               lineup from: proj_source "ours" gets adj_factor 1.0 in lineup.optimize_lineup.
               Only the DEFENSE part is applied, because ours already prices Vegas through
               forward.GAMMA; the full tag would count Vegas twice.
  2. avail     P(plays) recalibrated by bucket. In 2026 wks 3-9, "p_active = 1.0" players
               played 95.2% of the time (930 player-weeks).
  3. odds      team-mean gap shrink in the title-odds sim. The weekly team spread is right
               (sim 20.2 vs 19.5 actual), so overconfident win odds mean the GAPS between
               team means are too wide. lam = 1/s, from z' = z / s on graded games.

Every knob is shrunk toward "no change" (or its historical fit) by a prior worth N0 samples,
so nine weeks of noise can't swing it; gates mean an un-earned knob stays off.

Input rows are the scorecard's graded player-weeks (scorecard.grade_projections rows) plus:
  played (bool), p_active (as published), rating (opp defense rating as published).
The scorecard passes `ours_raw` as `ours`, so a knob never learns from its own output.

Written by mega/scorecard.build to data/build/live_calib.json; read through `load()` by
mega/proj_ros.build (defense, availability) and mega/title_odds (odds). No scipy: the Render
service imports title_odds and its requirements don't carry it, so the one-dimensional fits
use a golden-section search and the normal curve comes from the standard library.
"""
from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import DATA

CALIB = DATA / "build" / "live_calib.json"
POS = ("QB", "RB", "WR", "TE")
MIN_TESTED = 4          # walk-forward weeks a knob must win before it switches on (live_blend's bar)
N0_DEF = 300            # player-weeks of the 2016-25 exponent in every in-season beta
N0_AVAIL = 200          # player-weeks of "as published" in every availability bucket
ODDS_PRIOR_SD = 0.25    # log(s) ~ N(0, 0.25^2): s in [0.6, 1.6] at 2 sd
MIN_GAMES = 30          # graded league games before the odds knob may switch on
BUCKETS = [(-0.01, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]


_N = NormalDist()


def _ppf(p) -> np.ndarray:
    return np.array([_N.inv_cdf(float(x)) for x in np.atleast_1d(p)])


def _logcdf(x) -> np.ndarray:
    return np.log(np.array([_N.cdf(float(v)) for v in np.atleast_1d(x)]))


def _argmin(f, lo: float, hi: float, tol: float = 1e-7) -> float:
    """Minimum of a one-humped f on [lo, hi]: golden-section search (scipy's bounded
    minimize_scalar does the same job with Brent's method)."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def load(season: int) -> dict:
    """The knobs the last scorecard build wrote, for this season; {} when there are none, so
    every caller behaves exactly as before."""
    try:
        d = json.loads(CALIB.read_text())
    except (OSError, ValueError):
        return {}
    return d if int(d.get("season") or 0) == int(season) else {}


def b_hist() -> dict:
    """The 2016-25 defense exponents (data/matchup_constants.json b_def) every beta shrinks toward."""
    from .matchup_model import constants
    c = constants()
    return {p: float(c[p]["b_def"]) for p in POS}


# ------------------------------------------------------------------ 1. defense exponent
def _beta_mle(y, mu0, r) -> float:
    """Poisson MLE of beta in E[y] = mu0 * r**beta (one parameter, bounded)."""
    nll = lambda b: float(np.sum(mu0 * r ** b - y * np.log(mu0 * r ** b)))
    return _argmin(nll, -3, 3)


def beta_shrunk(d: pd.DataFrame, pos: str, b_hist: float) -> float:
    x = d[(d["pos"] == pos) & d["played"] & (d["ours"] > 0) & d["rating"].notna()]
    if len(x) < 20:
        return b_hist
    b = _beta_mle(x["actual"].clip(lower=0).values, x["ours"].values, x["rating"].values)
    return (len(x) * b + N0_DEF * b_hist) / (len(x) + N0_DEF)


def _dev(mu, y) -> float:
    y = np.clip(y, 0, None)
    return float(2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0.0) - (y - mu)))


def fit_defense(rows: list[dict], b_hist: dict) -> dict:
    """pos -> {beta, tested, dev_gain_pct, mae_gain, active}. Walk-forward: each week is scored
    with the beta fitted on the weeks before it. Active only when BOTH deviance (is the mean
    right?) and MAE (the scorecard's own metric) improve — a knob that helps one and hurts the
    other isn't earning its place in the lineup."""
    d = pd.DataFrame(rows)
    d = d[d["ours"] > 0]
    weeks = sorted(d["week"].unique())
    out = {}
    for pos in POS:
        dev0 = dev1 = mae0 = mae1 = 0.0
        tested = 0
        for w in weeks[1:]:
            t = d[(d["week"] == w) & (d["pos"] == pos) & d["rating"].notna()]
            if t.empty:
                continue
            b = beta_shrunk(d[d["week"] < w], pos, b_hist[pos])
            adj = t["ours"] * t["rating"] ** b
            p = t["played"]
            dev0 += _dev(t.loc[p, "ours"].values, t.loc[p, "actual"].values)
            dev1 += _dev(adj[p].values, t.loc[p, "actual"].values)
            mae0 += float((t["ours"] - t["actual"]).abs().sum())
            mae1 += float((adj - t["actual"]).abs().sum())
            tested += 1
        n = max(len(d[(d["pos"] == pos) & (d["week"] > weeks[0])]), 1)
        dev_gain = 100 * (dev0 - dev1) / dev0 if dev0 else 0.0
        mae_gain = (mae0 - mae1) / n
        out[pos] = {"beta": round(beta_shrunk(d, pos, b_hist[pos]), 3), "tested": tested,
                    "dev_gain_pct": round(dev_gain, 2), "mae_gain": round(mae_gain, 3),
                    "active": tested >= MIN_TESTED and dev_gain > 0 and mae_gain > 0}
    return out


# ------------------------------------------------------------------ 2. availability
def _bucket(p: float) -> int:
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo < p <= hi)


def fit_availability(rows: list[dict]) -> dict:
    """bucket -> {said, p, n}: P(plays) as published, shrunk toward itself by N0_AVAIL.
    Active when the walk-forward Brier score beats the published numbers."""
    d = pd.DataFrame(rows)
    d = d.assign(pa=d["p_active"].fillna(1.0).astype(float))
    d["b"] = d["pa"].map(_bucket)

    def table(x):
        t = {}
        for b, g in x.groupby("b"):
            said = float(g["pa"].mean())
            t[int(b)] = {"said": round(said, 3), "n": len(g),
                         "p": round((float(g["played"].sum()) + N0_AVAIL * said) / (len(g) + N0_AVAIL), 3)}
        return t

    weeks = sorted(d["week"].unique())
    br0 = br1 = 0.0
    tested = 0
    for w in weeks[1:]:
        tb, t = table(d[d["week"] < w]), d[d["week"] == w]
        q = t.apply(lambda r: r["pa"] * (tb[r["b"]]["p"] / tb[r["b"]]["said"]) if r["b"] in tb and tb[r["b"]]["said"] else r["pa"], axis=1)
        y = t["played"].astype(float)
        br0 += float(((t["pa"] - y) ** 2).sum())
        br1 += float(((q.clip(0, 1) - y) ** 2).sum())
        tested += 1
    return {"buckets": table(d), "tested": tested, "brier_gain": round((br0 - br1) / max(len(d), 1), 5),
            "active": tested >= MIN_TESTED and br1 < br0}


def apply_availability(p_active: float, calib: dict) -> float:
    """What proj_ros uses: the published P(plays), rescaled by its bucket's learned ratio."""
    if not calib.get("active"):
        return p_active
    t = calib["buckets"].get(_bucket(p_active)) or calib["buckets"].get(str(_bucket(p_active)))
    return p_active if not t or not t["said"] else min(1.0, p_active * t["p"] / t["said"])


# ------------------------------------------------------------------ 3. win-odds gap shrink
def fit_odds(games: list[dict]) -> dict:
    """games: {p (sim's P(team A wins) at kickoff), won (A won)}. Fits s in P' = Phi(Phi^-1(p)/s)
    with a log-normal prior at 1, and reports lam = 1/s, the factor the sim shrinks each team's
    weekly mean toward the league mean by. Active once MIN_GAMES are graded AND the shrunk fit
    beats s=1 walk-forward on log-loss."""
    if not games:
        return {"s": 1.0, "lam": 1.0, "n": 0, "active": False}
    g = pd.DataFrame(games)

    def fit(x):
        z, y = _ppf(x["p"].clip(0.01, 0.99)), x["won"].values.astype(bool)
        nll = lambda ls: -float(np.sum(np.where(y, _logcdf(z / math.exp(ls)), _logcdf(-z / math.exp(ls)))))
        return math.exp(_argmin(lambda ls: nll(ls) + ls ** 2 / (2 * ODDS_PRIOR_SD ** 2), -1.5, 1.5))

    ll0 = ll1 = 0.0
    for w in sorted(g["week"].unique())[1:]:
        s = fit(g[g["week"] < w])
        t = g[g["week"] == w]
        z, y = _ppf(t["p"].clip(0.01, 0.99)), t["won"].values.astype(bool)
        ll0 += float(np.sum(np.where(y, _logcdf(z), _logcdf(-z))))
        ll1 += float(np.sum(np.where(y, _logcdf(z / s), _logcdf(-z / s))))
    s = fit(g)
    return {"s": round(s, 3), "lam": round(1 / s, 3), "n": len(g), "logloss_gain": round((ll1 - ll0) / len(g), 4),
            "active": len(g) >= MIN_GAMES and ll1 > ll0}


def shrink_team_week(total: np.ndarray, league_mean: float, lam: float) -> np.ndarray:
    """title_odds.Model.team_week: pull this team's simulated week toward the league mean by
    (1 - lam) of its OWN expected gap, leaving its spread around its mean untouched."""
    return total + (1.0 - lam) * (league_mean - float(total.mean()))


# ------------------------------------------------------------------ the weekly run
def build(rows: list[dict], games: list[dict], b_hist: dict) -> dict:
    return {"defense": fit_defense(rows, b_hist), "availability": fit_availability(rows), "odds": fit_odds(games)}


def demo() -> None:
    """Self-check on synthetic data with known answers."""
    rng = np.random.default_rng(0)
    rows = []
    for w in range(3, 13):
        for i in range(300):
            pos = POS[i % 4]
            r = float(rng.choice([0.8, 0.9, 1.0, 1.1, 1.25]))
            ours = float(rng.uniform(6, 20))
            played = bool(rng.random() < 0.95)
            # truth: QB defense exponent 1.0, everyone else 0 (ours already right)
            mu = ours * (r ** 1.0 if pos == "QB" else 1.0)
            rows.append({"week": w, "pos": pos, "ours": ours, "rating": r, "p_active": 1.0, "played": played,
                         "actual": float(rng.poisson(mu)) if played else 0.0})
    hist = {"QB": 0.3, "RB": 0.3, "WR": 0.3, "TE": 0.3}
    dfn = fit_defense(rows, hist)
    assert dfn["QB"]["active"] and dfn["QB"]["beta"] > 0.6, dfn["QB"]
    assert all(not dfn[p]["active"] or abs(dfn[p]["beta"]) < 0.35 for p in ("RB", "WR", "TE")), dfn
    av = fit_availability(rows)
    assert 0.94 < av["buckets"][3]["p"] < 0.99 and av["buckets"][3]["said"] == 1.0, av
    assert apply_availability(1.0, {"active": False}) == 1.0
    # odds: a sim twice as confident as reality -> s near 2 once there's enough data
    games = []
    for w in range(1, 15):
        for _ in range(6):
            z = float(rng.normal(0, 1))
            games.append({"week": w, "p": float(_N.cdf(z)), "won": bool(rng.random() < _N.cdf(z / 2))})
    od = fit_odds(games)
    assert od["s"] > 1.2 and od["lam"] < 0.85 and od["n"] == 84, od
    assert fit_odds(games[:18])["active"] is False                     # too few games to switch on
    t = shrink_team_week(np.array([100.0, 120.0]), 100.0, 0.5)
    assert np.allclose(t, [95.0, 115.0])                                # mean 110 -> 105, spread kept
    print("live_calib demo: ok")


if __name__ == "__main__":
    demo()
