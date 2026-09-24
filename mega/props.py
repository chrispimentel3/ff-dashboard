"""§20 Vegas player props → a weekly fantasy projection.

A sportsbook prices every starter's receiving yards, receptions, rush yards and chance of
scoring. That is a projection built by people with money at stake, updated to the hour, and
it already contains the injury news, the weather and the game plan. This module turns those
lines into half-PPR points.

The naive version of this — add up the posted lines and score them — is wrong three
separate ways, and each correction here is measured against 2021-2025 nflverse weekly data
rather than assumed. `tools/fit_props.py` re-derives every constant below.

**1. A line is a median, not a mean.** Books hang the number that splits the betting, and
weekly yardage is right-skewed: a WR priced at 30.5 yards averaged 29.6 but had a median of
24. Taking the line as the projection understates low-volume players badly (~29% on
receiving yards, ~42% on rushing yards) and high-volume players barely at all (~7%). The
skew shrinks as the line rises because volume averages out. Passing yards are the exception
— they are near-symmetric at every level (ratio 1.00), so no correction is applied.

**2. Anytime touchdown is not expected touchdowns.** `player_anytime_td` prices P(scores at
least once). Scoring it as 6 × p undercounts the multi-score games. The textbook fix is a
Poisson inversion, λ = −ln(1−p), and it is wrong in the direction that flatters stars: at
p = 0.56 Poisson implies 1.46 touchdowns per scoring game where the data says 1.31 — half
a point handed to exactly the players already in the lineup. What is
actually true is E[TD] = p × E[TD | scored], and that multiplier is measurable — it runs
from 1.04 for a touchdown-light player to 1.31 for a goal-line back (`TD_MULT`).

**3. Counts are not skewed, they are Poisson.** Receptions and passing touchdowns both
check out against Poisson at the lines books actually hang (pass TDs at 1.5: 0.206 / 0.360
/ 0.498 observed against 0.203 / 0.356 / 0.474 predicted; receptions at 3.5: 0.113 / 0.385
/ 0.668 against 0.110 / 0.395 / 0.686). So a count market is inverted to its mean directly
and the skew correction above is deliberately NOT applied to it — doing both would count
the same effect twice.

Prices are de-vigged before any of this. A -140/+110 pair does not mean P = 0.583; it means
the book is holding about 4%, and the honest read is 0.556.

No scipy: Streamlit Cloud installs from requirements.txt and this app does not carry it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import SCORING

# ---------------------------------------------------------------- fitted constants
# (level, mean/median) from 2021-2025 weekly outcomes, binned by a player-season's own
# mean. Interpolated, clamped at both ends. See tools/fit_props.py.
SKEW = {
    "rec_yds":  ((30.0, 1.293), (47.0, 1.148), (61.0, 1.084), (82.0, 1.068)),
    "rush_yds": ((29.0, 1.420), (51.0, 1.128), (67.0, 1.089), (94.0, 1.025)),
    "pass_yds": ((200.0, 1.00), (285.0, 1.00)),     # symmetric; no correction
    # A quarterback's rushing yards are a different distribution from a running back's and
    # have to be fitted apart. Scrambles arrive at a steadier rate than carries in a
    # committee do, so the skew is far milder: at a 22-yard line it is 1.19, where the
    # running-back curve is still pinned at 1.42. Routing a rushing QB through the RB
    # curve adds about a point out of nothing, on exactly the position where a
    # rushing floor is the whole reason he is started.
    "rush_yds_qb": ((12.8, 1.469), (21.6, 1.193), (32.0, 1.106), (51.9, 1.075)),
}
# (level, sd/mean) — only used to shift a line off its posted number when the two prices
# are lopsided, so precision here matters far less than in SKEW.
CV = {
    "rec_yds":  ((30.0, 0.835), (47.0, 0.664), (61.0, 0.578), (82.0, 0.535)),
    "rush_yds": ((29.0, 0.876), (51.0, 0.623), (67.0, 0.543), (94.0, 0.462)),
    "pass_yds": ((220.0, 0.318), (256.0, 0.271), (284.0, 0.269)),
    "rush_yds_qb": ((12.8, 1.062), (21.6, 0.790), (32.0, 0.662), (51.9, 0.577)),
}
# (P(anytime TD), E[TD | scored at least one]) pooled over 21,458 QB/RB/WR/TE player-games.
TD_MULT = ((0.0752, 1.0375), (0.1376, 1.1018), (0.2207, 1.1289),
           (0.2983, 1.1531), (0.3875, 1.2065), (0.5574, 1.3056))

# Every book posts anytime-TD as "Yes" only — all six of them, on all 16 games of the first
# real sweep — so there is no other side to de-vig against and this haircut is doing the
# whole job on the single biggest component. It is therefore calibrated, not guessed, by
# the one constraint that can check it: a team's players' expected touchdowns have to add
# up to the touchdowns its Vegas total implies (0.1085 offensive TD per implied point,
# fitted over 2,689 team-games). At 0.025 the sum ran 3.4% hot; 0.0306 centres it, which
# implies a ~6.1% two-way hold — high, and normal for this market.
#
# Calibrated on one week (16 games, 32 teams), so it is rounded to 0.030 rather than
# carrying a precision that week cannot support. `python -m tools.fit_props --hold` re-runs
# the check against whatever props are on file; re-run it monthly.
ONE_SIDED_HOLD = 0.030

MARKETS = {
    "player_pass_yds": "pass_yds", "player_pass_tds": "pass_tds",
    "player_pass_interceptions": "pass_int", "player_rush_yds": "rush_yds",
    "player_reception_yds": "rec_yds", "player_receptions": "rec",
    "player_anytime_td": "anytime_td",
}
# What each component is worth, straight off the league's own scoring table.
WEIGHTS = {
    "pass_yds": SCORING["pass_yd"], "pass_tds": SCORING["pass_td"],
    "pass_int": SCORING["pass_int"], "rush_yds": SCORING["rush_yd"],
    "rec_yds": SCORING["rec_yd"], "rec": SCORING["rec"],
    "anytime_td": SCORING["rush_td"],     # rush and rec touchdowns both score 6
}
COUNT = ("pass_tds", "pass_int", "rec")
YARDS = ("pass_yds", "rush_yds", "rec_yds")

# Which markets a book could plausibly post for each position. This decides two things and
# neither is about scoring: what to fill when the book stayed silent, and what "complete"
# means. A quarterback has no receptions market and never will, so counting its absence as
# a gap marked every quarterback in the league incomplete — which made the flag useless on
# the first real sweep, where 332 of 420 players read as patched when almost none were.
# Anything actually priced is still scored, whatever the position: receivers do carry
# rushing lines on jet sweeps.
RELEVANT = {
    "QB": ("pass_yds", "pass_tds", "pass_int", "rush_yds", "anytime_td"),
    "RB": ("rush_yds", "rec_yds", "rec", "anytime_td"),
    "WR": ("rec_yds", "rec", "anytime_td"),
    "TE": ("rec_yds", "rec", "anytime_td"),
}


def relevant_for(pos: object) -> tuple:
    """The markets that position can have. An unknown position asks for everything."""
    return RELEVANT.get(str(pos).upper().strip(), tuple(WEIGHTS))


def _interp(x: float, anchors) -> float:
    """Piecewise linear through `anchors`, held flat beyond either end.

    Extrapolating the skew curve would be actively harmful: continuing the receiving-yards
    slope past the last anchor drives the ratio below 1, which would mean a 120-yard line
    projects under its own number.
    """
    if not np.isfinite(x):
        return float(anchors[0][1])
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(x, xs, ys))


# ---------------------------------------------------------------- odds math
def _curve(table: dict, market: str, pos: str = "") -> tuple:
    """The fitted curve for this market, position-specific where one was fitted."""
    key = f"{market}_qb" if (market == "rush_yds" and str(pos).upper() == "QB") else market
    return table.get(key) or table.get(market) or table["rec_yds"]


def american_to_prob(odds: float) -> float:
    """American price → implied probability, vig included."""
    o = float(odds)
    if not np.isfinite(o) or o == 0:
        return float("nan")
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def devig(p_over: float, p_under: float) -> float:
    """Strip the hold proportionally and return the fair P(over).

    Proportional (a.k.a. multiplicative) de-vigging splits the overround in ratio to each
    side's raw price. It is the standard choice and the only one that needs no extra
    parameter; at the near-even prices props are hung at, the alternatives differ by well
    under a point.
    """
    a, b = float(p_over), float(p_under)
    if np.isfinite(a) and np.isfinite(b) and (a + b) > 0:
        return a / (a + b)
    if np.isfinite(a):                      # only the over is posted
        return max(0.0, min(1.0, a - ONE_SIDED_HOLD))
    if np.isfinite(b):
        return max(0.0, min(1.0, 1.0 - (b - ONE_SIDED_HOLD)))
    return float("nan")


def _phi_inv(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, ~1e-9 absolute)."""
    if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
        return 0.0
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _pois_sf(k: int, lam: float) -> float:
    """P(X > k) for integer k under Poisson(lam)."""
    if lam <= 0:
        return 0.0
    term, cdf = math.exp(-lam), math.exp(-lam)
    for i in range(1, int(k) + 1):
        term *= lam / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


# ---------------------------------------------------------------- line → mean
def mean_from_count(line: float, p_over: float) -> float:
    """Invert a count market to its mean under Poisson.

    `over 4.5 receptions` settles on 5 or more, so the target is P(X > floor(line)).
    Monotone in lambda, so a bisection is exact to tolerance and needs no derivative.
    """
    if not (np.isfinite(line) and np.isfinite(p_over)):
        return float("nan")
    k = int(math.floor(float(line)))
    p = min(max(float(p_over), 1e-6), 1 - 1e-6)
    lo, hi = 1e-6, max(2.0, 4.0 * (k + 1))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _pois_sf(k, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_from_yards(line: float, p_over: float, market: str, pos: str = "") -> float:
    """Yardage line → expected yards: recentre on the median, then correct for skew.

    Two steps, deliberately separate. The price only says where the true median sits
    relative to the posted number (a lopsided -140 means the centre is above the line); the
    skew ratio then converts that median into a mean. Folding them together would hide
    which correction is doing the work.
    """
    if not np.isfinite(line):
        return float("nan")
    L = float(line)
    p = float(p_over) if np.isfinite(p_over) else 0.5
    sigma = _interp(L, _curve(CV, market, pos)) * L
    median = L + sigma * _phi_inv(p)
    median = max(0.0, median)
    return median * _interp(median, _curve(SKEW, market, pos))


def expected_td(p_anytime: float) -> float:
    """E[touchdowns] = P(scores) × E[TD | scored].

    Poisson's λ = −ln(1−p) is the tempting closed form and it overstates the top of the
    range by ~12%, because a player who scores twice in a game is rarer than independence
    implies — the second score needs a second trip inside the ten.
    """
    if not np.isfinite(p_anytime):
        return float("nan")
    p = max(0.0, min(1.0, float(p_anytime)))
    return p * _interp(p, TD_MULT)


def poisson_td(p_anytime: float) -> float:
    """The Poisson reading of an anytime-TD price. Kept for the comparison test only."""
    p = max(0.0, min(1.0 - 1e-9, float(p_anytime)))
    return -math.log(1.0 - p)


# ---------------------------------------------------------------- assembly
def to_means(lines: pd.DataFrame) -> pd.DataFrame:
    """One de-vigged mean per player per component.

    `lines` is the long form written by mega.odds: gsis_id, player, market, line,
    price_over, price_under. Books disagree, so each (player, market) is reduced to the
    median across books before inversion — a single stale book cannot then drag a
    projection, and the median needs no weighting scheme to defend.
    """
    cols = ["gsis_id", "player", "team", "pos", "component", "mean", "line", "p_over", "books"]
    if lines is None or lines.empty:
        return pd.DataFrame(columns=cols)
    d = lines.copy()
    d["component"] = d["market"].map(MARKETS)
    d = d[d["component"].notna()]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["p_raw_over"] = pd.to_numeric(d.get("price_over"), errors="coerce").map(american_to_prob)
    d["p_raw_under"] = pd.to_numeric(d.get("price_under"), errors="coerce").map(american_to_prob)
    d["p_over"] = [devig(a, b) for a, b in zip(d["p_raw_over"], d["p_raw_under"])]
    d["line"] = pd.to_numeric(d["line"], errors="coerce")

    if "pos" not in d.columns:
        d["pos"] = ""
    d["pos"] = d["pos"].fillna("").astype(str)
    keys = ["gsis_id", "component"]
    agg = d.groupby(keys, dropna=False).agg(
        player=("player", "first"), team=("team", "first"), pos=("pos", "first"),
        line=("line", "median"), p_over=("p_over", "median"), books=("line", "size"),
    ).reset_index()

    means = []
    for _, r in agg.iterrows():
        c = r["component"]
        if c == "anytime_td":
            # a Yes/No market: the "line" is meaningless, the price is everything
            means.append(expected_td(r["p_over"]))
        elif c in COUNT:
            means.append(mean_from_count(r["line"], r["p_over"]))
        elif c in YARDS:
            means.append(mean_from_yards(r["line"], r["p_over"], c, r.get("pos", "")))
        else:
            means.append(float("nan"))
    agg["mean"] = means
    return agg[cols]


def project(means: pd.DataFrame, fill: pd.DataFrame | None = None) -> pd.DataFrame:
    """Component means → half-PPR points per player.

    `fill` supplies the model's own expectation for components the book did not price, keyed
    the same way. A WR with receiving yards and receptions posted but no touchdown market is
    the common case, and scoring his touchdowns as zero would quietly dock him two or three
    points — worse than having no Vegas number at all, because it looks precise. Anything
    filled is counted in `vegas_parts` so the caller can say how much of the number is
    really the market's.
    """
    out_cols = ["gsis_id", "player", "team", "pos", "vegas", "vegas_parts", "vegas_filled",
                "vegas_complete"]
    if means is None or means.empty:
        return pd.DataFrame(columns=out_cols)
    wide = means.pivot_table(index="gsis_id", columns="component", values="mean",
                            aggfunc="first")
    agg = {"player": ("player", "first"), "team": ("team", "first")}
    if "pos" in means.columns:
        agg["pos"] = ("pos", "first")
    who = means.groupby("gsis_id").agg(**agg)
    if "pos" not in who.columns:
        who["pos"] = ""
    want = who["pos"].reindex(wide.index).map(relevant_for)

    fills = pd.DataFrame(index=wide.index)
    if fill is not None and not fill.empty and "gsis_id" in fill.columns:
        f = fill.set_index("gsis_id")
        for c in WEIGHTS:
            if c in f.columns:
                fills[c] = pd.to_numeric(f[c], errors="coerce").reindex(wide.index)

    total = pd.Series(0.0, index=wide.index)
    parts = pd.Series(0, index=wide.index, dtype=int)
    filled = pd.Series(0, index=wide.index, dtype=int)
    missing = pd.Series(0, index=wide.index, dtype=int)
    for comp, w in WEIGHTS.items():
        have = (pd.to_numeric(wide[comp], errors="coerce") if comp in wide.columns
                else pd.Series(np.nan, index=wide.index))
        applies = want.map(lambda r, c=comp: c in r)
        parts += have.notna().astype(int)          # anything priced counts, and is scored
        gap = have.isna() & applies
        missing += gap.astype(int)
        if comp in fills.columns:
            use = have.where(~gap, fills[comp])
            filled += (gap & fills[comp].notna()).astype(int)
        else:
            use = have
        total += w * use.fillna(0.0)

    out = pd.DataFrame({"vegas": total.round(2), "vegas_parts": parts,
                        "vegas_filled": filled}).join(who)
    out["vegas_complete"] = missing == 0
    return out.reset_index()[out_cols]
