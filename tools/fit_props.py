"""Re-derive every fitted constant in mega/props.py from nflverse weekly outcomes.

    python -m tools.fit_props [--seasons 2021 2022 2023 2024 2025]

Prints the three tables the module's constants come from, in a form that can be pasted
straight back into SKEW, CV and TD_MULT. Run it after a season ends; the numbers drift
slowly, and a constant nobody can reproduce is a constant nobody can challenge.

Two notes on method, because both are easy to get wrong:

**No split-half for the touchdown multiplier.** The first cut of this estimated P(scores)
from odd weeks and the realised mean from even weeks, to avoid using the same games twice.
That looked rigorous and was badly biased: a player's odd-week rate is a noisy estimate, so
the high bins are partly luck and their even-week mean regresses to the middle. It read as
Poisson overstating expected touchdowns by 60%. Measured on the same games — which is legitimate,
because E[TD] = P(>=1) x E[TD | >=1] is a fact about a distribution, not a forecast — the
real gap is about 12% at the top and near zero at the bottom.

**No filter on games the player left early.** A prop is priced for a player expected to
play, but a fantasy projection has to carry the risk that he does not finish. Dropping
those games would lift every mean above what a manager actually collects.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DEFAULT_SEASONS = [2021, 2022, 2023, 2024, 2025]


def load(seasons) -> pd.DataFrame:
    import nflreadpy as nfl
    d = nfl.load_player_stats(seasons=list(seasons))
    d = d.to_pandas() if hasattr(d, "to_pandas") else d
    pos = "position" if "position" in d.columns else "position_group"
    pid = "player_id" if "player_id" in d.columns else "gsis_id"
    d = d.rename(columns={pos: "pos", pid: "pid"})
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d = d[d["week"].between(1, 18)].copy()
    d["key"] = d["pid"].astype(str) + "_" + d["season"].astype(str)
    return d


def skew_table(d: pd.DataFrame, col: str, poss, bands, min_games: int = 8) -> None:
    x = d[d["pos"].isin(poss)].copy()
    x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=[col])
    g = x.groupby("key")[col].agg(n="size", mu="mean", med="median")
    g = g[g.n >= min_games]
    print(f"\n--- {col} ({'/'.join(poss)}) ---")
    print(f"{'level':>12} {'player-szn':>11} {'mean':>8} {'median':>8} {'mean/med':>9} {'CV':>7}")
    for lo, hi in bands:
        sel = g[(g.mu >= lo) & (g.mu < hi)]
        if len(sel) < 15:
            continue
        sub = x[x.key.isin(sel.index)]
        ratio = (sel.mu / sel.med.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        mu, sd = sub[col].mean(), sub[col].std()
        print(f"{f'{lo}-{hi}':>12} {len(sel):11d} {mu:8.2f} {sub[col].median():8.2f} "
              f"{ratio.mean():9.3f} {sd / max(mu, 1e-9):7.3f}")


def td_table(d: pd.DataFrame, min_games: int = 8) -> None:
    """E[TD | scored] by scorer quality — the TD_MULT anchors."""
    x = d[d["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    for c in ("rushing_tds", "receiving_tds"):
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    x["td"] = x["rushing_tds"] + x["receiving_tds"]   # what player_anytime_td settles on
    g = x.groupby("key")["td"].agg(p_any=lambda s: (s > 0).mean(), n="size")
    g = g[g.n >= min_games]
    x = x[x.key.isin(g.index)].merge(g[["p_any"]], on="key")
    print("\n--- TD_MULT anchors: (P(anytime TD), E[TD | scored]) ---")
    print(f"{'anchor':>28} {'games':>8} {'Poisson would say':>19}")
    for b, gg in x.groupby(pd.cut(x.p_any, [0, .10, .18, .26, .34, .45, 1.01]), observed=True):
        s = gg[gg.td > 0]
        if len(s) < 80:
            continue
        p = (gg.td > 0).mean()
        poi = -np.log(max(1e-9, 1 - p)) / p
        print(f"    ({p:.4f}, {s.td.mean():.4f}),{'':>6} {len(gg):8d} {poi:19.4f}")


def count_check(d: pd.DataFrame) -> None:
    """Is Poisson good enough at the lines books actually hang?"""
    def poi_sf(k, m):
        t = np.exp(-m); c = t
        for i in range(1, k + 1):
            t = t * m / i; c = c + t
        return 1 - c
    q = d[d["pos"] == "QB"].copy()
    q["ptd"] = pd.to_numeric(q["passing_tds"], errors="coerce").fillna(0)
    q["att"] = pd.to_numeric(q.get("attempts", 0), errors="coerce").fillna(0)
    q = q[q.att >= 15]
    g = q.groupby("key")["ptd"].agg(n="size", mu="mean"); g = g[g.n >= 8]
    q = q[q.key.isin(g.index)].merge(g[["mu"]], on="key")
    print("\n--- passing TDs vs Poisson at the 1.5 line ---")
    print(f"{'bin':>12} {'games':>7} {'mean':>6} {'obs P(>1.5)':>12} {'Poisson':>9}")
    for b, gg in q.groupby(pd.cut(q.mu, [0, 1.0, 1.4, 1.8, 5]), observed=True):
        if len(gg) < 50:
            continue
        m = gg.ptd.mean()
        print(f"{str(b):>12} {len(gg):7d} {m:6.3f} {(gg.ptd > 1.5).mean():12.3f} {poi_sf(1, m):9.3f}")

    r = d[d["pos"].isin(["WR", "TE", "RB"])].copy()
    r["rec"] = pd.to_numeric(r["receptions"], errors="coerce").fillna(0)
    r = r[pd.to_numeric(r.get("targets", 0), errors="coerce").fillna(0) >= 1]
    g2 = r.groupby("key")["rec"].agg(n="size", mu="mean"); g2 = g2[g2.n >= 8]
    r = r[r.key.isin(g2.index)].merge(g2[["mu"]], on="key")
    print("\n--- receptions vs Poisson at the 3.5 line ---")
    print(f"{'bin':>12} {'games':>7} {'mean':>6} {'var/mean':>9} {'obs P(>3.5)':>12} {'Poisson':>9}")
    for b, gg in r.groupby(pd.cut(r.mu, [1, 2.5, 4, 5.5, 15]), observed=True):
        if len(gg) < 50:
            continue
        m = gg.rec.mean()
        print(f"{str(b):>12} {len(gg):7d} {m:6.3f} {gg.rec.var() / m:9.3f} "
              f"{(gg.rec > 3.5).mean():12.3f} {poi_sf(3, m):9.3f}")


TD_PER_POINT = 0.1085          # offensive (rush+rec) TDs per point, 2,689 team-games


def hold_check(season: int, week: int) -> None:
    """Calibrate ONE_SIDED_HOLD against the one constraint that can test it.

    Every book posts anytime touchdown as "Yes" only, so there is no other side to de-vig
    against and the haircut is pure assumption — on the biggest single component of the
    projection. But it is not unfalsifiable: a team's players' expected touchdowns have to
    add up to the touchdowns its own Vegas total implies. If the haircut is too small every
    team runs hot, and by a measurable amount.
    """
    import numpy as np
    import pandas as pd

    from mega import forward as fw
    from mega import odds as O
    from mega import props as P
    from mega import season as S

    df = O.cached(season, week)
    if df is None or df.empty:
        print(f"no props on file for {season} week {week} — sweep one first")
        return
    df = df[df["gsis_id"].notna()]
    tmap = S.player_week(season).sort_values("week").groupby("gsis_id")["team"].last().to_dict()
    td = df[df["market"] == "player_anytime_td"].copy()
    td["t"] = td["gsis_id"].map(tmap)
    td = td[td["t"].notna()]
    imp = fw.implied(S.schedules(season))
    imp = imp[imp["week"] == int(week)].set_index("team")["implied"]

    def ratio(hold: float) -> float:
        old = P.ONE_SIDED_HOLD
        P.ONE_SIDED_HOLD = hold
        try:
            m = P.to_means(td)
        finally:
            P.ONE_SIDED_HOLD = old
        m["t"] = m["gsis_id"].map(tmap)
        c = pd.DataFrame({"s": m.groupby("t")["mean"].sum()}).join(
            imp.rename("i"), how="inner").dropna()
        return float((c.s / (c.i * TD_PER_POINT)).median())

    print()
    print(f"--- one-sided hold, {season} week {week} ({td['t'].nunique()} teams) ---")
    print(f"{'hold':>8} {'median sum/implied':>20}")
    for h in (0.020, 0.025, 0.030, 0.035, 0.040):
        print(f"{h:8.3f} {ratio(h):20.3f}")
    lo, hi = 0.005, 0.12
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ratio(mid) > 1.0 else (lo, mid)
    fit = (lo + hi) / 2
    print()
    print(f"centres at {fit:.4f} (two-way hold ~{2 * fit:.1%}); "
          f"module carries {P.ONE_SIDED_HOLD:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--hold", nargs=2, type=int, metavar=("SEASON", "WEEK"),
                    help="calibrate ONE_SIDED_HOLD against swept props instead of fitting")
    a = ap.parse_args()
    if a.hold:
        hold_check(a.hold[0], a.hold[1])
        return
    d = load(a.seasons)
    print(f"{len(d):,} player-weeks, seasons {min(a.seasons)}-{max(a.seasons)}")
    skew_table(d, "receiving_yards", ["WR", "TE"], [(20, 40), (40, 55), (55, 70), (70, 200)])
    skew_table(d, "rushing_yards", ["RB"], [(20, 40), (40, 60), (60, 80), (80, 300)])
    skew_table(d, "passing_yards", ["QB"], [(150, 200), (200, 240), (240, 270), (270, 500)])
    skew_table(d, "receptions", ["WR", "TE", "RB"], [(2, 3.5), (3.5, 5), (5, 6.5), (6.5, 20)])
    td_table(d)
    count_check(d)


if __name__ == "__main__":
    main()
