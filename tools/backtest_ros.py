"""Walk-forward backtest of mega/proj_ros.py against FantasyPros ROS ECR (HANDOFF v1.3 §4.3).

    python -m tools.backtest_ros            # fit, evaluate, write params + report

At each (season, week) for 2021–2025, weeks 3–14, the model sees only games played before
that week and ECR from the DynastyProcess scrape dated before that Sunday, and predicts
rest-of-season half-PPR points per game. The target is what each player actually scored
per game played over weeks w..17 (at least 3 games). The pool is ECR's top 30 QB/TE and
top 60 RB/WR at that date — the players a fantasy manager actually argues about.

Writes:
  config/proj_ros_params.json   fitted model params, blend weights, p10/p90 multipliers,
                                which disagreement drivers may be shown, schedule-lens test
  docs/backtest_v13.md          the report (tables), with SVG charts in docs/img/

The ship rules (§4.3) are applied here, in code, not left to judgment later:
  * production value = w·ours + (1−w)·ECR with w fitted per position × week band; where
    w comes out ~0 the report says we ship ECR there
  * a "we're higher/lower than consensus" tag is allowed only for a driver type whose
    disagreement hit rate beat 55%
  * the schedule lens is enabled per position only where the forward-schedule test passed
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mega import proj_ros as P

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMG = DOCS / "img"
SEASONS = [2021, 2022, 2023, 2024, 2025]
WEEKS = range(3, 15)
POOL = {"QB": 30, "TE": 30, "RB": 60, "WR": 60}
MIN_ROS_GAMES = 3
DISAGREE_RANKS = 8           # handoff §4.3
TAG_HIT_RATE = 0.55          # handoff §4.3 ship rule
TAG_MIN_N = 30               # a hit rate on fewer disagreements than this isn't evidence
ECR_STALE_DAYS = 10          # a scrape older than this before kickoff is not "that week's" ECR
GRID = {"h": [1, 2, 3, 4, 6, 8, 12, 20], "k_s": [0, 1, 2, 4, 8, 16], "k_t": [0, 2, 4, 8, 16, 32],
        "k_eff": [0, 8, 16, 32, 64, 128, math.inf]}
ECR_FILE = Path("/private/tmp") / "fpecr.parquet"      # overridden by --ecr


# ======================================================================== data
def load_ecr(path: Path) -> pd.DataFrame:
    import nflreadpy as nfl
    h = pd.read_parquet(path)
    h = h[h["fp_page"].astype(str).str.contains("ros-ppr-overall")].copy()
    h["scrape_date"] = pd.to_datetime(h["scrape_date"])
    ids = nfl.load_ff_playerids().to_pandas()[["fantasypros_id", "gsis_id"]].dropna()
    ids["fantasypros_id"] = pd.to_numeric(ids["fantasypros_id"], errors="coerce")
    h["fantasypros_id"] = pd.to_numeric(h["id"], errors="coerce")
    h = h.merge(ids.drop_duplicates("fantasypros_id"), on="fantasypros_id", how="left")
    return h[h["pos"].isin(P.POS)][["scrape_date", "gsis_id", "pos", "ecr", "player_owned_avg", "player"]]


def ecr_at(ecr: pd.DataFrame, sunday: pd.Timestamp) -> pd.DataFrame:
    """The last scrape before that Sunday, with a position rank."""
    before = ecr[(ecr["scrape_date"] < sunday) & (ecr["scrape_date"] >= sunday - pd.Timedelta(days=ECR_STALE_DAYS))]
    if before.empty:
        return pd.DataFrame()
    snap = before[before["scrape_date"] == before["scrape_date"].max()].dropna(subset=["gsis_id"]).copy()
    snap = snap.drop_duplicates("gsis_id")
    snap["ecr_pos_rank"] = snap.groupby("pos")["ecr"].rank(method="first").astype(int)
    return snap


def sundays(season: int) -> dict:
    from mega import season as S
    s = S.schedules(season)
    s = s[s["game_type"] == "REG"]
    first = pd.to_datetime(s.groupby("week")["gameday"].min())
    return {int(w): d + pd.Timedelta(days=(6 - d.weekday()) % 7) for w, d in first.items()}


class Data:
    def __init__(self, ecr_path: Path):
        self.g = {s: P.games(s) for s in [SEASONS[0] - 1, *SEASONS]}
        self.prior = {s: P.priors(self.g[s - 1]) for s in SEASONS}
        ecr = load_ecr(ecr_path)
        self.snap, self.target = {}, {}
        for s in SEASONS:
            sun = sundays(s)
            g = self.g[s]
            for w in WEEKS:
                e = ecr_at(ecr, sun.get(w, pd.NaT)) if w in sun else pd.DataFrame()
                if e.empty:
                    continue
                fut = g[(g["week"] >= w) & (g["week"] <= P.LAST_WEEK)]
                t = fut.groupby("gsis_id").agg(act=("pts", "mean"), n=("pts", "size"), tot=("pts", "sum"))
                self.snap[(s, w)] = e
                self.target[(s, w)] = t


# ======================================================================== evaluation
def evaluate(d: Data, params: dict) -> pd.DataFrame:
    rows = []
    for (s, w), e in d.snap.items():
        pj = P.project(d.g[s], w, d.prior[s], params)
        if pj.empty:
            continue
        pj = pj.dropna(subset=["pos"])
        pj["our_rank"] = pj.groupby("pos")["pts_pg"].rank(ascending=False, method="first")
        pj["use_rank"] = pj.groupby("pos")["xfp_proj"].rank(ascending=False, method="first")
        m = e.merge(pj.drop(columns=["pos"]), on="gsis_id", how="inner")
        m = m[m["ecr_pos_rank"] <= m["pos"].map(POOL)]
        if m.empty:
            continue
        m["ecr_pts"] = P.ecr_points(pj, m).values
        t = d.target[(s, w)]
        m = m.merge(t[t["n"] >= MIN_ROS_GAMES], left_on="gsis_id", right_index=True, how="inner")
        m["season"], m["week"], m["band"] = s, w, P.band(w)
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def mae(x: pd.DataFrame, col: str) -> float:
    return float((x[col] - x["act"]).abs().mean())


def spearman(x: pd.DataFrame, col: str) -> float:
    """Mean within-(season, week, position) rank correlation — ranking is the job."""
    vals = []
    for _, g in x.groupby(["season", "week", "pos"]):
        if len(g) >= 8:
            vals.append(g[col].rank().corr(g["act"].rank()))
    return float(np.nanmean(vals)) if vals else float("nan")


def fit_params(d: Data) -> tuple[dict, list]:
    """Coordinate descent on pooled MAE of our projection, two sweeps."""
    params, trail = dict(P.DEFAULT_PARAMS), []
    best = mae(evaluate(d, params), "pts_pg")
    for _ in range(2):
        for k, grid in GRID.items():
            for v in grid:
                trial = {**params, k: v}
                score = mae(evaluate(d, trial), "pts_pg")
                trail.append({**trial, "mae": round(score, 4)})
                if score < best - 1e-6:
                    best, params = score, trial
    return params, trail


def fit_blend(ev: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """w per position × band on all seasons, plus leave-one-season-out MAE for honesty."""
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)

    def best_w(x):
        return float(min(grid, key=lambda w: ((w * x["pts_pg"] + (1 - w) * x["ecr_pts"]) - x["act"]).abs().mean()))

    weights = {pos: {b: best_w(g) for b, g in gp.groupby("band")} for pos, gp in ev.groupby("pos")}
    cv = []
    for s in SEASONS:
        tr, te = ev[ev["season"] != s], ev[ev["season"] == s]
        for (pos, b), g in te.groupby(["pos", "band"]):
            trg = tr[(tr["pos"] == pos) & (tr["band"] == b)]
            if trg.empty:
                continue
            w = best_w(trg)
            cv.append(g.assign(blend_cv=w * g["pts_pg"] + (1 - w) * g["ecr_pts"]))
    return weights, pd.concat(cv, ignore_index=True)


def disagreement(ev: pd.DataFrame) -> pd.DataFrame:
    d = ev.copy()
    d["gap"] = d["ecr_pos_rank"] - d["our_rank"]            # + = we rank him higher
    d = d[d["gap"].abs() >= DISAGREE_RANKS].copy()
    d["use_gap"] = d["ecr_pos_rank"] - d["use_rank"]
    d["driver"] = np.where((np.sign(d["use_gap"]) == np.sign(d["gap"])) & (d["use_gap"].abs() >= DISAGREE_RANKS),
                           "usage", "efficiency")
    d["ours_closer"] = (d["pts_pg"] - d["act"]).abs() < (d["ecr_pts"] - d["act"]).abs()
    return d


def calibration(ev: pd.DataFrame, d: Data, blend_w: dict) -> tuple[dict, dict]:
    """p10/p90 as multipliers of the per-game projection, by position and tier, from the
    empirical spread of single-week scores. Fitted on 2021–23, coverage checked on 2024–25,
    then refitted on everything for production."""
    rows = []
    for (s, w), g in ev.groupby(["season", "week"]):
        fut = d.g[s][(d.g[s]["week"] >= w) & (d.g[s]["week"] <= P.LAST_WEEK)][["gsis_id", "pts"]]
        x = g[["gsis_id", "pos", "pts_pg", "ecr_pts", "band"]].merge(fut, on="gsis_id")
        wv = x.apply(lambda r: blend_w[r["pos"]][r["band"]], axis=1)
        x["pred"] = wv * x["pts_pg"] + (1 - wv) * x["ecr_pts"]
        x["season"] = s
        rows.append(x)
    wk = pd.concat(rows, ignore_index=True)
    wk = wk[wk["pred"] > 1]
    edges = {pos: [round(float(q), 2) for q in g["pred"].quantile([1 / 3, 2 / 3])] for pos, g in wk.groupby("pos")}
    wk["tier"] = [P.tier(p, pos, edges) for p, pos in zip(wk["pred"], wk["pos"])]
    wk["ratio"] = wk["pts"] / wk["pred"]

    def fit(x):
        q = {pos: {t: [round(float(v), 3) for v in g["ratio"].quantile([0.1, 0.5, 0.9])]
                   for t, g in gp.groupby("tier")} for pos, gp in x.groupby("pos")}
        q["_tier_edges"] = edges
        return q

    train = fit(wk[wk["season"] <= 2023])
    test = wk[wk["season"] >= 2024].copy()
    lo = test.apply(lambda r: train[r["pos"]][r["tier"]][0], axis=1) * test["pred"]
    hi = test.apply(lambda r: train[r["pos"]][r["tier"]][2], axis=1) * test["pred"]
    test["inside"] = (test["pts"] >= lo) & (test["pts"] <= hi)
    coverage = {pos: round(float(g["inside"].mean()), 3) for pos, g in test.groupby("pos")}
    return fit(wk), coverage


def schedule_test(d: Data) -> dict:
    """Does the week-w read of a team's remaining schedule predict what that schedule
    actually did to it? Defense strength vs a position = points allowed per game to it
    over games before w, shrunk 4 games toward league average. Predicted SOS = mean over
    the remaining opponents; realized = the offense's position-group points in those games
    over its own season average."""
    K = 4.0
    out = {}
    for pos in P.POS:
        pairs = []
        for s in SEASONS:
            from mega import season as S
            sc = S.schedules(s)
            sc = sc[sc["game_type"] == "REG"]
            g = d.g[s][d.g[s]["pos"] == pos]
            grp = g.groupby(["team", "week"])["pts"].sum().rename("pts").reset_index()
            opp = pd.concat([sc[["week", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opp"}),
                             sc[["week", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opp"})])
            opp["team"], opp["opp"] = opp["team"].map(P._canon), opp["opp"].map(P._canon)
            grp = grp.merge(opp, on=["team", "week"], how="left")
            season_avg = grp.groupby("team")["pts"].mean()
            for w in WEEKS:
                past, fut = grp[grp["week"] < w], grp[(grp["week"] >= w) & (grp["week"] <= P.LAST_WEEK)]
                if past.empty or fut.empty:
                    continue
                lg = past["pts"].mean()
                allowed = past.groupby("opp")["pts"].agg(["sum", "size"])
                rating = ((allowed["sum"] + K * lg) / (allowed["size"] + K)) / lg
                for team, f in fut.groupby("team"):
                    pred = f["opp"].map(rating).fillna(1.0).mean()
                    real = (f["pts"] / season_avg.get(team, np.nan)).mean()
                    pairs.append((s, w, team, pred, real))
        x = pd.DataFrame(pairs, columns=["season", "week", "team", "pred", "real"]).dropna()
        r = float(x["pred"].rank().corr(x["real"].rank()))
        n_eff = x.groupby(["season", "team"]).ngroups
        t = r * math.sqrt(max(n_eff - 2, 1)) / math.sqrt(max(1e-9, 1 - r * r))
        out[pos] = {"spearman": round(r, 3), "n_team_seasons": int(n_eff), "t": round(t, 2),
                    "passes": bool(r > 0.10 and t > 2.0)}
    return out


def early_signal(d: Data) -> pd.DataFrame:
    """Rising-usage flag + low ownership: how often did he finish ROS top-36 (RB/WR) or
    top-15 (TE), against the same position's unflagged low-owned players?"""
    from tools.fit_signal_rates import SIGNALS, _roles_through
    from mega import season as S
    ecr = {k: v.set_index("gsis_id")["player_owned_avg"] for k, v in d.snap.items()}
    top = {"RB": 36, "WR": 36, "TE": 15}
    rows = []
    for s in SEASONS:
        pw, ffo = S.player_week(s), S.ff_opportunity(s)
        for w in WEEKS:
            if (s, w) not in d.snap:
                continue
            tab = _roles_through(pw, ffo, w - 1)
            fut = d.g[s][(d.g[s]["week"] >= w) & (d.g[s]["week"] <= P.LAST_WEEK)]
            tot = fut.groupby(["gsis_id"])["pts"].sum()
            posmap = d.g[s].drop_duplicates("gsis_id").set_index("gsis_id")["pos"]
            rk = tot.to_frame("tot").join(posmap).groupby("pos")["tot"].rank(ascending=False)
            own = ecr[(s, w)]
            for r in tab.itertuples():
                if r.pos not in top:
                    continue
                o = own.get(r.gsis_id)
                if o is not None and pd.notna(o) and o >= 30:
                    continue
                flagged = any(f in SIGNALS for f in (r.flags if isinstance(r.flags, list) else []))
                rows.append({"pos": r.pos, "flagged": flagged,
                             "hit": bool(rk.get(r.gsis_id, 999) <= top[r.pos])})
    return pd.DataFrame(rows)


# ======================================================================== report
def svg_bars(title: str, groups: list[str], series: dict[str, list[float]], path: Path,
             fmt="{:.2f}", ymax=None) -> str:
    colors = ["#013369", "#9aa4b2", "#0f9d58", "#e69500"]
    W, H, pad = 640, 300, 48
    ymax = ymax or max(max(v) for v in series.values()) * 1.15
    gw = (W - 2 * pad) / len(groups)
    bw = gw * 0.8 / len(series)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui" font-size="11">',
             f'<rect width="{W}" height="{H}" fill="#fff"/>',
             f'<text x="{pad}" y="20" font-size="13" font-weight="600" fill="#101828">{title}</text>',
             f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="#e4e7ec"/>']
    for gi, g in enumerate(groups):
        x0 = pad + gi * gw + gw * 0.1
        for si, (name, vals) in enumerate(series.items()):
            v = vals[gi]
            if v is None or not math.isfinite(v):
                continue
            h = (H - 2 * pad) * v / ymax
            x = x0 + si * bw
            parts.append(f'<rect x="{x:.1f}" y="{H - pad - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" rx="3" fill="{colors[si % 4]}"/>')
            parts.append(f'<text x="{x + bw / 2 - 1:.1f}" y="{H - pad - h - 4:.1f}" text-anchor="middle" fill="#344054">{fmt.format(v)}</text>')
        parts.append(f'<text x="{x0 + gw * 0.4:.1f}" y="{H - pad + 16}" text-anchor="middle" fill="#344054">{g}</text>')
    for si, name in enumerate(series):
        parts.append(f'<rect x="{pad + si * 120}" y="{H - 18}" width="10" height="10" rx="2" fill="{colors[si % 4]}"/>'
                     f'<text x="{pad + si * 120 + 14}" y="{H - 9}" fill="#344054">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return f"![{title}](img/{path.name})"


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecr", type=Path, required=True, help="DynastyProcess db_fpecr.parquet")
    a = ap.parse_args(argv)
    IMG.mkdir(parents=True, exist_ok=True)

    d = Data(a.ecr)
    print(f"{len(d.snap)} season-weeks with an ECR snapshot")
    params, trail = fit_params(d)
    print("params", params)
    ev = evaluate(d, params)
    weights, cv = fit_blend(ev)
    dis = disagreement(ev)
    quant, coverage = calibration(ev, d, weights)
    sched = schedule_test(d)
    es = early_signal(d)

    # ---- per-position table
    tab = []
    for pos in P.POS:
        x, c = ev[ev["pos"] == pos], cv[cv["pos"] == pos]
        tab.append({"pos": pos, "n": len(x), "mae_ours": mae(x, "pts_pg"), "mae_ecr": mae(x, "ecr_pts"),
                    "mae_blend_cv": mae(c, "blend_cv"), "rho_ours": spearman(x, "pts_pg"),
                    "rho_ecr": spearman(x, "ecr_pts"), "rho_blend_cv": spearman(c, "blend_cv")})
    tab = pd.DataFrame(tab)

    hit = (dis.groupby(["pos", "driver"])["ours_closer"].agg(["mean", "size"]).reset_index()
           .rename(columns={"mean": "hit_rate", "size": "n"}))
    allowed = {}
    for pos in P.POS:
        allowed[pos] = {}
        for drv in ("usage", "efficiency"):
            r = hit[(hit["pos"] == pos) & (hit["driver"] == drv)]
            ok = bool(len(r) and r["n"].iloc[0] >= TAG_MIN_N and r["hit_rate"].iloc[0] > TAG_HIT_RATE)
            allowed[pos][drv] = ok

    es_tab = (es.groupby(["pos", "flagged"])["hit"].agg(["mean", "size"]).reset_index()
              .rename(columns={"mean": "hit_rate", "size": "n"})) if not es.empty else pd.DataFrame()

    out = {
        "_doc": "Fitted by tools/backtest_ros.py (walk-forward 2021-2025, weeks 3-14, vs FantasyPros "
                "ROS ECR). Read by mega/proj_ros.py. See docs/backtest_v13.md.",
        "fitted": dt.date.today().isoformat(),
        "params": {k: (None if (isinstance(v, float) and math.isinf(v)) else v) for k, v in params.items()},
        "blend_w": weights,
        "quantiles": quant, "coverage_2024_25": coverage,
        "disagreement_allowed": allowed,
        "schedule_lens": {pos: v["passes"] for pos, v in sched.items()},
        "schedule_test": sched,
        "metrics": json.loads(tab.round(4).to_json(orient="records")),
    }
    P.PARAMS.write_text(json.dumps(out, indent=1))

    # ---- report
    charts = [
        svg_bars("Rest-of-season MAE, pts/g (lower is better)", list(P.POS),
                 {"ours": tab["mae_ours"].tolist(), "ECR": tab["mae_ecr"].tolist(),
                  "blend (CV)": tab["mae_blend_cv"].tolist()}, IMG / "bt_mae.svg"),
        svg_bars("Rank correlation with what happened (higher is better)", list(P.POS),
                 {"ours": tab["rho_ours"].tolist(), "ECR": tab["rho_ecr"].tolist(),
                  "blend (CV)": tab["rho_blend_cv"].tolist()}, IMG / "bt_rho.svg", ymax=1.0),
        svg_bars("Blend weight on our projection, by week band", list(P.POS),
                 {b: [weights[p].get(b, float("nan")) for p in P.POS] for b in ("3-6", "7-10", "11-14")},
                 IMG / "bt_blend.svg", ymax=1.1),
        svg_bars("When we disagree by 8+ ranks: how often we were closer", list(P.POS),
                 {drv: [float(hit[(hit.pos == p) & (hit.driver == drv)]["hit_rate"].iloc[0])
                        if len(hit[(hit.pos == p) & (hit.driver == drv)]) else float("nan") for p in P.POS]
                  for drv in ("usage", "efficiency")}, IMG / "bt_disagree.svg", fmt="{:.0%}", ymax=1.0),
    ]
    L = ["# Backtest — our rest-of-season projection vs FantasyPros ECR (HANDOFF v1.3 §4.3)", "",
         f"_Generated {dt.date.today().isoformat()} by `tools/backtest_ros.py`. Walk-forward 2021–2025, "
         f"weeks 3–14, {len(d.snap)} season-weeks with an ECR scrape dated before kickoff. "
         f"Pool: ECR top 30 QB/TE, top 60 RB/WR. Target: actual half-PPR pts per game played, weeks w–17 "
         f"(min {MIN_ROS_GAMES} games)._", "",
         "## Fitted model", "",
         f"- EWMA half-life **h = {params['h']} games**; share shrink **k_s = {params['k_s']}** games toward last "
         f"season; team-volume shrink **k_t = {params['k_t']}** games; efficiency shrink **k_eff = {params['k_eff']}** games.",
         "- ECR is converted to points by giving the k-th ranked player our k-th highest projection at his "
         "position that week (§4.2), so the comparison is in points while ECR stays a pure (PPR) ordering.", "",
         "## Accuracy by position", "",
         "| Pos | n | MAE ours | MAE ECR | MAE blend (CV) | ρ ours | ρ ECR | ρ blend (CV) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        L.append(f"| {r.pos} | {r.n} | {r.mae_ours:.2f} | {r.mae_ecr:.2f} | {r.mae_blend_cv:.2f} | "
                 f"{r.rho_ours:.3f} | {r.rho_ecr:.3f} | {r.rho_blend_cv:.3f} |")
    L += ["", "Blend is scored leave-one-season-out: its weight is fitted on the other four seasons.", "",
          charts[0], "", charts[1], "", "## Blend weights (share on our projection)", "",
          "| Pos | 3–6 | 7–10 | 11–14 |", "|---|---|---|---|"]
    for pos in P.POS:
        L.append(f"| {pos} | " + " | ".join(f"{weights[pos].get(b, float('nan')):.2f}" for b in ("3-6", "7-10", "11-14")) + " |")
    zero = [f"{p} {b}" for p in P.POS for b, v in weights[p].items() if v <= 0.05]
    L += ["", ("Ship rule: where w ≈ 0 we ship ECR and say so — " + ", ".join(zero) + "." if zero
               else "No position-band came out at w ≈ 0, so our projection carries weight everywhere."),
          "", charts[2], "", "## Disagreement hit rate (≥ 8 position ranks apart)", "",
          "| Pos | Driver | n | Ours closer | Tag allowed |", "|---|---|---|---|---|"]
    for r in hit.itertuples():
        L.append(f"| {r.pos} | {r.driver} | {r.n} | {r.hit_rate:.0%} | {'yes' if allowed[r.pos][r.driver] else 'no'} |")
    L += ["", f"Driver: *usage* when the usage-only projection (expected points, no efficiency term) is "
          f"itself {DISAGREE_RANKS}+ ranks from ECR in the same direction; otherwise *efficiency*. A tag is shown "
          f"only where the hit rate beats {TAG_HIT_RATE:.0%} on at least {TAG_MIN_N} disagreements; everything "
          "else reads \"consensus view\".", "", charts[3], "",
          "## Early-signal test", "",
          "Players with a rising-usage flag (TGT, AIR, SNAP, LEAD, GL, ROLE+) and < 30% rostered (FantasyPros "
          "average; unranked players count as low-owned): how often they finished ROS top-36 RB/WR or top-15 TE, "
          "against unflagged low-owned players at the same position.", "",
          "| Pos | Flagged | n | Finished top |", "|---|---|---|---|"]
    for r in es_tab.itertuples():
        L.append(f"| {r.pos} | {'yes' if r.flagged else 'no'} | {r.n} | {r.hit_rate:.1%} |")
    L += ["", "## p10 / p90 calibration", "",
          "Weekly scores as multiples of the per-game projection, by position and projection tier, fitted "
          "on 2021–23 and checked on 2024–25 (target ≈ 80% of weeks inside the band):", "",
          "| Pos | 2024–25 coverage |", "|---|---|"]
    for pos, v in coverage.items():
        L.append(f"| {pos} | {v:.1%} |")
    L += ["", "## Forward-schedule reliability (from HANDOFF_matchups.md)", "",
          "Does the week-w read of a team's remaining opponents predict what those games did to its "
          "position group? Pass = Spearman > 0.10 with t > 2 on team-seasons as the effective sample.", "",
          "| Pos | Spearman | team-seasons | t | Schedule lens |", "|---|---|---|---|---|"]
    for pos, v in sched.items():
        L.append(f"| {pos} | {v['spearman']:.3f} | {v['n_team_seasons']} | {v['t']:.1f} | {'on' if v['passes'] else 'off'} |")
    L += ["", "## Caveats", "",
          "- DynastyProcess scrapes are Fridays, after Thursday night's game, so ECR has seen one game ours has not "
          "in each walk-forward week. That favours ECR slightly.",
          "- Rookies with no NFL snaps before week w have no projection of ours and are excluded from the "
          "comparison; in production they fall back to ECR.",
          "- ECR is a PPR ranking; only its order is used, mapped onto our half-PPR scale."]
    (DOCS / "backtest_v13.md").write_text("\n".join(L) + "\n")
    print(tab.round(3).to_string(index=False))
    print(hit.to_string(index=False))
    print("coverage", coverage)
    print("schedule", sched)
    print(es_tab.to_string(index=False) if not es_tab.empty else "no early-signal rows")


if __name__ == "__main__":
    main()
