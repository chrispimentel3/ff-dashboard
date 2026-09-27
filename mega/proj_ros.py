"""HANDOFF v1.3 Pass 2 — our own rest-of-season projection.

    pts/g(i) = Σ_c Share_c(i) × TeamVol_c(team) + shrunk efficiency residual
    c ∈ {rec, rush, pass}  (ffopportunity's expected-points components)

  * **Share_c** — the player's share of his team's expected points in component c, an
    exponentially weighted mean over his games (half-life `h` games, fitted), shrunk toward
    his previous-season share with `k_s` pseudo-games. The rec component carries targets and
    air yards, rush carries carries and red-zone touches, pass the QB's dropbacks — so this
    is the handoff's targets / air yards / carries / rz touches, weighted by what each is
    worth in expected half-PPR points rather than counted raw.
  * **TeamVol_c** — the team's expected points per game in c, same EWMA, shrunk toward the
    team's previous season with `k_t` games.
  * **Efficiency** — (actual − expected) per game across this season and last, multiplied
    by n/(n + k_eff). `k_eff` is fitted; it comes out large, which is the formula for "trust
    usage, fade efficiency".

`tools/backtest_ros.py` fits h, k_s, k_t, k_eff walk-forward on 2021–2025 and writes them,
with the blend weights against FantasyPros ROS ECR and the p10/p90 multipliers, to
config/proj_ros_params.json. `build()` then projects the live season and writes
data/proj_ros_<season>.json, which the trade and waiver engines read.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .forward import COMPONENTS, xfp_components

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "config" / "proj_ros_params.json"
POS = ("QB", "RB", "WR", "TE")
LAST_WEEK = 17
PRIOR_MIN_GAMES = 4           # a previous season shorter than this is not a prior worth using

DEFAULT_PARAMS = {"h": 3.0, "k_s": 2.0, "k_t": 4.0, "k_eff": 32.0}


# ======================================================================== inputs
@functools.lru_cache(maxsize=8)
def games(season: int) -> pd.DataFrame:
    """One row per player-game: expected points by component for him and his team, and
    the half-PPR points he actually scored."""
    from . import season as S
    ffo = S.ff_opportunity(season)
    if ffo is None or ffo.empty:
        return pd.DataFrame()
    f = ffo[ffo["position"].isin(POS)].copy()
    f["week"] = pd.to_numeric(f["week"], errors="coerce").astype(int)
    f = f[f["week"] <= 18]
    me = xfp_components(f)
    tm = xfp_components(f, team=True)
    out = pd.DataFrame({
        "season": season, "week": f["week"].values, "gsis_id": f["player_id"].values,
        "pos": f["position"].values, "team": f["posteam"].map(_canon).values,
    })
    for c in COMPONENTS:
        out[f"x_{c}"] = me[f"xfp_{c}"].values
        out[f"t_{c}"] = tm[f"xfp_{c}"].values
    out["xfp"] = out[[f"x_{c}" for c in COMPONENTS]].sum(axis=1)
    pw = S.player_week(season)
    pts = pw.groupby(["gsis_id", "week"])["half_ppr"].sum()
    out = out.merge(pts.rename("pts").reset_index(), on=["gsis_id", "week"], how="left")
    out["pts"] = out["pts"].fillna(0.0)
    return out.drop_duplicates(["gsis_id", "week"]).reset_index(drop=True)


def _canon(t):
    from .ids import canon_team
    return canon_team(t)


def _ewm(values: np.ndarray, h: float) -> float:
    """Mean with weight 0.5^(age/h), newest game age 0."""
    n = len(values)
    if n == 0:
        return float("nan")
    w = 0.5 ** (np.arange(n)[::-1] / max(h, 1e-9))
    return float(np.sum(w * values) / np.sum(w))


# ======================================================================== the model
def priors(prev: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    """Last season's per-player shares, per-team volumes and per-player residuals."""
    if prev is None or prev.empty:
        return {}, {}, pd.DataFrame(columns=["gsis_id", "res_sum", "res_n"])
    p = prev.copy()
    share = {}
    g = p.groupby("gsis_id")
    for c in COMPONENTS:
        s = (g[f"x_{c}"].sum() / g[f"t_{c}"].sum().replace(0, np.nan)).fillna(0.0)
        share[c] = s
    n = g.size()
    keep = n[n >= PRIOR_MIN_GAMES].index
    last = p.sort_values("week").groupby("gsis_id").tail(1).set_index("gsis_id")
    shares = {gid: {**{c: float(share[c].get(gid, 0.0)) for c in COMPONENTS},
                    "_pos": last.at[gid, "pos"], "_team": last.at[gid, "team"]} for gid in keep}
    team = p.drop_duplicates(["team", "week"]).groupby("team")[[f"t_{c}" for c in COMPONENTS]].mean()
    vols = {t: {c: float(r[f"t_{c}"]) for c in COMPONENTS} for t, r in team.iterrows()}
    res = p.assign(r=p["pts"] - p["xfp"]).groupby("gsis_id")["r"].agg(["sum", "size"])
    res = res.rename(columns={"sum": "res_sum", "size": "res_n"}).reset_index()
    return shares, vols, res


def project(cur: pd.DataFrame, week: int, prior: tuple, params: dict,
            team_of: dict | None = None) -> pd.DataFrame:
    """Rest-of-season pts/g for every player with a game this season before `week`, or a
    usable previous season. `cur` is this season's games(); only weeks < `week` are read."""
    h, k_s, k_t, k_eff = (float(params[k]) for k in ("h", "k_s", "k_t", "k_eff"))
    shares_p, vols_p, res_p = prior
    hist = cur[cur["week"] < week] if cur is not None and not cur.empty else pd.DataFrame(columns=cur.columns if cur is not None else [])

    # ---- team volume
    tv = {}
    if not hist.empty:
        tg = hist.drop_duplicates(["team", "week"]).sort_values("week")
        for team, g in tg.groupby("team"):
            n = len(g)
            tv[team] = {}
            for c in COMPONENTS:
                e = _ewm(g[f"t_{c}"].to_numpy(float), h)
                pr = vols_p.get(team, {}).get(c)
                tv[team][c] = e if pr is None else (e * n + pr * k_t) / (n + k_t)
    for team, v in vols_p.items():
        tv.setdefault(team, dict(v))

    # ---- player shares
    rows = {}
    if not hist.empty:
        hs = hist.sort_values("week")
        for gid, g in hs.groupby("gsis_id"):
            n = len(g)
            pr = shares_p.get(gid)
            sh = {}
            for c in COMPONENTS:
                t = g[f"t_{c}"].to_numpy(float)
                x = g[f"x_{c}"].to_numpy(float)
                s = np.divide(x, t, out=np.zeros_like(x), where=t > 0)
                e = _ewm(s, h)
                sh[c] = e if pr is None else (e * n + pr[c] * k_s) / (n + k_s)
            rows[gid] = {"pos": g["pos"].iloc[-1], "team": g["team"].iloc[-1], "n": n, "share": sh,
                         "res_sum": float((g["pts"] - g["xfp"]).sum())}
    for gid, sh in shares_p.items():
        if gid not in rows:
            rows[gid] = {"pos": sh["_pos"], "team": sh["_team"], "n": 0,
                         "share": {c: sh[c] for c in COMPONENTS}, "res_sum": 0.0}

    res_prev = dict(zip(res_p["gsis_id"], zip(res_p["res_sum"], res_p["res_n"]))) if len(res_p) else {}
    out = []
    for gid, r in rows.items():
        team = (team_of or {}).get(gid) or r["team"]
        if team is None or team not in tv:
            continue
        vol = tv[team]
        parts = {c: r["share"][c] * vol[c] for c in COMPONENTS}
        xproj = sum(parts.values())
        ps, pn = res_prev.get(gid, (0.0, 0))
        n_res = r["n"] + pn
        res_mean = (r["res_sum"] + ps) / n_res if n_res else 0.0
        shrink = n_res / (n_res + k_eff) if math.isfinite(k_eff) else 0.0
        out.append({"gsis_id": gid, "pos": r["pos"], "team": team, "games": r["n"],
                    "xfp_proj": xproj, "res_raw": res_mean, "res_shrunk": res_mean * shrink,
                    "pts_pg": xproj + res_mean * shrink,
                    **{f"share_{c}": r["share"][c] for c in COMPONENTS},
                    **{f"vol_{c}": vol[c] for c in COMPONENTS}})
    return pd.DataFrame(out)


# ======================================================================== consensus
def ecr_points(ours: pd.DataFrame, ecr: pd.DataFrame) -> pd.Series:
    """ECR mapped onto our points scale (§4.2): the k-th ranked player at a position gets
    our k-th highest projection at that position. Monotone, per position, per week — so
    ours and consensus can be compared in points while ECR stays a pure ordering (the
    rankings are PPR; this league is half-PPR, so only ECR's order is used)."""
    out = pd.Series(np.nan, index=ecr.index)
    for pos, g in ecr.groupby("pos"):
        vals = np.sort(ours.loc[ours["pos"] == pos, "pts_pg"].dropna().to_numpy())[::-1]
        if not len(vals):
            continue
        k = g["ecr_pos_rank"].astype(int).clip(1, len(vals)) - 1
        out.loc[g.index] = vals[k.to_numpy()]
    return out


ECR_URL = "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr_latest.csv"
ECR_CACHE = ROOT / "data" / "build" / "ecr_ros_dp.csv"


def latest_ecr() -> pd.DataFrame:
    """FantasyPros ROS overall ECR from DynastyProcess, with gsis_id and position rank.

    Refreshed from GitHub on each build and cached (ro rows only, ~500) so a failed
    download falls back to the last good scrape rather than to nothing."""
    try:
        import io
        import urllib.request
        with urllib.request.urlopen(ECR_URL, timeout=60) as r:
            raw = pd.read_csv(io.BytesIO(r.read()))
        ro = raw[raw["ecr_type"] == "ro"][["id", "player", "pos", "team", "ecr", "sd", "player_owned_avg", "scrape_date"]]
        if len(ro) > 100:
            ECR_CACHE.parent.mkdir(parents=True, exist_ok=True)
            ro.to_csv(ECR_CACHE, index=False)
    except Exception:
        pass
    if not ECR_CACHE.is_file():
        return pd.DataFrame(columns=["gsis_id", "pos", "ecr", "ecr_pos_rank"])
    ro = pd.read_csv(ECR_CACHE)
    import nflreadpy as nfl
    ids = nfl.load_ff_playerids().to_pandas()[["fantasypros_id", "gsis_id"]].dropna()
    ids["fantasypros_id"] = pd.to_numeric(ids["fantasypros_id"], errors="coerce")
    ro["fantasypros_id"] = pd.to_numeric(ro["id"], errors="coerce")
    ro = ro.merge(ids.drop_duplicates("fantasypros_id"), on="fantasypros_id", how="left")
    ro = ro[ro["pos"].isin(POS)].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").copy()
    ro["ecr_pos_rank"] = ro.groupby("pos")["ecr"].rank(method="first").astype(int)
    return ro


def _current_teams(season: int) -> dict:
    """gsis_id -> team this season (offseason moves), from nflverse weekly rosters."""
    try:
        import nflreadpy as nfl
        rw = nfl.load_rosters_weekly(seasons=[season]).to_pandas()
        rw = rw.sort_values("week").groupby("gsis_id").tail(1)
        return {g: _canon(t) for g, t in zip(rw["gsis_id"], rw["team"]) if pd.notna(g)}
    except Exception:
        return {}


def build(season: int, now: int, write: bool = True) -> dict:
    """The live projection: data/proj_ros_<season>.json.

    Per player: ours, ECR on our scale, the blend the backtest earned, and per remaining
    week a median (if he plays), p10/p90, P(plays) and the expected value the engines use.
    """
    from . import contingency as cg
    from . import forward as fw
    from . import season as S
    from .waiver_value import availability, bye_weeks, _birthdays, _rosters_weekly

    cfg = load_params()
    params = {**DEFAULT_PARAMS, **{k: (math.inf if v is None else v) for k, v in (cfg.get("params") or {}).items()}}
    cur, prev = games(season), games(season - 1)
    ours = project(cur, now, priors(prev), params, _current_teams(season))
    ours = ours.dropna(subset=["pos"])
    ours["our_rank"] = ours.groupby("pos")["pts_pg"].rank(ascending=False, method="first").astype(int)
    ours["use_rank"] = ours.groupby("pos")["xfp_proj"].rank(ascending=False, method="first").astype(int)

    ecr = latest_ecr()
    e = ecr[["gsis_id", "pos", "ecr", "ecr_pos_rank", "player_owned_avg"]].copy()
    e["ecr_pts"] = ecr_points(ours, e).values
    m = ours.merge(e.drop(columns=["pos"]), on="gsis_id", how="outer")
    m["pos"] = m["pos"].fillna(m["gsis_id"].map(dict(zip(e["gsis_id"], e["pos"]))))
    b = band(now)
    wts = cfg.get("blend_w") or {}
    m["w"] = [float((wts.get(p) or {}).get(b, 0.5)) for p in m["pos"]]
    m["ros_pg"] = np.where(m["pts_pg"].notna() & m["ecr_pts"].notna(),
                           m["w"] * m["pts_pg"] + (1 - m["w"]) * m["ecr_pts"],
                           m["pts_pg"].fillna(m["ecr_pts"]))
    m["basis"] = np.where(m["pts_pg"].notna() & m["ecr_pts"].notna(), "blend",
                          np.where(m["pts_pg"].notna(), "ours only (unranked by ECR)", "ECR only (no NFL usage yet)"))

    sched = S.schedules(season)
    byes = bye_weeks(sched)
    weeks = list(range(int(now), LAST_WEEK + 1))
    vol = fw.team_volume(S.ff_opportunity(season), sched, weeks)
    ratio = {(r.team, int(r.week)): float(r.implied_ratio) for r in vol.itertuples()} if not vol.empty else {}
    avail = availability(S.injuries(season), _rosters_weekly(season), now)
    inj = S.injuries(season)
    status_now = {}
    if inj is not None and not inj.empty:
        cw = inj[pd.to_numeric(inj["week"], errors="coerce") == int(now)]
        status_now = dict(zip(cw["gsis_id"], cw["report_status"].fillna("")))
    kick = pd.Timestamp(f"{season}-09-01")
    ages = {g: (kick - d).days / 365.25 for g, d in _birthdays().items() if pd.notna(d)}
    quant = cfg.get("quantiles") or {}
    edges = quant.get("_tier_edges") or {}
    allowed = cfg.get("disagreement_allowed") or {}
    team_now = _current_teams(season)

    players = {}
    for r in m.itertuples():
        if pd.isna(r.ros_pg) or r.pos not in POS:
            continue
        team = team_now.get(r.gsis_id) or (r.team if isinstance(r.team, str) else None)
        q = (quant.get(r.pos) or {}).get(tier(r.ros_pg, r.pos, edges)) or [0.35, 0.9, 1.75]
        a = avail.get(r.gsis_id)
        wk = {}
        for w in weeks:
            if byes.get(team) == w:
                wk[w] = {"median": 0.0, "p10": 0.0, "p90": 0.0, "p_active": 0.0, "expected": 0.0, "bye": True}
                continue
            if a and w < a["back"]:
                pa = 0.0
            elif w == now:
                pa = cg.status_p_active(status_now.get(r.gsis_id, "")) if status_now.get(r.gsis_id) else 1.0
            else:
                pa = 1.0 - cg.miss_hazard(r.pos, ages.get(r.gsis_id), w - now)
            base = float(r.ros_pg) * ratio.get((team, w), 1.0) ** fw.GAMMA
            wk[w] = {"median": round(base * q[1], 2), "p10": round(base * q[0], 2), "p90": round(base * q[2], 2),
                     "p_active": round(pa, 3), "expected": round(base * pa, 2)}
        drivers, driver, gap = [], None, None
        if pd.notna(r.pts_pg):
            drivers.append(f"usage: {r.xfp_proj:.1f} expected pts/g from his share of {team}'s volume")
            if abs(r.res_raw) >= 0.5:
                drivers.append(f"efficiency: {r.res_raw:+.1f} pts/g over expected so far, counted as {r.res_shrunk:+.1f}")
        if pd.notna(r.pts_pg) and pd.notna(r.ecr_pos_rank):
            gap = int(r.ecr_pos_rank - r.our_rank)
            ug = int(r.ecr_pos_rank - r.use_rank)
            driver = "usage" if (np.sign(ug) == np.sign(gap) and abs(ug) >= 8) else "efficiency"
        tag = None
        if gap is not None and abs(gap) >= 8 and (allowed.get(r.pos) or {}).get(driver):
            tag = "higher" if gap > 0 else "lower"
        players[r.gsis_id] = {
            "pos": r.pos, "team": team, "ros_pg": round(float(r.ros_pg), 2), "basis": r.basis,
            "ours_pg": None if pd.isna(r.pts_pg) else round(float(r.pts_pg), 2),
            "ecr_pg": None if pd.isna(r.ecr_pts) else round(float(r.ecr_pts), 2),
            "w_ours": round(float(r.w), 2),
            "our_rank": None if pd.isna(r.our_rank) else int(r.our_rank),
            "ecr_rank": None if pd.isna(r.ecr_pos_rank) else int(r.ecr_pos_rank),
            "usage_rank": None if pd.isna(r.use_rank) else int(r.use_rank),
            "disagreement": {"gap": gap, "driver": driver, "tag": tag},
            "owned_avg": None if pd.isna(r.player_owned_avg) else float(r.player_owned_avg),
            "components": None if pd.isna(r.pts_pg) else {
                "xfp_proj": round(float(r.xfp_proj), 2), "res_raw": round(float(r.res_raw), 2),
                "res_shrunk": round(float(r.res_shrunk), 2), "games": int(r.games),
                **{f"share_{c}": round(float(getattr(r, f"share_{c}")), 4) for c in COMPONENTS},
                **{f"vol_{c}": round(float(getattr(r, f"vol_{c}")), 2) for c in COMPONENTS}},
            "drivers": drivers, "weeks": wk,
        }
    out = {"meta": {"season": season, "week": now, "band": b, "params": cfg.get("params"),
                    "fitted": cfg.get("fitted"), "players": len(players),
                    "ecr_scrape": str(ecr["scrape_date"].max()) if "scrape_date" in ecr else None},
           "players": players}
    if write:
        (ROOT / "data" / f"proj_ros_{season}.json").write_text(json.dumps(out, separators=(",", ":")))
    return out


@functools.lru_cache(maxsize=2)
def cached(season: int) -> dict:
    """The last built projection file, or {} — engines must run without it."""
    try:
        return json.loads((ROOT / "data" / f"proj_ros_{season}.json").read_text())
    except Exception:
        return {}


def load_params() -> dict:
    try:
        return json.loads(PARAMS.read_text())
    except Exception:
        return {"params": DEFAULT_PARAMS}


def band(week: int) -> str:
    """Blend weights are fitted for weeks 3–14; weeks 1–2 borrow 3–6, 15–17 borrow 11–14."""
    return "3-6" if week <= 6 else "7-10" if week <= 10 else "11-14"


def tier(pred: float, pos: str, edges: dict) -> str:
    lo, hi = (edges.get(pos) or [0, 0])[:2]
    return "low" if pred < lo else "mid" if pred < hi else "high"
