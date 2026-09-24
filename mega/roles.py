"""§12 role context: who a player is on his own offence, and how his usage reads against
the baseline for that role and against the NFL average at his position.

A raw number means little without the role attached to it. A 20% target share is a strong
season for a WR3 and a warning sign for a WR1 — the same figure, two opposite readings. So
every metric here comes back three ways:

    x̂     the player's own rate, shrunk toward his role's baseline (small samples lie)
    idx   100 x̂ / μ_role   — 100 is average FOR HIS ROLE
    idxP  100 x̂ / μ_pos    — 100 is the NFL average at his position
    z     (x̂ − μ_role) / σ_role, so metrics in different units can be averaged (§14)

Every ratio is a rolling sum of numerators over a rolling sum of denominators, never the
mean of weekly ratios — §12.2 requires it and the rest of this app already works that way.
Two games of 10-of-40 and 2-of-20 targets is 20.0%, not 17.5%.

Missing Next Gen Stats rows give `None`, never 0. NGS only publishes qualifying players, so
a zero would say "this player separates badly" when the truth is "nobody measured him".

Pure pandas over frames the caller supplies, so it can be tested without Streamlit or the
network. `mega.ask.player_week` already carries most of the numerators and denominators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config (§19.2)
WINDOW_ROLES = 3          # games played, for deciding who a player is
WINDOW_METRICS = 4        # games played, for measuring how he is used
MIN_GAMES = 2             # fewer than this and the depth chart decides the role
ROLE_FLAG_IDX = 120.0     # role index at or above this is "running above his role"

# Shrinkage constants, in the metric's own denominator units (§12.4). Judgment defaults —
# §19.4 calibrates them from 2026 split-half reliability after week 8.
K = {
    "target_share": 60.0, "air_share": 500.0, "fd_per_target": 30.0, "fd_per_touch": 40.0,
    "tprr": 100.0, "fd_rr": 120.0, "carry_share": 40.0, "inside10_share": 6.0,
    "yacoe": 25.0, "ryoe": 60.0, "separation": 25.0, "snap_pct": 60.0, "xfp_share": 2.0,
    "adot": 30.0, "wopr": 60.0,
}

POS_ALL = ("QB", "RB", "WR", "TE")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    num: str                       # numerator column in the player-week frame
    den: str                       # denominator column
    pos: tuple[str, ...]
    higher_better: bool = True
    pct: bool = False
    fmt: str = "{:.3f}"
    note: str = ""


METRICS: tuple[Metric, ...] = (
    Metric("target_share", "target share", "targets", "team_targets", ("WR", "TE", "RB"),
           pct=True, fmt="{:.1%}"),
    Metric("air_share", "air-yards share", "receiving_air_yards", "team_air_yards", ("WR", "TE"),
           pct=True, fmt="{:.1%}"),
    Metric("fd_per_target", "1D per target", "receiving_first_downs", "targets", ("WR", "TE", "RB"),
           fmt="{:.3f}"),
    Metric("fd_per_touch", "1D per touch", "touch_first_downs", "touches", ("RB",), fmt="{:.3f}"),
    Metric("tprr", "TPRR (est.)", "targets", "routes", ("WR", "TE"), fmt="{:.3f}",
           note="Routes are estimated from snap share — see mega/routes.py and §12.6."),
    Metric("fd_rr", "1D/RR (est.)", "receiving_first_downs", "routes", ("WR", "TE"), fmt="{:.3f}",
           note="Doctrine anchor: a WR at or above 12% is a league-winner signal."),
    Metric("adot", "aDOT", "receiving_air_yards", "targets", ("WR", "TE"), fmt="{:.1f}"),
    Metric("yacoe", "YAC over exp", "yacoe_num", "receptions", ("WR", "TE", "RB"), fmt="{:+.2f}"),
    Metric("separation", "separation", "separation_num", "targets", ("WR", "TE"), fmt="{:.2f}"),
    Metric("carry_share", "carry share", "carries", "team_carries", ("RB", "QB"),
           pct=True, fmt="{:.1%}"),
    Metric("inside10_share", "inside-10 share", "i10_carries", "team_i10_carries", ("RB", "QB"),
           pct=True, fmt="{:.1%}"),
    Metric("ryoe", "rush yds over exp/att", "ryoe_num", "carries", ("RB",), fmt="{:+.2f}"),
    Metric("snap_pct", "snap %", "offense_snaps", "team_snaps", POS_ALL, pct=True, fmt="{:.1%}"),
    Metric("xfp_share", "xFP share", "half_ppr_exp", "team_xfp", POS_ALL, pct=True, fmt="{:.1%}"),
)
METRIC_BY_KEY = {m.key: m for m in METRICS}

# WOPR is built from two other metrics rather than a numerator over a denominator (§12.2),
# so it is computed after shrinkage instead of sitting in the table above.
WOPR_W = (1.5, 0.7)


def _n(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)


def _z(df: pd.DataFrame, c: str) -> pd.Series:
    """Numerator-style: a missing column is a zero, not a gap."""
    return _n(df, c).fillna(0.0)


# ---------------------------------------------------------------- the player-week frame
def frame(pw: pd.DataFrame, ffo: pd.DataFrame | None = None,
          ngs_rec: pd.DataFrame | None = None, ngs_rush: pd.DataFrame | None = None) -> pd.DataFrame:
    """`mega.ask.player_week` plus the denominators and NGS rates §12 needs.

    NGS publishes per-game rates, so they are turned back into numerators here (rate x the
    weight the spec asks for) and divided by the summed weight later. Averaging a rate
    across games would weight a two-target night the same as a twelve-target one.
    """
    if pw is None or pw.empty:
        return pd.DataFrame()
    d = pw.copy()
    d["touch_first_downs"] = _z(d, "receiving_first_downs") + _z(d, "rushing_first_downs")
    d["touches"] = _z(d, "carries") + _z(d, "receptions")

    # team air yards and team expected points come from ff_opportunity's own team columns
    d["team_air_yards"] = np.nan
    d["team_xfp"] = np.nan
    if ffo is not None and not ffo.empty:
        f = ffo.rename(columns={"player_id": "gsis_id", "posteam": "team"})
        f["week"] = pd.to_numeric(f["week"], errors="coerce").astype("Int64")
        cols = {}
        if "rec_air_yards_team" in f.columns:
            cols["team_air_yards"] = ("rec_air_yards_team", "max")
        if "total_fantasy_points_exp_team" in f.columns:
            cols["team_xfp"] = ("total_fantasy_points_exp_team", "max")
        if cols and "team" in f.columns:
            # a team column repeats the same team total on every one of its players' rows
            t = f.groupby(["team", "week"]).agg(**cols).reset_index()
            t["team"] = t["team"].map(_canon)
            d = d.drop(columns=[c for c in ("team_air_yards", "team_xfp") if c in d.columns])
            d = d.merge(t, on=["team", "week"], how="left")

    d = _join_ngs(d, ngs_rec, "avg_yac_above_expectation", "yacoe_num", "receptions")
    d = _join_ngs(d, ngs_rec, "avg_separation", "separation_num", "targets")
    d = _join_ngs(d, ngs_rush, "rush_yards_over_expected_per_att", "ryoe_num", "carries")
    return d


def _canon(t: object) -> str:
    try:
        from .ids import canon_team
        return canon_team(t)
    except Exception:
        return str(t).upper().strip()


def _join_ngs(d: pd.DataFrame, ngs, rate_col: str, out_col: str, weight_col: str) -> pd.DataFrame:
    """Turn an NGS per-game rate into a numerator: rate x that game's weight.

    Left as NaN where NGS has no row. NGS only publishes qualifying players, so filling a
    zero would read as "separates badly" when the truth is "was never measured" (§12.2,
    and test 24 pins it)."""
    d[out_col] = np.nan
    if ngs is None or getattr(ngs, "empty", True) or rate_col not in ngs.columns:
        return d
    g = ngs.rename(columns={"player_gsis_id": "gsis_id"})
    if "gsis_id" not in g.columns:
        return d
    g = g.assign(week=pd.to_numeric(g["week"], errors="coerce").astype("Int64"))
    g = g[g["week"] != 0]                       # week 0 is NGS's season-total row
    g = g.groupby(["gsis_id", "week"])[rate_col].mean().rename("_rate").reset_index()
    d = d.drop(columns=[out_col]).merge(g, on=["gsis_id", "week"], how="left")
    d[out_col] = d["_rate"] * _z(d, weight_col)
    return d.drop(columns=["_rate"])


def _window(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """The last `n` games each player actually played — not the last n weeks, which would
    punish a player for his team's bye and for the weeks he was hurt."""
    d = d.sort_values(["gsis_id", "week"])
    d["_rk"] = d.groupby("gsis_id").cumcount(ascending=False)
    return d[d["_rk"] < n].drop(columns=["_rk"])


# ---------------------------------------------------------------- §12.1 role assignment
ROLES = ("WR1", "WR2", "WR3", "WR4+", "TE1-REC", "TE1-BLK", "TE2",
         "LEAD", "COMMITTEE", "RECEIVING", "BACKUP", "STARTER", "QB-BACKUP")

# The role each role is measured against for ROLE+ / ROLE− (§12.5): the rung above and below.
LADDER = {
    "WR": ("WR1", "WR2", "WR3", "WR4+"),
    "TE": ("TE1-REC", "TE1-BLK", "TE2"),
    "RB": ("LEAD", "COMMITTEE", "RECEIVING", "BACKUP"),
    "QB": ("STARTER", "QB-BACKUP"),
}
# The metric that defines each position's pecking order (§12.5)
ROLE_METRIC = {"WR": "wopr", "TE": "target_share", "RB": "carry_share", "QB": "carry_share"}


def assign(pw: pd.DataFrame, depth: pd.DataFrame | None = None,
           window: int = WINDOW_ROLES, min_games: int = MIN_GAMES) -> pd.DataFrame:
    """One role per player, from his last `window` games played (§12.1).

    Players with fewer than `min_games` are placed from the depth chart instead — a rookie
    who has played once should not be called a WR1 because he happened to lead the team in
    targets in the game he appeared in.
    """
    if pw is None or pw.empty:
        return pd.DataFrame(columns=["gsis_id", "pos", "team", "role", "role_src"])
    w = _window(pw, window)
    g = w.groupby("gsis_id")
    agg = g.agg(pos=("pos", "last"), team=("team", "last"), games=("week", "nunique"),
                targets=("targets", "sum"), carries=("carries", "sum"),
                snaps=("offense_snaps", "sum"), team_snaps=("team_snaps", "sum"),
                team_targets=("team_targets", "sum"), team_carries=("team_carries", "sum"),
                attempts=("attempts", "sum"), air=("receiving_air_yards", "sum"),
                team_air=("team_air_yards", "sum")).reset_index()
    played = pw.groupby("gsis_id")["week"].nunique().rename("played_total")
    agg = agg.merge(played, on="gsis_id", how="left")

    agg["tgt_share"] = agg["targets"] / agg["team_targets"].replace(0, np.nan)
    agg["car_share"] = agg["carries"] / agg["team_carries"].replace(0, np.nan)
    agg["snap_share"] = agg["snaps"] / agg["team_snaps"].replace(0, np.nan)
    agg["air_share"] = agg["air"] / agg["team_air"].replace(0, np.nan)
    agg["wopr"] = WOPR_W[0] * agg["tgt_share"].fillna(0) + WOPR_W[1] * agg["air_share"].fillna(0)

    # QB starter share: of his own team's pass attempts. The spec says team dropbacks;
    # attempts is the same ordering and needs no play-by-play pass.
    tm_att = agg.groupby("team")["attempts"].transform("sum")
    agg["qb_share"] = agg["attempts"] / tm_att.replace(0, np.nan)

    agg["rank_tgt"] = agg[agg["pos"] == "WR"].groupby("team")["tgt_share"].rank(
        ascending=False, method="min").reindex(agg.index)
    agg["rank_te_snap"] = agg[agg["pos"] == "TE"].groupby("team")["snap_share"].rank(
        ascending=False, method="min").reindex(agg.index)

    agg["role"] = [_role_for(r) for _, r in agg.iterrows()]
    agg["role_src"] = "usage"

    thin = agg["played_total"].fillna(0) < min_games
    if thin.any() and depth is not None and not depth.empty:
        dep = _depth_roles(depth)
        fallback = agg.loc[thin, "gsis_id"].map(dep)
        agg.loc[thin, "role"] = fallback.fillna(agg.loc[thin, "role"])
        agg.loc[thin & fallback.notna(), "role_src"] = "depth chart"
    return agg


def _role_for(r: pd.Series) -> str:
    pos, ts, cs, ss = r["pos"], r["tgt_share"], r["car_share"], r["snap_share"]
    ts = 0.0 if pd.isna(ts) else ts
    cs = 0.0 if pd.isna(cs) else cs
    ss = 0.0 if pd.isna(ss) else ss
    if pos == "WR":
        rk = r.get("rank_tgt")
        if rk == 1:
            return "WR1"
        if rk == 2:
            return "WR2"
        if rk == 3 and ss >= 0.50:
            return "WR3"
        return "WR4+"
    if pos == "TE":
        if r.get("rank_te_snap") == 1:
            return "TE1-REC" if ts >= 0.12 else "TE1-BLK"
        return "TE2"
    if pos == "RB":
        if cs >= 0.50:
            return "LEAD"
        if cs >= 0.30:
            return "COMMITTEE"
        if ts >= 0.08:
            return "RECEIVING"
        return "BACKUP"
    if pos == "QB":
        return "STARTER" if (r.get("qb_share") or 0) >= 0.70 else "QB-BACKUP"
    return "BACKUP"


_DEPTH_ROLE = {
    ("WR", 1): "WR1", ("WR", 2): "WR2", ("WR", 3): "WR3",
    ("TE", 1): "TE1-REC", ("RB", 1): "LEAD", ("RB", 2): "BACKUP",
    ("QB", 1): "STARTER",
}


def _depth_roles(depth: pd.DataFrame) -> dict:
    """gsis_id -> role from the most recent depth chart snapshot."""
    d = depth.copy()
    if "dt" in d.columns:
        d = d[d["dt"] == d["dt"].max()]
    pos = d.get("pos_abb", d.get("pos_grp"))
    if pos is None or "pos_rank" not in d.columns:
        return {}
    d = d.assign(_p=pos.astype(str).str.upper().str[:2],
                 _r=pd.to_numeric(d["pos_rank"], errors="coerce"))
    d["_p"] = d["_p"].replace({"WR": "WR", "TE": "TE", "RB": "RB", "QB": "QB"})
    out = {}
    for _, row in d.iterrows():
        key = (row["_p"], int(row["_r"])) if pd.notna(row["_r"]) else None
        if key is None or row["_p"] not in LADDER:
            continue
        out[row["gsis_id"]] = _DEPTH_ROLE.get(key, LADDER[row["_p"]][-1])
    return out


# ---------------------------------------------------------------- §12.2–12.4 the table
def totals(pw: pd.DataFrame, roles: pd.DataFrame, window: int = WINDOW_METRICS) -> pd.DataFrame:
    """Summed numerators and denominators per player over his last `window` games."""
    if pw is None or pw.empty:
        return pd.DataFrame()
    w = _window(pw, window)
    cols = sorted({c for m in METRICS for c in (m.num, m.den)})
    have = [c for c in cols if c in w.columns]
    g = w.groupby("gsis_id")
    out = g[have].sum(min_count=1)
    # a numerator built from NGS stays NaN when NGS never measured him
    for c in ("yacoe_num", "separation_num", "ryoe_num"):
        if c in w.columns:
            out[c] = g[c].sum(min_count=1).where(g[c].count() > 0)
    out["games"] = g["week"].nunique()
    out = out.reset_index().merge(
        roles[["gsis_id", "pos", "team", "role", "role_src"]], on="gsis_id", how="left")
    return out


def baselines(tot: pd.DataFrame) -> dict:
    """§12.3 — the numbers you read a player against.

    For every metric, three things: the baseline for each ROLE, the NFL average for each
    POSITION, and the spread among players in that role. Each is a summed numerator over a
    summed denominator across all player-games in the group, and each carries the sample it
    came from so a baseline built on two players can be recognised as one.
    """
    out: dict = {"role": {}, "pos": {}, "spread": {}, "n": {}}
    if tot is None or tot.empty:
        return out
    for m in METRICS:
        if m.num not in tot.columns or m.den not in tot.columns:
            continue
        d = tot[tot["pos"].isin(m.pos)]
        d = d[d[m.den].fillna(0) > 0]
        if d.empty:
            continue
        for grp, key in (("role", "role"), ("pos", "pos")):
            g = d.groupby(key)
            num, den = g[m.num].sum(), g[m.den].sum()
            val = (num / den.replace(0, np.nan)).dropna()
            out[grp][m.key] = val.to_dict()
            out["n"].setdefault(grp, {})[m.key] = {
                k: {"players": int(v), "denominator": float(den.get(k, 0.0))}
                for k, v in g.size().items()}
        # spread among role qualifiers, on the shrunk value so one four-target night
        # does not set the width of the whole distribution
        sp = {}
        for role, sub in d.groupby("role"):
            mu = out["role"].get(m.key, {}).get(role)
            if mu is None or len(sub) < 2:
                continue
            k = K.get(m.key, 0.0)
            x = (sub[m.num].fillna(0) + k * mu) / (sub[m.den].fillna(0) + k)
            sp[role] = {"sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                        "p25": float(x.quantile(0.25)), "p50": float(x.quantile(0.50)),
                        "p75": float(x.quantile(0.75)), "players": int(len(x))}
        out["spread"][m.key] = sp

    # WOPR is composed from two shares rather than summed from a numerator, so it is not
    # in METRICS — but it IS the metric that ranks wide receivers (ROLE_METRIC), and
    # without a baseline here every WR ROLE+ / ROLE- test compared against None and
    # quietly did nothing at all.
    for grp in ("role", "pos"):
        if out[grp].get("target_share"):
            out[grp]["wopr"] = _wopr_baseline(out, grp)
    out["spread"]["wopr"] = {
        r: {"sd": v.get("sd", 0.0) * WOPR_W[0]}      # dominated by the target-share term
        for r, v in out["spread"].get("target_share", {}).items()}
    return out


def index(tot: pd.DataFrame, base: dict) -> pd.DataFrame:
    """§12.4 — shrink each player toward his role, then express it three ways.

        x̂    = (num + k μ_role) / (den + k)
        idx  = 100 x̂ / μ_role     (100 = average for his role)
        idxP = 100 x̂ / μ_pos      (100 = NFL average at his position)
        z    = (x̂ − μ_role) / σ_role
    """
    if tot is None or tot.empty:
        return pd.DataFrame()
    out = tot[["gsis_id", "pos", "team", "role", "role_src", "games"]].copy()
    for m in METRICS:
        if m.num not in tot.columns or m.den not in tot.columns:
            continue
        mu_r = tot["role"].map(base.get("role", {}).get(m.key, {}))
        mu_p = tot["pos"].map(base.get("pos", {}).get(m.key, {}))
        k = K.get(m.key, 0.0)
        num, den = tot[m.num], tot[m.den]
        raw = num / den.replace(0, np.nan)
        x = (num.fillna(0) + k * mu_r.fillna(0)) / (den.fillna(0) + k)
        # NGS never measured him: leave every derived figure missing rather than pretend
        x = x.where(num.notna() & (den.fillna(0) > 0))
        sd = tot["role"].map({r: v.get("sd") for r, v in base.get("spread", {}).get(m.key, {}).items()})
        out[m.key] = x
        out[f"{m.key}_raw"] = raw
        out[f"{m.key}_idx"] = 100.0 * x / mu_r.replace(0, np.nan)
        out[f"{m.key}_idxp"] = 100.0 * x / mu_p.replace(0, np.nan)
        out[f"{m.key}_z"] = (x - mu_r) / sd.replace(0, np.nan)

    # WOPR is composed after shrinkage, from the two shares it is made of (§12.2)
    if {"target_share", "air_share"} <= set(out.columns):
        out["wopr"] = WOPR_W[0] * out["target_share"] + WOPR_W[1] * out["air_share"].fillna(0.0)
        mu_r = out["role"].map(_wopr_baseline(base, "role"))
        mu_p = out["pos"].map(_wopr_baseline(base, "pos"))
        out["wopr_idx"] = 100.0 * out["wopr"] / mu_r.replace(0, np.nan)
        out["wopr_idxp"] = 100.0 * out["wopr"] / mu_p.replace(0, np.nan)
        sd = out.groupby("role")["wopr"].transform(lambda s: s.std(ddof=1) if len(s) > 1 else np.nan)
        out["wopr_z"] = (out["wopr"] - mu_r) / sd.replace(0, np.nan)
    return out


def _wopr_baseline(base: dict, grp: str) -> dict:
    ts = base.get(grp, {}).get("target_share", {})
    ays = base.get(grp, {}).get("air_share", {})
    return {k: WOPR_W[0] * v + WOPR_W[1] * ays.get(k, 0.0) for k, v in ts.items()}


# ---------------------------------------------------------------- §12.5 role flags
# The flat v1.1 thresholds survive as a FLOOR. Without one, a WR4 at 1.2x a tiny WR4
# baseline flags as though he were producing, which is how the old board filled up with
# players who had done nothing.
FLOOR = {
    "TGT": {"WR": 0.20, "TE": 0.15, "RB": 0.10},
    "AIR": {"WR": 0.25, "TE": 0.25},
    "SNAP": {"WR": 0.70, "TE": 0.70, "RB": 0.55},
    "LEAD": {"RB": 0.50},
    "GL": {"RB": 0.35, "QB": 0.35},
}
FLAG_METRIC = {"TGT": "target_share", "AIR": "air_share", "SNAP": "snap_pct",
               "LEAD": "carry_share", "GL": "inside10_share"}
WR_FD_RR_ANCHOR = 0.12      # doctrine anchor (§12.5)
RB_AGE_WARN = 28


def flags(idx: pd.DataFrame, base: dict) -> pd.DataFrame:
    """Role-relative flags. A flag fires on role index >= 120 AND the old flat floor."""
    if idx is None or idx.empty:
        return pd.DataFrame(columns=["gsis_id", "flags"])
    rows = []
    for _, r in idx.iterrows():
        got = []
        for name, metric in FLAG_METRIC.items():
            floor = FLOOR.get(name, {}).get(r["pos"])
            if floor is None:
                continue
            val, ix = r.get(metric), r.get(f"{metric}_idx")
            if pd.notna(val) and pd.notna(ix) and val >= floor and ix >= ROLE_FLAG_IDX:
                got.append(name)
        up, down = _neighbours(r["pos"], r["role"])
        rm = ROLE_METRIC.get(r["pos"])
        if r["pos"] == "RB" and r["role"] == "RECEIVING":
            rm = "target_share"
        val = r.get(rm) if rm else None
        if pd.notna(val):
            mu_up = base.get("role", {}).get(rm, {}).get(up) if up else None
            mu_dn = base.get("role", {}).get(rm, {}).get(down) if down else None
            if mu_up and val >= mu_up:
                got.append("ROLE+")
            elif mu_dn and val <= mu_dn:
                got.append("ROLE-")
        if r["pos"] == "WR" and pd.notna(r.get("fd_rr")) and r["fd_rr"] >= WR_FD_RR_ANCHOR:
            got.append("1D/RR")
        rows.append({"gsis_id": r["gsis_id"], "flags": got})
    return pd.DataFrame(rows)


def _neighbours(pos: str, role: str) -> tuple[str | None, str | None]:
    """The rung above and the rung below a player's role."""
    ladder = LADDER.get(pos, ())
    if role not in ladder:
        return None, None
    i = ladder.index(role)
    return (ladder[i - 1] if i > 0 else None,
            ladder[i + 1] if i + 1 < len(ladder) else None)


def persistence(pw: pd.DataFrame, window: int = WINDOW_ROLES) -> pd.DataFrame:
    """§4.2 — per flag: `sustained` when it held in >= 2 of the last `window` games played,
    `spike` when it held in only the most recent one.

    This is what stops one big afternoon from reading as a role change, which is the whole
    reason the old board filled with players who had done it once. A player with fewer than
    two games can only ever be a spike — there is no second game to sustain anything.

    Flags are tested per game against the flat floors, not the role index: a role index is
    a statement about a window, and asking whether it held "in that one game" is not a
    question it can answer.
    """
    empty = pd.DataFrame(columns=["gsis_id", "tags"])
    if pw is None or pw.empty:
        return empty
    w = _window(pw, window).sort_values(["gsis_id", "week"])
    rows = []
    for gid, sub in w.groupby("gsis_id"):
        pos = str(sub["pos"].iloc[-1])
        tags = {}
        for name, metric in FLAG_METRIC.items():
            floor = FLOOR.get(name, {}).get(pos)
            m = METRIC_BY_KEY.get(metric)
            if floor is None or m is None or m.num not in sub.columns or m.den not in sub.columns:
                continue
            den = pd.to_numeric(sub[m.den], errors="coerce")
            val = pd.to_numeric(sub[m.num], errors="coerce") / den.where(den > 0)
            hit = (val >= floor).fillna(False).tolist()
            if not any(hit):
                continue
            if len(hit) >= 2 and sum(hit) >= 2:
                tags[name] = "sustained"
            elif hit[-1]:
                tags[name] = "spike"
            else:
                tags[name] = "spike"
        rows.append({"gsis_id": gid, "tags": tags})
    return pd.DataFrame(rows) if rows else empty


# ---------------------------------------------------------------- one call for the app
def build(pw: pd.DataFrame, ffo=None, ngs_rec=None, ngs_rush=None, depth=None) -> dict:
    """Everything §12 produces, in one pass: the frame, the roles, the baselines and the
    indexed table with flags attached."""
    fr = frame(pw, ffo, ngs_rec, ngs_rush)
    if fr.empty:
        return {"frame": fr, "roles": pd.DataFrame(), "baselines": {}, "table": pd.DataFrame()}
    rl = assign(fr, depth)
    tot = totals(fr, rl)
    base = baselines(tot)
    tab = index(tot, base)
    fl = flags(tab, base)
    if not fl.empty:
        tab = tab.merge(fl, on="gsis_id", how="left")
    tg = persistence(fr)
    if not tg.empty:
        tab = tab.merge(tg, on="gsis_id", how="left")
    return {"frame": fr, "roles": rl, "baselines": base, "totals": tot, "table": tab}


def baseline_table(base: dict, which: str = "role") -> pd.DataFrame:
    """The baselines as something readable: one row per role (or position), one column per
    metric, with the sample each is built on."""
    src = base.get(which, {})
    if not src:
        return pd.DataFrame()
    groups = sorted({g for vals in src.values() for g in vals})
    order = [r for pos in LADDER.values() for r in pos] if which == "role" else list(POS_ALL)
    groups.sort(key=lambda g: order.index(g) if g in order else 99)
    rows = []
    for g in groups:
        row = {"role" if which == "role" else "pos": g}
        n = base.get("n", {}).get(which, {})
        row["players"] = max((n.get(m.key, {}).get(g, {}).get("players", 0) for m in METRICS),
                             default=0)
        for m in METRICS:
            v = src.get(m.key, {}).get(g)
            if v is not None:
                row[m.key] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- one player, for the card
def card(tab: pd.DataFrame, base: dict, gsis_id: str) -> pd.DataFrame:
    """One player's usage read three ways: his own figure, the baseline for his role, and
    the NFL average at his position — with the index against each.

    This is the whole point of §12. A 20% target share is a strong WR3 and a poor WR1, and
    the only way to see which is to put the role's baseline next to it.
    """
    if tab is None or tab.empty or gsis_id not in set(tab["gsis_id"]):
        return pd.DataFrame()
    r = tab[tab["gsis_id"] == gsis_id].iloc[0]
    pos, role = r["pos"], r["role"]
    rows = []
    for m in METRICS:
        if pos not in m.pos or m.key not in tab.columns or pd.isna(r.get(m.key)):
            continue
        mu_r = base.get("role", {}).get(m.key, {}).get(role)
        mu_p = base.get("pos", {}).get(m.key, {}).get(pos)
        rows.append({
            "metric": m.label,
            "him": r[m.key],
            f"{role} avg": mu_r,
            "vs role": r.get(f"{m.key}_idx"),
            f"NFL {pos} avg": mu_p,
            "vs NFL": r.get(f"{m.key}_idxp"),
            "_fmt": m.fmt,
        })
    return pd.DataFrame(rows)


def describe(tab: pd.DataFrame, gsis_id: str) -> str:
    """A sentence for the top of the card: what he is, and how it was decided."""
    if tab is None or tab.empty or gsis_id not in set(tab["gsis_id"]):
        return ""
    r = tab[tab["gsis_id"] == gsis_id].iloc[0]
    src = "from his usage" if r.get("role_src") == "usage" else "from the depth chart"
    bits = [f"**{r['role']}** ({src}, {int(r.get('games') or 0)} games)"]
    fl = r.get("flags")
    if isinstance(fl, list) and fl:
        tags = r.get("tags") if isinstance(r.get("tags"), dict) else {}
        bits.append(" · ".join(f"{f}{' (' + tags[f] + ')' if f in tags else ''}" for f in fl))
    return " — ".join(bits)
