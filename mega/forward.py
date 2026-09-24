"""§13 expected future opportunity and §14 the earned-opportunity adjustment.

v1.1 valued a player on a rolling xFP — what his role has been worth. That is backward
looking, and it prices a player into a schedule he has already played. §13 instead projects
each remaining week from three things that are actually knowable:

  * the share of his own offence's expected points he commands (shrunk, §12.4)
  * what that offence is expected to be worth that week, from the Vegas implied total
  * how the game script tilts it — favoured teams run more, trailing teams throw

§14 then asks whether he will keep that share. Current share says what a player gets;
targets per route, first downs and separation say whether he earns it. That adjustment is
deliberately small (±6% at the clamp) until there is 2026 data to fit it against — §19.4.

Every constant here marked (J) is a judgment default, not a fitted value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# §13 defaults (J)
GAMMA = 0.60                       # volume moves less than points do
BETA = {"rush": 0.05, "rec": -0.03, "pass": -0.03}
SPREAD_SCALE = 7.0                 # a touchdown's worth of spread is the unit
# §14 defaults (J)
LAMBDA = 0.04
Z_CLAMP = 1.5
OPP_SCORE_SD = 15.0                # Opportunity Score = 100 + 15 z

COMPONENTS = ("rec", "rush", "pass")

# half-PPR weights applied to the ff_opportunity component columns (§3.2). The package's
# own total_fantasy_points_exp is NOT half-PPR, which is why this is recomputed rather
# than read.
_XFP_TERMS = {
    "rec": ((0.10, "rec_yards_gained_exp"), (6.0, "rec_touchdown_exp"),
            (0.5, "receptions_exp"), (2.0, "rec_two_point_conv_exp")),
    "rush": ((0.10, "rush_yards_gained_exp"), (6.0, "rush_touchdown_exp"),
             (2.0, "rush_two_point_conv_exp")),
    "pass": ((0.04, "pass_yards_gained_exp"), (4.0, "pass_touchdown_exp"),
             (-1.0, "pass_interception_exp"), (2.0, "pass_two_point_conv_exp")),
}


def _n(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else pd.Series(0.0, index=df.index)


def xfp_components(ffo: pd.DataFrame, team: bool = False) -> pd.DataFrame:
    """Half-PPR expected points per player-week (or per team-week), split rec/rush/pass."""
    if ffo is None or ffo.empty:
        return pd.DataFrame()
    suffix = "_team" if team else ""
    out = pd.DataFrame(index=ffo.index)
    for comp, terms in _XFP_TERMS.items():
        total = pd.Series(0.0, index=ffo.index)
        for w, col in terms:
            total = total + w * _n(ffo, col + suffix)
        out[f"xfp_{comp}"] = total
    out["xfp_total"] = out[[f"xfp_{c}" for c in COMPONENTS]].sum(axis=1)
    return out


# ---------------------------------------------------------------- §13 implied totals
def implied(sched: pd.DataFrame) -> pd.DataFrame:
    """team, week, implied points and the spread from that team's side.

    Positive `spread_line` in nflverse means the HOME team is favoured, so the home side
    adds half the spread and the away side subtracts it. Getting this backwards silently
    inverts every game-script adjustment, so the sign is pinned by a test.
    """
    if sched is None or sched.empty:
        return pd.DataFrame(columns=["team", "week", "implied", "spread"])
    s = sched
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper() == "REG"]
    tl, sl = _nan(s, "total_line"), _nan(s, "spread_line")
    rows = []
    for side, sign in (("home_team", 1.0), ("away_team", -1.0)):
        rows.append(pd.DataFrame({
            "team": s[side],
            "week": pd.to_numeric(s["week"], errors="coerce").astype("Int64"),
            "implied": tl / 2.0 + sign * sl / 2.0,
            "spread": sign * sl,
        }))
    out = pd.concat(rows, ignore_index=True)
    out["team"] = out["team"].map(_canon)
    return out


def _nan(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)


def _canon(t: object) -> str:
    try:
        from .ids import canon_team
        return canon_team(t)
    except Exception:
        return str(t).upper().strip()


def team_volume(ffo: pd.DataFrame, sched: pd.DataFrame, weeks) -> pd.DataFrame:
    """Each team's expected half-PPR points per game by component, adjusted for the week.

        teamXFP_c,w = perGame_c * (implied_w / implied_avg) ^ gamma * (1 + beta_c * spread_w / 7)

    Weeks with no posted line fall back to the team's own average implied total, so the
    ratio is 1 and the game-script term is 0 — Vegas only prices a week or two ahead, and
    a missing line must not read as a low-scoring game.
    """
    if ffo is None or ffo.empty:
        return pd.DataFrame()
    f = ffo.rename(columns={"posteam": "team"})
    if "team" not in f.columns:
        return pd.DataFrame()
    f = f.assign(week=pd.to_numeric(f["week"], errors="coerce").astype("Int64"))
    comp = xfp_components(f, team=True)
    # a *_team column repeats the team's total on each of its players' rows
    tw = pd.concat([f[["team", "week"]], comp], axis=1).groupby(["team", "week"]).max().reset_index()
    tw["team"] = tw["team"].map(_canon)
    per_game = tw.groupby("team")[[f"xfp_{c}" for c in COMPONENTS]].mean()

    imp = implied(sched)
    avg = imp.groupby("team")["implied"].mean().rename("implied_avg")

    rows = []
    for team, base in per_game.iterrows():
        a = avg.get(team, np.nan)
        for w in weeks:
            hit = imp[(imp["team"] == team) & (imp["week"] == w)]
            i_w = float(hit["implied"].iloc[0]) if not hit.empty and pd.notna(hit["implied"].iloc[0]) else a
            sp = float(hit["spread"].iloc[0]) if not hit.empty and pd.notna(hit["spread"].iloc[0]) else 0.0
            ratio = (i_w / a) if (pd.notna(a) and a and pd.notna(i_w)) else 1.0
            row = {"team": team, "week": int(w), "implied": i_w, "spread": sp,
                   "implied_ratio": ratio}
            for c in COMPONENTS:
                row[f"xfp_{c}"] = float(base[f"xfp_{c}"]) * (ratio ** GAMMA) * (
                    1.0 + BETA[c] * sp / SPREAD_SCALE)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- §14 earned opportunity
EARN_Z = {
    "WR": ("tprr_z", "fd_rr_z", "wopr_z", "separation_z"),
    "TE": ("tprr_z", "fd_rr_z", "wopr_z", "separation_z"),
    "RB": ("target_share_z", "fd_per_touch_z", "ryoe_z", "snap_pct_z"),
    "QB": ("carry_share_z", "inside10_share_z"),
}


def earned(tab: pd.DataFrame) -> pd.DataFrame:
    """§14 — does he earn the share he has?

    A mean of role-relative z scores, nulls skipped rather than treated as zero: a missing
    Next Gen Stats row means nobody measured him, and scoring that as "exactly average"
    would quietly drag every unmeasured player toward the middle.

    `earnAdj` is capped at ±6% on purpose. It is a judgment weight until §19.4 can regress
    the next three weeks' share change on it, and a small wrong number is recoverable in a
    way that a large wrong number is not.
    """
    if tab is None or tab.empty:
        return pd.DataFrame(columns=["gsis_id", "z_earn", "earn_adj", "opp_score"])
    rows = []
    for _, r in tab.iterrows():
        cols = EARN_Z.get(str(r.get("pos")), ())
        vals = [r[c] for c in cols if c in tab.columns and pd.notna(r.get(c))]
        z = float(np.mean(vals)) if vals else np.nan
        adj = 1.0 + LAMBDA * float(np.clip(z, -Z_CLAMP, Z_CLAMP)) if pd.notna(z) else 1.0
        rows.append({
            "gsis_id": r["gsis_id"], "z_earn": z, "earn_adj": adj,
            "opp_score": 100.0 + OPP_SCORE_SD * z if pd.notna(z) else np.nan,
            "z_parts": len(vals),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- §13 per-player forward
def player_shares(tab: pd.DataFrame, pw: pd.DataFrame, ffo: pd.DataFrame,
                  window: int = 4) -> pd.DataFrame:
    """Each player's share of his own offence's expected points, by component.

    A player who changed teams uses only the games with his current team (§13); without
    that, a mid-season trade averages two offences into one meaningless share.
    """
    if ffo is None or ffo.empty or pw is None or pw.empty:
        return pd.DataFrame(columns=["gsis_id"] + [f"share_{c}" for c in COMPONENTS])
    f = ffo.rename(columns={"player_id": "gsis_id", "posteam": "team"})
    f = f.assign(week=pd.to_numeric(f["week"], errors="coerce").astype("Int64"))
    me = pd.concat([f[["gsis_id", "team", "week"]], xfp_components(f, team=False)], axis=1)
    tm = pd.concat([f[["team", "week"]], xfp_components(f, team=True)], axis=1)
    tm = tm.groupby(["team", "week"]).max().reset_index()
    tm.columns = ["team", "week"] + [f"tm_{c}" for c in tm.columns[2:]]
    me["team"] = me["team"].map(_canon)
    tm["team"] = tm["team"].map(_canon)

    now = pw.sort_values("week").groupby("gsis_id")["team"].last().rename("cur_team")
    me = me.merge(now, on="gsis_id", how="left")
    me = me[me["team"] == me["cur_team"]]                    # current team only

    me = me.sort_values(["gsis_id", "week"])
    me["_rk"] = me.groupby("gsis_id").cumcount(ascending=False)
    me = me[me["_rk"] < window].merge(tm, on=["team", "week"], how="left")

    g = me.groupby("gsis_id")
    out = pd.DataFrame(index=g.size().index)
    for c in COMPONENTS:
        num, den = g[f"xfp_{c}"].sum(), g[f"tm_xfp_{c}"].sum()
        out[f"share_{c}"] = num / den.replace(0, np.nan)
    out["team"] = g["team"].last()
    return out.reset_index()


def project(tab: pd.DataFrame, pw: pd.DataFrame, ffo: pd.DataFrame, sched: pd.DataFrame,
            weeks) -> pd.DataFrame:
    """§13 — expected half-PPR points per player per future week.

        xfp_fwd[p, w] = Σ_c share_c(p) * earnAdj(p) * teamXFP_c(team(p), w)
    """
    weeks = [int(w) for w in weeks]
    shares = player_shares(tab, pw, ffo)
    if shares.empty:
        return pd.DataFrame(columns=["gsis_id", "week", "xfp_fwd"])
    vol = team_volume(ffo, sched, weeks)
    if vol.empty:
        return pd.DataFrame(columns=["gsis_id", "week", "xfp_fwd"])
    adj = earned(tab)[["gsis_id", "earn_adj"]] if tab is not None and not tab.empty else None
    if adj is not None:
        shares = shares.merge(adj, on="gsis_id", how="left")
    shares["earn_adj"] = shares.get("earn_adj", pd.Series(1.0, index=shares.index)).fillna(1.0)

    out = shares.merge(vol, on="team", how="inner")
    out["xfp_fwd"] = sum(
        out[f"share_{c}"].fillna(0.0) * out["earn_adj"] * out[f"xfp_{c}"] for c in COMPONENTS)
    return out[["gsis_id", "team", "week", "xfp_fwd", "earn_adj", "implied", "spread"]]


def blend(proj_pg: pd.Series, xfp_fwd: pd.Series, games: pd.Series,
          max_weight: float = 0.40, ramp: int = 3) -> pd.Series:
    """§3.3, with §13's forward xFP in place of the backward rolling one.

        w   = maxWeight * min(g, ramp) / ramp
        ppg = (1 - w) * proj + w * xfp_fwd

    Usage earns its weight as the sample grows: nothing at g=0, a third of the way in by
    one game, full by the third. Expected points rather than actual, because xFP measures
    the role without the touchdown luck — the luck is the signal for sell-high (§4.3).
    """
    g = pd.to_numeric(games, errors="coerce").fillna(0.0).clip(lower=0)
    w = max_weight * np.minimum(g, ramp) / ramp
    p = pd.to_numeric(proj_pg, errors="coerce")
    x = pd.to_numeric(xfp_fwd, errors="coerce")
    return (1 - w) * p.fillna(x) + w * x.fillna(p)
