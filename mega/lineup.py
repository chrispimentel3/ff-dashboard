"""Defense-vs-position matchup model + start/sit optimizer.

- defense_vs_position(): half-PPR points each NFL defense allows to each position,
  per game, ranked 1–32 (1 = easiest / allows the most). Season-to-date with a
  prior-season fallback early in the year.
- optimize_lineup(): fills the best legal half-PPR lineup from a roster + projections,
  applies the matchup adjustment ONLY to nflverse estimates (FantasyPros projections
  already price the matchup), and flags close start/sit calls.
"""
from __future__ import annotations

import functools

import pandas as pd

from .config import FLEX_ELIGIBLE, LINEUP
from .intel import form_season, weekly

SKILL = ("QB", "RB", "WR", "TE")


@functools.lru_cache(maxsize=4)
def defense_vs_position(season: int) -> pd.DataFrame:
    """Columns: defense, pos, pa_pg (allowed/game), ease_rank (1=easiest), mult (vs league avg)."""
    fs = form_season(season)
    w = weekly(fs)
    if w.empty or "opponent_team" not in w.columns:
        return pd.DataFrame(columns=["defense", "pos", "pa_pg", "ease_rank", "mult"])
    w = w[w["pos"].isin(SKILL)].copy()
    per_wk = w.groupby(["opponent_team", "pos", "week"], as_index=False)["half_ppr"].sum()
    dvp = per_wk.groupby(["opponent_team", "pos"], as_index=False)["half_ppr"].mean().rename(
        columns={"opponent_team": "defense", "half_ppr": "pa_pg"}
    )
    lg = dvp.groupby("pos")["pa_pg"].transform("mean")
    dvp["mult"] = (dvp["pa_pg"] / lg).round(3)
    dvp["ease_rank"] = dvp.groupby("pos")["pa_pg"].rank(ascending=False, method="min").astype(int)
    return dvp


def _opponents(season: int, week: int) -> pd.DataFrame:
    """team -> (opponent, home/away) for a given week, from nflverse schedule."""
    import nflreadpy as nfl

    sched = nfl.load_schedules(seasons=[season]).to_pandas()
    wk = sched[sched["week"] == week]
    rows = []
    for _, g in wk.iterrows():
        rows.append(dict(team=g["home_team"], opp=g["away_team"], home=True))
        rows.append(dict(team=g["away_team"], opp=g["home_team"], home=False))
    return pd.DataFrame(rows)


def attach_matchup(players: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Add opponent, ease_rank, mult to a roster frame (needs cols: pos, nfl_team).

    `mult` prefers the fitted defense+Vegas model (`mega.matchup_model`) where it
    has an answer for that team/pos this week, and falls back to the plain
    points-allowed ratio otherwise (a bye-adjacent edge case, or the fitted model's
    network fetch failing) — never silently drops to 1.0 when a cruder number is
    available."""
    dvp = defense_vs_position(season)
    opp = _opponents(season, week)
    df = players.merge(opp, left_on="nfl_team", right_on="team", how="left")
    if not dvp.empty:
        df = df.merge(dvp[["defense", "pos", "ease_rank", "mult"]],
                      left_on=["opp", "pos"], right_on=["defense", "pos"], how="left").drop(columns=["defense"])
    df["mult"] = pd.to_numeric(df.get("mult"), errors="coerce").fillna(1.0)

    try:
        from .matchup_model import week_matchups
        fitted = week_matchups(season, week)
        fitted = fitted[fitted["week"] == week] if not fitted.empty else fitted
    except Exception:
        fitted = pd.DataFrame()
    if not fitted.empty:
        df = df.merge(
            fitted[["team", "pos", "mult", "pct", "basis"]].rename(
                columns={"team": "nfl_team", "mult": "mult_fitted", "pct": "pct_fitted", "basis": "matchup_basis"}),
            on=["nfl_team", "pos"], how="left",
        )
    else:
        df["mult_fitted"] = pd.NA
    return df


def optimize_lineup(roster: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """roster needs: player, pos, nfl_team, proj, proj_source (+ optional ecr/start_sit).
    Returns the roster with matchup-adjusted proj, lineup slot assignment, and flags."""
    df = attach_matchup(roster.copy(), season, week)
    # FantasyPros already prices the matchup; only nudge the nflverse estimates.
    # df.get returns a bare str when the column is absent, and str has no .eq
    is_est = df["proj_source"].eq("nflverse-est") if "proj_source" in df.columns \
        else pd.Series(False, index=df.index)
    effective_mult = pd.to_numeric(df.get("mult_fitted"), errors="coerce").fillna(df["mult"])
    adj_factor = (0.80 + 0.20 * effective_mult).where(is_est, 1.0)
    df["proj_adj"] = (pd.to_numeric(df["proj"], errors="coerce") * adj_factor).round(2)
    df = df.sort_values("proj_adj", ascending=False)

    # greedy legal fill
    assigned, used = {}, set()

    def take(pos_ok, n, label):
        picks = df[df["pos"].isin(pos_ok) & ~df.index.isin(used)].head(n)
        used.update(picks.index)
        for i in picks.index:
            assigned[i] = label

    take(["QB"], LINEUP["QB"], "QB")
    take(["RB"], LINEUP["RB"], "RB")
    take(["WR"], LINEUP["WR"], "WR")
    take(["TE"], LINEUP["TE"], "TE")
    take(list(FLEX_ELIGIBLE), LINEUP["W/R"], "FLEX")

    df["lineup"] = df.index.map(lambda i: assigned.get(i, "BENCH"))
    df["start"] = df["lineup"] != "BENCH"

    # close-call: a benched player within 2.0 adj pts of the weakest starter at an eligible slot
    flags = {}
    for i, r in df[~df["start"]].iterrows():
        elig = ["FLEX"] if r["pos"] in FLEX_ELIGIBLE else []
        elig.append(r["pos"])
        starters_same = df[(df["start"]) & (df["lineup"].isin(elig + [r["pos"]]))]
        if not starters_same.empty:
            weakest = starters_same["proj_adj"].min()
            if r["proj_adj"] >= weakest - 2.0:
                flags[i] = f"≈ {starters_same.sort_values('proj_adj').iloc[0]['player']}"
    df["close_call"] = df.index.map(lambda i: flags.get(i, ""))
    return df


if __name__ == "__main__":
    import sys

    import nflreadpy as nfl
    from .projections import blended_week

    s = int(sys.argv[1]) if len(sys.argv) > 1 else int(nfl.get_current_season())
    wk = int(sys.argv[2]) if len(sys.argv) > 2 else int(nfl.get_current_week())

    roster = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parent.parent / "roster.csv", dtype=str).fillna("")
    roster = roster[~roster["pos"].isin(["K", "DEF"])].rename(columns={"name": "player"})
    proj = blended_week(s, wk)
    from .intel import _norm
    roster["norm"] = roster["player"].map(_norm)
    roster = roster.merge(proj[["norm", "proj", "proj_source", "ecr", "start_sit"]], on="norm", how="left")
    roster["proj"] = roster["proj"].fillna(0)

    print("== DEFENSE VS POSITION (easiest WR matchups) ==")
    dvp = defense_vs_position(s)
    print(dvp[dvp["pos"] == "WR"].sort_values("ease_rank").head(6).to_string(), "\n")

    print(f"== OPTIMAL LINEUP  season {s} week {wk} ==")
    out = optimize_lineup(roster, s, wk)
    cols = ["lineup", "player", "pos", "nfl_team", "opp", "ease_rank", "proj", "proj_adj", "proj_source", "start_sit", "close_call"]
    print(out.sort_values(["start", "proj_adj"], ascending=[False, False])[cols].to_string())
