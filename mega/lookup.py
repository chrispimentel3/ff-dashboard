"""Player lookup: one player's bio, season line with positional ranks, and game log.

Pure pandas over frames the app already loads (weekly player stats, ff_opportunity,
snap counts, schedule, nflverse rosters), so it can be tested without Streamlit.

Every season rate is summed numerator over summed denominator, never a mean of weekly
ratios — the same rule as target share and WOPR elsewhere in the app. Team totals use
only the weeks the player played (games-played denominator, WOPR handoff D2).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

POS = ("QB", "RB", "WR", "TE")


def _n(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else pd.Series(0.0, index=df.index)


def _reg(w: pd.DataFrame) -> pd.DataFrame:
    if "season_type" in w.columns:
        w = w[w["season_type"].astype(str).str.upper() == "REG"]
    return w


# ---------------------------------------------------------------- search index
def player_index(stats: dict[int, pd.DataFrame], rosters: pd.DataFrame) -> pd.DataFrame:
    """Everyone searchable: skill players on a current NFL roster, plus anyone with stats in
    the seasons given (so a player who has left the league can still be looked up).
    Ordered by fantasy points, newest season first, so typing surfaces relevant names."""
    rows = []
    if rosters is not None and not rosters.empty:
        r = rosters[rosters["position"].isin(POS)]
        rows.append(pd.DataFrame({"gsis_id": r["gsis_id"], "name": r["full_name"],
                                  "pos": r["position"], "team": r["team"], "status": r["status"]}))
    pts = []
    for season, w in sorted(stats.items(), reverse=True):
        w = _reg(w)
        if w.empty:
            continue
        w = w[w["pos"].isin(POS)]
        g = w.sort_values("week").groupby("gsis_id").agg(
            name=("player", "last"), pos=("pos", "last"), team=("team", "last"), pts=("half_ppr", "sum"))
        rows.append(g.reset_index().assign(status=""))
        pts.append(g["pts"].rename(f"pts_{season}"))
    idx = pd.concat(rows, ignore_index=True).dropna(subset=["gsis_id"])
    idx = idx.drop_duplicates("gsis_id", keep="first")        # roster row wins: current team
    # Found only in old box scores = on no current roster. His last team is history; say so
    # rather than label him with it (Tyreek Hill read "WR · MIA" with no 2026 team).
    if rosters is not None and not rosters.empty:
        gone = ~idx["gsis_id"].isin(set(rosters["gsis_id"]))
        idx["last_team"] = idx["team"].where(gone, "")
        idx.loc[gone, "team"] = ""
    for p in pts:
        idx = idx.merge(p, left_on="gsis_id", right_index=True, how="left")
    sort_cols = [c for c in idx.columns if c.startswith("pts_")]
    idx = idx.sort_values(sort_cols, ascending=False, na_position="last") if sort_cols else idx
    idx["team"] = idx["team"].fillna("").astype(str)
    idx["label"] = idx["name"] + " · " + idx["pos"] + (" · " + idx["team"]).where(idx["team"] != "", " · no NFL team")
    return idx.reset_index(drop=True)


# ---------------------------------------------------------------- bio
def bio(gsis_id: str, rosters: pd.DataFrame) -> dict:
    if rosters is None or rosters.empty:
        return {}
    hit = rosters[rosters["gsis_id"] == gsis_id]
    if hit.empty:
        return {}
    r = hit.iloc[-1]
    out = {k: r.get(k) for k in ("full_name", "position", "team", "jersey_number", "status", "height",
                                 "weight", "college", "years_exp", "headshot_url", "draft_club",
                                 "draft_number", "entry_year", "depth_chart_position")}
    bd = pd.to_datetime(r.get("birth_date"), errors="coerce")
    if pd.notna(bd):
        today = dt.date.today()
        out["age"] = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    return out


STATUS_WORDS = {"ACT": "Active", "RES": "Reserve / IR", "INA": "Inactive", "DEV": "Practice squad",
                "CUT": "Released", "RET": "Retired", "EXE": "Exempt", "SUS": "Suspended",
                "TRC": "Reserve / COVID", "TRD": "Traded", "UFA": "Free agent (NFL)"}


# ---------------------------------------------------------------- season table
def season_table(w: pd.DataFrame, ffo: pd.DataFrame | None, snaps: pd.DataFrame | None,
                 crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per skill player for the season: counting stats, shares and rates.

    `crosswalk` supplies gsis_id -> pfr_id for joining snap counts.
    """
    w = _reg(w)
    if w.empty:
        return pd.DataFrame()
    w = w.copy()
    for c in ("targets", "receiving_air_yards", "carries", "attempts"):
        w[c] = _n(w, c)
    tt = (w.groupby(["team", "week"])[["targets", "receiving_air_yards", "carries"]]
          .sum().add_prefix("tm_").reset_index())
    s = w[w["pos"].isin(POS)].merge(tt, on=["team", "week"], how="left")

    agg = {
        "games": ("week", "nunique"), "pts": ("half_ppr", "sum"),
        "targets": ("targets", "sum"), "tm_targets": ("tm_targets", "sum"),
        "air": ("receiving_air_yards", "sum"), "tm_air": ("tm_receiving_air_yards", "sum"),
        "carries": ("carries", "sum"), "tm_carries": ("tm_carries", "sum"),
    }
    for c in ("receptions", "receiving_yards", "receiving_tds", "receiving_yards_after_catch",
              "receiving_first_downs", "receiving_epa", "rushing_yards", "rushing_tds",
              "rushing_first_downs", "rushing_epa", "completions", "attempts", "passing_yards",
              "passing_tds", "passing_interceptions", "sacks_suffered", "passing_epa",
              "passing_air_yards", "passing_first_downs"):
        if c in s.columns:
            s[c] = _n(s, c)
            agg[c] = (c, "sum")
    if "passing_cpoe" in s.columns:
        # CPOE is per-attempt; weight each game by its attempts rather than averaging games
        s["_cpoe_w"] = pd.to_numeric(s["passing_cpoe"], errors="coerce").fillna(0) * s["attempts"]
        agg["_cpoe_w"] = ("_cpoe_w", "sum")
    t = s.sort_values("week").groupby(["gsis_id"]).agg(
        name=("player", "last"), pos=("pos", "last"), team=("team", "last"), **agg).reset_index()

    if ffo is not None and not ffo.empty and "half_ppr_exp" in ffo.columns:
        f = ffo[ffo["gsis_id"].isin(t["gsis_id"])]
        t = t.merge(f.groupby("gsis_id")["half_ppr_exp"].sum().rename("xfp"), on="gsis_id", how="left")
    else:
        t["xfp"] = np.nan

    if snaps is not None and not snaps.empty and crosswalk is not None and "pfr_id" in snaps.columns:
        sn = snaps.copy()
        if "game_type" in sn.columns:
            sn = sn[sn["game_type"].astype(str).str.upper() == "REG"]
        sn["offense_snaps"] = _n(sn, "offense_snaps")
        pct = pd.to_numeric(sn.get("offense_pct"), errors="coerce")
        sn["tm_snaps"] = sn["offense_snaps"] / pct.where(pct > 0)
        sp = sn.groupby("pfr_id")[["offense_snaps", "tm_snaps"]].sum()
        sp["snap_pct"] = sp["offense_snaps"] / sp["tm_snaps"].replace(0, np.nan)
        xw = crosswalk.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("gsis_id")[["gsis_id", "pfr_id"]]
        t = t.merge(xw, on="gsis_id", how="left").merge(sp[["snap_pct"]], left_on="pfr_id",
                                                         right_index=True, how="left")
    else:
        t["snap_pct"] = np.nan

    g = t["games"].replace(0, np.nan)
    t["pts_pg"] = t["pts"] / g
    t["xfp_pg"] = t["xfp"] / g
    t["vs_exp_pg"] = (t["pts"] - t["xfp"]) / g
    t["tgt_share"] = t["targets"] / t["tm_targets"].replace(0, np.nan)
    t["ay_share"] = t["air"] / t["tm_air"].replace(0, np.nan)
    t["wopr"] = 1.5 * t["tgt_share"] + 0.7 * t["ay_share"]
    t["rush_share"] = t["carries"] / t["tm_carries"].replace(0, np.nan)
    t["tgt_pg"] = t["targets"] / g
    t["car_pg"] = t["carries"] / g
    t["adot"] = t["air"] / t["targets"].replace(0, np.nan)
    rec = t.get("receptions", pd.Series(0, index=t.index))
    t["catch_rate"] = rec / t["targets"].replace(0, np.nan)
    t["ypr"] = t.get("receiving_yards", 0) / rec.replace(0, np.nan)
    t["yac_pr"] = t.get("receiving_yards_after_catch", 0) / rec.replace(0, np.nan)
    t["ypc"] = t.get("rushing_yards", 0) / t["carries"].replace(0, np.nan)
    t["rec_epa_tgt"] = t.get("receiving_epa", 0) / t["targets"].replace(0, np.nan)
    t["rush_epa_car"] = t.get("rushing_epa", 0) / t["carries"].replace(0, np.nan)
    att = t.get("attempts", pd.Series(0, index=t.index))
    drop = att + t.get("sacks_suffered", 0)
    t["cmp_pct"] = t.get("completions", 0) / att.replace(0, np.nan)
    t["ypa"] = t.get("passing_yards", 0) / att.replace(0, np.nan)
    t["pass_epa_db"] = t.get("passing_epa", 0) / drop.replace(0, np.nan)
    t["cpoe"] = t["_cpoe_w"] / att.replace(0, np.nan) if "_cpoe_w" in t.columns else np.nan
    t["att_pg"] = att / g
    return t


# What the season card shows, per position: (column, label, format, what it means)
CARD = {
    "QB": [("pts_pg", "Pts/game", "{:.1f}", "Half-PPR fantasy points per game."),
           ("xfp_pg", "Expected pts/g", "{:.1f}", "What his volume should score."),
           ("att_pg", "Pass att/g", "{:.1f}", "Dropback volume."),
           ("cmp_pct", "Completion %", "{:.1%}", ""),
           ("cpoe", "CPOE", "{:+.1f}", "Completion % over expected, given throw difficulty. Plus is accurate."),
           ("ypa", "Yds/att", "{:.1f}", ""),
           ("pass_epa_db", "EPA/dropback", "{:+.2f}", "Points added per dropback, sacks included. Above +0.10 is good."),
           ("car_pg", "Carries/g", "{:.1f}", "Rushing volume — the fantasy QB cheat code."),
           ("rushing_yards", "Rush yds", "{:.0f}", ""),
           ("passing_interceptions", "INT", "{:.0f}", "")],
    "RB": [("pts_pg", "Pts/game", "{:.1f}", "Half-PPR fantasy points per game."),
           ("xfp_pg", "Expected pts/g", "{:.1f}", "What his touches should score."),
           ("snap_pct", "Snap share", "{:.0%}", "Share of his offense's plays he was on the field for."),
           ("rush_share", "Carry share", "{:.0%}", "His share of his team's carries."),
           ("car_pg", "Carries/g", "{:.1f}", ""),
           ("ypc", "Yds/carry", "{:.1f}", ""),
           ("rush_epa_car", "Rush EPA/carry", "{:+.2f}", "Points added per carry. Above 0 is rare for backs."),
           ("tgt_share", "Target share", "{:.1%}", "Receiving role — worth more than carries in half-PPR."),
           ("tgt_pg", "Targets/g", "{:.1f}", ""),
           ("rushing_tds", "Rush TD", "{:.0f}", "")],
    "WR": [("pts_pg", "Pts/game", "{:.1f}", "Half-PPR fantasy points per game."),
           ("xfp_pg", "Expected pts/g", "{:.1f}", "What his targets should score."),
           ("snap_pct", "Snap share", "{:.0%}", "Share of his offense's plays he was on the field for."),
           ("tgt_share", "Target share", "{:.1%}", "25%+ is a No. 1 receiver's role."),
           ("ay_share", "Air-yards share", "{:.1%}", "Share of the team's downfield targets."),
           ("wopr", "WOPR", "{:.2f}", "1.5 × target share + 0.7 × air-yards share. About 0.5+ is a starter's role."),
           ("adot", "aDOT", "{:.1f}", "Average depth of target, in yards."),
           ("catch_rate", "Catch rate", "{:.0%}", ""),
           ("yac_pr", "YAC/rec", "{:.1f}", "Yards after catch per reception."),
           ("rec_epa_tgt", "EPA/target", "{:+.2f}", "Points added per target.")],
}
CARD["TE"] = CARD["WR"]


def ranks(table: pd.DataFrame, gsis_id: str, pos: str) -> dict[str, str]:
    """'#5 of 64' for each card stat, among players at the position with at least half
    the games of the most-used player — so a one-game fill-in doesn't top a rate."""
    pool = table[table["pos"] == pos]
    if pool.empty or gsis_id not in set(pool["gsis_id"]):
        return {}
    pool = pool[pool["games"] >= max(1, int(np.ceil(pool["games"].max() / 2)))]
    out = {}
    for col, *_ in CARD.get(pos, []):
        if col not in pool.columns or gsis_id not in set(pool["gsis_id"]):
            continue
        v = pd.to_numeric(pool[col], errors="coerce")
        if v.notna().sum() < 3:
            continue
        # lower is better only for interceptions
        r = v.rank(ascending=(col == "passing_interceptions"), method="min")
        me = r[pool["gsis_id"] == gsis_id]
        if not me.empty and pd.notna(me.iloc[0]):
            out[col] = f"#{int(me.iloc[0])} of {int(v.notna().sum())}"
    return out


# ---------------------------------------------------------------- game log
def game_log(w: pd.DataFrame, gsis_id: str, ffo: pd.DataFrame | None, snaps: pd.DataFrame | None,
             sched: pd.DataFrame | None, pfr_id: str | None) -> pd.DataFrame:
    w = _reg(w)
    team_week = w.groupby(["team", "week"])[["targets"]].sum().rename(columns={"targets": "tm_tgt"})
    g = w[w["gsis_id"] == gsis_id].sort_values("week").copy()
    if g.empty:
        return g
    g = g.merge(team_week, left_on=["team", "week"], right_index=True, how="left")
    g["tgt_pct"] = _n(g, "targets") / g["tm_tgt"].replace(0, np.nan)
    if ffo is not None and not ffo.empty:
        e = ffo[ffo["gsis_id"] == gsis_id].groupby("week")["half_ppr_exp"].sum()
        g["xfp"] = g["week"].map(e)
    else:
        g["xfp"] = np.nan
    g["vs_exp"] = g["half_ppr"] - g["xfp"]
    if snaps is not None and not snaps.empty and pfr_id:
        sp = snaps[snaps["pfr_id"] == pfr_id].groupby("week")["offense_pct"].max()
        g["snap_pct"] = g["week"].map(sp)
    else:
        g["snap_pct"] = np.nan
    if sched is not None and not sched.empty:
        home = {(r.week, r.home_team) for r in sched.itertuples()}
        g["game"] = [("vs " if (wk, tm) in home else "@ ") + str(op)
                     for wk, tm, op in zip(g["week"], g["team"], g["opponent_team"])]
    else:
        g["game"] = g["opponent_team"]
    return g


LOG_COLS = {
    "QB": ["week", "game", "snap_pct", "completions", "attempts", "passing_yards", "passing_tds",
           "passing_interceptions", "sacks_suffered", "carries", "rushing_yards", "rushing_tds",
           "half_ppr", "xfp", "vs_exp"],
    "RB": ["week", "game", "snap_pct", "carries", "rushing_yards", "rushing_tds", "targets", "tgt_pct",
           "receptions", "receiving_yards", "receiving_tds", "half_ppr", "xfp", "vs_exp"],
    "WR": ["week", "game", "snap_pct", "targets", "tgt_pct", "receptions", "receiving_yards",
           "receiving_tds", "receiving_air_yards", "receiving_yards_after_catch", "half_ppr", "xfp", "vs_exp"],
}
LOG_COLS["TE"] = LOG_COLS["WR"]
