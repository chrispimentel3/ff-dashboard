"""Weekly intelligence: waiver board, buy-low/sell-high, draft value delta, trade finder.

Runs with or without the Yahoo scrape. Without it, current rosters are approximated
from the draft board (plus any adds we can read from the transactions feed).
"""
from __future__ import annotations

import functools
import re

import nflreadpy as nfl
import numpy as np
import pandas as pd

from .config import FLEX_ELIGIBLE, LINEUP, MY_SEAT, SCORING, TEAM_BY_SEAT
from .draft_board import load_draft
from .sources import fantasycalc_values, sleeper_players, sleeper_trending


def _norm(s: object) -> str:
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"[^a-z ]", " ", s).strip()


def _n(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else pd.Series(0.0, index=df.index)


def score_half_ppr(df: pd.DataFrame) -> pd.Series:
    s = SCORING
    ints = df["passing_interceptions"] if "passing_interceptions" in df.columns else df.get("interceptions", 0)
    ints = pd.to_numeric(ints, errors="coerce").fillna(0.0) if hasattr(ints, "fillna") else 0.0
    return (
        s["pass_yd"] * _n(df, "passing_yards") + s["pass_td"] * _n(df, "passing_tds") + s["pass_int"] * ints
        + s["rush_yd"] * _n(df, "rushing_yards") + s["rush_td"] * _n(df, "rushing_tds")
        + s["rec"] * _n(df, "receptions") + s["rec_yd"] * _n(df, "receiving_yards") + s["rec_td"] * _n(df, "receiving_tds")
        + s["fum_lost"] * (_n(df, "rushing_fumbles_lost") + _n(df, "receiving_fumbles_lost") + _n(df, "sack_fumbles_lost"))
        + s["two_pt"] * (_n(df, "passing_2pt_conversions") + _n(df, "rushing_2pt_conversions") + _n(df, "receiving_2pt_conversions"))
        + s["ret_td"] * _n(df, "special_teams_tds")
    )


# ────────────────────────────────────────────────────────── nflverse frames
@functools.lru_cache(maxsize=8)
def _weeks_available(season: int) -> int:
    w = weekly(season)
    return 0 if w.empty else int(w["week"].max())


def form_season(season: int, min_weeks: int = 2) -> int:
    """Season to use for recent-form signals: fall back a year until early-season."""
    return season if _weeks_available(season) >= min_weeks else season - 1


@functools.lru_cache(maxsize=4)
def weekly(season: int) -> pd.DataFrame:
    df = nfl.load_player_stats(seasons=[season]).to_pandas()
    ren, claimed = {}, set(df.columns)
    for a, b in [("player_display_name", "player"), ("player_name", "player"),
                 ("recent_team", "team"), ("position", "pos"), ("player_id", "gsis_id")]:
        if a in df.columns and b not in claimed:
            ren[a] = b
            claimed.add(b)
    df = df.rename(columns=ren)
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().isin(["REG", "POST"])]
    df["half_ppr"] = score_half_ppr(df)
    df["norm"] = df["player"].map(_norm)
    return df


@functools.lru_cache(maxsize=2)
def ff_opportunity(season: int) -> pd.DataFrame:
    try:
        df = nfl.load_ff_opportunity(seasons=[season], stat_type="weekly", model_version="latest").to_pandas()
    except Exception:
        return pd.DataFrame()
    df = df.rename(columns={"player_id": "gsis_id", "full_name": "player"})
    df["norm"] = df["player"].map(_norm)
    return df


@functools.lru_cache(maxsize=2)
def injuries(season: int) -> pd.DataFrame:
    try:
        df = nfl.load_injuries(seasons=[season]).to_pandas()
    except Exception:
        return pd.DataFrame()
    df = df.rename(columns={"full_name": "player", "gsis_id": "gsis_id"})
    df["norm"] = df["player"].map(_norm)
    return df


# ────────────────────────────────────────────────────────── rosters
def rosters_from_draft() -> pd.DataFrame:
    """Approximate current rosters = draft board (until the Yahoo scrape is wired)."""
    d = load_draft()
    return d[["seat", "drafted_by", "player", "pos", "nfl_team"]].rename(columns={"drafted_by": "team"}).assign(
        norm=lambda x: x["player"].map(_norm)
    )


def current_rosters(yahoo_rosters: pd.DataFrame | None = None) -> pd.DataFrame:
    if yahoo_rosters is not None and not yahoo_rosters.empty:
        r = yahoo_rosters.copy()
        r["norm"] = r["player"].map(_norm)
        if "seat" not in r.columns and "team" in r.columns:
            from .config import SEAT_BY_TEAM
            r["seat"] = r["team"].map(SEAT_BY_TEAM)
        return r
    return rosters_from_draft()


# ────────────────────────────────────────────────────────── waiver board
def _recent_form(season: int, weeks: int = 3) -> pd.DataFrame:
    w = weekly(form_season(season))
    if w.empty:
        return pd.DataFrame()
    maxwk = int(w["week"].max())
    lo = max(1, maxwk - weeks + 1)
    recent = w[w["week"].between(lo, maxwk)]
    g = recent.groupby(["norm", "player", "pos"], as_index=False).agg(
        gms=("week", "nunique"),
        pg_recent=("half_ppr", "mean"),
        tgt_pg=("targets", "mean") if "targets" in recent.columns else ("half_ppr", "size"),
        carry_pg=("carries", "mean") if "carries" in recent.columns else ("half_ppr", "size"),
    )
    return g.merge(_team_target_share(recent), on="norm", how="left")


def _team_target_share(recent: pd.DataFrame, key: str = "norm") -> pd.DataFrame:
    """Share of his own offense's targets over the window, and his rank on that team.

    Summed numerator over summed denominator — a mean of weekly shares lets one
    low-volume game dominate.
    """
    if "targets" not in recent.columns or "team" not in recent.columns:
        return pd.DataFrame(columns=[key, "tgt_pct", "tm_rank"])
    r = recent.copy()
    r["targets"] = _n(r, "targets")
    team_tot = r.groupby("team")["targets"].sum().rename("tm_tgt").reset_index()
    p = (
        r.sort_values("week")
        .groupby(key)
        .agg(targets=("targets", "sum"), team=("team", "last"))
        .reset_index()
    )
    p = p[p["targets"] > 0].merge(team_tot, on="team", how="left")
    p["tgt_pct"] = p["targets"] / p["tm_tgt"].replace(0, np.nan)
    p["tm_rank"] = p.groupby("team")["targets"].rank(method="min", ascending=False)
    return p[[key, "tgt_pct", "tm_rank"]]


def waiver_board(season: int, rostered_norms: set[str], top: int = 20) -> pd.DataFrame:
    form = _recent_form(season)
    if form.empty:
        return pd.DataFrame()
    fc = fantasycalc_values()[["norm", "value", "pos_rank", "trend_30d"]]
    tr = sleeper_trending("add", limit=80)[["norm", "add_rank", "sleeper_adds", "injury_status"]]

    b = form.merge(fc, on="norm", how="left").merge(tr, on="norm", how="left")
    b = b[~b["norm"].isin(rostered_norms)]
    b = b[b["pos"].isin(["QB", "RB", "WR", "TE"])]

    # vacated volume: teammates ruled out this week
    inj = injuries(season)
    hurt = set()
    if not inj.empty and "report_status" in inj.columns:
        latest = inj.sort_values("week").groupby("norm").tail(1)
        hurt = set(latest[latest["report_status"].isin(["Out", "Doubtful", "IR"])]["norm"])

    def z(s):
        s = s.fillna(s.median() if s.notna().any() else 0)
        return (s - s.mean()) / (s.std() + 1e-9)

    b["add_score"] = (
        1.4 * z(b["pg_recent"])
        + 0.8 * z(b["value"])
        + 0.7 * z(-b["add_rank"].fillna(b["add_rank"].max() if b["add_rank"].notna().any() else 999))
        + 0.5 * z(b["trend_30d"])
        + 0.4 * z(b["tgt_pg"] + b["carry_pg"])
    )
    b = b.sort_values("add_score", ascending=False).head(top).reset_index(drop=True)

    def why(r):
        bits = []
        if r["pg_recent"] >= 8:
            bits.append(f"{r['pg_recent']:.1f} half-PPR/g last {int(r['gms'])}")
        if pd.notna(r["add_rank"]) and r["add_rank"] <= 40:
            bits.append(f"#{int(r['add_rank'])} adds industry-wide")
        if pd.notna(r["trend_30d"]) and r["trend_30d"] > 150:
            bits.append("value rising")
        if pd.notna(r.get("tgt_pg")) and r["tgt_pg"] >= 6:
            bits.append(f"{r['tgt_pg']:.1f} tgt/g")
        if pd.notna(r.get("tm_rank")) and r["tm_rank"] <= 2 and pd.notna(r.get("tgt_pct")):
            bits.append(f"#{int(r['tm_rank'])} target on {r['tgt_pct']:.0%} share")
        if pd.notna(r.get("carry_pg")) and r["carry_pg"] >= 10:
            bits.append(f"{r['carry_pg']:.1f} carry/g")
        return "; ".join(bits) or "role trending up"

    b["why"] = b.apply(why, axis=1)
    return b[["player", "pos", "pg_recent", "tgt_pg", "tgt_pct", "tm_rank", "carry_pg",
              "value", "add_rank", "trend_30d", "add_score", "why"]]


# ────────────────────────────────────────────────────────── buy low / sell high
def buy_low_sell_high(season: int, through_week: int | None = None) -> pd.DataFrame:
    season = form_season(season)
    w = weekly(season)
    ffo = ff_opportunity(season)
    if w.empty or ffo.empty:
        return pd.DataFrame()
    mw = through_week or int(w["week"].max())
    w = w[w["pos"].isin(["QB", "RB", "WR", "TE"]) & w["norm"].str.len().gt(0)]
    act = w[w["week"] <= mw].groupby("norm", as_index=False).agg(
        player=("player", "first"), pos=("pos", "first"), gms=("week", "nunique"), actual=("half_ppr", "sum")
    )
    # half-PPR expected from ff_opportunity component columns
    s = SCORING
    ffo = ffo.copy()
    ffo["exp_hp"] = (
        s["pass_yd"] * _n(ffo, "pass_yards_gained_exp") + s["pass_td"] * _n(ffo, "pass_touchdown_exp")
        + s["pass_int"] * _n(ffo, "pass_interception_exp")
        + s["rush_yd"] * _n(ffo, "rush_yards_gained_exp") + s["rush_td"] * _n(ffo, "rush_touchdown_exp")
        + s["rec"] * _n(ffo, "receptions_exp") + s["rec_yd"] * _n(ffo, "rec_yards_gained_exp")
        + s["rec_td"] * _n(ffo, "rec_touchdown_exp")
    )
    exp = ffo[ffo["week"] <= mw].groupby("norm", as_index=False)["exp_hp"].sum().rename(columns={"exp_hp": "expected"})
    m = act.merge(exp, on="norm", how="inner")
    m = m[m["gms"] >= 2]
    m["diff"] = m["actual"] - m["expected"]
    m["diff_pg"] = m["diff"] / m["gms"].clip(lower=1)
    m["signal"] = np.where(m["diff_pg"] <= -2.0, "BUY LOW", np.where(m["diff_pg"] >= 2.5, "SELL HIGH", ""))
    return m.sort_values("diff_pg").reset_index(drop=True)


# ────────────────────────────────────────────────────────── draft value delta
def draft_value_delta(season: int) -> pd.DataFrame:
    d = load_draft()
    fc = fantasycalc_values()[["norm", "value", "overall_rank", "trend_30d"]]
    w = weekly(season)
    pts = w.groupby("norm", as_index=False).agg(season_hp=("half_ppr", "sum"), gms=("week", "nunique")) if not w.empty else pd.DataFrame(columns=["norm", "season_hp", "gms"])

    d = d.assign(norm=d["player"].map(_norm)).merge(fc, on="norm", how="left").merge(pts, on="norm", how="left")
    # Self-calibrating baseline: the k-th overall pick "should" hold the k-th-highest
    # current trade value. Delta = how far a player has risen/fallen vs their draft slot.
    vals_sorted = d["value"].fillna(0.0).sort_values(ascending=False).to_numpy()
    d["exp_value_for_slot"] = d["overall"].sub(1).clip(upper=len(vals_sorted) - 1).map(lambda i: vals_sorted[int(i)])
    d["value_delta"] = d["value"].fillna(0.0) - d["exp_value_for_slot"]
    d["mine"] = d["seat"].eq(MY_SEAT)
    return d.sort_values("value_delta", ascending=False)[
        ["overall", "round", "seat", "drafted_by", "player", "pos", "value", "exp_value_for_slot",
         "value_delta", "trend_30d", "season_hp", "gms", "mine"]
    ].reset_index(drop=True)


# ────────────────────────────────────────────────────────── trade finder
def _starter_value(players: pd.DataFrame) -> tuple[float, dict]:
    """Sum FantasyCalc value of the best legal starting lineup; return (total, by-slot picks)."""
    pool = players.dropna(subset=["value"]).sort_values("value", ascending=False)
    used, total, picks = set(), 0.0, {}
    for pos, n in LINEUP.items():
        if pos in ("K", "DEF", "W/R"):
            continue
        got = pool[(pool["pos"] == pos) & (~pool["player"].isin(used))].head(n)
        used |= set(got["player"])
        total += got["value"].sum()
        picks[pos] = list(got["player"])
    flex = pool[(pool["pos"].isin(FLEX_ELIGIBLE)) & (~pool["player"].isin(used))].head(LINEUP["W/R"])
    total += flex["value"].sum()
    picks["W/R"] = list(flex["player"])
    return total, picks


def _positional_strength(ros: pd.DataFrame) -> pd.DataFrame:
    """Per team-position: summed value of that team's best `LINEUP` starters at the position."""
    rows = []
    for seat, g in ros.groupby("seat"):
        for pos in ("QB", "RB", "WR", "TE"):
            need = LINEUP.get(pos, 1) + (1 if pos in FLEX_ELIGIBLE else 0)
            top = g[g["pos"] == pos].nlargest(need, "value")["value"].sum()
            rows.append(dict(seat=seat, pos=pos, starter_value=top, depth=(g["pos"] == pos).sum()))
    return pd.DataFrame(rows)


def trade_finder(season: int, yahoo_rosters: pd.DataFrame | None = None, max_ideas: int = 15) -> pd.DataFrame:
    """Need + value + momentum driven 1-for-1 ideas. Approximate until real rosters are scraped."""
    ros = current_rosters(yahoo_rosters)
    fc = fantasycalc_values()[["norm", "value", "pos_rank", "trend_30d"]]
    ros = ros.merge(fc, on="norm", how="left")
    ros = ros.dropna(subset=["value"])

    strength = _positional_strength(ros)
    lg_avg = strength.groupby("pos")["starter_value"].mean()
    mine_str = strength[strength["seat"] == MY_SEAT].set_index("pos")["starter_value"]
    my_need = (mine_str - lg_avg).sort_values()           # most negative = biggest need
    my_surplus_pos = list(my_need[my_need > 0].index)     # positions I can trade from
    need_pos = list(my_need[my_need < 0].index) or list(my_need.index[:2])

    mine = ros[ros["seat"] == MY_SEAT].sort_values("value", ascending=False)
    ideas = []
    for seat, name in TEAM_BY_SEAT.items():
        if seat == MY_SEAT:
            continue
        theirs = ros[ros["seat"] == seat]
        their_str = strength[strength["seat"] == seat].set_index("pos")["starter_value"]
        their_need = (their_str - lg_avg).sort_values()
        their_need_pos = list(their_need[their_need < 0].index)

        # I target their startable surplus at a position I need
        for _, get in theirs[theirs["pos"].isin(need_pos)].nlargest(6, "value").iterrows():
            # what I give: my best expendable at a position of surplus (or a position they need)
            give_pool = mine[mine["pos"].isin(set(my_surplus_pos) | set(their_need_pos))]
            give_pool = give_pool[give_pool["player"] != get["player"]]
            for _, give in give_pool.iterrows():
                gv, tv = give["value"], get["value"]
                if max(gv, tv) == 0 or abs(gv - tv) > 0.18 * max(gv, tv):
                    continue
                ideas.append(dict(
                    partner=name,
                    give=give["player"], give_pos=give["pos"], give_val=int(gv), give_trend=give.get("trend_30d"),
                    get=get["player"], get_pos=get["pos"], get_val=int(tv), get_trend=get.get("trend_30d"),
                    addresses=f"my {get['pos']} need",
                    they_need=", ".join(their_need_pos[:2]) or "—",
                    fairness=round(1 - abs(gv - tv) / max(gv, tv), 2),
                ))
    out = pd.DataFrame(ideas)
    if out.empty:
        return out
    out["edge"] = out["get_val"] - out["give_val"] + 0.15 * (out["get_trend"].fillna(0) - out["give_trend"].fillna(0))
    return (out.sort_values(["fairness", "edge"], ascending=False)
            .drop_duplicates(["get", "give"]).head(max_ideas).reset_index(drop=True))


def roster_strength(season: int, rosters: pd.DataFrame) -> pd.DataFrame:
    """Each team's expected weekly points from its best legal lineup.

    Strength independent of record: two 1-0 teams are not the same team. Positions
    come from the stats join rather than the roster page, which often omits them.
    """
    # Full-season per-game, not _recent_form: in week 1 the 3-week window collapses to
    # the tail of last season and matches barely a third of each roster.
    w = weekly(form_season(season))
    if w.empty or rosters is None or rosters.empty:
        return pd.DataFrame()
    form = (
        w[w["pos"].isin(["QB", "RB", "WR", "TE"])]
        .groupby(["norm", "pos"], as_index=False)
        .agg(pg_recent=("half_ppr", "mean"), gms=("week", "nunique"))
    )
    form = form.sort_values("gms", ascending=False).drop_duplicates("norm")
    r = rosters[["norm", "team"]].drop_duplicates().merge(form, on="norm", how="left")
    r["pg_recent"] = pd.to_numeric(r["pg_recent"], errors="coerce").fillna(0.0)

    rows = []
    for team, grp in r.groupby("team"):
        used: set = set()
        pts = 0.0
        for slot, n in LINEUP.items():
            if slot in ("K", "DEF", "W/R"):
                continue
            pool = grp[(grp["pos"] == slot) & (~grp.index.isin(used))].nlargest(n, "pg_recent")
            pts += float(pool["pg_recent"].sum())
            used |= set(pool.index)
        flex = grp[grp["pos"].isin(FLEX_ELIGIBLE) & (~grp.index.isin(used))].nlargest(
            LINEUP.get("W/R", 1), "pg_recent")
        pts += float(flex["pg_recent"].sum())
        used |= set(flex.index)
        bench = grp[~grp.index.isin(used)].nlargest(3, "pg_recent")
        rows.append(dict(
            team=team,
            starters_pg=round(pts, 1),
            bench_pg=round(float(bench["pg_recent"].sum()), 1),
            matched=int(grp["pg_recent"].gt(0).sum()),
            size=len(grp),
        ))
    out = pd.DataFrame(rows).sort_values("starters_pg", ascending=False).reset_index(drop=True)
    out.insert(0, "power_rank", range(1, len(out) + 1))
    return out


if __name__ == "__main__":
    import sys
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    print(f"season={S}  form_season={form_season(S)}  (weeks avail: {_weeks_available(S)})\n")
    dr = rosters_from_draft()
    rn = set(dr["norm"])
    print("== WAIVER BOARD ==")
    print(waiver_board(S, rn, top=12).to_string(), "\n")
    print("== DRAFT VALUE DELTA (top/bottom 8) ==")
    dv = draft_value_delta(S)
    print(pd.concat([dv.head(8), dv.tail(8)]).to_string(), "\n")
    print("== TRADE IDEAS (draft-roster approximation) ==")
    print(trade_finder(S).to_string(), "\n")
    print("== BUY LOW / SELL HIGH ==")
    bl = buy_low_sell_high(S)
    if not bl.empty:
        print(pd.concat([bl.head(8), bl.tail(6)]).to_string())
