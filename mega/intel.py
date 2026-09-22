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

from .config import FLEX_ELIGIBLE, LINEUP, MY_SEAT, MY_TEAM, SCORING, TEAM_BY_SEAT
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


# The rolling window is 3 games. Below that the current season cannot fill it, so
# every "recent form" number would be a one- or two-game sample dressed up as a
# trend. Fall back to last season until week 3, then switch over automatically.
MIN_WEEKS_CURRENT = 3


def form_season(season: int, min_weeks: int = MIN_WEEKS_CURRENT) -> int:
    """Season to use for recent-form signals: fall back a year until early-season."""
    return season if _weeks_available(season) >= min_weeks else season - 1


def season_basis(season: int, min_weeks: int = MIN_WEEKS_CURRENT) -> dict:
    """Which season the form signals actually read, plus a label for the UI.

    Every tab that talks about "recent form" should say which season that form is
    from — in week 1 it is last season's tail, and a number with no provenance is
    worse than no number.
    """
    have = _weeks_available(season)
    used = season if have >= min_weeks else season - 1
    if used == season:
        return dict(season=used, current=True, weeks=have,
                    label=f"{season} form · {have} wk played")
    return dict(season=used, current=False, weeks=have,
                label=f"{used} form — {season} has only {have} wk, needs {min_weeks}")


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
    # Half-PPR expected points, scored here once. projections.nflverse_estimate looked for
    # this column and never found it, so its 25% xFP weight silently fell back to points.
    s = SCORING
    df["half_ppr_exp"] = (
        s["pass_yd"] * _n(df, "pass_yards_gained_exp") + s["pass_td"] * _n(df, "pass_touchdown_exp")
        + s["pass_int"] * _n(df, "pass_interception_exp")
        + s["rush_yd"] * _n(df, "rush_yards_gained_exp") + s["rush_td"] * _n(df, "rush_touchdown_exp")
        + s["rec"] * _n(df, "receptions_exp") + s["rec_yd"] * _n(df, "rec_yards_gained_exp")
        + s["rec_td"] * _n(df, "rec_touchdown_exp")
    )
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
    """League-wide rosters, resolved to nflverse ids.

    The scrape leaves `pos` and `nfl_team` blank on roughly two-thirds of rows, and
    _positional_strength() groups by `pos` — so without the backfill every team's
    positional strength was computed from the third of its roster that happened to
    carry a position. The match report rides along on .attrs.
    """
    if yahoo_rosters is not None and not yahoo_rosters.empty:
        from .ids import resolve

        r, rep = resolve(yahoo_rosters.copy(), name_col="player")
        if "team" in r.columns:
            from .config import SEAT_BY_TEAM
            # Fill blanks, not just a missing column: the scrape stamps seats with the name
            # map as it stood at pull time, so a renamed team stays seatless until re-pulled.
            seat = pd.to_numeric(r["seat"], errors="coerce") if "seat" in r.columns else pd.Series(float("nan"), index=r.index)
            r["seat"] = seat.fillna(r["team"].map(SEAT_BY_TEAM))
        r.attrs["match"] = rep
        return r
    return rosters_from_draft()


# ────────────────────────────────────────────────────────── waiver board
def _recent_form(season: int, weeks: int = 3) -> pd.DataFrame:
    """Rolling form over the last `weeks` **regular-season** weeks.

    weekly() keeps the postseason on purpose, and leaving it in here was silently fatal: a
    2025 season runs to week 22, so the last three weeks were the conference championships
    and the Super Bowl. Form was therefore measured on the sixty-odd players still playing
    in late January, which is how a waiver board came to rank Bills and Seahawks starters
    above everyone else in the league.
    """
    w = weekly(form_season(season))
    if w.empty:
        return pd.DataFrame()
    if "season_type" in w.columns:
        w = w[w["season_type"].astype(str).str.upper() == "REG"]
    if w.empty:
        return pd.DataFrame()
    maxwk = int(w["week"].max())
    lo = max(1, maxwk - weeks + 1)
    recent = w[w["week"].between(lo, maxwk)]
    recent = recent.sort_values("week")
    agg = dict(
        gms=("week", "nunique"),
        pg_recent=("half_ppr", "mean"),
        tgt_pg=("targets", "mean") if "targets" in recent.columns else ("half_ppr", "size"),
        carry_pg=("carries", "mean") if "carries" in recent.columns else ("half_ppr", "size"),
    )
    if "team" in recent.columns:
        agg["team"] = ("team", "last")   # his current offence, for the vacated-volume signal
    g = recent.groupby(["norm", "player", "pos"], as_index=False).agg(**agg)
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


def waiver_signals(season: int) -> pd.DataFrame:
    """Talent-and-trend score for every skill player, with no knowledge of any roster.

    This is the old waiver board's scoring, lifted out so two callers can share it. On its
    own it answers "who is playing well and who is the industry buying", which is the right
    question for finding a breakout and the wrong one for deciding whether *you* should add
    him — that part now lives in mega.needs, which knows your lineup.

    Scored over the whole player population rather than whatever subset happens to be free
    this week, so a score means the same thing from one week to the next.
    """
    form = _recent_form(season)
    if form.empty:
        return pd.DataFrame()
    fc = fantasycalc_values()[["norm", "value", "pos_rank", "trend_30d"]]
    tr = sleeper_trending("add", limit=80)[["norm", "add_rank", "sleeper_adds", "injury_status"]]

    b = form.merge(fc, on="norm", how="left").merge(tr, on="norm", how="left")
    b = b[b["pos"].isin(["QB", "RB", "WR", "TE"])]
    from .status import out_for_season

    b = b[~b["norm"].isin(out_for_season())]   # data/player_status.csv

    b["vacated"] = _vacated_volume(season, b)

    def z(s):
        s = s.fillna(s.median() if s.notna().any() else 0)
        return (s - s.mean()) / (s.std() + 1e-9)

    b["add_score"] = (
        1.4 * z(b["pg_recent"])
        + 0.8 * z(b["value"])
        + 0.7 * z(-b["add_rank"].fillna(b["add_rank"].max() if b["add_rank"].notna().any() else 999))
        + 0.5 * z(b["trend_30d"])
        + 0.4 * z(b["tgt_pg"] + b["carry_pg"])
        + 0.6 * z(b["vacated"])
    )

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
        if r.get("vacated", 0) >= 1:
            bits.append(f"{r['vacated']:.0f} tgt/g vacated by an injured teammate")
        return "; ".join(bits) or "role trending up"

    b["upside"] = b.apply(why, axis=1)
    return b[["norm", "player", "pos", "pg_recent", "gms", "tgt_pg", "tgt_pct", "tm_rank",
              "carry_pg", "value", "add_rank", "trend_30d", "vacated", "add_score", "upside"]]


def _vacated_volume(season: int, b: pd.DataFrame) -> pd.Series:
    """Targets per game belonging to teammates who are Out/Doubtful/IR.

    The old board computed this set and then never used it, so the comment promising
    "vacated volume" was the only part of the feature that shipped. Volume is the thing
    that actually transfers when someone goes down, so it earns a real weight here.
    """
    zero = pd.Series(0.0, index=b.index)
    inj = injuries(season)
    if inj.empty or "report_status" not in inj.columns or "team" not in b.columns:
        return zero
    latest = inj.sort_values("week").groupby("norm").tail(1)
    hurt = set(latest[latest["report_status"].isin(["Out", "Doubtful", "IR"])]["norm"])
    if not hurt:
        return zero
    tg = b[["norm", "team", "tgt_pg"]].copy()
    tg["tgt_pg"] = pd.to_numeric(tg["tgt_pg"], errors="coerce").fillna(0)
    freed = tg[tg["norm"].isin(hurt)].groupby("team")["tgt_pg"].sum()
    out = b["team"].map(freed).fillna(0.0)
    # Only pass-catchers inherit targets. A quarterback whose WR1 is out has lost a weapon,
    # not gained volume, so crediting him here would have the sign exactly backwards.
    out = out.where(b["pos"].isin(["WR", "TE", "RB"]), 0.0)
    return out.where(~b["norm"].isin(hurt), 0.0)


def waiver_board(season: int, rostered_norms: set[str], top: int = 20) -> pd.DataFrame:
    """Roster-blind board: who is available and playing well, in league-wide terms.

    Kept for the league-wide view and for mega/intel.py's own __main__. What it cannot tell
    you is whether an add helps *your* team — `mega.needs.board` answers that, and it is
    what the dashboard and the Tuesday digest use.
    """
    b = waiver_signals(season)
    if b.empty:
        return b
    b = b[~b["norm"].isin(rostered_norms)]
    b = b.sort_values("add_score", ascending=False).head(top).reset_index(drop=True)
    return b.rename(columns={"upside": "why"})


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
    exp = (ffo[ffo["week"] <= mw].groupby("norm", as_index=False)["half_ppr_exp"].sum()
           .rename(columns={"half_ppr_exp": "expected"}))
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
    """Per team-position: summed value of that team's best `LINEUP` starters at the position.

    Keyed by team name as well as seat. Seat comes from a hard-coded map that goes
    stale the moment a manager renames their team, and a NaN seat silently drops that
    manager out of every comparison here.
    """
    rows = []
    for (seat, team), g in ros.groupby(["seat", "team"], dropna=False):
        for pos in ("QB", "RB", "WR", "TE"):
            need = LINEUP.get(pos, 1) + (1 if pos in FLEX_ELIGIBLE else 0)
            top = g[g["pos"] == pos].nlargest(need, "value")["value"].sum()
            rows.append(dict(seat=seat, team=team, pos=pos, starter_value=top,
                             depth=(g["pos"] == pos).sum()))
    return pd.DataFrame(rows)


def trade_finder(season: int, yahoo_rosters: pd.DataFrame | None = None, max_ideas: int = 15) -> pd.DataFrame:
    """Need + value + momentum driven 1-for-1 ideas. Approximate until real rosters are scraped."""
    ros = current_rosters(yahoo_rosters)
    from .status import out_for_season
    ros = ros[~ros["norm"].isin(out_for_season())]
    fc = fantasycalc_values()[["norm", "value", "pos_rank", "trend_30d"]]
    ros = ros.merge(fc, on="norm", how="left")
    ros = ros.dropna(subset=["value"])

    strength = _positional_strength(ros)
    lg_avg = strength.groupby("pos")["starter_value"].mean()
    mine_str = strength[strength["team"] == MY_TEAM].set_index("pos")["starter_value"]
    my_need = (mine_str - lg_avg).sort_values()           # most negative = biggest need
    my_surplus_pos = list(my_need[my_need > 0].index)     # positions I can trade from
    need_pos = list(my_need[my_need < 0].index) or list(my_need.index[:2])

    mine = ros[ros["seat"] == MY_SEAT].sort_values("value", ascending=False)
    ideas = []
    # Iterate the teams actually present, not config.TEAM_BY_SEAT — a manager who
    # renames their team loses its seat mapping and would otherwise vanish from the
    # finder entirely, with no error and no empty section to notice.
    for seat, name in ros[["seat", "team"]].drop_duplicates().itertuples(index=False):
        if seat == MY_SEAT or str(name) == MY_TEAM:
            continue
        theirs = ros[ros["team"] == name]
        their_str = strength[strength["team"] == name].set_index("pos")["starter_value"]
        their_need = (their_str - lg_avg).sort_values()
        their_need_pos = list(their_need[their_need < 0].index)

        # I target their startable surplus at a position I need
        for _, get in theirs[theirs["pos"].isin(need_pos)].nlargest(6, "value").iterrows():
            # what I give: my best expendable at a position of surplus (or a position they need)
            # A partner needing QB does not make my own QB expendable — without this
            # the finder happily offers my starter to "fill my QB need".
            give_positions = (set(my_surplus_pos) | set(their_need_pos)) - set(need_pos)
            give_pool = mine[mine["pos"].isin(give_positions)]
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


def _valued_rosters(yahoo_rosters: pd.DataFrame | None) -> pd.DataFrame:
    """League rosters with a trade value on each row, season-enders dropped."""
    from .status import out_for_season

    ros = current_rosters(yahoo_rosters)
    ros = ros[~ros["norm"].isin(out_for_season())]
    fc = fantasycalc_values()[["norm", "value", "pos_rank", "trend_30d"]]
    return ros.merge(fc, on="norm", how="left").dropna(subset=["value"])


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
    from .status import out_for_season
    live = rosters[~rosters["norm"].isin(out_for_season())]
    r = live[["norm", "team"]].drop_duplicates().merge(form, on="norm", how="left")
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
