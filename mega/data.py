"""Pure, Streamlit-free data loaders — nflverse tables, roster resolution, ownership.

Extracted from app.py so a caller with no active Streamlit run context (the Ask API
service in `service/`) can build the same player-week index and ownership picture that
the live app uses. app.py re-wraps every loader here with `st.cache_data`; nothing in
this module changes what it returns — this is a relocation, not a rewrite.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import nflreadpy as nfl
except ImportError:  # pragma: no cover
    nfl = None

from . import ids as player_ids

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ROSTER_CSV = ROOT / "roster.csv"
SEASON_DEFAULT = 2025

# Yahoo default half-PPR scoring. Edit if your league differs.
SCORING = dict(
    pass_yd=0.04, pass_td=4, pass_int=-1,
    rush_yd=0.10, rush_td=6,
    rec=0.5, rec_yd=0.10, rec_td=6,
    fum_lost=-2, two_pt=2, ret_td=6,
)


def to_pandas(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    try:
        return df.to_pandas()
    except Exception:
        return pd.DataFrame(df)


def first_col(df: pd.DataFrame, *cands: str) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def numc(df: pd.DataFrame, *cols: str) -> pd.Series:
    """First present column as numeric, else zeros."""
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def score_actual(df: pd.DataFrame) -> pd.Series:
    s = SCORING
    return (
        s["pass_yd"] * num(df, "passing_yards")
        + s["pass_td"] * num(df, "passing_tds")
        + s["pass_int"] * numc(df, "passing_interceptions", "interceptions")
        + s["rush_yd"] * num(df, "rushing_yards")
        + s["rush_td"] * num(df, "rushing_tds")
        + s["rec"] * num(df, "receptions")
        + s["rec_yd"] * num(df, "receiving_yards")
        + s["rec_td"] * num(df, "receiving_tds")
        + s["fum_lost"] * (num(df, "rushing_fumbles_lost") + num(df, "receiving_fumbles_lost") + num(df, "sack_fumbles_lost"))
        + s["two_pt"] * (num(df, "passing_2pt_conversions") + num(df, "rushing_2pt_conversions") + num(df, "receiving_2pt_conversions"))
        + s["ret_td"] * num(df, "special_teams_tds")
    )


def score_expected(df: pd.DataFrame) -> pd.Series:
    """Half-PPR expected points from ff_opportunity component _exp columns."""
    s = SCORING
    rec_exp = num(df, "receptions_exp") if "receptions_exp" in df.columns else num(df, "rec_attempt_exp")
    pts = (
        s["pass_yd"] * num(df, "pass_yards_gained_exp")
        + s["pass_td"] * num(df, "pass_touchdown_exp")
        + s["pass_int"] * num(df, "pass_interception_exp")
        + s["rush_yd"] * num(df, "rush_yards_gained_exp")
        + s["rush_td"] * num(df, "rush_touchdown_exp")
        + s["rec"] * rec_exp
        + s["rec_yd"] * num(df, "rec_yards_gained_exp")
        + s["rec_td"] * num(df, "rec_touchdown_exp")
    )
    if pts.abs().sum() == 0 and "total_fantasy_points_exp" in df.columns:
        pts = num(df, "total_fantasy_points_exp")
    return pts


# --------------------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------------------
def load_player_stats(season: int) -> pd.DataFrame:
    df = to_pandas(nfl.load_player_stats(seasons=[season]))
    ren = {}
    if c := first_col(df, "player_display_name", "player_name", "full_name"):
        ren[c] = "player"
    if c := first_col(df, "recent_team", "team", "posteam"):
        ren[c] = "team"
    if c := first_col(df, "position", "position_group"):
        ren[c] = "pos"
    if c := first_col(df, "player_id", "gsis_id"):
        ren[c] = "gsis_id"
    df = df.rename(columns=ren)
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().isin(["REG", "POST"])]
    df["half_ppr"] = score_actual(df)
    return df


def load_ff_opportunity(season: int) -> pd.DataFrame:
    try:
        df = to_pandas(nfl.load_ff_opportunity(seasons=[season], stat_type="weekly", model_version="latest"))
    except Exception:
        return pd.DataFrame()
    ren = {}
    if c := first_col(df, "player_id", "gsis_id"):
        ren[c] = "gsis_id"
    if c := first_col(df, "full_name", "player_name"):
        ren[c] = "player"
    df = df.rename(columns=ren)
    df["half_ppr_exp"] = score_expected(df)
    return df


def load_snaps(season: int) -> pd.DataFrame:
    try:
        df = to_pandas(nfl.load_snap_counts(seasons=[season]))
    except Exception:
        return pd.DataFrame()
    ren = {}
    if c := first_col(df, "pfr_player_id", "pfr_id"):
        ren[c] = "pfr_id"
    df = df.rename(columns=ren)
    return df


def load_routes(season: int) -> pd.DataFrame:
    """Weekly WR/TE routes, targets and receiving first downs. Estimated — see mega/routes.py.
    Empty frame if play-by-play (the source of team dropbacks) can't be reached."""
    from mega import routes as rz
    try:
        return rz.weekly(load_player_stats(season), load_snaps(season),
                         player_ids.crosswalk(), rz.team_dropbacks(season))
    except Exception:
        return pd.DataFrame()


def load_redzone(season: int) -> pd.DataFrame:
    """Carries and targets inside the 20, the 10 and the 5, from play-by-play.

    It has to come from play-by-play: the weekly `rushing_10` / `rushing_20` columns look
    like red zone stats and count runs of 10+ and 20+ YARDS. See mega/redzone.py."""
    from mega import redzone as rzn
    try:
        return rzn.weekly(season)
    except Exception:
        return pd.DataFrame()


def load_injuries(season: int) -> pd.DataFrame:
    try:
        df = to_pandas(nfl.load_injuries(seasons=[season]))
    except Exception:
        return pd.DataFrame()
    ren = {}
    if c := first_col(df, "gsis_id", "player_id"):
        ren[c] = "gsis_id"
    if c := first_col(df, "report_status", "game_status"):
        ren[c] = "report_status"
    df = df.rename(columns=ren)
    return df


def load_schedule(season: int) -> pd.DataFrame:
    df = to_pandas(nfl.load_schedules(seasons=[season]))
    return df


def load_rosters(season: int) -> pd.DataFrame:
    try:
        return to_pandas(nfl.load_rosters(seasons=[season]))
    except Exception:
        return pd.DataFrame()


def current_week(season: int, fallback: int) -> int:
    """The week you're setting a lineup for: the first regular-season week with a game
    still to play. Box scores arrive game by game, so "latest week with stats + 1" jumps
    ahead the moment Thursday night's game posts — on the Friday of Week 2 the dashboard
    was optimizing Week 3."""
    try:
        s = load_schedule(season)
        s = s[s["game_type"].astype(str).str.upper() == "REG"]
        open_weeks = s.loc[s["result"].isna(), "week"]
        return int(open_weeks.min()) if not open_weeks.empty else int(s["week"].max())
    except Exception:
        return fallback


# --------------------------------------------------------------------------------------
# Roster mapping
# --------------------------------------------------------------------------------------
def map_roster(roster: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Resolve a roster to nflverse ids and backfill position + NFL team.

    The resolution order and the match report live in mega.ids so the league-wide
    rosters that feed the trade finder and the power rankings go through exactly the
    same path as this one — a player who resolves here and not there is how the same
    name ends up on two different teams.
    """
    return player_ids.resolve(roster, name_col="name")


def my_roster() -> tuple[pd.DataFrame, str]:
    """My roster, live scrape first. `roster.csv` is hand-maintained and drifts the
    moment a waiver clears — it is the fallback, not the source of truth."""
    try:
        from mega.config import MY_TEAM
        from mega.yahoo import cached_rosters

        r = cached_rosters()
        r = r[r["team"] == MY_TEAM] if not r.empty else r
        if not r.empty:
            # Carry yahoo_id through — it is the exact key, and dropping it forced
            # every one of these rows down the name-matching path.
            cols = [c for c in ("name", "slot", "pos", "nfl_team", "yahoo_id") if c in
                    r.rename(columns={"player": "name"}).columns]
            out = r.rename(columns={"player": "name"})[cols].copy()
            # The roster page often omits pos; the starting slot implies it for everyone
            # but flex, and the name match fills the rest in downstream.
            out["pos"] = out["pos"].where(
                out["pos"].notna() & out["pos"].astype(str).ne("nan"),
                out["slot"].where(out["slot"].isin(["QB", "RB", "WR", "TE", "K", "DEF"]), ""),
            )
            out["gsis_id"] = ""
            return out.fillna(""), "live Yahoo scrape"
    except Exception:
        pass
    return pd.read_csv(ROSTER_CSV, dtype=str).fillna(""), "roster.csv (stale — run the scrape)"


def my_gsis_ids(roster_raw: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Skill-position gsis_ids for a roster, and the rows that couldn't be resolved.

    Same filter app.py applies before it builds any stat view: kicker/DST excluded (no
    offensive stats to show), unresolved names dropped (no id to key on).
    """
    mapped, _match = map_roster(roster_raw)
    skill = mapped[~mapped["pos"].isin(["K", "DEF"]) & ~mapped["slot"].isin(["K", "DEF"])].copy()
    dropped = skill[~skill["resolved"] & ~skill["unmapped"]]
    skill = skill[skill["resolved"]].copy()
    return [g for g in skill["gsis_id"].dropna().tolist()], dropped


def ownership() -> tuple[dict, dict]:
    """gsis_id -> (fantasy team, slot) from the scraped rosters, and gsis_id -> Yahoo FA
    status. Both go through the id resolver: matching the FA list by name missed players
    Yahoo and nflverse spell differently (Joshua vs Josh Palmer)."""
    from mega.intel import current_rosters
    from mega.yahoo import cached_rosters

    r = current_rosters(cached_rosters())
    own = {g: (t, sl) for g, t, sl in zip(r["gsis_id"], r["team"], r["slot"]) if pd.notna(g) and g}
    fa, path = {}, DATA / "yahoo_free_agents.csv"
    if path.is_file():
        f, _ = player_ids.resolve(pd.read_csv(path, dtype=str).fillna(""), name_col="player")
        fa = {g: st_ for g, st_ in zip(f["gsis_id"], f["roster_status"]) if pd.notna(g) and g}
    return own, fa


def ask_player_week(season: int) -> pd.DataFrame:
    """The curated player-week index `mega.ask` queries against."""
    from mega import ask as ASK

    return ASK.player_week(load_player_stats(season), load_snaps(season),
                           load_ff_opportunity(season), load_routes(season),
                           player_ids.crosswalk(), load_redzone(season))


def nflverse_table(table: str, season: int) -> pd.DataFrame:
    """Any catalogued nflverse table, for questions outside the curated metrics."""
    from . import catalog

    return catalog.load(table, season)
