"""
Fantasy football dashboard for a single Yahoo team (default: half-PPR).

Data source: nflverse-data (https://github.com/nflverse/nflverse-data) via nflreadpy.
Run:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# On Streamlit Community Cloud there is no .env file — secrets live in the app's
# Secrets manager (st.secrets). Mirror any flat string secrets into the process
# environment so every module that reads os.environ (projections, digest, …)
# keeps working unchanged. Local runs have no st.secrets and fall back to .env.
try:
    for _k, _v in dict(st.secrets).items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

try:
    import nflreadpy as nfl
except ImportError:  # pragma: no cover
    nfl = None

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
HERE = Path(__file__).parent
ROSTER_CSV = HERE / "roster.csv"
SEASON_DEFAULT = 2025
CACHE_TTL = dt.timedelta(hours=6)
SIM_TOP = 12   # §18.3: only the best candidates by pts/wk are worth simulating

# Yahoo default half-PPR scoring. Edit if your league differs.
SCORING = dict(
    pass_yd=0.04, pass_td=4, pass_int=-1,
    rush_yd=0.10, rush_td=6,
    rec=0.5, rec_yd=0.10, rec_td=6,
    fum_lost=-2, two_pt=2, ret_td=6,
)

STARTER_SLOTS = ["QB", "RB", "WR", "TE", "W/R", "W/R/T", "FLEX"]

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(s: object) -> str:
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"[^a-z ]", " ", s)
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


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
# Data loaders (cached)
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def target_share(season: int, through_week: int, roll: int) -> pd.DataFrame:
    """Target share and rank inside the player's own NFL offense, over the last `roll` weeks.

    Shares are summed numerator / summed denominator — averaging weekly ratios lets one
    low-volume game swing a player's season number.
    """
    df = load_player_stats(season)
    if df.empty or "targets" not in df.columns:
        return pd.DataFrame(columns=["gsis_id", "tgt_pct", "tm_rank"])
    df = df[(df["week"] <= through_week) & (df["week"] > through_week - roll)].copy()
    df["targets"] = pd.to_numeric(df["targets"], errors="coerce").fillna(0.0)
    team_tot = df.groupby("team")["targets"].sum().rename("tm_tgt").reset_index()
    p = (
        df.sort_values("week")
        .groupby("gsis_id")
        .agg(targets=("targets", "sum"), team=("team", "last"))
        .reset_index()
    )
    p = p[p["targets"] > 0].merge(team_tot, on="team", how="left")
    p["tgt_pct"] = p["targets"] / p["tm_tgt"].replace(0, pd.NA)
    p["tm_rank"] = p.groupby("team")["targets"].rank(method="min", ascending=False)
    return p[["gsis_id", "tgt_pct", "tm_rank"]]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
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


@st.cache_data(ttl=CACHE_TTL, show_spinner="Estimating routes run…")
def load_routes(season: int) -> pd.DataFrame:
    """Weekly WR/TE routes, targets and receiving first downs. Estimated — see mega/routes.py.
    Empty frame if play-by-play (the source of team dropbacks) can't be reached."""
    from mega import routes as rz
    try:
        return rz.weekly(load_player_stats(season), load_snaps(season),
                         player_ids.crosswalk(), rz.team_dropbacks(season))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_schedule(season: int) -> pd.DataFrame:
    df = to_pandas(nfl.load_schedules(seasons=[season]))
    return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_rosters(season: int) -> pd.DataFrame:
    try:
        return to_pandas(nfl.load_rosters(seasons=[season]))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=dt.timedelta(minutes=30), show_spinner=False)
def _league_scores() -> pd.DataFrame:
    """Every team's score in every completed week, from the weekly scrape.

    The import is inside the try on purpose. Streamlit Cloud can be serving an older copy of
    mega/ than of app.py, and an ImportError at module scope takes the whole dashboard down
    rather than hiding one section — which is exactly what it did the first time.
    """
    try:
        from mega.yahoo import cached_scores

        return cached_scores()
    except Exception:
        return pd.DataFrame()
# ---- shared loaders: role context, red zone, and the Ask index ------------------------
# Defined here with the other cached loaders rather than beside the Ask tab, because the
# player card reads _role_ctx long before that tab's code is reached and Streamlit runs
# this file top to bottom — a loader defined below its first use does not exist yet.

# A question box over the nflverse tables. Deterministic (mega/ask.py) rather than a
# language model: no API key, nothing to pay per question, and — the reason it is built
# this way — it can always print how it read the question, so a misparse is visible on
# screen instead of arriving as a confident wrong table.
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_nextgen(season: int, kind: str) -> pd.DataFrame:
    try:
        return to_pandas(nfl.load_nextgen_stats(seasons=[season], stat_type=kind))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_depth_charts(season: int) -> pd.DataFrame:
    try:
        return to_pandas(nfl.load_depth_charts(seasons=[season]))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Working out everyone's role…")
def _role_ctx(season: int) -> dict:
    """§12 role context: role per player, role baselines, NFL positional averages and the
    shrunk index of each against both."""
    from mega import roles as RL

    return RL.build(_ask_pw(season), load_ff_opportunity(season),
                    load_nextgen(season, "receiving"), load_nextgen(season, "rushing"),
                    load_depth_charts(season))


@st.cache_data(ttl=CACHE_TTL, show_spinner="Reading the red zone…")
def load_redzone(season: int) -> pd.DataFrame:
    """Carries and targets inside the 20, the 10 and the 5, from play-by-play.

    It has to come from play-by-play: the weekly `rushing_10` / `rushing_20` columns look
    like red zone stats and count runs of 10+ and 20+ YARDS. See mega/redzone.py."""
    from mega import redzone as rzn
    try:
        return rzn.weekly(season)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Indexing every player-week…")
def _ask_pw(season: int) -> pd.DataFrame:
    from mega import ask as ASK

    return ASK.player_week(load_player_stats(season), load_snaps(season),
                           load_ff_opportunity(season), load_routes(season),
                           player_ids.crosswalk(), load_redzone(season))


@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching that nflverse table…")
def _nflverse_table(table: str, season: int) -> pd.DataFrame:
    """Any catalogued nflverse table, for questions outside the curated metrics."""
    from mega import catalog

    return catalog.load(table, season)
@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Simulating the rest of the season…")
def _season_model(season: int):
    """§18 — this league's remaining season, ready to simulate."""
    from mega import sim as SIM
    from mega.yahoo import cached_scores

    from mega.yahoo import cached_fixtures

    sc = cached_scores()
    st_path = HERE / "data" / "yahoo_standings.csv"
    if sc is None or sc.empty or not st_path.is_file():
        return None
    stand = pd.read_csv(st_path)
    ppw = sc.groupby("team")["points"].mean().to_dict()
    left = range(int(next_week), 15)
    try:
        fx = cached_fixtures()
    except Exception:
        fx = None
    return SIM.from_league(sc, stand, ppw, left, fixtures=fx)


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner=False)
def _odds(season: int) -> pd.DataFrame:
    from mega import sim as SIM

    s = _season_model(season)
    return SIM.simulate(s, n=6000) if s is not None else pd.DataFrame()


@st.cache_data(ttl=dt.timedelta(minutes=30), show_spinner="Pricing trades in playoff odds…")
def _trade_odds(season: int, rows: tuple) -> dict:
    """(give, get, dMe, dThem) -> what the deal does to your chances.

    Points per week is the ranking; this is the thing that actually matters. A point added
    to a team already 99% safe is worth less than the same point on the bubble."""
    from mega import sim as SIM
    from mega.config import MY_TEAM

    s = _season_model(season)
    if s is None or MY_TEAM not in s.teams:
        return {}
    out = {}
    for key, partner, d_me, d_them in rows:
        if partner not in s.teams:
            continue
        out[key] = SIM.trade_delta(s, MY_TEAM, partner, float(d_me), float(d_them), n=1500)
    return out








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


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Mega Bowl · Command Center", page_icon="🏈", layout="wide")

from mega import ids as player_ids  # noqa: E402
from mega import ui  # noqa: E402

ui.inject_css()

if nfl is None:
    st.error("`nflreadpy` is not installed. Run:  `pip install -r requirements.txt`")
    st.stop()

def _my_roster() -> tuple[pd.DataFrame, str]:
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


roster_raw, ROSTER_SRC = _my_roster()

try:
    _cur_season = int(nfl.get_current_season())
except Exception:
    _cur_season = SEASON_DEFAULT

with st.sidebar:
    st.header("Settings")
    season = st.number_input("Season", min_value=2015, max_value=2030, value=_cur_season, step=1)
    stats = load_player_stats(int(season))
    max_wk = int(stats["week"].max()) if not stats.empty else 1
    if max_wk <= 1:
        week = max_wk
        st.caption(f"Through week {week} (only week {max_wk} available)")
    else:
        week = st.slider("Through week", 1, max_wk, max_wk)
    next_week = st.slider("Matchup week", 1, 18, min(current_week(int(season), max_wk + 1), 18),
                          help="The week Start/Sit and Matchups plan for. Defaults to the first week "
                               "with games still to play.")
    roll = st.slider("Rolling window (weeks)", 2, 6, 3)
    # Read by ui.table below, so it has to be rendered before the first one — app.py runs
    # top to bottom every rerun.
    ui.detail_toggle()
    if st.button("♻️ Clear data cache"):
        st.cache_data.clear()
        try:
            nfl.clear_cache()
        except Exception:
            pass
        st.rerun()
    st.caption("Positions")
    ui.pos_legend()

mapped, MATCH = map_roster(roster_raw)
# Filter on slot too — the scrape leaves pos blank on plenty of rows, so a pos-only
# test lets the kicker and defense through into the skill-player views.
skill = mapped[~mapped["pos"].isin(["K", "DEF"]) & ~mapped["slot"].isin(["K", "DEF"])].copy()
# An unresolved player has no stats to show and no position to be slotted at, so he
# would sit in every view as a silent blank row. Drop him here and say so loudly —
# that is the whole point of the match check.
dropped = skill[~skill["resolved"] & ~skill["unmapped"]]
skill = skill[skill["resolved"]].copy()
gsis_list = [g for g in skill["gsis_id"].dropna().tolist()]

if not dropped.empty:
    st.error(
        "**Couldn't match " + str(len(dropped)) + " rostered player(s) to nflverse:** "
        + ", ".join(f"`{n}`" for n in dropped["name"])
        + ".  They're excluded from every stat view below — add a row to "
        "`data/id_overrides.csv` (`yahoo_id,name,gsis_id,note`) to fix or to mark them "
        "intentionally unmapped."
    )

sw = stats[(stats["gsis_id"].isin(gsis_list)) & (stats["week"] <= week)].copy()
ffo = load_ff_opportunity(int(season))
if not ffo.empty:
    ffo = ffo[(ffo["gsis_id"].isin(gsis_list)) & (ffo["week"] <= week)]
snaps = load_snaps(int(season))
inj = load_injuries(int(season))
sched = load_schedule(int(season))

name_by_id = dict(zip(skill["gsis_id"], skill["name"]))
slot_by_id = dict(zip(skill["gsis_id"], skill["slot"]))
team_by_id = dict(zip(skill["gsis_id"], skill["nfl_team"]))
pos_by_id = dict(zip(skill["gsis_id"], skill["pos"]))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _player_teams(season: int) -> dict[str, str]:
    """normalized name -> current NFL team, for the logo beside every player.

    Layered freshest-last: ff_playerids covers nearly everyone, this season's box scores
    catch moves since that file was built, and the league rosters are what Yahoo says
    today. All of it goes through canon_team so LAR/JAC/GBP land on the right logo.
    """
    xw = player_ids.crosswalk()
    out = dict(zip(xw["norm"], xw["team"].map(player_ids.canon_team)))
    st_ = load_player_stats(season)
    if not st_.empty and {"player", "team", "week"} <= set(st_.columns):
        last = st_.sort_values("week").groupby("player")["team"].last()
        out.update({player_ids.norm(n): player_ids.canon_team(t) for n, t in last.items()})
    try:
        from mega.intel import current_rosters
        from mega.yahoo import cached_rosters

        lr = current_rosters(cached_rosters())
        if not lr.empty:
            lr = lr[lr["nfl_team"].astype(str).str.len() > 0]
            out.update(dict(zip(lr["norm"], lr["nfl_team"])))
    except Exception:
        pass
    return {k: v for k, v in out.items() if k and v}


try:
    ui.set_player_teams(_player_teams(int(season)))
except Exception:
    pass  # logos are decoration; never block the dashboard on them


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Working out everyone's role…")
def _player_roles(season: int) -> dict[str, str]:
    """normalized name -> "WR3 · playing up", for the role column beside every player.

    Read from mega.season.role_lookup — the same role context the waiver board prices
    with — so a flag on a table and a flag on the board can never disagree. Worded by
    mega.glossary, which is the one place that decides what a code says in English."""
    from mega.glossary import cell as role_cell
    from mega.season import role_lookup

    rl = role_lookup(season)
    if not rl:
        return {}
    xw = player_ids.crosswalk()
    xw = xw[xw["gsis_id"].isin(rl.keys())].drop_duplicates("gsis_id")
    out = {}
    for gid, n in zip(xw["gsis_id"], xw["norm"]):
        rc = rl[gid]
        if not rc.get("role"):
            continue
        out[n] = role_cell(rc["role"], rc.get("flags"), rc.get("tags"))
    return out


try:
    ui.set_player_roles(_player_roles(int(season)))
except Exception:
    pass  # role context is enrichment; a missing season must not blank the tables

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _season_basis(season: int) -> dict:
    from mega.intel import season_basis
    return season_basis(season)


try:
    BASIS = _season_basis(int(season))
except Exception:
    BASIS = {"season": int(season), "current": True, "weeks": 0, "label": f"{int(season)} form"}

ui.masthead([
    "Half-PPR · team <strong>TaylorMade</strong>",
    f"{int(season)} season · through wk {week}",
    f"matchup wk {next_week}",
    f"roster: {ROSTER_SRC}",
    BASIS["label"],
])

# Grouped by the decision you're making, not by where the data came from. Streamlit
# tabs are containers, so the `with tab_*:` bodies further down render into these
# wherever they appear in the file.
sec_now, sec_team, sec_get, sec_league, sec_players, sec_ask, sec_more = st.tabs(
    ["This Week", "My Team", "Get Better", "League", "Players", "Ask", "More"]
)
with sec_now:
    tab_action, tab_start, tab_match = st.tabs(["Action Board", "Start / Sit", "Matchups"])
with sec_team:
    tab_over, tab_axe, tab_use = st.tabs(["Roster", "Points vs Opportunity", "Usage Trends"])
with sec_get:
    tab_wire, tab_trade, tab_wopr, tab_arch = st.tabs(
        ["Waiver Wire", "Trade Finder", "WOPR", "Archetypes"])
with sec_league:
    tab_league, tab_draft = st.tabs(["Standings", "Draft Value"])
with sec_more:
    tab_digest, tab_news, tab_raw = st.tabs(["Weekly Digest", "News", "Raw Data"])


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Scoring league-winner archetypes…")
def _archetypes(season: int) -> pd.DataFrame:
    from mega.archetypes import score
    return score(season)


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Computing WOPR (opportunity) targets…")
def _wopr(season: int) -> dict:
    from mega.wopr import summary
    return summary(season)


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Ranking rosters…")
def _power(season: int) -> pd.DataFrame:
    from mega.intel import roster_strength
    from mega.yahoo import cached_rosters
    return roster_strength(season, cached_rosters())


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Pulling projections…")
def _blended_proj(season: int, week: int) -> pd.DataFrame:
    from mega.projections import blended_week
    return blended_week(season, week)


@st.cache_data(ttl=dt.timedelta(hours=2), show_spinner="Reading the sportsbook…")
def _vegas(season: int, week: int) -> pd.DataFrame:
    """§20 — the week's props scored into half-PPR points. Empty when none are on file."""
    from mega.vegas import week as vweek
    return vweek(season, week)


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner=False)
def _dvp(season: int) -> pd.DataFrame:
    from mega.lineup import defense_vs_position
    return defense_vs_position(season)


def _my_roster_projected(season: int, week: int) -> pd.DataFrame:
    """My skill roster joined to blended projections (by gsis_id, name fallback)."""
    proj = _blended_proj(season, week)
    for c in ("proj", "proj_source", "ecr", "start_sit", "norm"):
        if c not in proj.columns:
            proj[c] = pd.NA
    r = skill.rename(columns={"name": "player"})[["player", "slot", "pos", "nfl_team", "gsis_id"]].copy()
    pj = proj[["gsis_id", "proj", "proj_source", "ecr", "start_sit"]].dropna(subset=["gsis_id"])
    r = r.merge(pj, on="gsis_id", how="left")
    miss = r["proj"].isna()
    if miss.any():  # fallback join on normalized name
        from mega.intel import _norm
        r["_n"] = r["player"].map(_norm)
        pn = proj.dropna(subset=["proj"]).drop_duplicates("norm")[["norm", "proj", "proj_source", "ecr", "start_sit"]]
        r = r.merge(pn, left_on="_n", right_on="norm", how="left", suffixes=("", "_n"))
        for c in ["proj", "proj_source", "ecr", "start_sit"]:
            r[c] = r[c].where(r[c].notna(), r[f"{c}_n"])
        r = r[[c for c in r.columns if not c.endswith("_n") and c != "norm"]]
    r["proj"] = pd.to_numeric(r["proj"], errors="coerce").fillna(0.0)
    v = _vegas(season, week)
    if not v.empty:
        r = r.merge(v[["gsis_id", "vegas", "vegas_parts", "vegas_complete"]],
                    on="gsis_id", how="left")
        r["vegas_edge"] = (pd.to_numeric(r["vegas"], errors="coerce") - r["proj"]).round(1)
    return r

# ---- Roster aggregate ----------------------------------------------------------------
# Built once here rather than inside the Roster tab — the Action board needs the same
# points-vs-opportunity numbers to pick sell-high and buy-low candidates.
def _build_agg() -> pd.DataFrame:
    if sw.empty:
        return pd.DataFrame()
    sws = sw.sort_values("week")
    for col in ("targets", "carries"):
        if col not in sws.columns:
            sws[col] = 0.0
    agg = (
        sws.groupby("gsis_id")
        .agg(
            games=("week", "nunique"),
            half_ppr_tot=("half_ppr", "sum"),
            half_ppr_pg=("half_ppr", "mean"),
            last_wk=("half_ppr", "last"),
            roll_pg=("half_ppr", lambda s: s.tail(roll).mean()),
            tgt_pg=("targets", "mean"),
            carry_pg=("carries", "mean"),
        )
        .reset_index()
    )
    agg["player"] = agg["gsis_id"].map(name_by_id)
    agg["slot"] = agg["gsis_id"].map(slot_by_id)
    agg["pos"] = agg["gsis_id"].map(pos_by_id)
    agg = agg.merge(target_share(int(season), int(week), int(roll)), on="gsis_id", how="left")

    if not ffo.empty:
        exp = ffo.groupby("gsis_id")["half_ppr_exp"].sum().reset_index().rename(columns={"half_ppr_exp": "xfp_tot"})
        agg = agg.merge(exp, on="gsis_id", how="left")
        agg["xfp_diff"] = agg["half_ppr_tot"] - agg["xfp_tot"]

    if not inj.empty and "report_status" in inj.columns:
        latest = (
            inj[inj["gsis_id"].isin(gsis_list)]
            .sort_values("week")
            .groupby("gsis_id")
            .tail(1)[["gsis_id", "report_status"]]
        )
        agg = agg.merge(latest, on="gsis_id", how="left")

    # Chris's overrides (data/player_status.csv) beat the injury report, which lags.
    from mega.status import out_for_week
    _out = out_for_week(int(next_week))
    if _out:
        lab = agg["player"].map(lambda n: _out.get(player_ids.norm(n)))
        agg["report_status"] = lab.fillna(agg["report_status"]) if "report_status" in agg else lab

    if not sched.empty:  # next opponent + Vegas implied team total
        wk = sched[sched["week"] == next_week]
        opp_rows = []
        for _, gm in wk.iterrows():
            tl, sl = gm.get("total_line"), gm.get("spread_line")
            h_imp = (tl / 2 + sl / 2) if pd.notna(tl) and pd.notna(sl) else None
            a_imp = (tl / 2 - sl / 2) if pd.notna(tl) and pd.notna(sl) else None
            opp_rows.append({"team": gm["home_team"], "opp": "vs " + str(gm["away_team"]), "implied": h_imp})
            opp_rows.append({"team": gm["away_team"], "opp": "@ " + str(gm["home_team"]), "implied": a_imp})
        agg["nfl_team"] = agg["gsis_id"].map(team_by_id)
        agg = agg.merge(pd.DataFrame(opp_rows), left_on="nfl_team", right_on="team",
                        how="left", suffixes=("", "_x"))

    order = {s: i for i, s in enumerate(["QB", "RB", "WR", "TE", "W/R", "BN"])}
    agg["_o"] = agg["slot"].map(order).fillna(9)
    return agg.sort_values(["_o", "half_ppr_pg"], ascending=[True, False])


agg = _build_agg()

# ---- Roster --------------------------------------------------------------------------
with tab_over:
    if agg.empty:
        st.info("No stats yet for this season/week range.")
    else:
        starters = agg[agg["slot"] != "BN"]
        proj = starters["roll_pg"].sum()
        best = starters.loc[starters["roll_pg"].idxmax()] if not starters.empty else None
        worst = starters.loc[starters["roll_pg"].idxmin()] if not starters.empty else None
        xdelta = starters["xfp_diff"].sum() if "xfp_diff" in starters.columns else None
        n_hurt = (
            int(starters["report_status"].isin(["Out", "Doubtful", "Questionable", "IR"]).sum())
            if "report_status" in starters.columns else 0
        )
        ui.kpi_row([
            (f"Starters · L{roll} proj", f"{proj:.0f}", "sum of rolling avg"),
            ("Roster Act−xFP", f"{xdelta:+.0f}" if xdelta is not None else "—",
             "over/under his usage" if xdelta is not None else "xFP n/a"),
            ("Top starter", ui.short_name(best["player"]) if best is not None else "—",
             f"{best['roll_pg']:.1f}/g" if best is not None else ""),
            ("Coldest starter", ui.short_name(worst["player"]) if worst is not None else "—",
             f"{worst['roll_pg']:.1f}/g" if worst is not None else ""),
            ("Injury flags", str(n_hurt), "starters Q or worse"),
        ])
        st.write("")

        show_cols = [c for c in [
            "slot", "player", "pos", "games", "half_ppr_pg", "roll_pg", "last_wk",
            "xfp_tot", "xfp_diff", "tgt_pg", "tgt_pct", "tm_rank", "carry_pg",
            "report_status", "opp", "implied",
        ] if c in agg.columns]
        roll_lbl = f"L{roll}"
        ui.lede(
            "Your roster, with <b>how much work each player is getting</b> next to what he scored. "
            "Big gaps between the two are where the decisions are."
        )
        ui.table(
            agg[show_cols],
            rename={"roll_pg": roll_lbl},
            diverging=["xFP±"], sequential=["TGT%"], pos_cols=["POS"],
            fmt={c: "{:.1f}" for c in ["PPG", roll_lbl, "LAST", "xFP", "xFP±", "TGT", "CAR", "IMP"]}
                | {"TGT%": "{:.1%}"},
            labels={roll_lbl: f"Last {roll}/g"},
            help={roll_lbl: f"Fantasy points per game over his last {roll} games."},
            # POS stays: the role tag usually carries the position, but a player with
            # too few games to have a role would otherwise show none at all
            essential=["POS", roll_lbl, "xFP±", "TGT%", "ST"],
        )
        st.caption(
            f"**Target share** and **Team tgt rank** cover the last {roll} weeks. A #1 target rank on a high "
            "share is a true alpha. Good points on a low target share is usually touchdown luck that won't "
            "hold. **Vs expected**: red = scoring above his usage (sell high) · navy = below it (hold or buy)."
        )

        # ---- route usage: WR and TE only. Routes are estimated (mega/routes.py), and the
        # estimate is too crude for backs, who are on the field for runs they never route on.
        _rw = load_routes(int(season))
        if not _rw.empty:
            from mega import routes as rz

            _win = range(max(1, int(week) - int(roll) + 1), int(week) + 1)
            _rt = rz.totals(_rw[_rw["gsis_id"].isin(gsis_list)], weeks=_win)
            if not _rt.empty and _rt["routes"].notna().any():
                _rt["slot"] = _rt["gsis_id"].map(slot_by_id)
                _rt["route_flag"] = _rt.apply(rz.flag, axis=1)
                if "tgt_pct" in agg.columns:
                    _rt = _rt.merge(agg[["gsis_id", "tgt_pct"]], on="gsis_id", how="left")
                # small samples last: a half-game cameo can top any per-route rate
                _rt = _rt.sort_values(["qualified", "fd_rr"], ascending=False, na_position="last")
                ui.h("Route usage")
                ui.lede(
                    "Target share tells you how big his slice is. These tell you <b>how good the slice "
                    "is</b>: how often he's on the field in a route, how often the ball comes when he is, "
                    "and how often that moves the chains. <b>1st downs per route</b> is the one to read "
                    "first — 12%+ is the league-winner line for a WR."
                )

                # Your receivers against the league, on the two axes the decision actually turns
                # on: is he on the field, and is it worth anything when he is. The yellow rule is
                # the first-down line — 12% of routes — drawn the way television draws one.
                _pool = rz.totals(_rw, weeks=_win)
                _pool = _pool[_pool["qualified"] & _pool["fd_rr"].notna()]
                if len(_pool) >= 8:
                    _me = _pool[_pool["gsis_id"].isin(gsis_list)].copy()
                    _me["short"] = _me["player"].map(ui.short_name)
                    ui.route_plot(
                        _pool, _me, threshold=rz.WR_FD_FLAG,
                        title=f"Route usage — every qualified WR and TE, last {roll} weeks",
                        caption=(
                            "**Up** is doing more with each route; **right** is being on the field for "
                            "more of them. Top-right is where you want your starters. Top-left is a "
                            "player one role change from a league-winner — the waiver claim to make."
                        ))
                # The rates lead: they're the point of the table, and on a laptop the last
                # columns of a wide table sit off the right edge until you scroll.
                _rcols = [c for c in ["slot", "player", "pos", "team", "fd_rr", "tprr", "tgt_pct",
                                      "targets", "routes", "routes_pg", "route_flag"] if c in _rt.columns]
                # No heat shading on the rates: it would paint a half-game cameo's 50%
                # target rate the darkest cell in the table, which is the opposite of true.
                ui.table(
                    _rt[_rcols], pos_cols=["POS"],
                    fmt={"RTE": "{:.0f}", "RTE/G": "{:.1f}", "TARGETS": "{:.0f}",
                         "TGT%": "{:.1%}", "TPRR": "{:.1%}", "1D/RR": "{:.1%}"},
                )
                st.caption(
                    "Routes are **estimated** — his snap share × his team's dropbacks — because nflverse "
                    "publishes no charted route count. The estimate counts blocking snaps, so tight ends "
                    f"read low: compare a TE with other TEs. Under {rz.MIN_ROUTES} routes in the window, "
                    "treat the rates as noise and no flag is given. Backs are left out — the estimate "
                    "can't tell their run snaps from their pass snaps."
                )

# ---- League (live Yahoo API) ---------------------------------------------------------
with tab_league:
    # §18.2 — where this is all heading. Points per week is the working currency; this is
    # the one that decides the season.
    try:
        _o = _odds(int(season))
    except Exception as _e:
        _o = pd.DataFrame()
        st.caption(f"Playoff odds unavailable: {type(_e).__name__}: {_e}")
    if not _o.empty:
        from mega.config import MY_TEAM as _MT
        from mega.sim import schedule_note as _sched_note

        ui.h("Playoff odds")
        _mine_odds = _o[_o["team"] == _MT]
        if not _mine_odds.empty:
            _r = _mine_odds.iloc[0]
            ui.kpi_row([
                ("Make the playoffs", f"{_r['p_playoffs']:.0%}", "6 of 12 get in"),
                ("First-round bye", f"{_r['p_bye']:.0%}", "top 2 seeds"),
                ("Win it all", f"{_r['p_title']:.0%}", ""),
                ("Average seed", f"{_r['mean_seed']:.1f}", "across the simulations"),
            ])
        # logos off: these are fantasy teams, and the NFL logo lookup blanks the column
        # rather than admitting it cannot match "Crabcakes and Football" to a shield.
        # "team" would map to TM and render under an "NFL" header; these are fantasy teams
        ui.table(_o[["team", "p_playoffs", "p_bye", "p_title", "mean_seed"]],
                 rename={"team": "TEAM"}, sequential=["PLAYOFFS", "TITLE"],
                 fmt={"PLAYOFFS": "{:.0%}", "BYE": "{:.0%}", "TITLE": "{:.0%}",
                      "SEED": "{:.1f}"}, legend=False, logos=False, roles=False)
        st.caption(
            "6,000 simulated seasons. Each week's score is drawn around the team's own "
            "average, with this league's own spread. "
            + f"**{_sched_note(_season_model(int(season)))}**"
        )
        st.write("")

    from mega import yahoo_api as _ya
    from mega.yahoo import cached_standings as _cs

    _scraped = _cs()
    if not _ya.available() and not _scraped.empty:
        from mega.config import LEAGUE_ID, LEAGUE_URL, MY_SEAT

        st.caption(f"Scraped Yahoo standings · [league {LEAGUE_ID}]({LEAGUE_URL})")
        me = _scraped[_scraped["seat"] == MY_SEAT]
        if not me.empty:
            r = me.iloc[0]
            ui.kpi_row([
                ("Record", f"{int(r['wins'])}-{int(r['losses'])}-{int(r['ties'])}", str(r["team"])),
                ("Standing", f"#{int(r['rank'])}", f"of {len(_scraped)}"),
                ("Manager", str(r["manager"]), "you"),
            ])
            st.write("")
        ui.h("Standings")
        ui.table(
            _scraped[["rank", "team", "manager", "wins", "losses", "ties"]],
            rename={"team": "TEAM"}, fmt={c: "{:.0f}" for c in ["W", "L", "T"]}, logos=False,
        )
        st.caption("Records and ranks come from the weekly scrape.")

        _scores = _league_scores()
        if not _scores.empty:
            ui.h("Power rankings — expected wins")
            ui.lede(
                "A fantasy record is mostly schedule. This throws the schedule out and asks what "
                "each week's score was worth <b>against the whole league</b>: a close score counts "
                "as a coin flip, a blowout as near-certain. <b>Luck</b> is real wins minus expected "
                "ones — plus means the record is flattering them and should come back."
            )
            from mega.xwins import power_table

            _xw = power_table(_scores).rename(columns={"luck": "luck_w"})
            ui.table(
                _xw[["power_rank", "team", "xwins", "power", "wins", "luck_w", "pf", "ppg", "cv"]],
                rename={"team": "TEAM"}, diverging=["LUCK W"], sequential=["xW%"],
                fmt={"xW": "{:.2f}", "xW%": "{:.1%}", "W": "{:.0f}", "LUCK W": "{:+.2f}",
                     "PF": "{:.1f}", "PPG": "{:.1f}", "SWING": "{:.0%}"},
                logos=False,
            )
            st.caption(
                f"Through week {int(_scores['week'].max())}. The model is the one from Chris's "
                "Expected Wins workbook — a logistic on the score gap, scaled to that week's own "
                "spread — and this implementation reproduces the workbook's 2025 numbers exactly."
            )

        ui.h("Power rankings — roster strength")
        ui.lede(
            "Every roster scored by what its <b>best legal lineup</b> is worth per game, ignoring "
            "record entirely. <b>Luck</b> is the gap between where a team sits and how good it is: "
            "a big positive number means the record is flattering them, and they'll come back."
        )
        try:
            _rs = _power(int(season))
        except Exception as e:
            _rs = pd.DataFrame()
            st.caption(f"Power rankings unavailable: {e}")

        if not _rs.empty:
            _rs = _rs.merge(_scraped[["team", "rank"]], on="team", how="left")
            # Both ranks count 1 = best, so power_rank - rank is positive when a team
            # sits higher in the table than its roster justifies.
            _rs["luck"] = _rs["power_rank"] - _rs["rank"]
            ui.table(
                _rs[["power_rank", "team", "starters_pg", "bench_pg", "rank", "luck", "matched"]],
                rename={"team": "TEAM"},
                sequential=["LINEUP"], diverging=["LUCK"],
                fmt={"LINEUP": "{:.1f}", "BENCH": "{:.1f}", "LUCK": "{:+.0f}", "MATCHED": "{:.0f}"},
                logos=False,
            )
        st.divider()

    if not _ya.available():
        st.info(
            "No live Yahoo data yet.\n\n"
            "**The official API is blocked** — Yahoo removed Fantasy Sports from the app "
            "permissions console, so new apps can't be granted the scope and every token comes "
            "back `additional_authorization_required`. `pull_league.py` is ready if that ever "
            "changes.\n\n"
            "**Use the browser scrape instead** — sign in once, then pull:\n\n"
            "```\n.venv/bin/python -m mega.yahoo login\n.venv/bin/python -m mega.yahoo pull\n```\n\n"
            "That fills ownership for Waiver wire, Trades and WOPR (replacing the draft-board "
            "approximation). Standings and weekly results below need the API and stay empty."
        )
    else:
        from mega.config import LEAGUE_ID, LEAGUE_URL, MY_SEAT

        st.caption(f"Live Yahoo API · [league {LEAGUE_ID}]({LEAGUE_URL}) · pulled through week {_ya.week()}")

        standings = _ya.standings_df()
        if not standings.empty:
            me = standings[standings["seat"] == MY_SEAT]
            if not me.empty:
                r = me.iloc[0]
                ui.kpi_row([
                    ("Record", f"{int(r['wins'] or 0)}-{int(r['losses'] or 0)}", "TaylorMade"),
                    ("Standing", f"#{int(r['rank'] or 0)}", f"of {len(standings)}"),
                    ("Points for", f"{float(r['points_for'] or 0):.0f}", "season total"),
                    ("Points against", f"{float(r['points_against'] or 0):.0f}", "season total"),
                ])
                st.write("")

            ui.h("Standings")
            scols = [c for c in ["rank", "team", "wins", "losses", "ties", "points_for",
                                 "points_against", "streak", "faab_balance", "moves", "trades"]
                     if c in standings.columns]
            ui.table(
                standings[scols], rename={"team": "TEAM"}, sequential=["PF"],
                fmt={"PF": "{:.1f}", "PA": "{:.1f}", "W": "{:.0f}", "L": "{:.0f}", "T": "{:.0f}"},
                logos=False,
            )

        mu = _ya.matchups_df()
        if not mu.empty:
            ui.h("Weekly results")
            wk_pick = st.selectbox("Week", sorted(mu["week"].unique(), reverse=True))
            wv = mu[mu["week"] == wk_pick]
            mcols = [c for c in ["team", "opponent", "points", "opp_points", "proj", "result"]
                     if c in wv.columns]
            ui.table(
                wv[mcols].sort_values("points", ascending=False), rename={"team": "TEAM"},
                sequential=["PTS"], fmt={"PTS": "{:.1f}", "OPP PTS": "{:.1f}", "PROJ": "{:.1f}"},
                logos=False,
            )

            _mu = mu.dropna(subset=["points"])
            _teams = sorted(_mu["team"].dropna().unique())
            # imported here, not borrowed from the playoff-odds block above: that one sits
            # inside `if not _o.empty`, so it is unbound whenever the sim returns nothing
            from mega.config import MY_TEAM as _MINE
            _pick = st.multiselect(
                "Follow", _teams, key="pbw",
                default=[t for t in (_MINE,) if t in _teams] or _teams[:1],
                help="Every team is drawn; the ones you pick here are the ones named.")
            ui.line_chart(
                _mu, x="week", y="points", color="team", y_title="Points", x_title="Week",
                height=340, highlight=_pick, title="Points by week",
                caption=("The rest of the league is the grey backdrop. Flat and high beats "
                         "spiky and high — a team that swings wildly loses weeks it should win."))

        tx = _ya.transactions_df()
        if not tx.empty:
            ui.h("Recent transactions")
            ui.table(tx.head(40), rename={"team": "TEAM"}, logos=False)

# ---- Start / Sit -------------------------------------------------------------------
with tab_start:
    st.caption(
        f"Projections: FantasyPros (studs) + nflverse estimate (everyone else), "
        f"matchup-adjusted via defense-vs-position. Optimizing week {next_week}."
    )
    try:
        from mega.lineup import optimize_lineup
        rp = _my_roster_projected(int(season), int(next_week)).copy()
        # Applied outside the cached projection so an edit to data/player_status.csv
        # takes effect on the next rerun.
        from mega.status import out_for_week
        _out = out_for_week(int(next_week))
        rp["report_status"] = rp["player"].map(lambda n: _out.get(player_ids.norm(n)))
        rp.loc[rp["report_status"].notna(), "proj"] = 0.0
        lu = optimize_lineup(rp, int(season), int(next_week))
    except Exception as e:
        lu = None
        st.warning(f"Projections unavailable: {e}")

    _vg = _vegas(int(season), int(next_week))
    if _vg.empty:
        from mega import odds as _O
        st.caption(
            "Vegas column is off — " + ("no ODDS_API_KEY is set." if not _O.available()
                                        else f"no props swept for week {next_week} yet.")
        )
    else:
        _full = int(_vg["vegas_complete"].sum())
        st.caption(
            f"**Vegas** = this week's sportsbook player props scored in half-PPR: "
            f"{len(_vg):,} players priced, {_full:,} with every market posted. Lines are read as "
            "medians and corrected to means, so a projection sits above its own posted line."
        )

    if lu is not None and not lu.empty:
        starters = lu[lu["start"]]
        bench = lu[~lu["start"]]
        proj_total = starters["proj_adj"].sum()
        n_close = int((bench["close_call"] != "").sum())
        fp_cov = int((starters["proj_source"] == "FantasyPros").sum())
        ui.kpi_row([
            (f"Wk {next_week} proj total", f"{proj_total:.1f}", "optimal starting 9 (skill)"),
            ("Close calls", str(n_close), "bench within 2 pts of a starter"),
            ("FantasyPros-backed", f"{fp_cov}/{len(starters)}", "starters with FP projection"),
        ])
        st.write("")

        # Opportunity alongside the projection. Below the FantasyPros free-tier cutoff
        # (top 10 per position) everything is an estimate, and a 1-2 point gap between
        # two estimates is noise — target share is the steadier tiebreaker.
        _ts = target_share(int(season), int(week), int(roll))
        lu = lu.merge(_ts, on="gsis_id", how="left") if "gsis_id" in lu.columns else lu
        for _c in ("tgt_pct", "tm_rank"):
            if _c not in lu.columns:
                lu[_c] = pd.NA
        starters, bench = lu[lu["start"]], lu[~lu["start"]]

        slot_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLEX": 4}
        cols = [c for c in ["lineup", "player", "pos", "nfl_team", "report_status", "opp", "ease_rank",
                            "proj", "proj_adj", "proj_source", "vegas", "vegas_edge",
                            "tgt_pct", "tm_rank", "start_sit", "close_call"] if c in lu.columns]

        lu_fmt = {"PROJ": "{:.1f}", "PROJ*": "{:.1f}", "TGT%": "{:.1%}",
                  "VEGAS": "{:.1f}", "VEG±": "{:+.1f}"}
        lu_help = {"VEGAS": "Half-PPR points implied by this week's sportsbook player props.",
                   "VEG±": "Vegas minus our projection. Positive = the market likes him more "
                           "than the usage model does."}

        ui.h("✅ Recommended starters")
        sview = starters.assign(_o=starters["lineup"].map(slot_order)).sort_values("_o")[cols]
        ui.table(sview, sequential=["PROJ*"], diverging=["VEG±"], pos_cols=["POS"],
                 fmt=lu_fmt, help=lu_help, view="startsit_start")

        ui.h("🪑 Bench")
        bview = bench.sort_values("proj_adj", ascending=False)[cols]
        ui.table(bview, diverging=["VEG±"], pos_cols=["POS"], fmt=lu_fmt, help=lu_help,
                 view="startsit_bench")

        _est = int((starters["proj_source"] != "FantasyPros").sum())
        if _est:
            st.warning(
                f"**{_est} of {len(starters)} starters are running on estimates, not real projections.** "
                "FantasyPros' free tier stops at the top 10 per position, so everyone below that is "
                "modelled from usage. Two estimates within ~2 points of each other is a coin flip, not "
                "a recommendation — when it's that close, start the player with the bigger **target share** "
                "and the better **team target rank**, because volume holds up week to week and points don't."
            )
        if n_close:
            st.info("**Close calls** are flagged in the **Close call** column — a bench player projecting within ~2 points "
                    "of a starter in the same slot. Worth an injury and matchup check before lock.")

# ---- Actual vs expected ------------------------------------------------------------
with tab_axe:
    if ffo.empty:
        st.info("ff_opportunity data unavailable for this season.")
    else:
        a = ffo.groupby("gsis_id")["half_ppr_exp"].sum().rename("expected")
        b = sw.groupby("gsis_id")["half_ppr"].sum().rename("actual")
        cmp = pd.concat([a, b], axis=1).reset_index()
        cmp["player"] = cmp["gsis_id"].map(name_by_id)
        cmp["diff"] = cmp["actual"] - cmp["expected"]
        cmp = cmp.dropna(subset=["player"]).sort_values("diff")
        _long = cmp.melt(id_vars="player", value_vars=["expected", "actual"],
                         var_name="kind", value_name="pts")
        _long["kind"] = _long["kind"].map({"expected": "Expected", "actual": "Actual"})
        _long["player"] = _long["player"].map(ui.short_name)
        ui.bar_compare(
            _long, category="player", series="kind", value="pts",
            sort=cmp["player"].map(ui.short_name).tolist(),  # keeps the diff order
            domain=["Expected", "Actual"],
            palette=(ui.INK3, ui.NAVY), x_title="half-PPR points",
            title=f"Actual vs expected, through week {week}",
            caption=("Grey is what his usage should have produced; navy is what he scored. "
                     "A navy bar well short of grey is a hold, not a cut — the work is there "
                     "and the points normally follow."))
        ui.table(cmp[["player", "expected", "actual", "diff"]], diverging=["xFP±"],
                 fmt={"xFP": "{:.1f}", "ACT": "{:.1f}", "xFP±": "{:+.1f}"})

# ---- Usage trends ----------------------------------------------------------------
with tab_use:
    metric = st.selectbox("Metric", ["snap share", "target share", "half-PPR points", "targets", "carries"])
    players_sel = st.multiselect("Players", skill["name"].tolist(), default=skill[skill["slot"] != "BN"]["name"].tolist())
    sel_ids = [k for k, v in name_by_id.items() if v in players_sel]

    # A share is a fraction between 0 and 1; a count is not. One y_title used to serve
    # both, and neither axis said "%", so snap share read as 0.8 where every other surface
    # in the app says 80%.
    _pct = metric in ("snap share", "target share")
    _cap = {
        "snap share": "How much of his offence's snaps he is on the field for. The floor "
                      "under everything else — a player off the field cannot be targeted.",
        "target share": "His cut of his team's targets. The stickiest week-to-week signal "
                        "there is; points move around it, not the other way.",
        "half-PPR points": "What he actually scored. Spiky is not the same as good.",
        "targets": "Raw looks per game, before any share maths.",
        "carries": "Raw carries per game.",
    }[metric]

    if metric == "snap share" and not snaps.empty:
        pfr_map = dict(zip(skill["pfr_id"], skill["name"]))
        # honour the Players picker here too: this branch used to plot the whole roster
        # whatever was selected, so on this one metric the control did nothing
        _want = [p for p, n in pfr_map.items() if p and n in players_sel]
        s = snaps[snaps["pfr_id"].isin(_want)].copy()
        col = first_col(s, "offense_pct", "off_pct")
        if col:
            s["value"] = pd.to_numeric(s[col], errors="coerce")
            s["player"] = s["pfr_id"].map(pfr_map).map(ui.short_name)
            ui.line_chart(s.dropna(subset=["value"]), x="week", y="value", color="player",
                          y_title="Snap share", x_title="Week", percent=True, height=320,
                          title=f"Snap share by week — {season}", caption=_cap)
        else:
            st.info("No snap-share column found.")
    else:
        base = sw[sw["gsis_id"].isin(sel_ids)].copy()
        base["player"] = base["gsis_id"].map(name_by_id).map(ui.short_name)
        field = {
            "target share": "target_share", "half-PPR points": "half_ppr",
            "targets": "targets", "carries": "carries", "snap share": None,
        }[metric]
        if field and field in base.columns:
            _d = base.dropna(subset=[field])
            ui.line_chart(_d, x="week", y=field, color="player",
                          y_title=ui.title_case(metric), x_title="Week", percent=_pct,
                          height=320,
                          title=f"{ui.title_case(metric)} by week — {season}", caption=_cap)
        else:
            st.info(f"Column '{field}' not available.")

# ---- Matchups ------------------------------------------------------------------
with tab_match:
    if sched.empty:
        st.info("Schedule unavailable.")
    else:
        teams = sorted({t for t in team_by_id.values() if t})
        wk = sched[sched["week"] == next_week].copy()
        rows = []
        for _, gm in wk.iterrows():
            tl, sl = gm.get("total_line"), gm.get("spread_line")
            for side, opp_side in (("home_team", "away_team"), ("away_team", "home_team")):
                t = gm[side]
                if t not in teams:
                    continue
                is_home = side == "home_team"
                imp = None
                if pd.notna(tl) and pd.notna(sl):
                    imp = tl / 2 + (sl / 2 if is_home else -sl / 2)
                rows.append({
                    "team": t,
                    "players": ", ".join(sorted(n for i, n in name_by_id.items() if team_by_id.get(i) == t)),
                    "matchup": ("vs " if is_home else "@ ") + gm[opp_side],
                    "total": tl, "spread": sl if is_home else (-sl if pd.notna(sl) else None),
                    "implied_pts": imp,
                })
        mt = pd.DataFrame(rows).sort_values("implied_pts", ascending=False)
        ui.h(f"Week {next_week} — team game environment (Vegas)")
        st.caption("Implied team total = Vegas's expected points for that offense. Higher = more scoring to go around.")
        ui.table(mt, sequential=["IMP"], fmt={"TOT": "{:.1f}", "SPRD": "{:+.1f}", "IMP": "{:.1f}"},
                 help={"IMP": "Vegas's expected points for this offense. Higher = more scoring to go around."})

        # ---- §20 every player the book priced this week -----------------------------
        _vb = _vegas(int(season), int(next_week))
        if _vb.empty:
            from mega import odds as _O
            ui.h(f"Week {next_week} — Vegas player projections")
            st.info(
                "No player props on file for this week. " + (
                    "Set `ODDS_API_KEY` in `.env` (the free plan at the-odds-api.com covers "
                    "about one sweep of the slate per week) and the daily task will fill this in."
                    if not _O.available() else
                    "The next scheduled refresh will sweep them.")
            )
        else:
            ui.h(f"Week {next_week} — Vegas player projections")
            st.caption(
                "Half-PPR points implied by the sportsbook's own player props. A posted line is "
                "the **median** outcome, so it is corrected to a **mean** before scoring — which is "
                "why a projection sits above the line you would see on the app. "
                "**MKTS** counts how many markets were actually priced; anything the book did not "
                "post is filled from the player's own expected-points rate, and **FULL** marks the "
                "players whose number is entirely the market's."
            )
            _vb = _vb.copy()
            _own = {}
            try:
                from mega.yahoo import cached_rosters
                _r = cached_rosters()
                _own = dict(zip(_r.get("gsis_id", []), _r.get("team", [])))
            except Exception:
                pass
            from mega.config import MY_TEAM as _MINE   # imported here: the module-level
            # alias below is defined further down the file, and app.py runs top to bottom
            _vb["owner"] = _vb["gsis_id"].map(_own).fillna("FA")
            _c1, _c2 = st.columns([1, 2])
            _only_mine = _c1.checkbox("My roster only", value=False, key="vegas_mine")
            _min_mkts = _c2.slider("Minimum markets priced", 1, 5, 1, key="vegas_mkts")
            _view = _vb[_vb["vegas_parts"] >= _min_mkts]
            if _only_mine:
                _view = _view[_view["owner"] == _MINE]
            _view = _view.sort_values("vegas", ascending=False).head(200)
            ui.table(
                _view[["player", "team", "owner", "vegas", "vegas_parts", "vegas_complete"]],
                sequential=["VEGAS"], fmt={"VEGAS": "{:.1f}"}, logos=False,
                rename={"team": "TM", "owner": "OWNER"},
                help={"VEGAS": "Half-PPR points implied by this week's player props.",
                      "MKTS": "How many prop markets the book actually posted for him.",
                      "FULL": "True = every scoring market was priced; nothing was filled in."},
            )

        # per-player defense-vs-position matchup
        dvp = _dvp(int(season))
        if not dvp.empty:
            ui.h(f"Week {next_week} — player matchup difficulty (defense vs position)")
            pr = skill.rename(columns={"name": "player"})[["player", "pos", "nfl_team"]].copy()
            teamopp = {}
            for _, gm in wk.iterrows():
                teamopp[gm["home_team"]] = "vs " + str(gm["away_team"])
                teamopp[gm["away_team"]] = "@ " + str(gm["home_team"])
            pr["matchup"] = pr["nfl_team"].map(teamopp)
            pr["opp"] = pr["matchup"].str.replace("vs ", "", regex=False).str.replace("@ ", "", regex=False)
            pr = pr.merge(dvp[["defense", "pos", "pa_pg", "ease_rank"]],
                          left_on=["opp", "pos"], right_on=["defense", "pos"], how="left")
            pr["verdict"] = pd.cut(pr["ease_rank"], [0, 8, 16, 24, 32],
                                   labels=["🟢 great", "🙂 good", "😐 tough", "🔴 avoid"])
            show = pr[["player", "pos", "matchup", "ease_rank", "pa_pg", "verdict"]].sort_values("ease_rank")
            ui.table(show, pos_cols=["POS"], fmt={"PA/G": "{:.1f}"})

# ---- League intelligence (Mega Bowl) ---------------------------------------------
@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Crunching league intel…")
def _intel_bundle(season: int):
    from mega import intel as _i
    from mega import yahoo_api as _ya
    from mega.draft_board import load_draft as _ld

    draft = _ld()
    # Ownership, best source first: official API -> browser scrape -> draft board.
    ros = _ya.rosters_df() if _ya.available() else None
    roster_src = f"live Yahoo rosters (week {_ya.week()})"
    if ros is None or ros.empty:
        from mega.yahoo import cached_rosters as _cr

        ros = _cr()
        roster_src = "scraped Yahoo rosters"
    if ros is None or ros.empty:
        ros = None
        rostered = {_i._norm(p) for p in draft["player"]}
        roster_src = "draft-board approximation"
    else:
        rostered = set(ros["norm"])
    return dict(
        draft=draft,
        waivers=_waiver_board(season, ros, rostered),
        trades=_i.trade_finder(season, yahoo_rosters=ros),
        draft_delta=_i.draft_value_delta(season),
        buysell=_i.buy_low_sell_high(season),
        basis=_i.season_basis(season),
        roster_src=roster_src,
    )


def _waiver_board(season: int, ros, rostered: set) -> pd.DataFrame:
    """Free agents priced against *this* roster, not against the league in the abstract.

    Falls back to the roster-blind board if the valuation engine can't build (no rosters,
    no projections), because a worse board still beats an empty tab.
    """
    # intel is imported here, not borrowed from _intel_bundle's local scope. It was, and the
    # fallback below then raised NameError instead of falling back — which took out the whole
    # intel bundle and blanked every tab that reads it.
    from mega import intel as _i
    from mega import needs

    try:
        b = needs.board(season, current_week(season, 1), yahoo_rosters=ros, top=25)
        if not b.empty:
            return b
        st.caption("Waiver board: the valuation engine returned nothing; showing the "
                   "roster-blind board.")
    except Exception as e:
        # Say why. A silent except here once hid the real failure behind a fallback that
        # then failed for an unrelated reason.
        st.caption(f"Waiver board fell back to the roster-blind view: {type(e).__name__}: {e}")
    return _i.waiver_board(season, rostered, top=20)


try:
    IB = _intel_bundle(int(season))
except Exception as e:  # network / dependency issue — keep the core dashboard usable
    IB = None
    _intel_err = str(e)


# ---- Action board --------------------------------------------------------------------
with tab_action:
    ui.lede(
        "What's worth doing this week, and why. Everything here is pulled from the other "
        "tabs — the one idea running through it is that <b>opportunity is sticky and points "
        "are noisy</b>, so a gap between the two is usually a chance to buy or sell."
    )

    if not BASIS["current"]:
        st.info(
            f"**Form numbers below are {BASIS['season']}, not {int(season)}.**  "
            f"{int(season)} has {BASIS['weeks']} week(s) played and the rolling window needs 3 — "
            f"a {BASIS['weeks']}-week sample would read as a trend. This switches over on its own at week 3."
        )

    mine = agg.copy() if not agg.empty else pd.DataFrame()
    if not mine.empty and "xfp_diff" in mine.columns:
        mine["per_g"] = mine["xfp_diff"] / mine["games"].clip(lower=1)

        def _why_sell(r):
            bits = [f"scoring {r['per_g']:+.1f}/g more than his opportunity"]
            if pd.notna(r.get("tgt_pct")) and r["tgt_pct"] < 0.20:
                bits.append(f"only {r['tgt_pct']:.0%} of targets")
            if pd.notna(r.get("tm_rank")) and r["tm_rank"] >= 3:
                bits.append(f"#{int(r['tm_rank'])} option on his own offense")
            return "; ".join(bits)

        def _why_buy(r):
            bits = [f"scoring {abs(r['per_g']):.1f}/g less than his opportunity"]
            if pd.notna(r.get("tgt_pct")) and r["tgt_pct"] >= 0.20:
                bits.append(f"{r['tgt_pct']:.0%} target share")
            if pd.notna(r.get("tm_rank")) and r["tm_rank"] <= 2:
                bits.append(f"his team's #{int(r['tm_rank'])} option")
            return "; ".join(bits)

        acols = ["player", "pos", "slot", "half_ppr_pg", "per_g", "tgt_pct", "tm_rank", "why"]
        afmt = {"PPG": "{:.1f}", "xFP±/G": "{:+.1f}", "TGT%": "{:.1%}"}

        sell = mine[mine["per_g"] >= 2.0].sort_values("per_g", ascending=False).head(5)
        ui.h("Shop these — points are ahead of the work")
        if sell.empty:
            st.caption("Nobody on your roster is meaningfully outscoring his opportunity right now.")
        else:
            sell = sell.assign(why=sell.apply(_why_sell, axis=1))
            ui.table(sell[acols], diverging=["xFP±/G"], pos_cols=["POS"], fmt=afmt)
            st.caption("Their value to a leaguemate is at its peak. Sell the name, not the role.")

        buy = mine[mine["per_g"] <= -1.5].sort_values("per_g").head(5)
        ui.h("Hold these — the work is there, the points aren't yet")
        if buy.empty:
            st.caption("Nobody is notably underperforming his opportunity.")
        else:
            buy = buy.assign(why=buy.apply(_why_buy, axis=1))
            ui.table(buy[acols], diverging=["xFP±/G"], pos_cols=["POS"], fmt=afmt)
            st.caption("Don't sell into a cold streak — the usage says the points are coming.")
    else:
        st.info("Expected-points data isn't available yet this season, so sell/hold flags are off.")

    if IB is not None:
        wv = IB["waivers"]
        if not wv.empty and "bid" in wv.columns:
            ui.h("Best waiver claims")
            worth = wv[wv["bid"] >= 1]
            if worth.empty:
                st.caption(
                    "Nothing on the wire improves your starting lineup this week, so there's "
                    "nothing worth bidding on. Your budget keeps."
                )
            else:
                ui.table(worth.head(5)[["player", "pos", "gain", "bid", "max_bid", "drop", "why"]],
                         sequential=["GAIN"], pos_cols=["POS"],
                         fmt={"GAIN": "{:+.2f}", "BID": "${:.0f}", "MAX": "${:.0f}"})
        elif not wv.empty:
            ui.h("Best waiver claims")
            ui.table(wv.head(5)[["player", "pos", "pg_recent", "tgt_pct", "tm_rank", "add_score", "why"]],
                     sequential=["SCORE"], pos_cols=["POS"],
                     fmt={"PPG": "{:.1f}", "TGT%": "{:.1%}", "TM#": "{:.0f}", "SCORE": "{:.2f}"})

        tr = IB["trades"]
        if not tr.empty:
            ui.h("Best trades to offer")
            tr5 = tr.head(5).copy()
            tr5["give"] = tr5["give"] + " (" + tr5["give_pos"] + ")"
            tr5["get"] = tr5["get"] + " (" + tr5["get_pos"] + ")"
            tr5["addresses"] = tr5["addresses"].str.replace(r"^my (\S+) need$", r"\1", regex=True)
            ui.table(
                tr5[["partner", "give", "give_val", "get", "get_val", "addresses", "fairness"]],
                sequential=["FAIR"],
                fmt={"GIVE VAL": "{:.0f}", "GET VAL": "{:.0f}", "FAIR": "{:.2f}"},
            )
            st.caption(f"Full list, grouped by manager, under **Get better → Trade finder**. "
                       f"Rosters: {IB['roster_src']}.")
    else:
        st.warning(f"League intel unavailable: {_intel_err}")

with tab_wire:
    if IB is None:
        st.warning(f"League intel unavailable: {_intel_err}")
    else:
        wv = IB["waivers"].copy()
        need_aware = "bid" in wv.columns

        if need_aware:
            from mega import faab as _fb

            _r, _m = _fb.rivals(), _fb.market_summary()
            worth = wv[wv["bid"] >= 1]
            spec = wv[wv["bid"] < 1]
            ui.kpi_row([
                ("Your FAAB", f"${_r['mine']}" if _r.get("known") else "—",
                 f"of ${_fb.BUDGET} · {_r['richer']} of {_r['teams'] - 1} teams hold more"
                 if _r.get("known") else "balances not cached"),
                ("Worth bidding on", str(len(worth)),
                 "free agents who'd change your lineup" if len(worth) != 1
                 else "free agent who'd change your lineup"),
                ("League has paid", f"${_m['median']:.0f}" if _m.get("claims") else "—",
                 f"median of {_m['claims']} settled claims" if _m.get("claims") else "no claims yet"),
            ])
            st.write("")
            ui.lede(
                "Free agents priced against <b>your actual lineup</b>. A player is worth what he adds "
                "to an optimal starting eleven once you account for who he displaces and who you'd cut "
                "for him — so a third tight end is worth nothing here however good he is, because the "
                "flex is RB/WR and he can never start."
            )
            ui.h("Worth bidding on")
            if worth.empty:
                st.info(
                    "Nothing available improves your starting lineup this week. That is a real "
                    "answer, not a missing one — hold the budget for a week when it isn't true."
                )
            else:
                ui.table(
                    worth[["player", "pos", "nfl_team", "ppg", "gain", "bid", "max_bid", "drop", "why"]],
                    sequential=["GAIN", "BID"], pos_cols=["POS"],
                    fmt={"PPG": "{:.1f}", "GAIN": "{:+.2f}", "BID": "${:.0f}", "MAX": "${:.0f}"},
                )
                st.caption(
                    "**Bid** spends a share of your budget that scales with the points the player adds "
                    "between now and week 17; **Walk-away** is the most he could justify. Both are capped "
                    "by what you actually hold."
                )

            ui.h("Speculative")
            ui.lede(
                "These add nothing to your lineup today, so they're ranked by who is trending instead. "
                "A dollar at most, and only for a bench spot you don't mind wasting."
            )
            ui.table(
                spec[["player", "pos", "nfl_team", "ppg", "add_score", "upside", "why"]].head(15),
                sequential=["SCORE"], pos_cols=["POS"],
                fmt={"PPG": "{:.1f}", "SCORE": "{:.2f}"},
                labels={"WHY": "Why not now"},
            )
            if _m.get("unlisted_spend"):
                # Escaped dollars: Streamlit reads $...$ in markdown as LaTeX and swallows
                # both the signs and everything between them.
                st.caption(
                    rf"Yahoo's FAB feed lists only claims that went to a waiver run, so "
                    rf"\${_m['unlisted_spend']:.0f} of the \${_m['league_spend']:.0f} this league "
                    rf"has actually spent never appears on it. The real market runs dearer than "
                    rf"the \${_m['median']:.0f} median suggests."
                )
        else:
            top3 = wv.head(3)
            ui.kpi_row([
                (f"#{i+1} target", ui.short_name(r["player"]), f"{r['pos']} · {r['why'][:38]}")
                for i, (_, r) in enumerate(top3.iterrows())
            ] or [("—", "no candidates", "")])
            st.write("")
            st.caption(f"Recent form: {IB['basis']['label']} · adds via Sleeper · values via FantasyCalc")
            ui.lede(
                "Free agents ranked by <b>whether their role actually changed</b>, not just whether they had one "
                "good week. Check <b>Team tgt rank</b> and <b>Target share</b> before you spend a claim."
            )
            ui.table(
                wv.drop(columns=["norm"], errors="ignore"),
                sequential=["SCORE", "TGT%"], diverging=["TR30"], pos_cols=["POS"],
                fmt={"PPG": "{:.1f}", "TGT": "{:.1f}", "CAR": "{:.1f}", "TGT%": "{:.1%}",
                     "TM#": "{:.0f}", "VAL": "{:.0f}", "ADD#": "{:.0f}", "TR30": "{:+.0f}",
                     "SCORE": "{:.2f}"},
                view="waivers_blind",
            )
            st.caption(
                "A **#1–2 team target rank** on a rising **target share** is the strongest sign a role has "
                "genuinely changed. **Claim score** blends that with recent points, trade value and how fast "
                "he's being added elsewhere."
            )

from mega.config import MY_TEAM as MY_TEAM_LABEL   # noqa: E402  (the trade tab reads it)




@st.cache_data(ttl=dt.timedelta(minutes=30), show_spinner=False)
def _trade_pool() -> pd.DataFrame:
    """Everyone rostered who has a trade value, labelled with who owns him."""
    from mega.intel import _valued_rosters
    from mega.yahoo import cached_rosters

    r = _valued_rosters(cached_rosters())
    if r.empty:
        return r
    r = r.sort_values("value", ascending=False)
    from mega.trade_league import _pid

    who = r["team"].map(lambda t: "yours" if str(t) == MY_TEAM_LABEL else str(t))
    r["label"] = r["player"] + " · " + r["pos"].astype(str) + " · " + who
    r["pid"] = [_pid(row) for _, row in r.iterrows()]   # same id the engine keys on
    return r[["label", "player", "pos", "team", "value", "pid"]]


@st.cache_resource(ttl=dt.timedelta(minutes=30), show_spinner="Building the league's rosters…")
def _trade_engine(season: int):
    """The trade engine over this league. Cached as a resource: it holds a memo table that
    makes the search fast, and rebuilding it per rerun would throw that away."""
    from mega.trade_engine import create_engine
    from mega.trade_league import build_league, engine_config
    from mega.yahoo import cached_rosters

    league = build_league(season, cached_rosters())
    if not league["teams"]:
        return None, league["report"]
    return create_engine(league, engine_config()), league["report"]


def _ppg_note(rep: dict) -> str:
    """Where the points-per-game behind every number came from — a projection nobody can
    see the provenance of is a projection nobody should act on."""
    src = rep.get("ppg_sources") or {}
    bits = [f"**{src.get('fantasypros_ros', 0)}** from FantasyPros rest-of-season",
            f"**{src.get('nflverse', 0)}** from nflverse form (FantasyPros' free tier stops at "
            "each position's top 10)"]
    if src.get("none"):
        bits.append(f"**{src['none']}** with no projection at all"
                    + (" — " + ", ".join(rep.get("unpriced") or []) if rep.get("unpriced") else ""))
    return "Points per game behind every trade: " + " · ".join(bits) + "."


@st.cache_data(ttl=dt.timedelta(minutes=30), show_spinner="Rebuilding both rosters for every trade…")
def _trade_search(season: int, pid: str, mine: bool, flags: tuple, shapes: tuple,
                  order: str = "accept") -> dict:
    from mega.trade_league import find_for_their_player, find_from_my_player

    eng, rep = _trade_engine(season)
    if eng is None:
        return {"table": pd.DataFrame(), "results": [], "matched": 0, "evaluated": 0, "padded": 0}
    me = rep["my_team_id"]
    if mine:
        out = find_from_my_player(eng, me, [pid], includeFlags=list(flags), shapes=list(shapes), topN=40)
    else:
        out = find_for_their_player(eng, me, pid, include_flags=flags, shapes=shapes, top_n=40)
    # The engine sorts by what you gain, which puts the biggest raids on top — and those are
    # the ones nobody accepts. Default to ordering by whether the other manager would say yes,
    # then by what you gain inside each group.
    if order == "accept":
        rank = {"LIKELY": 0, "NEEDS_PITCH": 1, "EXPLOIT": 2, "LONGSHOT": 3}
        pairs = sorted(zip(out["results"], range(len(out["results"]))),
                       key=lambda rv: (rank.get(rv[0]["flag"], 9), -rv[0]["dMe"]))
        out["results"] = [r for r, _ in pairs]
        from mega.trade_league import _rows
        out["table"] = _rows(eng, out["results"])
    out["explain"] = [eng.explain(r) for r in out["results"][:3]]
    return out


with tab_trade:
    if IB is None:
        st.warning("League intel unavailable.")
    elif IB["trades"].empty:
        st.info("No trade ideas cleared the fairness filter this run.")
    else:
        ui.h("Trade around one player")
        ui.lede(
            "Pick anyone in the league. If he's <b>yours</b>, this is what could come back for him. "
            "If he's <b>someone else's</b>, it's what it would take to get him — and whether giving "
            "that up opens a hole you can't afford."
        )
        _pool = _trade_pool()
        if _pool.empty:
            st.caption("No rosters with trade values yet — run the Tuesday scrape.")
        else:
            _pick = st.selectbox("Player", _pool["label"].tolist(), index=None, key="trade_pick",
                                 placeholder="Search — e.g. Kelce, Bijan, Nabers…")
            _c1, _c2 = st.columns([2, 1])
            _flags = _c1.pills("Show", ["Likely", "Exploit", "Needs pitch"],
                               selection_mode="multi", default=["Likely", "Exploit", "Needs pitch"],
                               key="trade_flags") or ["Likely", "Exploit", "Needs pitch"]
            _two = _c2.toggle("Allow two-player packages", value=True, key="trade_two")
            _order = _c2.radio("Order by", ["Most likely accepted", "Best for me"],
                               horizontal=True, key="trade_order", label_visibility="collapsed")
            if _pick:
                _row = _pool[_pool["label"] == _pick].iloc[0]
                _mine = str(_row["team"]) == MY_TEAM_LABEL
                _flagset = tuple(f.upper().replace(" ", "_") for f in _flags)
                _shapes = ("1-for-1", "2-for-1") if _two else ("1-for-1",)
                _res = _trade_search(int(season), str(_row["pid"]), _mine, _flagset, _shapes,
                                     "accept" if _order.startswith("Most") else "gain")
                st.markdown(
                    f"**{_row['player']}** · {_row['pos']} · "
                    + ("yours" if _mine else f"on {_row['team']}")
                    + f" · {_res['evaluated']:,} trades evaluated"
                    + (f", {_res['padded']} dropped as padding" if _res["padded"] else "")
                )
                if _res["table"].empty:
                    st.info(
                        "Nothing clears the filters. Every offer has to leave your lineup better "
                        "off — try allowing two-player packages, or widen the likelihood filter."
                    )
                else:
                    _tt = _res["table"].copy()
                    # §18.3 — rank on points per week, then price the survivors in the only
                    # currency that counts. The top rows get the sim; the rest do not need it.
                    try:
                        _keyrows = tuple(
                            (i, str(r["partner"]), float(r["d_me"]), float(r["d_them"]))
                            for i, r in _tt.head(SIM_TOP).iterrows())
                        _od = _trade_odds(int(season), _keyrows)
                    except Exception:
                        _od = {}
                    if _od:
                        _tt["odds"] = _tt.index.map(lambda i: _od.get(i, {}).get("d_playoffs"))
                        _tt["their_odds"] = _tt.index.map(
                            lambda i: _od.get(i, {}).get("their_d_playoffs"))
                        _tt["watch"] = _tt.index.map(
                            lambda i: ("arms a rival" if _od.get(i, {}).get("arms_rival")
                                       else _od.get(i, {}).get("their_tag", "")))
                    _tcols = ["partner", "shape", "give", "get", "d_me", "d_them"] + \
                        (["odds", "their_odds", "watch"] if _od else []) + ["mkt_ratio", "flag"]
                    ui.table(
                        _tt[_tcols], diverging=["THEM ±", "ODDS±", "THEIR ODDS±"],
                        sequential=["YOU ±"],
                        fmt={"YOU ±": "{:+.2f}", "THEM ±": "{:+.2f}", "MARKET": "{:.2f}",
                             "ODDS±": "{:+.1%}", "THEIR ODDS±": "{:+.1%}"},
                        labels={"MANAGER": "Trade with"}, logos=False,
                    )
                    if _od:
                        st.caption(
                            "**ODDS±** is what the deal does to your chance of making the "
                            "playoffs, from 1,500 simulated seasons run on the same dice "
                            "before and after — so an offer that changes nothing reads as "
                            "exactly zero. **THEIR ODDS±** is the same for the other side; "
                            "*arms a rival* means it helps someone you are actually racing."
                        )
                    st.caption(
                        "Both rosters are rebuilt for every offer — forced back to legal size, "
                        "re-optimised, and compared with where they started. **Your lineup ±** is "
                        "what you gain per week; **their lineup ±** is what it costs them, which is "
                        "what decides whether the offer gets accepted."
                    )
                    with st.expander("The lineups behind the top offers"):
                        st.code("\n\n".join(_res["explain"]), language="text")
                    st.caption(_ppg_note(_trade_engine(int(season))[1]))
            st.divider()

        ui.h("Offers the league is set up for")
        ui.lede(
            "Built from <b>your league's actual rosters</b> — who has a surplus where you're "
            "thin, and what they're short of in return. This is the part a generic ranking site "
            "can't do for you."
        )
        tt = IB["trades"].copy()
        # Position folded into the name — two "POS" columns would collide on rename.
        tt["give"] = tt["give"] + " (" + tt["give_pos"] + ")"
        tt["get"] = tt["get"] + " (" + tt["get_pos"] + ")"
        tt["addresses"] = tt["addresses"].str.replace(r"^my (\S+) need$", r"\1", regex=True)
        tfmt = {"GIVE VAL": "{:.0f}", "GET VAL": "{:.0f}", "FAIR": "{:.2f}", "EDGE": "{:+.0f}"}
        tcols = ["give", "give_val", "get", "get_val", "addresses", "fairness", "edge"]

        by_mgr = st.toggle("Group by manager", value=True)
        if by_mgr:
            for partner, grp in tt.groupby("partner", sort=False):
                need = grp["they_need"].iloc[0] if "they_need" in grp.columns else "—"
                ui.h(f"{partner}")
                st.caption(
                    f"Thin at **{need}** — lead with that when you pitch it."
                    if need and need != "—"
                    else "No clear positional hole — pitch this one on value, not need."
                )
                ui.table(grp[tcols], sequential=["FAIR"], diverging=["EDGE"], fmt=tfmt)
        else:
            ui.table(tt[["partner"] + tcols], sequential=["FAIR"], diverging=["EDGE"], fmt=tfmt)

        st.caption(
            f"Rosters: {IB['roster_src']}. **Fairness** near 1.00 means an even swap by FantasyCalc value — "
            "offers below ~0.85 get filtered out, so everything here should at least get a reply."
        )

with tab_draft:
    if IB is None:
        st.warning("League intel unavailable.")
    else:
        dd = IB["draft_delta"]
        mine_only = st.checkbox("My picks only", value=False)
        view = dd[dd["mine"]] if mine_only else dd
        view = view[view["value"] > 0]
        dcols = ["player", "pos", "drafted_by", "round", "value", "value_delta"]
        dfmt = {"VAL": "{:.0f}", "VAL±": "{:+.0f}", "RD": "{:.0f}"}
        c1, c2 = st.columns(2)
        c1.markdown("**▲ Risers vs draft slot**")
        ui.table(view.head(15)[dcols], diverging=["VAL±"], pos_cols=["POS"], fmt=dfmt,
                 legend=False, container=c1)
        c2.markdown("**▼ Fallers vs draft slot**")
        ui.table(view.sort_values("value_delta").head(15)[dcols], diverging=["VAL±"], pos_cols=["POS"],
                 fmt=dfmt, container=c2)
        st.markdown("**Your regression watch** — actual vs expected half-PPR")
        from mega.intel import _norm as _mnorm
        bs = IB["buysell"]
        my_norm = {_mnorm(n) for n in skill["name"]}
        mine_bs = bs[bs["norm"].isin(my_norm)].sort_values("diff_pg")[
            ["player", "pos", "gms", "actual", "expected", "diff_pg", "signal"]] if not bs.empty else pd.DataFrame()
        if mine_bs.empty:
            st.caption("Not enough games yet to compare.")
        else:
            ui.table(mine_bs, diverging=["xFP±/G"], pos_cols=["POS"],
                     fmt={"ACT": "{:.1f}", "xFP": "{:.1f}", "xFP±/G": "{:+.1f}"})

with tab_arch:
    st.caption(
        "Every player scored 0–100 against the Mega Bowl **league-winner blueprint**, "
        "with thresholds prorated **per game**. WR uses a labeled first-down proxy (routes-run isn't in free data)."
    )
    with st.expander("The blueprint"):
        st.markdown(
            "- **QB** — rushing volume: **55+ rush att/season (3.2/g)** tags `RUSH`, **100+ (5.9/g)** tags `RUSH+`. A rushing stud beats a late-round pocket QB.\n"
            "- **RB** — early ADP (R1–3) + **young**: target exp years 1–3, avg winner age **25**, avoid **27+**.\n"
            "- **WR** — **sticky first-down producer** (blueprint: FD/route 12%+; here FD-rate + FD/g proxy) in the **breakout window (exp yrs 3–6)**, avoid age **32+** (except Evans/Adams).\n"
            "- **TE** — **alpha on a WR-thin, high-scoring offense**: no team WR inside top-60 ADP, **20+ team ppg**, big target share (except prime Kelce)."
        )
    try:
        arch = _archetypes(int(season))
    except Exception as e:
        arch = pd.DataFrame()
        st.warning(f"Archetype scoring unavailable: {e}")

    if not arch.empty:
        from mega.intel import _norm as _an
        my_norms = {_an(n) for n in skill["name"]}
        arch = arch.assign(mine=arch["norm"].isin(my_norms))

        # my roster's archetype fit
        ui.h("Your roster — archetype fit")
        mine = arch[arch["mine"]].sort_values("arch_fit", ascending=False)
        acols = ["player", "pos", "team", "arch_fit", "tags", "carries_pg", "tgt_share",
                 "tm_rank", "age", "exp_yrs", "why"]
        ui.table(mine[acols], sequential=["ARCH FIT"], pos_cols=["POS"], view="archetypes",
                 fmt={"ARCH FIT": "{:.0f}", "CAR": "{:.1f}", "TGT%": "{:.1%}", "AGE": "{:.0f}", "EXP": "{:.0f}"})

        # target board by position
        pos_sel = st.radio("Position", ["QB", "RB", "WR", "TE"], horizontal=True)
        rostered = {_an(p) for p in IB["draft"]["player"]} if IB is not None else set()
        pool = arch[arch["pos"] == pos_sel].copy()
        pool["status"] = pool.apply(
            lambda r: "🟡 mine" if r["mine"] else ("rostered" if r["norm"] in rostered else "🟢 available"), axis=1)
        ui.h(f"Best {pos_sel} fits — who to target")
        st.caption("🟢 available = not on any draft-board roster (verify against live adds). 🟡 mine = already yours.")
        ui.table(pool.head(20)[["status", "player", "team", "arch_fit", "tags",
                                "half_ppr_pg", "proj_ppg", "tgt_share", "tm_rank", "why"]],
                 sequential=["ARCH FIT"], fmt={"ARCH FIT": "{:.0f}", "PPG": "{:.1f}", "PROJ": "{:.1f}", "TGT%": "{:.1%}"},
                 labels={"PROJ": "Proj pts/g"}, help={"PROJ": "Projected fantasy points per game this season."})

with tab_wopr:
    st.caption(
        "**Weighted Opportunity Rating** — how much receiving opportunity each WR/TE earns "
        "(target share + air-yards share), split by who owns them. **WOPR** blends last season "
        "with this one on a 3-game prior; **xPPG±** flags points running ahead of / behind the "
        "underlying role."
    )
    try:
        W = _wopr(int(season))
    except Exception as e:
        W = None
        st.warning(f"WOPR unavailable: {e}")

    if W and not W["df"].empty:
        from mega.wopr import SCHEMA

        meta = W["meta"]
        pct = meta.get("percentiles", {})
        st.caption(
            f"Ownership: **{meta.get('ownership_source', '')}** · baseline {meta.get('base_season')} "
            f"· value axis = board VOR (preseason)."
        )

        with st.expander("How to read this — formula, tags, percentiles"):
            st.markdown(
                "**WOPR** = 1.5 × target share + 0.7 × air-yards share (numerators and denominators "
                "summed across the window, then divided — never an average of weekly ratios).\n\n"
                "**Tags** — `UNDERPRICED`: opportunity beats draft cost (board rank ≥6 worse) · "
                "`BUY_LOW`: strong role, points lagging · `SELL_HIGH`: points ahead of role · "
                "`RISER`: late-season opportunity trending up · `ROLE_JUMP`: new-season breakout · "
                "`FADE`: drafted high but thin opportunity.\n\n"
                "**Percentiles (board-matched, ≥6 games):** "
                + " · ".join(
                    f"{p} p50 {v['p50']:.2f} / p75 {v['p75']:.2f} / p90 {v['p90']:.2f}"
                    for p, v in pct.items()
                )
            )

        _mcols = ["name", "pos", "team_2026_nfl", "wopr_anchored", "wopr_posrank",
                  "board_posrank", "rank_delta", "ppg_minus_xppg", "tags"]
        _fmt = {"WOPR": "{:.3f}", "xPPG±": "{:+.1f}", "GAP": "{:+.0f}"}

        def _show(frame, cols):
            if frame is None or frame.empty:
                st.info("Nothing flagged here right now.")
                return
            ui.table(frame[cols], pos_cols=["POS"], sequential=["WOPR"], diverging=["GAP", "xPPG±"],
                     fmt=_fmt, view="wopr",
                     help={"TAGS": "Opportunity flags — see 'How to read this' above."})

        ui.h("Your WR/TE — sell / hold")
        st.caption("`SELL_HIGH` / `FADE` = points ran ahead of opportunity, shop them. "
                   "`BUY_LOW` = hold, don't sell low.")
        _show(W["mine"], _mcols)

        ui.h("Trade targets on other rosters")
        st.caption("Players whose opportunity outstrips their price or is trending up — grouped by manager.")
        _show(W["opp"], ["owner"] + _mcols)

        ui.h("Waiver adds (free agents)")
        st.caption("Unrostered WR/TE clearing a startable opportunity bar or jumping in role.")
        _show(W["fa"], _mcols)

        if not W["unknown"].empty:
            ui.h("Match review (unmatched)")
            ui.table(W["unknown"][["name", "pos", "tags"]], legend=False)

        _full = W["df"][[c for c in SCHEMA if c in W["df"].columns]]
        st.download_button(
            "Download full WOPR table (CSV)", _full.to_csv(index=False),
            file_name=f"wopr_targets_{int(season)}.csv", mime="text/csv",
        )

with tab_digest:
    ui.lede(
        "One page you can read on Tuesday morning — waivers, trades, and regression watch "
        "in plain text. Generate it here, or run <code>tuesday.py --email</code> to have it sent."
    )
    if st.button("Build this week's digest", type="primary"):
        try:
            from mega.digest import build_digest, write_digest
            from mega.yahoo import cached_rosters as _cr2

            with st.spinner("Crunching…"):
                md = build_digest(int(season), yahoo_rosters=_cr2())
            path = write_digest(md, int(season))
            st.success(f"Saved to `{path}`")
            st.download_button("Download the digest (.md)", md,
                               file_name=Path(path).name, mime="text/markdown")
            st.markdown("---")
            st.markdown(md)
        except Exception as e:
            st.error(f"Digest failed: {e}")
    else:
        _prev = sorted(Path(HERE / "data").glob("digest_*.md"))
        if _prev:
            st.caption(f"Last built: `{_prev[-1].name}`")
            with st.expander("Show the last one"):
                st.markdown(_prev[-1].read_text(encoding="utf-8"))


with tab_news:
    from mega.sources import news_for_players, news_items

    scope = st.radio("Scope", ["My roster", "Watchlist + roster", "All NFL"], horizontal=True)
    watch = [w.strip() for w in st.text_input("Watchlist (comma-separated)", "").split(",") if w.strip()]
    names = list(skill["name"])
    if scope == "Watchlist + roster":
        names += watch
    try:
        nn = news_items() if scope == "All NFL" else news_for_players(names)
        st.dataframe(nn[["published", "source", "title", "link"]].head(60) if not nn.empty else pd.DataFrame(),
                     width="stretch", hide_index=True,
                     column_config={
                         "published": st.column_config.DatetimeColumn("Published", format="MMM D, h:mm a"),
                         "source": st.column_config.Column("Source", width="small"),
                         "title": st.column_config.Column("Headline", width="large"),
                         "link": st.column_config.LinkColumn("Link", display_text="Open"),
                     })
    except Exception as e:
        st.warning(f"News feeds unavailable: {e}")


# ---- Player lookup -------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner="Indexing players…")
def _player_index(cur: int) -> pd.DataFrame:
    from mega.lookup import player_index
    return player_index({cur: load_player_stats(cur), cur - 1: load_player_stats(cur - 1)},
                        load_rosters(cur))


@st.cache_data(ttl=CACHE_TTL, show_spinner="Crunching the season…")
def _season_table(season: int) -> pd.DataFrame:
    from mega.lookup import season_table
    snaps_s = load_snaps(season)
    return season_table(load_player_stats(season), load_ff_opportunity(season), snaps_s,
                        player_ids.crosswalk(), load_routes(season))


@st.cache_data(ttl=dt.timedelta(minutes=30), show_spinner=False)
def _ownership() -> tuple[dict, dict]:
    """gsis_id -> (fantasy team, slot) from the scraped rosters, and gsis_id -> Yahoo FA
    status. Both go through the id resolver: matching the FA list by name missed players
    Yahoo and nflverse spell differently (Joshua vs Josh Palmer)."""
    from mega.intel import current_rosters
    from mega.yahoo import cached_rosters

    r = current_rosters(cached_rosters())
    own = {g: (t, sl) for g, t, sl in zip(r["gsis_id"], r["team"], r["slot"]) if pd.notna(g) and g}
    fa, path = {}, HERE / "data" / "yahoo_free_agents.csv"
    if path.is_file():
        f, _ = player_ids.resolve(pd.read_csv(path, dtype=str).fillna(""), name_col="player")
        fa = {g: st_ for g, st_ in zip(f["gsis_id"], f["roster_status"]) if pd.notna(g) and g}
    return own, fa


def _height(v) -> str:
    try:
        v = int(float(v))
        return f"{v // 12}'{v % 12}\""
    except (TypeError, ValueError):
        return ""


with sec_players:
    from mega import glossary as GL
    from mega import logos as _logos
    from mega import lookup as LK
    from mega.config import MY_TEAM
    from mega.status import note_for, out_for_week

    tab_lookup, tab_gloss = st.tabs(["Player Lookup", "Glossary"])

    with tab_gloss:
        ui.lede(
            "Every tag the dashboard puts next to a player, in plain English. "
            "If a label anywhere needs this page to make sense, that is a fault in the "
            "label \u2014 tell me and I will fix the wording, not the glossary."
        )
        st.markdown(GL.HEADLINE)
        st.write("")
        _g = GL.frame()
        for _grp, _title, _lede in (
            ("Role", "Roles \u2014 the job he has",
             "One per player. Worked out from his last three games, not from where he was "
             "drafted, so it changes during the season when his usage does."),
            ("Flag", "Flags \u2014 what he is doing well",
             "A player can carry several. These are measured against others in the SAME "
             "role, so a third receiver is judged against other third receivers."),
            ("How long", "How long it has held",
             "The difference between a pattern and a good afternoon."),
            ("Vegas", "Vegas — what the betting market says",
             GL.VEGAS_HEADLINE),
        ):
            ui.h(_title, 5)
            st.caption(_lede)
            _sub = _g[_g["group"] == _grp][["tag", "what it means", "why it matters"]]
            st.dataframe(_sub, width="stretch", hide_index=True,
                         column_config={
                             "tag": st.column_config.TextColumn("TAG", width=150),
                             "what it means": st.column_config.TextColumn("WHAT IT MEANS", width=380),
                             "why it matters": st.column_config.TextColumn("WHY IT MATTERS", width=340),
                         })
            st.write("")
        st.caption(
            "Roles and flags are computed once and shared, so the tag beside a player in "
            "the Waiver Wire, the Trade Finder and his own card is always the same tag."
        )

with tab_lookup:
    ui.lede(
        "Look up any QB, RB, WR or TE — who has him in Mega Bowl, how he's actually being used, "
        "and every game he's played. <b>Type part of a name.</b>"
    )
    _cur = int(season)
    try:
        _idx = _player_index(_cur)
    except Exception as e:
        _idx = pd.DataFrame()
        st.warning(f"Player index unavailable: {e}")

    if not _idx.empty:
        c_pick, c_season = st.columns([3, 1])
        _pick = c_pick.selectbox("Player", _idx["label"].tolist(), index=None, key="lookup_pick",
                                 placeholder="Search — e.g. Puka, Bijan, Kelce…")
        _season_pick = c_season.segmented_control("Season", [_cur, _cur - 1], default=_cur,
                                                  key="lookup_season") or _cur
        _mine = _idx[_idx["gsis_id"].isin(gsis_list)]["label"].tolist()
        _quick = st.pills("Or pick one of yours", _mine, key="lookup_quick") if _mine else None
        _label = _pick or _quick

        if not _label:
            st.caption("Nothing selected yet. Everyone on an NFL roster is searchable, plus anyone "
                       f"who played in {_cur - 1}.")
        else:
            prow = _idx[_idx["label"] == _label].iloc[0]
            gid, ppos, pteam = prow["gsis_id"], prow["pos"], prow["team"]
            b = LK.bio(gid, load_rosters(_cur))

            # ---- who he is, and whose he is
            own, fa = _ownership()
            n = player_ids.norm(prow["name"])
            if gid in own:
                t, sl = own[gid]
                owner = f"<b>{'Yours' if t == MY_TEAM else t}</b> · {'bench' if sl == 'BN' else sl}"
            elif str(fa.get(gid, "")).startswith("W"):
                owner = f"<b>On waivers</b> until {str(fa[gid])[1:].strip(' ()')}"   # "W (Sep 19)"
            elif not pteam:
                owner = "<b>No NFL team</b>"
            else:
                owner = "<b>Free agent</b> in Mega Bowl"
            _out_now = out_for_week(int(next_week)).get(n)
            nfl_status = LK.STATUS_WORDS.get(str(b.get("status") or ""), str(b.get("status") or ""))
            if not inj.empty and "gsis_id" in inj.columns:
                _ir = inj[inj["gsis_id"] == gid].sort_values("week").tail(1)
                if not _ir.empty and pd.notna(_ir["report_status"].iloc[0]):
                    nfl_status += f" · wk {int(_ir['week'].iloc[0])} report: {_ir['report_status'].iloc[0]}"
            status_html = (f'<span class="mb-flag">{_out_now}</span>' if _out_now else "") + nfl_status

            facts = [ppos]
            if b.get("jersey_number") and pd.notna(b.get("jersey_number")):
                facts.append(f"#{int(float(b['jersey_number']))}")
            if b.get("age"):
                facts.append(f"age {b['age']}")
            hw = " ".join(x for x in (_height(b.get("height")),
                                      f"{int(b['weight'])} lb" if pd.notna(b.get("weight")) else "") if x)
            if hw:
                facts.append(hw)
            if b.get("college"):
                facts.append(str(b["college"]).split(";")[0])
            if pd.notna(b.get("draft_number")):
                facts.append(f"pick {int(b['draft_number'])} ({int(b['entry_year'])})"
                             if pd.notna(b.get("entry_year")) else f"pick {int(b['draft_number'])}")
            elif pd.notna(b.get("entry_year")):
                facts.append(f"undrafted ({int(b['entry_year'])})")
            logo = _logos.logo_url(pteam)
            logo_html = f'<img src="{logo}" class="mb-card-logo" alt="{pteam}">' if logo else ""
            team_txt = pteam or f"no team (last: {prow.get('last_team') or '—'})"

            h_img, h_txt = st.columns([1, 5])
            if isinstance(b.get("headshot_url"), str) and b["headshot_url"].startswith("http"):
                h_img.image(b["headshot_url"], width=110)
            h_txt.markdown(
                f'<div class="mb-card"><div class="mb-card-name">{prow["name"]}</div>'
                f'<div class="mb-card-team">{logo_html}<b>{team_txt}</b></div>'
                f'<div class="mb-card-facts">{" · ".join(facts)}</div>'
                f'<div class="mb-card-row"><span>Mega Bowl</span>{owner}</div>'
                f'<div class="mb-card-row"><span>NFL status</span>{status_html}</div></div>',
                unsafe_allow_html=True,
            )

            # ---- season at a glance
            T = _season_table(int(_season_pick))
            me = T[T["gsis_id"] == gid] if not T.empty else pd.DataFrame()
            if me.empty:
                st.info(f"No {_season_pick} regular-season games for {prow['name']}.")
            else:
                r = me.iloc[0]
                rk = LK.ranks(T, gid, ppos)
                usage = {"QB": ("att_pg", "Pass att/g", "{:.1f}"), "RB": ("rush_share", "Carry share", "{:.0%}")}.get(
                    ppos, ("tgt_share", "Target share", "{:.1%}"))
                fmt = lambda c, f: (f.format(r[c]) if pd.notna(r.get(c)) else "—")
                ui.kpi_row([
                    ("Games", f"{int(r['games'])}", f"{_season_pick} regular season"),
                    ("Pts/game", fmt("pts_pg", "{:.1f}"), rk.get("pts_pg", "not enough games to rank")),
                    ("Expected/g", fmt("xfp_pg", "{:.1f}"), rk.get("xfp_pg", "what his usage should score")),
                    ("Vs expected/g", fmt("vs_exp_pg", "{:+.1f}"),
                     "running hot" if (r.get("vs_exp_pg") or 0) >= 2 else
                     "due to bounce back" if (r.get("vs_exp_pg") or 0) <= -1.5 else "about what his role earns"),
                    (usage[1], fmt(usage[0], usage[2]), rk.get(usage[0], "")),
                ])
                st.write("")

                # ---- role context (HANDOFF §12): what he is, and how his usage reads
                # against the baseline for that role and the NFL average at his position.
                # A 20% target share is a strong WR3 and a poor WR1 — the number alone
                # cannot tell you which, so the baselines sit beside it.
                if int(_season_pick) == _cur:
                    from mega import roles as RL
                    try:
                        _RC = _role_ctx(_cur)
                    except Exception as _e:
                        _RC = None
                        st.caption(f"Role context unavailable: {_e}")
                    if _RC is not None and not _RC["table"].empty:
                        _rrow = _RC["table"][_RC["table"]["gsis_id"] == gid]
                        if not _rrow.empty:
                            from mega import glossary as GL

                            _r0 = _rrow.iloc[0]
                            ui.h("Role", 5)
                            # The tag, not a paragraph. What each one means lives once, in
                            # Players -> Glossary; repeating it on every card is noise the
                            # second time you read it.
                            st.markdown("### " + GL.cell(
                                _r0.get("role"),
                                _r0.get("flags") if isinstance(_r0.get("flags"), list) else [],
                                _r0.get("tags") if isinstance(_r0.get("tags"), dict) else {}))
                            _src = _r0.get("role_src")
                            st.caption(
                                (f"From his last {int(_r0.get('games') or 0)} games."
                                 if _src == "usage" else
                                 "Too few games to read his usage, so this is his depth-chart spot.")
                                + "  Tags explained under **Players → Glossary**.")
                            _cd = RL.card(_RC["table"], _RC["baselines"], gid)
                            if not _cd.empty:
                                _fmts = dict(zip(_cd["metric"], _cd["_fmt"]))
                                _show = _cd.drop(columns=["_fmt"])
                                _val_cols = [c for c in _show.columns if c != "metric"]
                                _sty = _show.style.format(
                                    {c: (lambda v, c=c: "—" if pd.isna(v) else
                                         (f"{v:.0f}" if c.startswith("vs ") else f"{v:,.3f}"))
                                     for c in _val_cols})
                                st.dataframe(_sty, width="stretch", hide_index=True,
                                             column_config={
                                                 "vs role": st.column_config.NumberColumn(
                                                     "VS ROLE", help="100 = average for his role. "
                                                     "120+ is the §12.5 flag line."),
                                                 "vs NFL": st.column_config.NumberColumn(
                                                     "VS NFL", help="100 = the NFL average at his "
                                                     "position, across every player-game this season."),
                                             })
                                st.caption(
                                    "Shrunk toward his role's baseline, so a two-game sample is "
                                    "pulled to the middle rather than believed. **vs role** and "
                                    "**vs NFL** are 100 at the average."
                                )
                st.write("")

                # ---- this week
                if int(_season_pick) == _cur and pteam:
                    wk = sched[(sched["week"] == next_week) & ((sched["home_team"] == pteam) | (sched["away_team"] == pteam))]
                    if wk.empty:
                        st.caption(f"**Week {next_week}:** bye.")
                    else:
                        g0 = wk.iloc[0]
                        opp = g0["away_team"] if g0["home_team"] == pteam else g0["home_team"]
                        bits = [f"**Week {next_week}:** {'vs' if g0['home_team'] == pteam else '@'} {opp}"]
                        try:
                            dv = _dvp(_cur)
                            m = dv[(dv["defense"] == opp) & (dv["pos"] == ppos)]
                            if not m.empty:
                                bits.append(f"matchup #{int(m['ease_rank'].iloc[0])} of 32 for {ppos}s (1 = easiest)")
                        except Exception:
                            pass
                        if _out_now:   # ruled out: a projection would only mislead
                            bits.append(f"**{_out_now}** — {note_for(prow['name']) or 'ruled out'}")
                        else:
                            try:
                                pj = _blended_proj(_cur, int(next_week))
                                pr = pj[pj["gsis_id"] == gid] if "gsis_id" in pj.columns else pd.DataFrame()
                                if not pr.empty and pd.notna(pr["proj"].iloc[0]):
                                    bits.append(f"projection {float(pr['proj'].iloc[0]):.1f} ({pr['proj_source'].iloc[0]})")
                            except Exception:
                                pass
                        st.markdown(" · ".join(bits))

                # ---- advanced
                ui.h("Season detail")
                # The stat names are values in a column, not headers, so they don't pass through
                # col_config — title_case them here so the card reads like every other table.
                card = pd.DataFrame([
                    {"stat": ui.title_case(lab),
                     "value_fmt": (f.format(r[c]) if pd.notna(r.get(c)) else "—"),
                     "pos_rank": rk.get(c, "—"), "means": means}
                    for c, lab, f, means in LK.CARD.get(ppos, [])
                ])
                ui.table(card, legend=False, logos=False)

                # ---- game log
                ui.h("Game log")
                pfr = r.get("pfr_id") if "pfr_id" in r.index else None
                log = LK.game_log(load_player_stats(int(_season_pick)), gid,
                                  load_ff_opportunity(int(_season_pick)), load_snaps(int(_season_pick)),
                                  load_schedule(int(_season_pick)), pfr if isinstance(pfr, str) else None,
                                  load_routes(int(_season_pick)))
                if log.empty:
                    st.caption("No games yet.")
                else:
                    lcols = [c for c in LK.LOG_COLS.get(ppos, LK.LOG_COLS["WR"]) if c in log.columns]
                    ui.table(log[lcols], diverging=["xFP±"], logos=False,
                             fmt={"SNAP%": "{:.0%}", "TGT%": "{:.0%}", "PTS": "{:.1f}", "xFP": "{:.1f}",
                                  "xFP±": "{:+.1f}", **{c: "{:.0f}" for c in
                                  ("CMP", "ATT", "PASSYD", "PASSTD", "INT", "SK", "CARRIES", "RUSHYD", "RUSHTD",
                                   "TARGETS", "REC", "RECYD", "RECTD", "AIRYD", "YAC", "RTE", "1D")}})
                    if len(log) >= 2:
                        lg = pd.concat([
                            log[["week", "half_ppr"]].rename(columns={"half_ppr": "pts"}).assign(series="Actual pts"),
                            log[["week", "xfp"]].rename(columns={"xfp": "pts"}).assign(series="Expected pts"),
                        ]).dropna(subset=["pts"])
                        # navy = actual, grey = expected, exactly as in the season
                        # comparison under My Team — one encoding for one idea
                        ui.line_chart(lg, x="week", y="pts", color="series",
                                      y_title="Half-PPR points", x_title="Week",
                                      palette=[ui.NAVY, ui.INK3],
                                      title="Actual vs expected, week by week",
                                      caption=("Navy is what he scored, grey is what his "
                                               "usage was worth. Navy under grey for weeks "
                                               "on end is a player due a correction up."))

# ---- Raw ---------------------------------------------------------------------
with tab_raw:
    st.write("Mapped roster")
    st.caption(f"Yahoo → nflverse id match: {player_ids.report_line(MATCH)}")
    st.dataframe(mapped, width="stretch", hide_index=True)
    st.download_button("weekly player_stats (your roster)", sw.to_csv(index=False), "player_stats.csv")
    if not ffo.empty:
        st.download_button("ff_opportunity (your roster)", ffo.to_csv(index=False), "ff_opportunity.csv")
    with st.expander("Coverage notes"):
        st.markdown(
            "- Kicker (Tyler Loop) and DST (Ravens) are **not** in the offensive datasets. "
            "Use `nfl.load_team_stats()` / special-teams play-by-play for those, or track them manually.\n"
            "- Snap share joins via `pfr_id` from `ff_playerids`; rookies can lag a week.\n"
            "- `xFP` uses the nflverse `ff_opportunity` model scored with the half-PPR weights in `SCORING`."
        )


with sec_ask:
    from mega import ask as ASK

    ui.lede(
        "Ask for any list this data can produce — <b>“list WRs by snap %”</b>, “top 10 RB by "
        "targets last 3 weeks”, “who leads the Rams in target share”. It repeats the question "
        "back in plain English before answering, so you can see how it was read."
    )

    _aq = st.text_input("Question", key="ask_q", label_visibility="collapsed",
                        placeholder="list WRs by snap %")
    _all_yrs = list(range(int(season), 2014, -1))
    _yrs = st.multiselect("Seasons", _all_yrs, default=[int(season)], key="ask_years",
                          help="Pick two or more to compare them side by side, with the "
                               "change between the outer two. Naming a year in the question "
                               "itself (\u201ctargets in 2024\u201d, \u201c2023-2025\u201d) "
                               "overrides this.")
    _ax = st.pills("Or start from one of these", list(ASK.EXAMPLES), key="ask_ex")
    _text = (_aq or "").strip() or (_ax or "")

    if not _text:
        st.caption("Type a question, or pick an example. Positions, NFL teams, week windows "
                   "(“last 3 weeks”, “in week 2”), a minimum (“min 20 targets”) and "
                   "“my team” / “free agents” all work.")
    else:
        try:
            _pw = _ask_pw(int(season))
        except Exception as e:
            _pw = pd.DataFrame()
            st.warning(f"Could not build the player-week index: {e}")

        _wks = tuple(sorted(int(w) for w in _pw["week"].dropna().unique())) if not _pw.empty else ()
        try:
            _own, _ = _ownership()
        except Exception:
            _own = {}

        try:
            _res = ASK.answer(_pw, _text, weeks_available=_wks, mine=set(gsis_list),
                              rostered={g: t for g, (t, _sl) in _own.items()},
                              loader=_nflverse_table,
                              xwalk=player_ids.crosswalk(),
                              pw_loader=_ask_pw, default_season=int(season),
                              default_seasons=tuple(_yrs))
        except ASK.AskError as e:
            _res = None
            st.warning(str(e))

        if _res is not None:
            _hdr = ASK.headers(_res.query)
            st.caption(f"**Read as:** {_res.restated}")
            if _res.df.empty:
                st.info("No rows. " + (" ".join(_res.warnings) or "Try widening the filters."))
            else:
                if len(_res.query.seasons) > 1:      # one column per season, plus the move
                    _cmp = {**_hdr, "change": "CHANGE",
                            **{str(y): str(y) for y in _res.query.seasons}}
                    ui.table(_res.df, rename=_cmp, fmt=_res.fmt,
                             sequential=[str(_res.query.seasons[-1])],
                             diverging=["CHANGE"], legend=False)
                else:
                    _metric = _hdr[_res.query.field.key]
                    ui.table(_res.df, rename=_hdr,
                             fmt={_metric: _res.query.field.fmt},
                             sequential=[_metric], legend=False)
                st.download_button("Download this answer (.csv)",
                                   _res.df.to_csv(index=False),
                                   "answer.csv", key="ask_dl")
            for _w in _res.warnings:
                st.caption("⚠︎ " + _w)
            if _res.note:
                st.caption(_res.note)

    with st.expander("Role baselines — what an average WR1, TE1 or lead back looks like"):
        from mega import roles as RL

        try:
            _RCB = _role_ctx(int(season))
        except Exception as _e:
            _RCB = None
            st.caption(f"Unavailable: {_e}")
        if _RCB is not None and _RCB["baselines"]:
            st.caption(
                "Every figure is a summed numerator over a summed denominator across every "
                "player-game in the group this season — never the average of weekly rates. "
                "These are the numbers a player's own usage is read against."
            )
            _bt1, _bt2 = st.tabs(["By role", "NFL average by position"])
            _cols = ["target_share", "air_share", "snap_pct", "carry_share",
                     "inside10_share", "tprr", "fd_rr", "xfp_share"]
            _pcts = {"target_share", "air_share", "snap_pct", "carry_share",
                     "inside10_share", "xfp_share"}
            _lbl = {m.key: m.label for m in RL.METRICS}

            def _show_base(which: str, first: str):
                _b = RL.baseline_table(_RCB["baselines"], which)
                if _b.empty:
                    st.info("Not enough games yet to build baselines.")
                    return
                _keep = [first, "players"] + [c for c in _cols if c in _b.columns]
                _d = _b[_keep].rename(columns={**_lbl, first: first.upper(),
                                               "players": "PLAYERS"})
                st.dataframe(
                    _d.style.format({_lbl[c]: ("{:.1%}" if c in _pcts else "{:.3f}")
                                     for c in _cols if c in _b.columns}, na_rep="—"),
                    width="stretch", hide_index=True)

            with _bt1:
                _show_base("role", "role")
                st.caption(
                    "Roles come from the last 3 games played (§12.1): WRs rank by target "
                    "share on their own team, tight ends split on whether they are actually "
                    "targeted (12%), backs on carry share. A player with under 2 games is "
                    "placed from the depth chart instead."
                )
            with _bt2:
                _show_base("pos", "pos")
                st.caption("Across every player-game at the position this season — the line "
                           "a player's **vs NFL** index is measured against.")

    with st.expander("What can I ask?"):
        from mega import catalog as CAT

        _t1, _t2, _t3 = st.tabs(["Curated stats", "Every nflverse column", "Tables"])
        with _t1:
            st.caption("These are computed the way this app computes them everywhere else — "
                       "the right denominator, a sensible minimum volume, the positions the "
                       "stat applies to. Say any spelling in the last column.")
            st.dataframe(ASK.catalogue(), width="stretch", hide_index=True)
            st.caption(
                "Rates are summed numerator over summed denominator — a player's season "
                "catch rate is his catches over his targets, never the average of his "
                "weekly rates. Routes are estimated (snap share × team dropbacks), so read "
                "route-based rates as a ranking rather than a measurement."
            )
        with _t2:
            _sc = CAT.schema()
            st.caption(f"Anything else in nflverse — {sum(r['n_cols'] for r in _sc.values()):,} "
                       f"columns across {len(_sc)} tables — is reachable by name. Those are "
                       "aggregated generically (summed, or averaged when the name says the "
                       "column is already a rate), and the answer says so.")
            _find = st.text_input("Search columns", key="ask_colsearch",
                                  placeholder="separation, air yards, epa, cushion…")
            _cols = CAT.columns(query=_find)
            st.caption(f"{len(_cols):,} column(s)")
            st.dataframe(_cols.head(400), width="stretch", hide_index=True)
        with _t3:
            st.caption("Where it all comes from. Rebuild the catalogue with "
                       "`PYTHONPATH=. .venv/bin/python tools/build_schema.py` when nflverse "
                       "adds columns.")
            st.dataframe(CAT.tables(), width="stretch", hide_index=True)
