"""
Fantasy football dashboard for a single Yahoo team (default: half-PPR).

Data source: nflverse-data (https://github.com/nflverse/nflverse-data) via nflreadpy.
Run:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import difflib
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
def load_ids() -> pd.DataFrame:
    df = to_pandas(nfl.load_ff_playerids())
    keep = [c for c in ["name", "gsis_id", "pfr_id", "position", "team", "sleeper_id", "yahoo_id"] if c in df.columns]
    df = df[keep].copy()
    df["norm"] = df["name"].map(norm_name)
    return df


# --------------------------------------------------------------------------------------
# Roster mapping
# --------------------------------------------------------------------------------------
def map_roster(roster: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    rows = []
    norms = ids["norm"].tolist()
    for _, r in roster.iterrows():
        gsis = str(r.get("gsis_id") or "").strip()
        hit = ids[ids["gsis_id"] == gsis] if gsis else ids.iloc[0:0]
        if hit.empty:
            n = norm_name(r["name"])
            hit = ids[ids["norm"] == n]
            if hit.empty:
                close = difflib.get_close_matches(n, norms, n=1, cutoff=0.87)
                if close:
                    hit = ids[ids["norm"] == close[0]]
        if len(hit) > 1 and str(r.get("pos")):
            pref = hit[hit["position"].astype(str).str.upper() == str(r["pos"]).upper()]
            hit = pref if not pref.empty else hit
        rec = dict(r)
        # The roster scrape leaves pos blank on most bench rows, and a player with no
        # position can't be slotted into a lineup at all — he silently vanishes from
        # Start/Sit rather than competing for the flex. Backfill from ff_playerids.
        if not str(rec.get("pos") or "").strip() and not hit.empty and "position" in hit:
            rec["pos"] = str(hit["position"].iloc[0] or "").upper()
        # Same for the NFL team: without it there's no opponent to look up, so the
        # matchup adjustment silently does nothing for that player.
        if not str(rec.get("nfl_team") or "").strip() and not hit.empty and "team" in hit:
            rec["nfl_team"] = str(hit["team"].iloc[0] or "").upper()
        rec["gsis_id"] = hit["gsis_id"].iloc[0] if not hit.empty else None
        rec["pfr_id"] = hit["pfr_id"].iloc[0] if (not hit.empty and "pfr_id" in hit) else None
        rec["matched_name"] = hit["name"].iloc[0] if not hit.empty else None
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Mega Bowl · Command Center", page_icon="🏈", layout="wide")

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
            out = r.rename(columns={"player": "name"})[["name", "slot", "pos", "nfl_team"]].copy()
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
    ids = load_ids()
    stats = load_player_stats(int(season))
    max_wk = int(stats["week"].max()) if not stats.empty else 1
    if max_wk <= 1:
        week = max_wk
        st.caption(f"Through week {week} (only week {max_wk} available)")
    else:
        week = st.slider("Through week", 1, max_wk, max_wk)
    next_week = st.slider("Matchup week", 1, 18, min(max_wk + 1, 18))
    roll = st.slider("Rolling window (weeks)", 2, 6, 3)
    if st.button("♻️ Clear data cache"):
        st.cache_data.clear()
        try:
            nfl.clear_cache()
        except Exception:
            pass
        st.rerun()
    st.caption("Positions")
    ui.pos_legend()

mapped = map_roster(roster_raw, ids)
# Filter on slot too — the scrape leaves pos blank on plenty of rows, so a pos-only
# test lets the kicker and defense through into the skill-player views.
skill = mapped[~mapped["pos"].isin(["K", "DEF"]) & ~mapped["slot"].isin(["K", "DEF"])].copy()
unmatched = skill[skill["gsis_id"].isna()]["name"].tolist()
gsis_list = [g for g in skill["gsis_id"].dropna().tolist()]

if unmatched:
    st.warning("Unmatched (add a `gsis_id` in roster.csv): " + ", ".join(unmatched))

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

ui.masthead([
    "Half-PPR · team <strong>TaylorMade</strong>",
    f"{int(season)} season · through wk {week}",
    f"matchup wk {next_week}",
    f"roster: {ROSTER_SRC}",
])

# Grouped by the decision you're making, not by where the data came from. Streamlit
# tabs are containers, so the `with tab_*:` bodies further down render into these
# wherever they appear in the file.
sec_now, sec_team, sec_get, sec_league, sec_more = st.tabs(
    ["This week", "My team", "Get better", "League", "More"]
)
with sec_now:
    tab_action, tab_start, tab_match = st.tabs(["Action board", "Start / Sit", "Matchups"])
with sec_team:
    tab_over, tab_axe, tab_use = st.tabs(["Roster", "Points vs opportunity", "Usage trends"])
with sec_get:
    tab_wire, tab_trade, tab_wopr, tab_arch = st.tabs(
        ["Waiver wire", "Trade finder", "WOPR", "Archetypes"])
with sec_league:
    tab_league, tab_draft = st.tabs(["Standings", "Draft value"])
with sec_more:
    tab_digest, tab_news, tab_raw = st.tabs(["Weekly digest", "News", "Raw data"])


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
             "＋ overperforming · − buy-low" if xdelta is not None else "xFP n/a"),
            ("Top starter", ui.short_name(best["player"]) if best is not None else "—",
             f"{best['roll_pg']:.1f}/g" if best is not None else ""),
            ("Coldest starter", ui.short_name(worst["player"]) if worst is not None else "—",
             f"{worst['roll_pg']:.1f}/g" if worst is not None else ""),
            ("Injury flags", str(n_hurt), "starters Q or worse"),
        ])
        st.write("")

        show_cols = [c for c in [
            "slot", "player", "games", "half_ppr_pg", "roll_pg", "last_wk",
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
            diverging=["xFP±"], sequential=["TGT%"], pos_cols=["SLOT"],
            fmt={c: "{:.1f}" for c in ["PPG", roll_lbl, "LAST", "xFP", "xFP±", "TGT", "CAR", "IMP"]}
                | {"TGT%": "{:.1%}", "TM#": "{:.0f}"},
            help={roll_lbl: f"points per game, last {roll} weeks"},
        )
        ui.col_key("G", "PPG", "xFP", "xFP±", "TGT", "TGT%", "TM#", "CAR", "ST", "OPP", "IMP",
                   **{roll_lbl: f"points per game, last {roll} weeks"})
        st.caption(
            f"**TGT%** and **TM#** cover the last {roll} weeks — his share of his own offense's targets, and where "
            "that ranks him on the team. **TM# 1** on a high **TGT%** is a true alpha. A good **PPG** on a low "
            "**TGT%** is usually touchdown luck that won't hold. **xFP±**: amber = outscoring his opportunity "
            "(sell high) · blue = underperforming it (buy low / hold)."
        )

# ---- League (live Yahoo API) ---------------------------------------------------------
with tab_league:
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
        st.markdown("#### Standings")
        st.dataframe(
            ui.style_df(
                ui.cols(_scraped[["rank", "team", "manager", "wins", "losses", "ties"]],
                        team="TEAM", manager="MGR"),
                fmt={c: "{:.0f}" for c in ["RANK", "W", "L", "T"]},
            ),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Points for / against and weekly results need the Yahoo API, which is "
            "currently blocked — see the note below. Records and ranks come from the scrape."
        )

        st.markdown("#### Power rankings — who's actually good")
        ui.lede(
            "Every roster scored by what its <b>best legal lineup</b> is worth per game, ignoring "
            "record entirely. <b>LUCK</b> is the gap between where a team sits and how good it is: "
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
                fmt={"LINEUP": "{:.1f}", "BENCH": "{:.1f}", "PWR": "{:.0f}",
                     "RANK": "{:.0f}", "LUCK": "{:+.0f}", "MATCHED": "{:.0f}"},
            )
            ui.col_key("PWR", "LINEUP", "BENCH", "LUCK", "MATCHED")
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

            st.markdown("#### Standings")
            scols = [c for c in ["rank", "team", "wins", "losses", "ties", "points_for",
                                 "points_against", "streak", "faab_balance", "moves", "trades"]
                     if c in standings.columns]
            st.dataframe(
                ui.style_df(
                    ui.cols(standings[scols], team="TEAM"),
                    sequential=["PF"], diverging=[],
                    fmt={"PF": "{:.1f}", "PA": "{:.1f}", "RANK": "{:.0f}",
                         "W": "{:.0f}", "L": "{:.0f}", "T": "{:.0f}"},
                ),
                width="stretch", hide_index=True,
            )
            ui.col_key("PF", "PA", "STRK", "FAAB", "MOV", "TRD")

        mu = _ya.matchups_df()
        if not mu.empty:
            st.markdown("#### Weekly results")
            wk_pick = st.selectbox("Week", sorted(mu["week"].unique(), reverse=True))
            wv = mu[mu["week"] == wk_pick]
            mcols = [c for c in ["team", "opponent", "points", "opp_points", "proj", "result"]
                     if c in wv.columns]
            st.dataframe(
                ui.style_df(
                    ui.cols(wv[mcols].sort_values("points", ascending=False),
                            team="TEAM", proj="PROJ"),
                    sequential=["PTS"],
                    fmt={"PTS": "{:.1f}", "OPP PTS": "{:.1f}", "PROJ": "{:.1f}"},
                ),
                width="stretch", hide_index=True,
            )

            st.markdown("#### Points by week")
            ui.line_chart(mu.dropna(subset=["points"]), x="week", y="points", color="team",
                          y_title="points")

        tx = _ya.transactions_df()
        if not tx.empty:
            st.markdown("#### Recent transactions")
            st.dataframe(ui.cols(tx.head(40), team="TEAM"), width="stretch", hide_index=True)

# ---- Start / Sit -------------------------------------------------------------------
with tab_start:
    st.caption(
        f"Projections: FantasyPros (studs) + nflverse estimate (everyone else), "
        f"matchup-adjusted via defense-vs-position. Optimizing week {next_week}."
    )
    try:
        from mega.lineup import optimize_lineup
        rp = _my_roster_projected(int(season), int(next_week))
        lu = optimize_lineup(rp, int(season), int(next_week))
    except Exception as e:
        lu = None
        st.warning(f"Projections unavailable: {e}")

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
        cols = [c for c in ["lineup", "player", "pos", "nfl_team", "opp", "ease_rank",
                            "proj", "proj_adj", "proj_source", "tgt_pct", "tm_rank",
                            "start_sit", "close_call"] if c in lu.columns]

        lu_fmt = {"PROJ": "{:.1f}", "PROJ*": "{:.1f}", "MU#": "{:.0f}",
                  "TGT%": "{:.1%}", "TM#": "{:.0f}"}

        st.markdown("#### ✅ Recommended starters")
        sview = starters.assign(_o=starters["lineup"].map(slot_order)).sort_values("_o")[cols]
        ui.table(sview, rename={"proj": "PROJ"}, sequential=["PROJ*"], pos_cols=["POS"], fmt=lu_fmt)
        ui.col_key("PROJ*", "MU#", "SRC", "GRADE", "TGT%", "TM#",
                   PROJ="raw projection", NOTE="close-call flag")

        st.markdown("#### 🪑 Bench")
        bview = bench.sort_values("proj_adj", ascending=False)[cols]
        ui.table(bview, rename={"proj": "PROJ"}, pos_cols=["POS"], fmt=lu_fmt)

        _est = int((starters["proj_source"] != "FantasyPros").sum())
        if _est:
            st.warning(
                f"**{_est} of {len(starters)} starters are running on estimates, not real projections.** "
                "FantasyPros' free tier stops at the top 10 per position, so everyone below that is "
                "modelled from usage. Two estimates within ~2 points of each other is a coin flip, not "
                "a recommendation — when it's that close, start the player with the bigger **TGT%** "
                "and the better **TM#**, because volume holds up week to week and points don't."
            )
        if n_close:
            st.info("**Close calls** are flagged in **NOTE** — a bench player projecting within ~2 points "
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
        st.subheader(f"Season total: actual half-PPR vs expected (through wk {week})")
        st.bar_chart(cmp.set_index("player")[["expected", "actual"]])
        st.dataframe(cmp[["player", "expected", "actual", "diff"]].round(1),
                     width="stretch", hide_index=True)

# ---- Usage trends ----------------------------------------------------------------
with tab_use:
    metric = st.selectbox("Metric", ["snap share", "target share", "half-PPR points", "targets", "carries"])
    players_sel = st.multiselect("Players", skill["name"].tolist(), default=skill[skill["slot"] != "BN"]["name"].tolist())
    sel_ids = [k for k, v in name_by_id.items() if v in players_sel]

    if metric == "snap share" and not snaps.empty:
        pfr_map = dict(zip(skill["pfr_id"], skill["name"]))
        s = snaps[snaps["pfr_id"].isin([p for p in skill["pfr_id"] if p])].copy()
        col = first_col(s, "offense_pct", "off_pct")
        if col:
            s["value"] = pd.to_numeric(s[col], errors="coerce")
            s["player"] = s["pfr_id"].map(pfr_map)
            ui.line_chart(s.dropna(subset=["value"]), x="week", y="value", color="player",
                          y_title="snap share")
        else:
            st.info("No snap-share column found.")
    else:
        base = sw[sw["gsis_id"].isin(sel_ids)].copy()
        base["player"] = base["gsis_id"].map(name_by_id)
        field = {
            "target share": "target_share", "half-PPR points": "half_ppr",
            "targets": "targets", "carries": "carries", "snap share": None,
        }[metric]
        if field and field in base.columns:
            ui.line_chart(base.dropna(subset=[field]), x="week", y=field, color="player",
                          y_title=metric)
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
        st.subheader(f"Week {next_week} — team game environment (Vegas)")
        st.caption("Implied team total = Vegas's expected points for that offense. Higher = more scoring to go around.")
        st.dataframe(
            ui.style_df(
                ui.cols(mt, players="PLAYERS", matchup="MU", total="TOT", spread="SPRD", implied_pts="IMP"),
                sequential=["IMP"],
                fmt={"TOT": "{:.1f}", "SPRD": "{:+.1f}", "IMP": "{:.1f}"},
            ),
            width="stretch", hide_index=True,
        )
        ui.col_key("IMP", TOT="Vegas game total", SPRD="point spread", MU="opponent")

        # per-player defense-vs-position matchup
        dvp = _dvp(int(season))
        if not dvp.empty:
            st.subheader(f"Week {next_week} — player matchup difficulty (defense vs position)")
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
            st.dataframe(
                ui.style_df(ui.cols(show, matchup="MU", verdict="VERDICT"), pos_cols=["POS"],
                            fmt={"MU#": "{:.0f}", "PA/G": "{:.1f}"}),
                width="stretch", hide_index=True,
            )
            ui.col_key("MU#", "PA/G", MU="opponent")
            st.caption("**MU#**: opponent's rank in half-PPR points allowed to that position — **1 = easiest** of 32, 32 = stingiest.")

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
        waivers=_i.waiver_board(season, rostered, top=20),
        trades=_i.trade_finder(season, yahoo_rosters=ros),
        draft_delta=_i.draft_value_delta(season),
        buysell=_i.buy_low_sell_high(season),
        form_season=_i.form_season(season),
        roster_src=roster_src,
    )


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

        acols = ["player", "slot", "half_ppr_pg", "per_g", "tgt_pct", "tm_rank", "why"]
        afmt = {"PPG": "{:.1f}", "xFP±/G": "{:+.1f}", "TGT%": "{:.1%}", "TM#": "{:.0f}"}
        aren = {"per_g": "xFP±/G", "slot": "SLOT"}

        sell = mine[mine["per_g"] >= 2.0].sort_values("per_g", ascending=False).head(5)
        st.markdown("#### Shop these — points are ahead of the work")
        if sell.empty:
            st.caption("Nobody on your roster is meaningfully outscoring his opportunity right now.")
        else:
            sell = sell.assign(why=sell.apply(_why_sell, axis=1))
            ui.table(sell[acols], rename=aren, diverging=["xFP±/G"], pos_cols=["SLOT"], fmt=afmt,
                     help={"xFP±/G": "points per game above what his usage predicts"})
            st.caption("Their value to a leaguemate is at its peak. Sell the name, not the role.")

        buy = mine[mine["per_g"] <= -1.5].sort_values("per_g").head(5)
        st.markdown("#### Hold these — the work is there, the points aren't yet")
        if buy.empty:
            st.caption("Nobody is notably underperforming his opportunity.")
        else:
            buy = buy.assign(why=buy.apply(_why_buy, axis=1))
            ui.table(buy[acols], rename=aren, diverging=["xFP±/G"], pos_cols=["SLOT"], fmt=afmt,
                     help={"xFP±/G": "points per game above what his usage predicts"})
            st.caption("Don't sell into a cold streak — the usage says the points are coming.")
    else:
        st.info("Expected-points data isn't available yet this season, so sell/hold flags are off.")

    if IB is not None:
        wv = IB["waivers"]
        if not wv.empty:
            st.markdown("#### Best waiver claims")
            ui.table(wv.head(5)[["player", "pos", "pg_recent", "tgt_pct", "tm_rank", "add_score", "why"]],
                     sequential=["SCORE"], pos_cols=["POS"],
                     fmt={"PPG": "{:.1f}", "TGT%": "{:.1%}", "TM#": "{:.0f}", "SCORE": "{:.2f}"})

        tr = IB["trades"]
        if not tr.empty:
            st.markdown("#### Best trades to offer")
            tr5 = tr.head(5).copy()
            tr5["give"] = tr5["give"] + " (" + tr5["give_pos"] + ")"
            tr5["get"] = tr5["get"] + " (" + tr5["get_pos"] + ")"
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
        top3 = wv.head(3)
        ui.kpi_row([
            (f"#{i+1} target", ui.short_name(r["player"]), f"{r['pos']} · {r['why'][:38]}")
            for i, (_, r) in enumerate(top3.iterrows())
        ] or [("—", "no candidates", "")])
        st.write("")
        st.caption(f"Recent-form data: {IB['form_season']} season · adds via Sleeper · values via FantasyCalc")
        ui.lede(
            "Free agents ranked by <b>whether their role actually changed</b>, not just whether they had one "
            "good week. Check <b>TM#</b> and <b>TGT%</b> before you spend a claim."
        )
        ui.table(
            wv,
            sequential=["SCORE", "TGT%"], diverging=["TR30"], pos_cols=["POS"],
            fmt={"PPG": "{:.1f}", "TGT": "{:.1f}", "CAR": "{:.1f}", "TGT%": "{:.1%}",
                 "TM#": "{:.0f}", "VAL": "{:.0f}", "ADD#": "{:.0f}", "TR30": "{:+.0f}",
                 "SCORE": "{:.2f}"},
        )
        ui.col_key("PPG", "TGT", "TGT%", "TM#", "CAR", "VAL", "ADD#", "TR30", "SCORE")
        st.caption(
            "**SCORE** blends recent points, target/carry volume, FantasyCalc value + 30-day trend, and industry "
            "add rank. **TM# 1–2** on a rising **TGT%** is the strongest signal a role has genuinely changed."
        )

with tab_trade:
    if IB is None:
        st.warning("League intel unavailable.")
    elif IB["trades"].empty:
        st.info("No trade ideas cleared the fairness filter this run.")
    else:
        ui.lede(
            "Offers built from <b>your league's actual rosters</b> — who has a surplus where you're "
            "thin, and what they're short of in return. This is the part a generic ranking site "
            "can't do for you."
        )
        tt = IB["trades"].copy()
        # Position folded into the name — two "POS" columns would collide on rename.
        tt["give"] = tt["give"] + " (" + tt["give_pos"] + ")"
        tt["get"] = tt["get"] + " (" + tt["get_pos"] + ")"
        tfmt = {"GIVE VAL": "{:.0f}", "GET VAL": "{:.0f}", "FAIR": "{:.2f}", "EDGE": "{:+.0f}"}
        tcols = ["give", "give_val", "get", "get_val", "addresses", "fairness", "edge"]

        by_mgr = st.toggle("Group by manager", value=True)
        if by_mgr:
            for partner, grp in tt.groupby("partner", sort=False):
                need = grp["they_need"].iloc[0] if "they_need" in grp.columns else "—"
                st.markdown(f"#### {partner}")
                st.caption(f"Thin at **{need}** — lead with that when you pitch it.")
                ui.table(grp[tcols], sequential=["FAIR"], diverging=["EDGE"], fmt=tfmt)
        else:
            ui.table(tt[["partner"] + tcols], sequential=["FAIR"], diverging=["EDGE"], fmt=tfmt)

        ui.col_key("MANAGER", "YOU GIVE", "YOU GET", "GIVE VAL", "GET VAL", "FAIR", "EDGE", "FILLS")
        st.caption(
            f"Rosters: {IB['roster_src']}. **FAIR** near 1.0 means an even swap by FantasyCalc value — "
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
        dfmt = {"value": "{:.0f}", "value_delta": "{:+.0f}"}
        c1, c2 = st.columns(2)
        c1.markdown("**▲ Risers vs draft slot**")
        c1.dataframe(ui.style_df(view.head(15)[dcols], diverging=["value_delta"], pos_cols=["pos"], fmt=dfmt),
                     width="stretch", hide_index=True)
        c2.markdown("**▼ Fallers vs draft slot**")
        c2.dataframe(ui.style_df(view.sort_values("value_delta").head(15)[dcols], diverging=["value_delta"], pos_cols=["pos"], fmt=dfmt),
                     width="stretch", hide_index=True)
        st.markdown("**Your regression watch** — actual vs expected half-PPR")
        from mega.intel import _norm as _mnorm
        bs = IB["buysell"]
        my_norm = {_mnorm(n) for n in skill["name"]}
        mine_bs = bs[bs["norm"].isin(my_norm)].sort_values("diff_pg")[
            ["player", "pos", "gms", "actual", "expected", "diff_pg", "signal"]] if not bs.empty else pd.DataFrame()
        st.dataframe(
            ui.style_df(mine_bs, diverging=["diff_pg"], pos_cols=["pos"],
                        fmt={"actual": "{:.1f}", "expected": "{:.1f}", "diff_pg": "{:+.1f}"})
            if not mine_bs.empty else mine_bs,
            width="stretch", hide_index=True,
        )

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
        st.markdown("#### Your roster — archetype fit")
        mine = arch[arch["mine"]].sort_values("arch_fit", ascending=False)
        acols = ["player", "pos", "team", "arch_fit", "tags", "carries_pg", "tgt_share",
                 "tm_rank", "age", "exp_yrs", "why"]
        st.dataframe(
            ui.style_df(ui.cols(mine[acols]), sequential=["FIT"], pos_cols=["POS"],
                        fmt={"FIT": "{:.0f}", "CAR": "{:.1f}", "TGT%": "{:.1%}",
                             "TM#": "{:.0f}", "AGE": "{:.0f}", "EXP": "{:.0f}"}),
            width="stretch", hide_index=True,
        )
        ui.col_key("FIT", "TAGS", "CAR", "TGT%", "TM#", "AGE", "EXP")

        # target board by position
        pos_sel = st.radio("Position", ["QB", "RB", "WR", "TE"], horizontal=True)
        rostered = {_an(p) for p in IB["draft"]["player"]} if IB is not None else set()
        pool = arch[arch["pos"] == pos_sel].copy()
        pool["status"] = pool.apply(
            lambda r: "🟡 mine" if r["mine"] else ("rostered" if r["norm"] in rostered else "🟢 available"), axis=1)
        st.markdown(f"#### Best {pos_sel} fits — who to target")
        st.caption("🟢 available = not on any draft-board roster (verify against live adds). 🟡 mine = already yours.")
        st.dataframe(
            ui.style_df(
                ui.cols(pool.head(20)[["status", "player", "team", "arch_fit", "tags",
                                       "half_ppr_pg", "proj_ppg", "tgt_share", "tm_rank", "why"]],
                        status="OWN"),
                sequential=["FIT"],
                fmt={"FIT": "{:.0f}", "PPG": "{:.1f}", "PROJ": "{:.1f}",
                     "TGT%": "{:.1%}", "TM#": "{:.0f}"}),
            width="stretch", hide_index=True,
        )
        ui.col_key("FIT", "TAGS", "PPG", "TGT%", "TM#", OWN="who holds him", PROJ="season projection per game")

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
        _fmt = {"WOPR": "{:.3f}", "xPPG±": "{:+.1f}", "GAP": "{:+.0f}",
                "WOPR#": "{:.0f}", "DRAFT#": "{:.0f}"}

        def _show(frame, cols):
            if frame is None or frame.empty:
                st.info("Nothing flagged here right now.")
                return
            st.dataframe(
                ui.style_df(ui.cols(frame[cols], name="PLAYER"), pos_cols=["POS"],
                            sequential=["WOPR"], diverging=["GAP", "xPPG±"], fmt=_fmt),
                width="stretch", hide_index=True,
            )
            ui.col_key("WOPR", "WOPR#", "DRAFT#", "GAP", "xPPG±",
                       TAGS="opportunity flags — see 'How to read this' above")

        st.markdown("#### Your WR/TE — sell / hold")
        st.caption("`SELL_HIGH` / `FADE` = points ran ahead of opportunity, shop them. "
                   "`BUY_LOW` = hold, don't sell low.")
        _show(W["mine"], _mcols)

        st.markdown("#### Trade targets on other rosters")
        st.caption("Players whose opportunity outstrips their price or is trending up — grouped by manager.")
        _show(W["opp"], ["owner"] + _mcols)

        st.markdown("#### Waiver adds (free agents)")
        st.caption("Unrostered WR/TE clearing a startable opportunity bar or jumping in role.")
        _show(W["fa"], _mcols)

        if not W["unknown"].empty:
            st.markdown("#### Match review (unmatched)")
            st.dataframe(ui.cols(W["unknown"][["name", "pos", "tags"]], name="PLAYER"),
                         width="stretch", hide_index=True)

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
                     column_config={"link": st.column_config.LinkColumn("link")})
    except Exception as e:
        st.warning(f"News feeds unavailable: {e}")

# ---- Raw ---------------------------------------------------------------------
with tab_raw:
    st.write("Mapped roster")
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
