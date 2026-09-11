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

roster_raw = pd.read_csv(ROSTER_CSV, dtype=str).fillna("")

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
skill = mapped[~mapped["pos"].isin(["K", "DEF"])].copy()
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
    "data: nflverse · Sleeper · FantasyCalc",
])

(tab_over, tab_start, tab_axe, tab_use, tab_match,
 tab_wire, tab_trade, tab_draft, tab_arch, tab_news, tab_raw) = st.tabs(
    ["Team overview", "Start / Sit", "Actual vs expected", "Usage trends", "Matchups",
     "Waiver wire", "Trades", "Draft value", "Archetypes", "News", "Raw data"]
)


@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Scoring league-winner archetypes…")
def _archetypes(season: int) -> pd.DataFrame:
    from mega.archetypes import score
    return score(season)


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

# ---- Team overview -------------------------------------------------------------------
with tab_over:
    if sw.empty:
        st.info("No stats yet for this season/week range.")
    else:
        sws = sw.sort_values("week")
        for col in ("targets", "carries", "target_share"):
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
                tgt_share=("target_share", lambda s: s.tail(roll).mean()),
            )
            .reset_index()
        )
        agg["player"] = agg["gsis_id"].map(name_by_id)
        agg["slot"] = agg["gsis_id"].map(slot_by_id)

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

        # next opponent + implied team total
        if not sched.empty:
            wk = sched[sched["week"] == next_week]
            opp_rows = []
            for _, gm in wk.iterrows():
                tl, sl = gm.get("total_line"), gm.get("spread_line")
                h_imp = (tl / 2 + sl / 2) if pd.notna(tl) and pd.notna(sl) else None
                a_imp = (tl / 2 - sl / 2) if pd.notna(tl) and pd.notna(sl) else None
                opp_rows.append({"team": gm["home_team"], "opp": "vs " + str(gm["away_team"]), "implied": h_imp})
                opp_rows.append({"team": gm["away_team"], "opp": "@ " + str(gm["home_team"]), "implied": a_imp})
            opp = pd.DataFrame(opp_rows)
            agg["nfl_team"] = agg["gsis_id"].map(team_by_id)
            agg = agg.merge(opp, left_on="nfl_team", right_on="team", how="left", suffixes=("", "_x"))

        order = {s: i for i, s in enumerate(["QB", "RB", "WR", "TE", "W/R", "BN"])}
        agg["_o"] = agg["slot"].map(order).fillna(9)
        agg = agg.sort_values(["_o", "half_ppr_pg"], ascending=[True, False])

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
            "xfp_tot", "xfp_diff", "tgt_pg", "carry_pg", "tgt_share",
            "report_status", "opp", "implied",
        ] if c in agg.columns]
        tbl = agg[show_cols].rename(columns={
            "half_ppr_pg": "HalfPPR/G", "roll_pg": f"L{roll}/G", "last_wk": "Last wk",
            "xfp_tot": "xFP", "xfp_diff": "Act−xFP", "tgt_pg": "Tgt/G", "carry_pg": "Carry/G",
            "tgt_share": f"TgtSh L{roll}", "report_status": "Status", "implied": f"Wk{next_week} impl",
        })
        st.dataframe(
            ui.style_df(
                tbl,
                diverging=["Act−xFP"],
                pos_cols=["slot"],
                fmt={c: "{:.1f}" for c in ["HalfPPR/G", f"L{roll}/G", "Last wk", "xFP", "Act−xFP",
                                           "Tgt/G", "Carry/G", f"Wk{next_week} impl"]} | {f"TgtSh L{roll}": "{:.0%}"},
            ),
            width="stretch", hide_index=True,
        )
        st.caption("**Act−xFP**: amber = overperforming its expected points (sell-high / regression risk) · blue = underperforming (buy-low).")

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

        slot_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLEX": 4}
        cols = ["lineup", "player", "pos", "nfl_team", "opp", "ease_rank", "proj", "proj_adj",
                "proj_source", "start_sit", "close_call"]

        st.markdown("#### ✅ Recommended starters")
        sview = starters.assign(_o=starters["lineup"].map(slot_order)).sort_values("_o")[cols]
        st.dataframe(
            ui.style_df(
                sview.rename(columns={"nfl_team": "team", "ease_rank": "matchup#", "proj_adj": "proj*",
                                      "proj_source": "source", "start_sit": "FP grade", "close_call": "note"}),
                sequential=["proj*"], pos_cols=["pos"],
                fmt={"proj": "{:.1f}", "proj*": "{:.1f}", "matchup#": "{:.0f}"},
            ),
            width="stretch", hide_index=True,
        )
        st.caption("**proj\\*** = matchup-adjusted · **matchup#** = opponent's rank allowing points to that position (1 = easiest of 32).")

        st.markdown("#### 🪑 Bench")
        bview = bench.sort_values("proj_adj", ascending=False)[cols]
        st.dataframe(
            ui.style_df(
                bview.rename(columns={"nfl_team": "team", "ease_rank": "matchup#", "proj_adj": "proj*",
                                      "proj_source": "source", "start_sit": "FP grade", "close_call": "note"}),
                pos_cols=["pos"],
                fmt={"proj": "{:.1f}", "proj*": "{:.1f}", "matchup#": "{:.0f}"},
            ),
            width="stretch", hide_index=True,
        )
        if n_close:
            st.info("**Close calls** flagged in `note` — a bench player projects within ~2 pts of a starter at the same slot. Worth a matchup/injury check before lock.")

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
            piv = s.pivot_table(index="week", columns="player", values="value", aggfunc="mean")
            st.line_chart(piv)
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
            piv = base.pivot_table(index="week", columns="player", values=field, aggfunc="sum")
            st.line_chart(piv)
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
            ui.style_df(mt, sequential=["implied_pts"],
                        fmt={"total": "{:.1f}", "spread": "{:+.1f}", "implied_pts": "{:.1f}"}),
            width="stretch", hide_index=True,
        )

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
                ui.style_df(show.rename(columns={"ease_rank": "matchup#", "pa_pg": "pts allowed/g"}),
                            diverging=[], pos_cols=["pos"],
                            fmt={"matchup#": "{:.0f}", "pts allowed/g": "{:.1f}"}),
                width="stretch", hide_index=True,
            )
            st.caption("**matchup#**: opponent's rank in half-PPR points allowed to that position — **1 = easiest** of 32, 32 = stingiest.")

# ---- League intelligence (Mega Bowl) ---------------------------------------------
@st.cache_data(ttl=dt.timedelta(hours=6), show_spinner="Crunching league intel…")
def _intel_bundle(season: int):
    from mega import intel as _i
    from mega import yahoo_api as _ya
    from mega.draft_board import load_draft as _ld

    draft = _ld()
    ros = _ya.rosters_df() if _ya.available() else None
    if ros is not None and not ros.empty:
        rostered = set(ros["norm"])
        roster_src = f"live Yahoo rosters (week {_ya.week()})"
    else:
        ros = None
        rostered = {_i._norm(p) for p in draft["player"]}
        roster_src = "draft-board approximation"
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
        wtbl = wv.rename(columns={
            "pg_recent": "HalfPPR/G", "tgt_pg": "Tgt/G", "carry_pg": "Carry/G",
            "value": "FC value", "add_rank": "Ind add#", "trend_30d": "30d trend",
            "add_score": "Add score", "why": "Why",
        })
        st.dataframe(
            ui.style_df(
                wtbl, sequential=["Add score"], diverging=["30d trend"], pos_cols=["pos"],
                fmt={"HalfPPR/G": "{:.1f}", "Tgt/G": "{:.1f}", "Carry/G": "{:.1f}",
                     "FC value": "{:.0f}", "Ind add#": "{:.0f}", "30d trend": "{:+.0f}", "Add score": "{:.2f}"},
            ),
            width="stretch", hide_index=True,
        )
        st.caption("Add score blends recent points, target/carry volume, FantasyCalc value + 30-day trend, and industry add rank.")

with tab_trade:
    if IB is None:
        st.warning("League intel unavailable.")
    elif IB["trades"].empty:
        st.info("No trade ideas cleared the fairness filter this run.")
    else:
        st.caption(f"Rosters: {IB['roster_src']}. Fairness = value parity (1.0 = even).")
        tt = IB["trades"].copy()
        tt.insert(0, "trade", tt["give"] + "  →  " + tt["get"])
        cols = [c for c in ["trade", "partner", "give_pos", "get_pos", "give_val", "get_val",
                            "addresses", "fairness", "edge"] if c in tt.columns]
        st.dataframe(
            ui.style_df(
                tt[cols], sequential=["fairness"], diverging=["edge"], pos_cols=["give_pos", "get_pos"],
                fmt={"give_val": "{:.0f}", "get_val": "{:.0f}", "fairness": "{:.2f}", "edge": "{:+.0f}"},
            ),
            width="stretch", hide_index=True,
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
        acols = ["player", "pos", "team", "arch_fit", "tags", "carries_pg", "tgt_share", "age", "exp_yrs", "why"]
        st.dataframe(
            ui.style_df(mine[acols], sequential=["arch_fit"], pos_cols=["pos"],
                        fmt={"arch_fit": "{:.0f}", "carries_pg": "{:.1f}", "tgt_share": "{:.0%}",
                             "age": "{:.0f}", "exp_yrs": "{:.0f}"}),
            width="stretch", hide_index=True,
        )

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
                pool.head(20)[["status", "player", "team", "arch_fit", "tags", "half_ppr_pg", "proj_ppg", "why"]],
                sequential=["arch_fit"],
                fmt={"arch_fit": "{:.0f}", "half_ppr_pg": "{:.1f}", "proj_ppg": "{:.1f}"}),
            width="stretch", hide_index=True,
        )

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
