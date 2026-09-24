"""One place that loads a season of nflverse and assembles it.

The dashboard has Streamlit's cache in front of its loaders; the scheduled task and the
command-line tools do not. Both need the same player-week frame and the same role context,
and two copies of "load these six tables and join them" is exactly the duplication that
drifts — so it lives here once, memoised per process, and the app wraps it in st.cache_data
for the cross-session layer.
"""
from __future__ import annotations

import functools

import pandas as pd

from .config import SCORING


def _tp(d):
    return d.to_pandas() if hasattr(d, "to_pandas") else d


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else pd.Series(0.0, index=df.index)


@functools.lru_cache(maxsize=4)
def stats(season: int) -> pd.DataFrame:
    """Weekly player stats, normalized to this app's column names, with half-PPR scored."""
    import nflreadpy as nfl

    df = _tp(nfl.load_player_stats(seasons=[season]))
    ren = {}
    for target, cands in (("gsis_id", ("player_id", "gsis_id")),
                          ("player", ("player_display_name", "player_name")),
                          ("pos", ("position", "position_group")),
                          ("team", ("team", "recent_team"))):
        if target in df.columns:
            continue
        for c in cands:
            if c in df.columns:
                ren[c] = target
                break
    df = df.rename(columns=ren)
    s = SCORING
    df["half_ppr"] = (
        s["pass_yd"] * _num(df, "passing_yards") + s["pass_td"] * _num(df, "passing_tds")
        + s["pass_int"] * _num(df, "passing_interceptions")
        + s["rush_yd"] * _num(df, "rushing_yards") + s["rush_td"] * _num(df, "rushing_tds")
        + s["rec"] * _num(df, "receptions") + s["rec_yd"] * _num(df, "receiving_yards")
        + s["rec_td"] * _num(df, "receiving_tds")
        + s["fum_lost"] * (_num(df, "rushing_fumbles_lost") + _num(df, "receiving_fumbles_lost")
                           + _num(df, "sack_fumbles_lost"))
        + s["two_pt"] * (_num(df, "passing_2pt_conversions") + _num(df, "rushing_2pt_conversions")
                         + _num(df, "receiving_2pt_conversions"))
        + s["ret_td"] * _num(df, "special_teams_tds")
    )
    return df


def _safe(fn, *a, **kw) -> pd.DataFrame:
    """A missing optional table degrades the answer; it must not take the app down."""
    try:
        out = _tp(fn(*a, **kw))
        return out if out is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@functools.lru_cache(maxsize=4)
def snaps(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    return _safe(nfl.load_snap_counts, seasons=[season])


@functools.lru_cache(maxsize=4)
def ff_opportunity(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    return _safe(nfl.load_ff_opportunity, seasons=[season], stat_type="weekly",
                 model_version="latest")


@functools.lru_cache(maxsize=8)
def nextgen(season: int, kind: str) -> pd.DataFrame:
    import nflreadpy as nfl
    return _safe(nfl.load_nextgen_stats, seasons=[season], stat_type=kind)


@functools.lru_cache(maxsize=4)
def depth_charts(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    return _safe(nfl.load_depth_charts, seasons=[season])


@functools.lru_cache(maxsize=4)
def schedules(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    return _safe(nfl.load_schedules, seasons=[season])


@functools.lru_cache(maxsize=4)
def injuries(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    return _safe(nfl.load_injuries, seasons=[season])


@functools.lru_cache(maxsize=4)
def player_week(season: int) -> pd.DataFrame:
    """The denormalised player-week table: stats, snaps, expected points, routes, red zone."""
    from . import ask, ids
    from . import redzone as rzn
    from . import routes as rz

    st = stats(season)
    sn = snaps(season)
    xw = ids.crosswalk()
    try:
        rt = rz.weekly(st, sn.rename(columns={"pfr_player_id": "pfr_id"}), xw,
                       rz.team_dropbacks(season))
    except Exception:
        rt = pd.DataFrame()
    try:
        rzw = rzn.weekly(season)
    except Exception:
        rzw = pd.DataFrame()
    return ask.player_week(st, sn, ff_opportunity(season), rt, xw, rzw, season=season)


@functools.lru_cache(maxsize=4)
def role_context(season: int) -> dict:
    """HANDOFF §12: roles, baselines, NFL positional averages and the indexed table."""
    from . import roles as rl

    return rl.build(player_week(season), ff_opportunity(season),
                    nextgen(season, "receiving"), nextgen(season, "rushing"),
                    depth_charts(season))


def role_lookup(season: int) -> dict[str, dict]:
    """gsis_id -> the handful of role fields the waiver board and trade finder want."""
    rc = role_context(season)
    tab = rc.get("table")
    if tab is None or tab.empty:
        return {}
    out = {}
    for _, r in tab.iterrows():
        fl = r.get("flags") if isinstance(r.get("flags"), list) else []
        tags = r.get("tags") if isinstance(r.get("tags"), dict) else {}
        out[r["gsis_id"]] = {
            "role": r.get("role"), "role_src": r.get("role_src"),
            "flags": fl, "tags": tags,
            "role_plus": "ROLE+" in fl, "role_minus": "ROLE-" in fl,
            "tgt_idx": r.get("target_share_idx"), "snap_idx": r.get("snap_pct_idx"),
            "xfp_idx": r.get("xfp_share_idx"),
        }
    return out


def clear() -> None:
    """Drop every memo — the scheduled task calls this after a fresh scrape."""
    for f in (stats, snaps, ff_opportunity, nextgen, depth_charts, schedules, injuries,
              player_week, role_context):
        f.cache_clear()
