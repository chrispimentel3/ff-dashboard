"""Route-based receiving usage: routes run, targets per route, first downs per route.

**Routes are an estimate.** nflverse has no charted route count. The proxy the targets
handoff specifies — "on the field for a team dropback", from the participation file —
can't be built for 2026: `load_participation` covers 2016–2025 only, so the handoff's
own fallback is used instead:

    routes(player, week) = snap share × his team's dropbacks that week

Team dropbacks come from play-by-play (`qb_dropback`, excluding spikes and two-point
plays), which counts sacks and scrambles — routes were run on those.

Known bias, same direction as the participation proxy but larger: a player is assumed to
be on the field for passing plays at his overall snap rate. Anyone who sits on obvious
passing downs, or stays in to block, gets too many routes, so his per-route rates read
low. That hits **tight ends** hardest. Only WR and TE are computed for that reason —
running backs would need a run/pass split the snap counts don't carry.

Every rate is summed numerator over summed denominator, never a mean of weekly ratios.

    python -m mega.routes          # validation + top 10 WR and TE by 1D/RR
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POS = ("WR", "TE")

MIN_ROUTES = 50        # routes in the window before a rate is worth reading (handoff §4)
PER_GAME_BAR = 20      # ... or this many per game, when the window is shorter than that takes
WR_FD_FLAG = 0.12      # 1D/RR at or above this is Chris's league-winner line for a WR


def route_bar(games) -> float:
    """How many routes a window needs before its rates mean anything.

    MIN_ROUTES is written for the three-week default. Over one or two games nobody could
    clear it, so the bar falls back to a part-timer's route load per game — enough to keep
    out a receiver who left in the first quarter. One rule, used by the season card's ranks,
    the roster table and the Targets page alike.
    """
    return np.minimum(MIN_ROUTES, PER_GAME_BAR * np.asarray(games, dtype=float))


def _n(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else pd.Series(0.0, index=df.index)


# ---------------------------------------------------------------- team dropbacks
def team_dropbacks(season: int) -> pd.DataFrame:
    """team, week, dropbacks — regular season, spikes and two-point plays removed."""
    import nflreadpy as nfl
    import polars as pl

    # Play-by-play is ~100 MB a season and nflreadpy caches in memory by default, which
    # would pin all of it for a frame that reduces to ~570 rows. Cache it on disk instead.
    prev = None
    try:
        from nflreadpy import config as _cfg
        prev = _cfg.get_config().cache_mode
        _cfg.update_config(cache_mode="filesystem")
    except Exception:
        prev = None
    try:
        pbp = nfl.load_pbp([season])
    finally:
        if prev is not None:
            try:
                from nflreadpy import config as _cfg
                _cfg.update_config(cache_mode=prev)
            except Exception:
                pass

    keep = ["season_type", "week", "posteam", "qb_dropback", "qb_spike", "two_point_attempt"]
    out = (
        pbp.select([c for c in keep if c in pbp.columns])
        .filter(
            (pl.col("season_type") == "REG")
            & (pl.col("qb_dropback").fill_null(0) == 1)
            & (pl.col("qb_spike").fill_null(0) != 1)
            & (pl.col("two_point_attempt").fill_null(0) != 1)
            & pl.col("posteam").is_not_null()
        )
        .group_by(["posteam", "week"])
        .len()
        .rename({"posteam": "team", "len": "dropbacks"})
    )
    return out.to_pandas().astype({"week": int, "dropbacks": int})


# ---------------------------------------------------------------- weekly routes
def weekly(stats: pd.DataFrame, snaps: pd.DataFrame, crosswalk: pd.DataFrame | None,
           dropbacks: pd.DataFrame) -> pd.DataFrame:
    """One row per WR/TE per game played: targets, receiving first downs, estimated routes.

    Raw counts only — rates are computed after the window is chosen, so the same weekly
    frame serves a season card and a three-week roster view.
    """
    if stats is None or stats.empty or dropbacks is None or dropbacks.empty:
        return pd.DataFrame()
    w = stats
    if "season_type" in w.columns:
        w = w[w["season_type"].astype(str).str.upper() == "REG"]
    w = w[w["pos"].isin(POS)].copy()
    if w.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "gsis_id": w["gsis_id"], "player": w["player"], "pos": w["pos"],
        "team": w["team"], "week": pd.to_numeric(w["week"], errors="coerce").astype("Int64"),
        "targets": _n(w, "targets"), "rec": _n(w, "receptions"),
        "fd": _n(w, "receiving_first_downs"), "air": _n(w, "receiving_air_yards"),
    })

    # snap share, via the pfr_id crosswalk (snap counts carry no gsis_id)
    out["snap_pct"] = np.nan
    if snaps is not None and not snaps.empty and crosswalk is not None and "pfr_id" in snaps.columns:
        sn = snaps
        if "game_type" in sn.columns:
            sn = sn[sn["game_type"].astype(str).str.upper() == "REG"]
        sp = (sn.assign(week=pd.to_numeric(sn["week"], errors="coerce").astype("Int64"))
                .groupby(["pfr_id", "week"])["offense_pct"].max().rename("snap_pct").reset_index())
        xw = crosswalk.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("gsis_id")[["gsis_id", "pfr_id"]]
        out = out.merge(xw, on="gsis_id", how="left").drop(columns=["snap_pct"])
        out = out.merge(sp, on=["pfr_id", "week"], how="left")

    out = out.merge(dropbacks.rename(columns={"dropbacks": "tm_db"}), on=["team", "week"], how="left")
    out["routes"] = pd.to_numeric(out["snap_pct"], errors="coerce") * out["tm_db"]
    # The proxy can land under a player's own target count — he was targeted on a snap the
    # estimate didn't credit him. Floor it so no rate exceeds 1; validate() counts these.
    out["routes_raw"] = out["routes"]
    out["routes"] = out[["routes", "targets"]].max(axis=1)
    return out


def validate(w: pd.DataFrame) -> list[str]:
    """The handoff's Pass 1 checks. Returns one line per problem; empty means clean."""
    msgs = []
    if w.empty:
        return ["no weekly rows"]
    under = w[w["routes_raw"].notna() & (w["routes_raw"] < w["targets"])]
    if not under.empty:
        msgs.append(f"{len(under)} player-weeks estimated fewer routes than targets "
                    f"(floored): {', '.join(under['player'].head(5))}"
                    + (" …" if len(under) > 5 else ""))
    bad = w[(w["fd"] > w["rec"]) | (w["rec"] > w["targets"])]
    if not bad.empty:
        msgs.append(f"{len(bad)} player-weeks break fd <= rec <= targets: {', '.join(bad['player'].head(5))}")
    over = w[w["routes"].notna() & w["tm_db"].notna() & (w["routes"] > w["tm_db"] + 0.5)]
    if not over.empty:
        msgs.append(f"{len(over)} player-weeks have more routes than team dropbacks")
    dup = w.duplicated(["gsis_id", "week"]).sum()
    if dup:
        msgs.append(f"{dup} duplicate player-weeks")
    miss = w["routes"].isna().mean()
    if miss > 0.25:
        msgs.append(f"{miss:.0%} of player-weeks have no snap share, so no route estimate")
    return msgs


# ---------------------------------------------------------------- season / window
def totals(w: pd.DataFrame, weeks: range | list[int] | None = None) -> pd.DataFrame:
    """Sum a weekly frame into per-player rates. `weeks` limits the window.

    Returns gsis_id, routes, targets, fd, rec, plus tprr, fd_rr, routes_pg and `qualified`
    (enough routes for the rate to mean anything).

    There is deliberately no "route participation" column. Under this proxy routes are snap
    share × dropbacks, so routes ÷ dropbacks is just snap share again — the app already
    shows that, and a second column saying the same thing in different words is worse than
    none at all.
    """
    if w is None or w.empty:
        return pd.DataFrame(columns=["gsis_id", "routes", "tprr", "fd_rr", "routes_pg"])
    if weeks is not None:
        w = w[w["week"].isin(list(weeks))]
    if w.empty:
        return pd.DataFrame(columns=["gsis_id", "routes", "tprr", "fd_rr", "routes_pg"])
    # A game with no snap share can't contribute routes, so it's counted separately from
    # games played — otherwise routes per game would be diluted by games it can't see.
    r = w.assign(_rg=w["routes"].notna().astype(int))
    t = r.sort_values("week").groupby("gsis_id").agg(
        player=("player", "last"), pos=("pos", "last"), team=("team", "last"),
        games=("week", "nunique"), route_games=("_rg", "sum"),
        routes=("routes", "sum"),
        targets=("targets", "sum"), fd=("fd", "sum"), rec=("rec", "sum"),
    ).reset_index()
    t.loc[t["route_games"] == 0, "routes"] = np.nan
    rt = t["routes"].replace(0, np.nan)
    t["tprr"] = t["targets"] / rt
    t["fd_rr"] = t["fd"] / rt
    t["routes_pg"] = t["routes"] / t["route_games"].replace(0, np.nan)
    t["qualified"] = t["routes"].fillna(0) >= route_bar(t["route_games"])
    return t


def flag(row: pd.Series) -> str:
    """Chris's WR line, and a warning on anyone whose sample can't carry a rate.

    A receiver who played half of one game can post a 50% target rate; saying nothing
    about him would leave that number looking like the best on the roster.
    """
    if pd.isna(row.get("routes")) or pd.isna(row.get("fd_rr")):
        return "no route data"
    if not row.get("qualified"):
        return f"under {route_bar(row.get('route_games', 0)):.0f} routes"
    return "league-winner 1D/RR" if row["pos"] == "WR" and row["fd_rr"] >= WR_FD_FLAG else ""


if __name__ == "__main__":
    import nflreadpy as nfl

    from . import ids as _ids

    season = int(nfl.get_current_season())
    st_ = nfl.load_player_stats(seasons=[season]).to_pandas().rename(
        columns={"player_display_name": "player", "position": "pos", "player_id": "gsis_id"})
    sn = nfl.load_snap_counts(seasons=[season]).to_pandas().rename(columns={"pfr_player_id": "pfr_id"})
    wk = weekly(st_, sn, _ids.crosswalk(), team_dropbacks(season))
    print(f"{season}: {len(wk)} WR/TE player-weeks, "
          f"{wk['routes'].notna().sum()} with a route estimate, weeks {wk['week'].min()}–{wk['week'].max()}")
    for m in validate(wk) or ["all checks passed"]:
        print("  -", m)
    t = totals(wk)
    for p in POS:
        top = t[(t["pos"] == p) & t["qualified"]].nlargest(10, "fd_rr")
        print(f"\n== top 10 {p} by 1D/RR (qualified) ==")
        print(top[["player", "team", "routes", "routes_pg", "targets", "fd", "tprr", "fd_rr"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
