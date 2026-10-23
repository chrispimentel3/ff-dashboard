"""Export dashboard data as JSON for the mega-bowl-web frontend.

Builds every file the site reads straight from the mega/ modules — the analysis lives there,
and each `mega/*_web.py` shapes one page's payload. Season, week and the rolling window are
the defaults the old Streamlit app opened on: the current season, through the last week with
stats, planning for the first week with games left, a 3-week window.

Writes to data/web/<name>.json. That directory is committed and public (same as the rest
of data/), so mega-bowl-web's Next.js build fetches these files straight off
raw.githubusercontent.com — no live backend, no second copy of any analysis logic.

Usage:
    .venv/bin/python tools/export_web.py
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "web"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mega import data  # noqa: E402
from mega import ids as player_ids  # noqa: E402

ROLL = 3      # the rolling window (weeks) the roster and target-share numbers use


def _warn(what: str, e: Exception) -> None:
    print(f"[export_web] {what} failed: {type(e).__name__}: {e}", file=sys.stderr)


# ---- loaders: each read once per run ------------------------------------------------------
stats = functools.cache(data.load_player_stats)
ffo_all = functools.cache(data.load_ff_opportunity)
snaps_all = functools.cache(data.load_snaps)
routes_all = functools.cache(data.load_routes)
schedule = functools.cache(data.load_schedule)
rosters = functools.cache(data.load_rosters)


@functools.cache
def blended_proj(season: int, week: int) -> pd.DataFrame:
    from mega.projections import blended_week
    return blended_week(season, week)


@functools.cache
def vegas(season: int, week: int) -> pd.DataFrame:
    """The week's props scored into half-PPR points. Empty when none are on file."""
    from mega.vegas import week as vweek
    return vweek(season, week)


@functools.cache
def dvp(season: int) -> pd.DataFrame:
    from mega.lineup import defense_vs_position
    return defense_vs_position(season)


def target_share(season: int, through_week: int, roll: int) -> pd.DataFrame:
    """Target share and rank inside the player's own NFL offense, over the last `roll` weeks.
    Summed numerator / summed denominator — averaging weekly ratios lets one low-volume game
    swing a player's season number."""
    df = stats(season)
    if df.empty or "targets" not in df.columns:
        return pd.DataFrame(columns=["gsis_id", "tgt_pct", "tm_rank"])
    df = df[(df["week"] <= through_week) & (df["week"] > through_week - roll)].copy()
    df["targets"] = pd.to_numeric(df["targets"], errors="coerce").fillna(0.0)
    team_tot = df.groupby("team")["targets"].sum().rename("tm_tgt").reset_index()
    p = df.sort_values("week").groupby("gsis_id").agg(targets=("targets", "sum"), team=("team", "last")).reset_index()
    p = p[p["targets"] > 0].merge(team_tot, on="team", how="left")
    p["tgt_pct"] = p["targets"] / p["tm_tgt"].replace(0, pd.NA)
    p["tm_rank"] = p.groupby("team")["targets"].rank(method="min", ascending=False)
    return p[["gsis_id", "tgt_pct", "tm_rank"]]


# ---- the run's context: season, weeks, my roster ------------------------------------------
def context() -> SimpleNamespace:
    try:
        import nflreadpy as nfl
        season = int(nfl.get_current_season())
    except Exception:
        season = data.SEASON_DEFAULT
    st_ = stats(season)
    week = int(st_["week"].max()) if not st_.empty else 1
    next_week = min(data.current_week(season, week + 1), 18)

    roster_raw, roster_src = data.my_roster()
    mapped, match = data.map_roster(roster_raw)
    # filter on slot too: the scrape leaves pos blank on plenty of rows
    skill = mapped[~mapped["pos"].isin(["K", "DEF"]) & ~mapped["slot"].isin(["K", "DEF"])].copy()
    dropped = skill[~skill["resolved"] & ~skill["unmapped"]]
    if not dropped.empty:
        print("[export_web] couldn't match to nflverse (add to data/id_overrides.csv): "
              + ", ".join(dropped["name"]), file=sys.stderr)
    skill = skill[skill["resolved"]].copy()
    gsis_list = skill["gsis_id"].dropna().tolist()

    ffo = ffo_all(season)
    if not ffo.empty:
        ffo = ffo[ffo["gsis_id"].isin(gsis_list) & (ffo["week"] <= week)]
    try:
        from mega.intel import season_basis
        basis = season_basis(season)
    except Exception:
        basis = {"season": season, "current": True, "weeks": 0, "label": f"{season} form"}
    c = SimpleNamespace(
        season=season, week=week, next_week=next_week, roll=ROLL, roster_src=roster_src,
        mapped=mapped, match=match, skill=skill, gsis_list=gsis_list,
        sw=st_[st_["gsis_id"].isin(gsis_list) & (st_["week"] <= week)].copy(), ffo=ffo,
        inj=data.load_injuries(season), sched=schedule(season), basis=basis,
        name_by_id=dict(zip(skill["gsis_id"], skill["name"])),
        slot_by_id=dict(zip(skill["gsis_id"], skill["slot"])),
        team_by_id=dict(zip(skill["gsis_id"], skill["nfl_team"])),
        pos_by_id=dict(zip(skill["gsis_id"], skill["pos"])))
    c.agg = roster_agg(c)
    try:
        c.ib = intel_bundle(season)
    except Exception as e:          # network / dependency issue: the intel-backed files go empty
        _warn("league intel", e)
        c.ib = None
    return c


def roster_agg(c) -> pd.DataFrame:
    """My roster's points against its opportunity — the roster page and the action board's
    sell-high / buy-low both read it."""
    if c.sw.empty:
        return pd.DataFrame()
    sws = c.sw.sort_values("week")
    for col in ("targets", "carries"):
        if col not in sws.columns:
            sws[col] = 0.0
    agg = sws.groupby("gsis_id").agg(
        games=("week", "nunique"), half_ppr_tot=("half_ppr", "sum"), half_ppr_pg=("half_ppr", "mean"),
        last_wk=("half_ppr", "last"), roll_pg=("half_ppr", lambda s: s.tail(c.roll).mean()),
        tgt_pg=("targets", "mean"), carry_pg=("carries", "mean")).reset_index()
    agg["player"] = agg["gsis_id"].map(c.name_by_id)
    agg["slot"] = agg["gsis_id"].map(c.slot_by_id)
    agg["pos"] = agg["gsis_id"].map(c.pos_by_id)
    agg = agg.merge(target_share(c.season, c.week, c.roll), on="gsis_id", how="left")
    if not c.ffo.empty:
        exp = c.ffo.groupby("gsis_id")["half_ppr_exp"].sum().reset_index().rename(columns={"half_ppr_exp": "xfp_tot"})
        agg = agg.merge(exp, on="gsis_id", how="left")
        agg["xfp_diff"] = agg["half_ppr_tot"] - agg["xfp_tot"]
    if not c.inj.empty and "report_status" in c.inj.columns:
        latest = (c.inj[c.inj["gsis_id"].isin(c.gsis_list)].sort_values("week")
                  .groupby("gsis_id").tail(1)[["gsis_id", "report_status"]])
        agg = agg.merge(latest, on="gsis_id", how="left")
    # Chris's overrides (data/player_status.csv) beat the injury report, which lags
    from mega.status import out_for_week
    out = out_for_week(c.next_week)
    if out:
        lab = agg["player"].map(lambda n: out.get(player_ids.norm(n)))
        agg["report_status"] = lab.fillna(agg["report_status"]) if "report_status" in agg else lab
    if not c.sched.empty:      # next opponent + Vegas implied team total
        rows = []
        for _, gm in c.sched[c.sched["week"] == c.next_week].iterrows():
            tl, sl = gm.get("total_line"), gm.get("spread_line")
            h = (tl / 2 + sl / 2) if pd.notna(tl) and pd.notna(sl) else None
            a = (tl / 2 - sl / 2) if pd.notna(tl) and pd.notna(sl) else None
            rows.append({"team": gm["home_team"], "opp": "vs " + str(gm["away_team"]), "implied": h})
            rows.append({"team": gm["away_team"], "opp": "@ " + str(gm["home_team"]), "implied": a})
        agg["nfl_team"] = agg["gsis_id"].map(c.team_by_id)
        agg = agg.merge(pd.DataFrame(rows), left_on="nfl_team", right_on="team", how="left", suffixes=("", "_x"))
    order = {s: i for i, s in enumerate(["QB", "RB", "WR", "TE", "W/R", "BN"])}
    agg["_o"] = agg["slot"].map(order).fillna(9)
    return agg.sort_values(["_o", "half_ppr_pg"], ascending=[True, False])


def intel_bundle(season: int) -> dict:
    """Waivers, trades and the draft board, on the best rosters on hand."""
    from mega import intel as I
    from mega import needs
    from mega import trade_theses as TT
    from mega import yahoo_api as ya
    from mega.draft_board import load_draft

    draft = load_draft()
    # ownership, best source first: official API -> browser scrape -> draft board
    ros = ya.rosters_df() if ya.available() else None
    roster_src = f"live Yahoo rosters (week {ya.week()})"
    if ros is None or ros.empty:
        from mega.yahoo import cached_rosters
        ros, roster_src = cached_rosters(), "scraped Yahoo rosters"
    if ros is None or ros.empty:
        ros, roster_src = None, "draft-board approximation"
        rostered = {I._norm(p) for p in draft["player"]}
    else:
        rostered = set(ros["norm"])
    wk = data.current_week(season, 1)
    try:
        waivers = needs.board(season, wk, yahoo_rosters=ros, top=25)
        if waivers.empty:
            print("[export_web] waiver board: the valuation engine returned nothing; "
                  "using the roster-blind board", file=sys.stderr)
    except Exception as e:
        _warn("waiver valuation (using the roster-blind board)", e)
        waivers = pd.DataFrame()
    if waivers.empty:
        waivers = I.waiver_board(season, rostered, top=20)
    try:
        theses = TT.run(season, wk, yahoo_rosters=ros)
        ww = theses.pop("_week_weights", None)
        if ww:                     # the live service prices trades on the same week weights
            from mega import title_odds as _T
            _T.write_week_weights(ww)
    except Exception as e:
        theses = {"available": False, "cards": [], "meta": {"error": f"{type(e).__name__}: {e}"}}
    return {"draft": draft, "waivers": waivers, "trades": I.trade_finder(season, yahoo_rosters=ros),
            "theses": theses, "roster_src": roster_src}


def my_roster_projected(c) -> pd.DataFrame:
    """My skill roster joined to this week's projections: ours (mega/proj_ros.py), with the
    consensus kept beside it, and the Vegas props."""
    season, week = c.season, c.next_week
    proj = blended_proj(season, week).copy()
    for col in ("proj", "proj_source", "ecr", "start_sit", "norm", "nfl_est"):
        if col not in proj.columns:
            proj[col] = pd.NA
    cols = ["proj", "proj_source", "ecr", "start_sit", "nfl_est"]
    r = c.skill.rename(columns={"name": "player"})[["player", "slot", "pos", "nfl_team", "gsis_id"]].copy()
    r = r.merge(proj[["gsis_id"] + cols].dropna(subset=["gsis_id"]), on="gsis_id", how="left")
    if r["proj"].isna().any():      # fallback join on normalized name
        from mega.intel import _norm
        r["_n"] = r["player"].map(_norm)
        pn = proj.dropna(subset=["proj"]).drop_duplicates("norm")[["norm"] + cols]
        r = r.merge(pn, left_on="_n", right_on="norm", how="left", suffixes=("", "_n"))
        for col in cols:
            r[col] = r[col].where(r[col].notna(), r[f"{col}_n"])
        r = r[[col for col in r.columns if not col.endswith("_n") and col != "norm"]]
    r["proj"] = pd.to_numeric(r["proj"], errors="coerce").fillna(0.0)
    # the lineup is set on our weekly projection — the scorecard grades it against the consensus
    from mega.proj_ros import week_expected
    ours = week_expected(season, week)
    r["consensus"] = r["proj"]
    if ours:
        hit = r["gsis_id"].map(ours)
        # the live blend (mega/scorecard.py): only once it has won the walk-forward test
        try:
            from mega.scorecard import LIVE_BLEND
            lb = json.loads(LIVE_BLEND.read_text())
            w = (lb.get("weights") or {}) if lb.get("active") and int(lb.get("season", 0)) == season else {}
        except Exception:
            w = {}
        if w:
            wt = r["pos"].map(lambda p: float(w.get(p, 1.0)))
            hit = (wt * hit + (1 - wt) * r["consensus"]).where(hit.notna() & r["consensus"].notna(), hit)
        r["proj"] = hit.where(hit.notna(), r["proj"])
        r["proj_source"] = r["proj_source"].where(hit.isna(), "ours")
    v = vegas(season, week)
    if not v.empty:
        r = r.merge(v[["gsis_id", "vegas", "vegas_parts", "vegas_complete"]], on="gsis_id", how="left")
        r["vegas_edge"] = (pd.to_numeric(r["vegas"], errors="coerce") - r["proj"]).round(1)
    return r


# ---- one function per file ----------------------------------------------------------------
def action_board(c) -> dict:
    from mega.action_board import build
    return {**build(c.agg, c.ib, c.basis, c.season), "season": c.season, "week": c.week,
            "next_week": c.next_week}


def start_sit(c) -> dict:
    from mega.start_sit import build
    try:
        from mega.lineup import optimize_lineup
        from mega.status import out_for_week
        rp = my_roster_projected(c)
        out = out_for_week(c.next_week)
        rp["report_status"] = rp["player"].map(lambda n: out.get(player_ids.norm(n)))
        rp.loc[rp["report_status"].notna(), "proj"] = 0.0
        lu = optimize_lineup(rp, c.season, c.next_week)
    except Exception as e:
        _warn("projections", e)
        lu = None
    return build(lu, target_share(c.season, c.week, c.roll), c.next_week)


def waivers(c) -> dict:
    from mega.waivers import build
    if c.ib is None:
        return build(None, None, None)
    wv = c.ib["waivers"].copy()
    if "lane" not in wv.columns:
        return build(wv, None, None)
    from mega import faab
    return build(wv, faab.rivals(), faab.market_summary())


def trades(c) -> dict:
    from mega.trades import build
    if c.ib is None:
        return build(None, None)
    if c.ib["trades"].empty and not (c.ib.get("theses") or {}).get("available"):
        return build(None, c.ib["roster_src"])
    return build(c.ib["trades"], c.ib["roster_src"], c.season, theses=c.ib.get("theses") or {})


def archetypes(c) -> dict:
    """Blueprint fit — the player card reads each player's score off the board."""
    from mega.archetypes import score
    from mega.archetypes_web import build
    from mega.intel import _norm
    try:
        arch = score(c.season)
    except Exception as e:
        _warn("blueprint scores", e)
        arch = pd.DataFrame()
    if arch.empty:
        return build(None, set(), set())
    mine = {_norm(n) for n in c.skill["name"]}
    rostered = {_norm(p) for p in c.ib["draft"]["player"]} if c.ib is not None else set()
    return build(arch.assign(mine=arch["norm"].isin(mine)), mine, rostered)


def roster(c) -> dict:
    from mega.roster_web import build
    return build(None if c.agg.empty else c.agg, c.roll)


def playoff_odds(c) -> pd.DataFrame:
    """The player-level title simulation — the same seasons that price trades and the rooting
    guide. The team-level mega/sim.py is the fallback when it can't be built."""
    from mega import sim as SIM
    from mega import title_odds as T
    from mega.yahoo import cached_fixtures, cached_rosters, cached_scores
    try:
        m = T.build(c.season, c.next_week, cached_rosters())
        if m is not None:
            return pd.DataFrame(m.league_table())
    except Exception as e:
        _warn("player-level odds (using the team-level sim)", e)
    sc = cached_scores()
    st_path = ROOT / "data" / "yahoo_standings.csv"
    if sc is None or sc.empty or not st_path.is_file():
        return pd.DataFrame()
    try:
        fx = cached_fixtures()
    except Exception:
        fx = None
    s = SIM.from_league(sc, pd.read_csv(st_path), sc.groupby("team")["points"].mean().to_dict(),
                        range(c.next_week, 15), fixtures=fx)
    return SIM.simulate(s, n=6000)


def league(c) -> dict:
    from mega import yahoo_api as ya
    from mega.config import MY_TEAM
    from mega.league_web import build
    from mega.xwins import power_table
    from mega.yahoo import cached_scores, cached_standings
    try:
        odds = playoff_odds(c)
    except Exception as e:
        _warn("playoff odds", e)
        odds = pd.DataFrame()
    standings = src = xw = xw_week = tx = None
    if ya.available():
        standings, src = ya.standings_df(), "live Yahoo API"
        tx = ya.transactions_df()
        if standings.empty:
            standings = None
        if tx.empty:
            tx = None
    else:
        scraped = cached_standings()
        if not scraped.empty:
            standings, src = scraped, "scraped Yahoo standings"
            scores = cached_scores()
            if scores is not None and not scores.empty:
                xw = power_table(scores).rename(columns={"luck": "luck_w"})
                xw_week = int(scores["week"].max())
    # weekly scores: the live API when it answers, else the weekly scrape
    mu, mu_src = pd.DataFrame(), "no source"
    try:
        if ya.available():
            mu, mu_src = ya.matchups_df(), "live Yahoo API"
    except Exception:
        pass
    if mu is None or mu.empty:
        try:
            mu, mu_src = cached_scores(), "weekly scrape"
        except Exception:
            mu, mu_src = pd.DataFrame(), "no source"
    mcols = [col for col in ["team", "opponent", "points", "opp_points", "proj", "result", "week"] if col in mu.columns]
    return build(MY_TEAM, odds, standings, src, xw, xw_week, tx, mu[mcols] if not mu.empty else mu, mu_src)


def glossary(c) -> dict:
    from mega.glossary_web import build
    return build()


def news(c) -> dict:
    from mega.news_web import build
    return build(list(c.skill["name"]))


def _owner_info(gid: str, pteam: str, own: dict, fa: dict) -> dict:
    """Who holds him in this league — nothing at all when there is no league on file, so
    the card is then simply an NFL player card."""
    if not own and not fa:
        return {}
    from mega.config import MY_TEAM
    if gid in own:
        team, slot = own[gid]
        return {"kind": "mine" if team == MY_TEAM else "other_team", "team": team, "slot": slot}
    if str(fa.get(gid, "")).startswith("W"):
        return {"kind": "waivers", "waiver_until": str(fa[gid])[1:].strip(" ()")}   # "W (Sep 19)"
    return {"kind": "free_agent"} if pteam else {"kind": "no_team"}


def players(c) -> dict:
    """Every player's card, up front: the site renders whichever one is asked for."""
    from mega.lookup import player_index, season_table
    from mega.player_web import build
    from mega.status import out_for_week
    cur = c.season
    idx = player_index({cur: stats(cur), cur - 1: stats(cur - 1)}, rosters(cur))
    if idx.empty:
        return {"available": False, "index": [], "players": {}}
    cw = player_ids.crosswalk()
    pfr_ids = ({g: (p if isinstance(p, str) else None) for g, p in zip(cw["gsis_id"], cw["pfr_id"])}
               if not cw.empty else {})
    seasons = {s: {"table": season_table(stats(s), ffo_all(s), snaps_all(s), cw, routes_all(s)),
                   "w": stats(s), "ffo": ffo_all(s), "snaps": snaps_all(s), "sched": schedule(s),
                   "routes": routes_all(s), "pfr_ids": pfr_ids} for s in (cur, cur - 1)}
    own, fa = data.ownership()
    owner_info = {r["gsis_id"]: _owner_info(r["gsis_id"], r["team"], own, fa) for _, r in idx.iterrows()}
    try:
        import nflreadpy as nfl
        from mega import roles as RL

        def nextgen(kind):
            try:
                return data.to_pandas(nfl.load_nextgen_stats(seasons=[cur], stat_type=kind))
            except Exception:
                return pd.DataFrame()
        try:
            depth = data.to_pandas(nfl.load_depth_charts(seasons=[cur]))
        except Exception:
            depth = pd.DataFrame()
        role_ctx = RL.build(data.ask_player_week(cur), ffo_all(cur), nextgen("receiving"), nextgen("rushing"), depth)
    except Exception as e:
        _warn("role context", e)
        role_ctx = None
    try:
        d = dvp(cur)
    except Exception:
        d = pd.DataFrame()
    try:
        bp = blended_proj(cur, c.next_week)
    except Exception:
        bp = pd.DataFrame()
    try:
        from mega.matchup_model import week_matchups
        mm = week_matchups(cur, c.next_week)
    except Exception:
        mm = pd.DataFrame()
    return build(idx, set(c.gsis_list), owner_info, seasons, rosters(cur), role_ctx, d, bp,
                 out_for_week(c.next_week), c.sched, c.next_week, cur, player_ids.norm, mm)


EXPORTS = {
    "action_board.json": action_board, "start_sit.json": start_sit,
    "waivers.json": waivers, "trades.json": trades, "archetypes.json": archetypes,
    "roster.json": roster, "league.json": league, "glossary.json": glossary, "news.json": news,
}


def _write_players(payload: dict) -> list[Path]:
    """One small search index plus one file per player — the whole set in one file runs to
    several MB, too much for one fetch; the site's /players/[gsisId] route reads one."""
    written = []
    index_path = OUT_DIR / "players_index.json"
    index_path.write_text(json.dumps(
        {"available": payload.get("available", False), "index": payload.get("index", [])},
        indent=2, default=str))
    written.append(index_path)

    players_dir = OUT_DIR / "players"
    players_dir.mkdir(parents=True, exist_ok=True)
    for gid, detail in payload.get("players", {}).items():
        path = players_dir / f"{gid}.json"
        path.write_text(json.dumps(detail, indent=2, default=str))
        written.append(path)
    return written


def main() -> None:
    # our rest-of-season projection first — the trade and waiver engines read it
    try:
        from mega import proj_ros as _pr
        import nflreadpy as _nfl0
        _s0 = int(_nfl0.get_current_season())
        _pr.build(_s0, min(data.current_week(_s0, 1), 17))
        print(f"wrote data/proj_ros_{_s0}.json")
    except Exception as e:
        print(f"[export_web] projection build failed, engines use the last file: {e}", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = context()
    written = []
    for filename, fn in EXPORTS.items():
        try:
            payload = fn(c)
        except Exception as e:
            _warn(filename, e)
            continue
        path = OUT_DIR / filename
        path.write_text(json.dumps(payload, indent=2, default=str))
        written.append(path)
        print(f"wrote {path.relative_to(ROOT)}")
    try:
        player_files = _write_players(players(c))
        written.extend(player_files)
        print(f"wrote data/web/players_index.json + {len(player_files) - 1} data/web/players/<gsis_id>.json files")
    except Exception as e:
        _warn("player cards", e)

    _data, _season, _next_week = data, c.season, c.next_week
    from mega import trend_web as _trend_web
    trends = _trend_web.build(_data.my_norms(_data.my_roster()[0]))
    path = OUT_DIR / "trade_value_trend.json"
    path.write_text(json.dumps(trends["trade_value"], indent=2, default=str))
    written.append(path)
    print(f"wrote {path.relative_to(ROOT)}")

    # Logic (methodology) content is static and Streamlit-free too — see mega/logic.py.
    from mega import logic_web as _logic_web

    logic_path = OUT_DIR / "logic.json"
    logic_path.write_text(json.dumps(_logic_web.build(), indent=2, default=str))
    written.append(logic_path)
    print(f"wrote {logic_path.relative_to(ROOT)}")

    # Player headshots, name-keyed — see mega/headshots_web.py.
    from mega import headshots_web as _headshots_web

    headshots_path = OUT_DIR / "headshots.json"
    headshots_path.write_text(json.dumps(_headshots_web.build(_data.load_rosters(_season)),
                                         indent=2, default=str))
    written.append(headshots_path)
    print(f"wrote {headshots_path.relative_to(ROOT)}")

    # Weekly rankings — league-wide, position-ranked by projection. Streamlit-free (same
    # tier as logic/headshots/trend); see mega/rankings_web.py.
    from mega import rankings_web as _rankings_web

    _gsis_list, _ = _data.my_gsis_ids(_data.my_roster()[0])
    rankings_path = OUT_DIR / "rankings.json"
    _rankings = _rankings_web.build(_season, _next_week, set(_gsis_list))

    # Kickers and defenses (mega/kdef.py): ranked alongside everyone else, folded into my
    # start/sit, streamed on the waivers page, and added to the stats table. Outside the
    # Streamlit run like the rest of this tier; a failure here leaves those files as built.
    try:
        from mega import faab as _fb
        from mega import kdef as _kdef
        from mega import start_sit as _start_sit
        from mega.config import MY_TEAM as _MY_TEAM
        from mega.yahoo import cached_rosters as _cros0

        _kros = _kdef.rostered(_cros0())
        _rankings["rows"] = list(_rankings.get("rows") or []) + _kdef.ranking_rows(_season, _next_week, _kros, _MY_TEAM)
        _ss_path = OUT_DIR / "start_sit.json"
        _ss_path.write_text(json.dumps(_start_sit.add_kdef(
            json.loads(_ss_path.read_text()), _kdef.lineup_rows(_season, _next_week, _kros, _MY_TEAM)),
            indent=2, default=str))
        _bud = _fb.cached_budgets()
        _left = dict(zip(_bud["team"], _bud["faab_left"])) if not _bud.empty else {}
        _wv_path = OUT_DIR / "waivers.json"
        _wv = json.loads(_wv_path.read_text())
        _wv["streamers"] = _kdef.streamers(_season, _next_week, _kros, _MY_TEAM, _left,
                                           int(_left.get(_MY_TEAM, _fb.BUDGET)))
        # role changes the books priced (mega/prop_moves.py): free agents and my own players
        from mega import prop_moves as _pm
        from mega.data import ownership as _own
        _owner = _own()[0]
        _wv["prop_movers"] = [{**m, "owner": (_owner.get(m["gsis_id"]) or ("FA",))[0]}
                              for m in _pm.movers(_season, _next_week)
                              if (_owner.get(m["gsis_id"]) or ("FA",))[0] in ("FA", _MY_TEAM)]
        _wv_path.write_text(json.dumps(_wv, indent=2, default=str))
        print("wrote K/DEF into rankings, start_sit and waivers")
    except Exception as e:
        print(f"[export_web] K/DEF failed: {type(e).__name__}: {e}", file=sys.stderr)

    rankings_path.write_text(json.dumps(_rankings, indent=2, default=str))
    written.append(rankings_path)
    print(f"wrote {rankings_path.relative_to(ROOT)}")

    # Next-four-games matchup outlook per team x position, for the waiver and trade pages
    # (rest-of-season question, not this week's start/sit) — see mega/schedule_web.py.
    try:
        from mega import schedule_web as _schedule_web
        from mega.lookup import player_index as _player_index
        _pidx = _player_index(
            {_season: _data.load_player_stats(_season), _season - 1: _data.load_player_stats(_season - 1)},
            _data.load_rosters(_season))
        schedule_path = OUT_DIR / "schedule.json"
        schedule_path.write_text(json.dumps(
            _schedule_web.build(_season, _next_week, _pidx), separators=(",", ":"), default=str))
        written.append(schedule_path)
        print(f"wrote {schedule_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"[export_web] schedule outlook failed: {type(e).__name__}: {e}", file=sys.stderr)

    # Past seasons (final standings, rosters, champions, records) — pulled one year at a time
    # by tools/pull_history.py, exported from whatever is on disk; see mega/history_web.py.
    try:
        from mega import history_web as _history_web
        from mega import yahoo_history as _yh
        history_path = OUT_DIR / "history.json"
        history_path.write_text(json.dumps(
            _history_web.build(sorted(_yh.SEASON_LEAGUE_IDS)), separators=(",", ":"), default=str))
        written.append(history_path)
        print(f"wrote {history_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"[export_web] history failed: {type(e).__name__}: {e}", file=sys.stderr)

    # The playoff plan: weeks 15-17 lineups, thin spots, stashes, handcuffs (mega/playoff_plan.py).
    try:
        from mega import playoff_plan as _playoff_plan
        from mega.yahoo import cached_rosters as _cros_pp
        pp_path = OUT_DIR / "playoffs.json"
        pp_path.write_text(json.dumps(_playoff_plan.build(_season, _next_week, _cros_pp()),
                                      separators=(",", ":"), default=str))
        written.append(pp_path)
        print(f"wrote {pp_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"[export_web] playoff plan failed: {type(e).__name__}: {e}", file=sys.stderr)

    # The scorecard: every finished week's calls, read out of git as they stood at kickoff and
    # graded against what happened (mega/scorecard.py).
    try:
        from mega import scorecard as _scorecard
        sc_path = OUT_DIR / "scorecard.json"
        sc_path.write_text(json.dumps(_scorecard.build(_season, _next_week), separators=(",", ":"), default=str))
        written.append(sc_path)
        print(f"wrote {sc_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"[export_web] scorecard failed: {type(e).__name__}: {e}", file=sys.stderr)

    # The weekly digest — the week just played and the one coming, in sentences. Built
    # from the league files and the league.json written above; see mega/digest_web.py.
    try:
        from mega import digest_web as _digest_web
        from mega.config import MY_TEAM as _MY_TEAM
        from mega.yahoo import cached_fixtures as _cfx, cached_scores as _csc

        _league = json.loads((OUT_DIR / "league.json").read_text())
        _through = int((_league.get("power_xwins") or {}).get("through_week") or 0)
        if _digest_web.snapshot_odds(_season, _through, _league.get("playoff_odds") or []):
            print(f"wrote {_digest_web.ODDS_CSV.relative_to(ROOT)} (week {_through} odds)")
        _ab = json.loads((OUT_DIR / "action_board.json").read_text())
        digest = _digest_web.build(_season, _MY_TEAM, _csc(), _cfx(), _league, _ab,
                                   _digest_web.odds_history(_season))
        digest_path = OUT_DIR / "digest.json"
        digest_path.write_text(json.dumps(digest, indent=2, default=str))
        written.append(digest_path)
        print(f"wrote {digest_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"[export_web] digest failed: {type(e).__name__}: {e}", file=sys.stderr)

    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
