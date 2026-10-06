"""Kickers and team defenses: Yahoo default scoring, weekly points, projections, free agents.

The rest of the site prices QB/RB/WR/TE through the trade engine; K and DEF sit outside it
on purpose (they don't trade, and a one-slot position has no bench-depth arithmetic). What
they do need is the weekly question: who do I start, and who on waivers beats him.

Scoring is Yahoo's default (Chris 2026-10-06: "Use Yahoo defaults"):

  K    FG 0-39 yds 3 · 40-49 4 · 50+ 5 · PAT 1 (no penalty for misses)
  DEF  sack 1 · interception 2 · fumble recovery 2 · TD (defense or return) 6 · safety 2
       · blocked kick 2 · points allowed 0: 10, 1-6: 7, 7-13: 4, 14-20: 1, 21-27: 0,
       28-34: -1, 35+: -4

Projection for week w, team T against opponent O:

  DEF  points-allowed tier, expected over Normal(O's implied total, PA_SD) — the Vegas line
       where there is one, else a shrunk season average — plus each event (sacks, picks,
       recoveries) at league rate x T's rate x what O gives up, each shrunk toward the
       league with K_SHRINK games. TDs, safeties and blocks are mostly noise: league rate.
  K    his points per game, shrunk toward the league kicker, scaled by his team's implied
       total over the league's (more points, more kicks).
"""
from __future__ import annotations

import functools
import math

import pandas as pd

K_FG = {"fg_made_0_19": 3, "fg_made_20_29": 3, "fg_made_30_39": 3, "fg_made_40_49": 4,
        "fg_made_50_59": 5, "fg_made_60_": 5}
K_PAT = 1
DEF_EVENTS = {"sacks": 1, "ints": 2, "fum_rec": 2}
DEF_TD, DEF_SAFETY, DEF_BLOCK = 6, 2, 2
PA_TIERS = [(0, 0, 10), (1, 6, 7), (7, 13, 4), (14, 20, 1), (21, 27, 0), (28, 34, -1), (35, 999, -4)]

PA_SD = 9.5          # (J) sd of an NFL team's score around its implied total
K_SHRINK = 4.0       # (J) pseudo-games of league average in every rate (5 played by week 6)
LEAGUE_PTS = 22.0    # fallback league scoring average when a season has no games yet


# ======================================================================== scoring
def pa_points(pa: float) -> int:
    for lo, hi, pts in PA_TIERS:
        if lo <= pa <= hi:
            return pts
    return PA_TIERS[-1][2]


def pa_expected(mu: float, sd: float = PA_SD) -> float:
    """Expected points-allowed score when the opponent scores ~ Normal(mu, sd), rounded to
    whole points (the tier edges sit on half points)."""
    cdf = lambda x: 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))
    return sum(pts * (cdf(hi + 0.5) - cdf(lo - 0.5 if lo > 0 else -math.inf))
               for lo, hi, pts in PA_TIERS)


def k_points(row) -> float:
    return sum(float(row.get(c) or 0) * v for c, v in K_FG.items()) + float(row.get("pat_made") or 0) * K_PAT


# ======================================================================== inputs
@functools.lru_cache(maxsize=1)
def _teams() -> dict:
    """Yahoo's DEF name (the nickname) -> team abbreviation, current franchises only."""
    import nflreadpy as nfl
    from .ids import canon_team
    t = nfl.load_teams().to_pandas()
    out = {}
    for abbr, nick in zip(t["team_abbr"], t["team_nick"]):
        out.setdefault(str(nick), canon_team(abbr))      # first row is the current franchise
    return out


def _games(season: int) -> pd.DataFrame:
    """team, week, opp, scored, allowed — regular-season games played."""
    from . import season as S
    from .ids import canon_team
    s = S.schedules(season)
    s = s[s["game_type"].astype(str).str.upper() == "REG"].dropna(subset=["home_score", "away_score"])
    rows = []
    for t, o, ts, os_ in (("home_team", "away_team", "home_score", "away_score"),
                          ("away_team", "home_team", "away_score", "home_score")):
        rows.append(pd.DataFrame({"team": s[t].map(canon_team), "opp": s[o].map(canon_team),
                                  "week": s["week"].astype(int), "scored": s[ts].astype(float),
                                  "allowed": s[os_].astype(float)}))
    return pd.concat(rows, ignore_index=True)


def _team_stats(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    from .ids import canon_team
    try:
        ts = nfl.load_team_stats(seasons=[season]).to_pandas()
    except Exception:
        return pd.DataFrame()
    ts = ts[ts.get("season_type", "REG").astype(str).str.upper() == "REG"] if "season_type" in ts else ts
    g = lambda c: pd.to_numeric(ts[c], errors="coerce").fillna(0) if c in ts.columns else 0.0
    return pd.DataFrame({
        "team": ts["team"].map(canon_team), "opp": ts["opponent_team"].map(canon_team),
        "week": ts["week"].astype(int),
        "sacks": g("def_sacks"), "ints": g("def_interceptions"), "fum_rec": g("fumble_recovery_opp"),
        "tds": g("def_tds") + g("special_teams_tds"), "safeties": g("def_safeties"),
        "blocks": g("def_fg_blocks") + g("def_punt_blocks") + g("def_pat_blocks"),
    })


# ======================================================================== history
def def_weekly(season: int) -> pd.DataFrame:
    """team, week, opp, the scoring events, points allowed and Yahoo DEF points."""
    st, gm = _team_stats(season), _games(season)
    if st.empty or gm.empty:
        return pd.DataFrame(columns=["team", "week", "opp", "pa", "pts"])
    d = st.merge(gm[["team", "week", "allowed"]], on=["team", "week"], how="inner").rename(columns={"allowed": "pa"})
    d["pts"] = (sum(d[c] * v for c, v in DEF_EVENTS.items()) + d["tds"] * DEF_TD
                + d["safeties"] * DEF_SAFETY + d["blocks"] * DEF_BLOCK + d["pa"].map(pa_points))
    return d


def k_weekly(season: int) -> pd.DataFrame:
    """gsis_id, player, team, week, fg_made, fg_att, pat_made and Yahoo K points."""
    import nflreadpy as nfl
    from .ids import canon_team
    ps = nfl.load_player_stats(seasons=[season]).to_pandas()
    ps = ps[(ps["position"] == "K") & (ps.get("season_type", "REG").astype(str).str.upper() == "REG")].copy()
    if ps.empty:
        return pd.DataFrame(columns=["gsis_id", "player", "team", "week", "pts"])
    ps["pts"] = [k_points(r) for r in ps.to_dict("records")]
    return pd.DataFrame({"gsis_id": ps["player_id"], "player": ps["player_display_name"],
                         "team": ps["team"].map(canon_team), "week": ps["week"].astype(int),
                         "fg_made": ps["fg_made"], "fg_att": ps["fg_att"], "pat_made": ps["pat_made"],
                         "pts": ps["pts"]})


# ======================================================================== projection
def _shrunk(total: float, n: float, prior: float) -> float:
    return (total + K_SHRINK * prior) / (n + K_SHRINK)


def project(season: int, weeks: list[int]) -> dict:
    return _project(int(season), tuple(int(w) for w in weeks))


@functools.lru_cache(maxsize=8)
def _project(season: int, weeks: tuple) -> dict:
    """{"DEF": DataFrame[team, week, opp, proj, pa_mu], "K": DataFrame[gsis_id, player,
    team, week, opp, proj]} for every team / current kicker in each of `weeks`."""
    from . import season as S
    from .forward import implied
    from .ids import canon_team

    d, gm = def_weekly(season), _games(season)
    sched = S.schedules(season)
    sched = sched[sched["game_type"].astype(str).str.upper() == "REG"]
    imp = implied(sched)
    imp_of = {(t, int(w)): float(v) for t, w, v in zip(imp["team"], imp["week"], imp["implied"]) if pd.notna(v)}
    opp_of = {}
    for h, a, w in zip(sched["home_team"], sched["away_team"], sched["week"]):
        opp_of[(canon_team(h), int(w))] = canon_team(a)
        opp_of[(canon_team(a), int(w))] = canon_team(h)

    lg_pts = float(gm["scored"].mean()) if not gm.empty else LEAGUE_PTS
    n_g = d.groupby("team").size() if not d.empty else pd.Series(dtype=float)
    lg = {e: float(d[e].mean()) for e in (*DEF_EVENTS, "tds", "safeties", "blocks")} if not d.empty else {}

    def rate(df, key, col, prior):
        g = df[df[key[0]] == key[1]]
        return _shrunk(float(g[col].sum()), len(g), prior)

    def mu_pa(o, w):
        if (o, w) in imp_of:
            return imp_of[(o, w)]
        scored = gm[gm["team"] == o]
        return _shrunk(float(scored["scored"].sum()), len(scored), lg_pts)

    def_rows = []
    for team in sorted(set(t for t, _ in opp_of)):
        for w in weeks:
            o = opp_of.get((team, w))
            if o is None:
                def_rows.append({"team": team, "week": w, "opp": None, "proj": 0.0, "pa_mu": None})
                continue
            mu = mu_pa(o, w)
            ev = 0.0
            for e, v in DEF_EVENTS.items():
                base = lg.get(e, 0.0)
                if base <= 0:
                    continue
                mine = rate(d, ("team", team), e, base)          # what T's defense makes
                given = rate(d, ("opp", o), e, base)             # what O's offense gives up
                ev += v * base * (mine / base) * (given / base)
            ev += lg.get("tds", 0) * DEF_TD + lg.get("safeties", 0) * DEF_SAFETY + lg.get("blocks", 0) * DEF_BLOCK
            def_rows.append({"team": team, "week": w, "opp": o, "proj": round(ev + pa_expected(mu), 2),
                             "pa_mu": round(mu, 1)})

    k = k_weekly(season)
    k_rows = []
    if not k.empty:
        lg_k = float(k["pts"].mean())
        last = k.sort_values("week").groupby("team").tail(1)          # the team's current kicker
        lg_imp = sum(imp_of.values()) / len(imp_of) if imp_of else lg_pts
        for r in last.itertuples():
            mine = k[k["gsis_id"] == r.gsis_id]
            ppg = _shrunk(float(mine["pts"].sum()), len(mine), lg_k)
            for w in weeks:
                o = opp_of.get((r.team, w))
                scale = (imp_of.get((r.team, w), lg_imp) / lg_imp) if o else 0.0
                k_rows.append({"gsis_id": r.gsis_id, "player": r.player, "team": r.team, "week": w,
                               "opp": o, "proj": round(ppg * scale, 2)})
    return {"DEF": pd.DataFrame(def_rows), "K": pd.DataFrame(k_rows)}


# ======================================================================== rosters
def rostered(yahoo_rosters: pd.DataFrame) -> pd.DataFrame:
    """fantasy team, slot, pos (K/DEF), and the key the projections use: the team abbreviation
    for a defense, the kicker's name for a kicker."""
    r = yahoo_rosters
    is_k = (r["slot"] == "K") | (r.get("pos") == "K")
    is_d = (r["slot"] == "DEF") | (r.get("pos") == "DEF")
    nick = _teams()
    out = []
    for row in r[is_k | is_d].itertuples():
        pos = "K" if bool(is_k.loc[row.Index]) else "DEF"
        key = row.player if pos == "K" else nick.get(str(row.player))
        out.append({"fantasy_team": row.team, "slot": row.slot, "pos": pos, "player": row.player, "key": key})
    return pd.DataFrame(out, columns=["fantasy_team", "slot", "pos", "player", "key"])


# ======================================================================== the pages
STREAM_WEEKS = 3     # (J) a streamer is judged on the next three weeks, not just the next game
STREAM_SHOW = 6      # free agents listed per position


def _week_table(season: int, week: int, ros: pd.DataFrame, my_team: str) -> list[dict]:
    """Every defense and current kicker in `week`: projection, opponent, rank within the
    position, and who holds him — the rows the rankings page and start/sit read."""
    p = project(season, [week])
    held = {(r.pos, r.key): r.fantasy_team for r in ros.itertuples()}
    nick = {v: k for k, v in _teams().items()}
    rows = []
    for pos, df in p.items():
        if df.empty:
            continue
        df = df.assign(rank=df["proj"].rank(ascending=False, method="min").astype(int)).sort_values("rank")
        for r in df.itertuples():
            key = r.team if pos == "DEF" else r.player
            owner = held.get((pos, key))
            rows.append({"gsis_id": f"DEF-{r.team}" if pos == "DEF" else r.gsis_id,
                         "player": nick.get(r.team, r.team) if pos == "DEF" else r.player,
                         "pos": pos, "team": r.team, "rank": int(r.rank), "proj": round(float(r.proj), 1),
                         "proj_source": "kdef", "opponent": r.opp if isinstance(r.opp, str) else None, "matchup_pct": None,
                         "matchup_basis": ("Vegas total" if pos == "K" else f"opp. implied {r.pa_mu}")
                         if isinstance(r.opp, str) else "bye",
                         "mine": owner == my_team, "owner": owner or "FA", "key": key})
    return rows


def ranking_rows(season: int, week: int, ros: pd.DataFrame, my_team: str) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in ("owner", "key")} for r in _week_table(season, week, ros, my_team)]


def lineup_rows(season: int, week: int, ros: pd.DataFrame, my_team: str) -> list[dict]:
    """My kickers and defenses for start/sit: the better of each starts (one K, one DEF)."""
    table = {(r["pos"], r["key"]): r for r in _week_table(season, week, ros, my_team)}
    mine = ros[ros["fantasy_team"] == my_team]
    out = []
    for pos in ("K", "DEF"):
        mine_pos = [(table.get((pos, k)) or {}) | {"player_name": n} for k, n in
                    zip(mine[mine["pos"] == pos]["key"], mine[mine["pos"] == pos]["player"])]
        mine_pos.sort(key=lambda r: -(r.get("proj") or 0.0))
        for i, r in enumerate(mine_pos):
            proj = float(r.get("proj") or 0.0)
            out.append({"lineup": pos if i == 0 else "BENCH", "player": r["player_name"], "pos": pos,
                        "nfl_team": r.get("team"), "opp": r.get("opponent"), "proj": proj,
                        "proj_adj": round(proj, 1), "proj_source": "kdef", "start": i == 0,
                        "close_call": ""})
    return out


def streamers(season: int, week: int, ros: pd.DataFrame, my_team: str, faab_left: dict,
              budget_left: int) -> dict:
    """The best free-agent defenses and kickers against the one I'd start, over the next
    STREAM_WEEKS weeks, each bid against the other rosters by the same rule as every other
    claim (waiver_value.bid_vs_rivals): a rival's interest is what the streamer adds over
    his own best at the position, priced on the same faab curve over the same window.

    A team needs one kicker and one defense, so each rival bids only on the streamer that
    helps him most — counting every rival on every streamer had all six "likely outbid"."""
    from . import faab as fb
    from .waiver_value import RIVAL_MIN, bid_vs_rivals

    weeks = [w for w in range(int(week), int(week) + STREAM_WEEKS) if w <= fb.LAST_WEEK]
    if not weeks:
        return {"available": False, "weeks": [], "K": None, "DEF": None}
    price_week = fb.LAST_WEEK - len(weeks) + 1              # faab.suggest prices len(weeks) weeks
    p = project(season, weeks)
    nick = {v: k for k, v in _teams().items()}
    out = {"available": True, "weeks": weeks}
    for pos, col in (("DEF", "team"), ("K", "player")):
        P = p[pos]
        by = {(k, int(w)): float(v) for k, w, v in zip(P[col], P["week"], P["proj"])}
        opp = {(k, int(w)): (o if isinstance(o, str) else None) for k, w, o in zip(P[col], P["week"], P["opp"])}
        held = ros[ros["pos"] == pos]
        taken = set(held["key"])

        def best(team, w):
            return max((by.get((k, w), 0.0) for k in held[held["fantasy_team"] == team]["key"]), default=0.0)

        rivals = [t for t in held["fantasy_team"].unique() if t != my_team]
        mine_w = {w: best(my_team, w) for w in weeks}
        free = sorted(set(P[col]) - taken)
        chase = {}                                        # free agent -> [(rival, his bid)]
        for t in rivals:
            rg = {k: sum(max(0.0, by.get((k, w), 0.0) - best(t, w)) for w in weeks) / len(weeks) for k in free}
            k_best = max(rg, key=rg.get, default=None)
            if k_best is not None and rg[k_best] >= RIVAL_MIN:
                chase.setdefault(k_best, []).append((t, fb.suggest(rg[k_best], int(faab_left.get(t, 0)), price_week)["bid"]))
        rows = []
        for k in free:
            gains = [by.get((k, w), 0.0) - mine_w[w] for w in weeks]
            g1, g3 = gains[0], sum(max(0.0, g) for g in gains) / len(gains)
            if g3 <= 0.05:
                continue
            my = fb.suggest(g3, budget_left, price_week)
            bids = chase.get(k, [])
            n = len(bids)
            who, top = max(bids, key=lambda x: x[1]) if bids else ("", 0)
            bid, note = bid_vs_rivals(top, who, n, g3, my["max_worth"])
            rows.append({"player": nick.get(k, k) if pos == "DEF" else k, "team": k if pos == "DEF" else
                         P[P["player"] == k]["team"].iloc[0], "pos": pos,
                         "weeks": [{"week": w, "opp": opp.get((k, w)), "proj": round(by.get((k, w), 0.0), 1)} for w in weeks],
                         "gain_next": round(g1, 1), "gain_avg": round(g3, 2),
                         "bid": bid, "max_bid": my["max_worth"], "rival_top": top, "rival_team": who,
                         "rivals_n": n, "bid_note": note})
        rows.sort(key=lambda r: (-r["gain_avg"], -r["gain_next"]))
        mine = held[held["fantasy_team"] == my_team]
        out[pos] = {"mine": [{"player": n, "weeks": [{"week": w, "opp": opp.get((k, w)), "proj": round(by.get((k, w), 0.0), 1)}
                                                     for w in weeks]} for n, k in zip(mine["player"], mine["key"])],
                    "rows": rows[:STREAM_SHOW]}
    return out


SEASON_METRICS = [
    # (column, label, kind, positions, means) — appended to mega/stats_web.py METRICS
    ("fg_made", "FG made", "int", ("K",), ""),
    ("fg_pct", "FG %", "pct", ("K",), ""),
    ("fg_50", "FG 50+ made", "int", ("K",), ""),
    ("pat_made", "Extra points made", "int", ("K",), ""),
    ("sacks_pg", "Sacks/game", "num", ("DEF",), ""),
    ("takeaways_pg", "Takeaways/game", "num", ("DEF",), "Interceptions + fumble recoveries."),
    ("pa_pg", "Points allowed/game", "num", ("DEF",), ""),
    ("def_tds", "Defense/return TD", "int", ("DEF",), ""),
]


def season_rows(season: int, ros: pd.DataFrame, my_team: str) -> list[dict]:
    """One row per kicker / defense for the stats table: games, Yahoo points and the stats
    behind them, keyed the way mega/stats_web.py rows are."""
    held = {(r.pos, r.key): r.fantasy_team for r in ros.itertuples()} if ros is not None else {}
    nick = {v: k for k, v in _teams().items()}
    out = []
    d = def_weekly(season)
    for team, g in d.groupby("team"):
        n = len(g)
        owner = held.get(("DEF", team))
        out.append({"gsis_id": f"DEF-{team}", "name": nick.get(team, team), "pos": "DEF", "team": team,
                    "season": season, "owner": owner or "FA", "mine": owner == my_team,
                    "games": n, "pts": float(g["pts"].sum()), "pts_pg": float(g["pts"].mean()),
                    "sacks_pg": float(g["sacks"].mean()), "takeaways_pg": float((g["ints"] + g["fum_rec"]).mean()),
                    "pa_pg": float(g["pa"].mean()), "def_tds": int(g["tds"].sum())})
    k = k_weekly(season)
    import nflreadpy as nfl
    try:
        ps = nfl.load_player_stats(seasons=[season]).to_pandas()
        ps = ps[ps["position"] == "K"]
        f50 = ps.assign(f=pd.to_numeric(ps["fg_made_50_59"], errors="coerce").fillna(0)
                        + pd.to_numeric(ps["fg_made_60_"], errors="coerce").fillna(0)).groupby("player_id")["f"].sum()
    except Exception:
        f50 = pd.Series(dtype=float)
    for gid, g in k.groupby("gsis_id"):
        name = g["player"].iloc[-1]
        owner = held.get(("K", name))
        att = float(pd.to_numeric(g["fg_att"], errors="coerce").sum())
        made = float(pd.to_numeric(g["fg_made"], errors="coerce").sum())
        out.append({"gsis_id": gid, "name": name, "pos": "K", "team": g["team"].iloc[-1], "season": season,
                    "owner": owner or "FA", "mine": owner == my_team, "games": len(g),
                    "pts": float(g["pts"].sum()), "pts_pg": float(g["pts"].mean()),
                    "fg_made": int(made), "fg_pct": made / att if att else None,
                    "fg_50": int(f50.get(gid, 0)), "pat_made": int(pd.to_numeric(g["pat_made"], errors="coerce").sum())})
    return out
