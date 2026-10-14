"""Playoff plan: TaylorMade's lineup for weeks 15-17, where it's thin, and what to stash.

Every number is the trade engine's weekly expected points (mega/proj_ros.py through
trade_league.build_league): 0 on a bye, discounted by the chance he plays, so a starter's
injury risk between now and the playoffs is already in it. Lineups are the engine's own
(Mega Bowl's slots, RB/WR flex), scored on starters only — bench depth wins no playoff game.

  * **Lineup by week**: the best lineup each playoff week from the roster plus anyone on IR
    (they're back by then or they aren't, and the projection says which), with each starter's
    opponent and how that defense has treated his position this season.
  * **League rank**: every roster's playoff lineup on the same terms.
  * **Thin spots**: positions where the playoff starters sit THIN_GAP or more below the
    league's average starters there.
  * **Stashes**: free agents ranked by what they'd add to the playoff lineup, summed over the
    three weeks — a player you'd start in week 16 against a soft defense shows up here even if
    he's no help now.
  * **Handcuffs**: the named backup behind each playoff starter, and who holds him.
"""
from __future__ import annotations

from .title_odds import PLAYOFF_WEEKS

THIN_GAP = 1.5       # (J) pts/wk below the league's average starters at a position to call it thin
STASHES = 8
CUFF_POS = ("RB", "QB")   # a backup who inherits the job: a receiver's "handcuff" inherits nothing
EASY, TOUGH = 8, 25  # opponent ease rank (1 = allows the most to the position) for the tags


def _opponents(season: int, weeks) -> dict:
    """(team, week) -> opponent."""
    import nflreadpy as nfl
    s = nfl.load_schedules(seasons=[season]).to_pandas()
    s = s[s["week"].isin(list(weeks)) & (s["game_type"] == "REG")]
    out = {}
    for _, g in s.iterrows():
        out[(g["home_team"], int(g["week"]))] = g["away_team"]
        out[(g["away_team"], int(g["week"]))] = g["home_team"]
    return out


def _ease(season: int) -> dict:
    """(defense, pos) -> ease rank, 1 = allows the most points to that position."""
    from .lineup import defense_vs_position
    d = defense_vs_position(season)
    return {(r.defense, r.pos): int(r.ease_rank) for r in d.itertuples()} if not d.empty else {}


def _tag(rank: int | None) -> str | None:
    if rank is None:
        return None
    return "easy" if rank <= EASY else "tough" if rank >= TOUGH else None


def build(season: int, now: int, yahoo_rosters=None) -> dict:
    from . import trade_engine as te
    from . import trade_league as tl

    lg = tl.build_league(season, yahoo_rosters)
    me = (lg.get("report") or {}).get("my_team_id")
    if not lg.get("teams") or me is None or now > PLAYOFF_WEEKS[-1]:
        return {"available": False}
    ctx = te.build_context(lg, tl.engine_config())
    weeks = list(PLAYOFF_WEEKS)
    opp, ease = _opponents(season, weeks), _ease(season)
    P = ctx.players

    def squad(tid):
        return list(ctx.teams[tid]["roster"]) + list(ctx.ir.get(tid, []))

    def pts(ids, w):
        return te.lineup(ids, ctx, w).starter_pts

    def starter_row(e, w, on_ir):
        pid, pos, val, slot = e[0], e[1], e[2], e[3]
        team = P[pid].get("nfl")
        o = opp.get((team, w))
        rank = ease.get((o, pos)) if o else None
        return {"name": P[pid]["name"], "pos": pos, "slot": slot, "pts": round(val, 1), "team": team,
                "opp": o, "ease": rank, "tag": _tag(rank) if o else "bye", "ir": pid in on_ir}

    mine = squad(me)
    on_ir = set(ctx.ir.get(me, []))
    lineups = []
    for w in weeks:
        lu = te.lineup(mine, ctx, w)
        lineups.append({"week": w, "total": round(lu.starter_pts, 1),
                        "starters": [starter_row(e, w, on_ir) for e in lu.starters]})

    # every roster on the same terms
    totals = {t["name"]: sum(pts(squad(tid), w) for w in weeks) / len(weeks) for tid, t in ctx.teams.items()}
    ranked = sorted(totals, key=lambda n: -totals[n])
    my_name = ctx.teams[me]["name"]

    # thin spots: my playoff starters at each position against the league's
    by_pos: dict = {}
    for tid in ctx.teams:
        for w in weeks:
            for e in te.lineup(squad(tid), ctx, w).starters:
                by_pos.setdefault(e[1], {}).setdefault(tid, []).append(e[2])
    thin = []
    for pos, teams in by_pos.items():
        avg = {tid: sum(v) / len(weeks) for tid, v in teams.items()}
        league = sum(avg.values()) / len(ctx.teams)
        gap = league - avg.get(me, 0.0)
        if gap >= THIN_GAP:
            thin.append({"pos": pos, "mine": round(avg.get(me, 0.0), 1), "league": round(league, 1),
                         "gap": round(gap, 1)})
    thin.sort(key=lambda r: -r["gap"])

    # stashes: what each free agent adds to the playoff lineup
    base = {w: pts(mine, w) for w in weeks}
    stash = []
    for f in ctx.fa:
        by_week = {w: pts(mine + [f], w) - base[w] for w in weeks}
        gain = sum(by_week.values())
        if gain > 0.5:
            team = P[f].get("nfl")
            stash.append({"name": P[f]["name"], "pos": P[f]["pos"], "team": team, "gain": round(gain, 1),
                          "by_week": [{"week": w, "gain": round(g, 1), "opp": opp.get((team, w)),
                                       "tag": _tag(ease.get((opp.get((team, w)), P[f]["pos"])))
                                       if opp.get((team, w)) else "bye"} for w, g in by_week.items()]})
    stash.sort(key=lambda r: -r["gain"])

    # handcuffs behind the playoff starters
    cuffs = []
    try:
        from . import season as S
        cuff = S.cuff_lookup(season)
        names = dict(zip(*(S.player_week(season)[c] for c in ("gsis_id", "player"))))
    except Exception:
        cuff, names = {}, {}
    owner = {pid: t["name"] for t in ctx.teams.values() for pid in list(t["roster"]) + list(ctx.ir.get(t["id"], []))}
    starters = {e["name"]: e for lu in lineups for e in lu["starters"]}
    by_starter = {c["cuff_of"]: (g, c) for g, c in cuff.items()}
    for pid in mine:
        if P[pid]["name"] in starters and pid in by_starter and P[pid]["pos"] in CUFF_POS:
            g, c = by_starter[pid]
            cuffs.append({"starter": P[pid]["name"], "cuff": (P.get(g) or {}).get("name") or names.get(g, g),
                          "clear_two": c["clear_two"], "owner": owner.get(g, "free agent"),
                          "mine": owner.get(g) == my_name})
    return {
        "available": True, "season": season, "weeks": weeks,
        "lineups": lineups,
        "rank": {"mine": ranked.index(my_name) + 1, "of": len(ranked), "avg": round(totals[my_name], 1),
                 "league": round(sum(totals.values()) / len(totals), 1),
                 "best": {"team": ranked[0], "avg": round(totals[ranked[0]], 1)}},
        "thin": thin, "stashes": stash[:STASHES], "handcuffs": cuffs,
    }
