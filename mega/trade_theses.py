"""HANDOFF v1.3 Pass 4 — trade theses: offers from the engine, ranked by title odds, each
carrying the reason it helps TaylorMade, where we disagree with consensus, and what would
prove it wrong.

The old "offers the league is set up for" list was FantasyCalc fairness with the lineup
delta printed beside it and ignored — 10 of the 15 offers it showed LOWERED our lineup.
Here the order is reversed: the engine (the same roster-rebuild-and-re-optimise valuation
as the live search) generates every 1-for-1 and 2-for-1 that raises our rest-of-season
points; the best 25 of those are played through the title-odds simulation for both teams;
and a card survives only if it raises OUR title odds beyond the simulation's own noise —
or is a CONSOLIDATION, where the value is the roster spot the deal opens.

Every surviving card carries:
  * 1–2 thesis tags (§6.2), the lead one templated into a one-line thesis
  * this week Δpts, ΔROS pts/wk and ΔTitle — for us AND for them, each labelled (the old
    card printed their number next to our player's name)
  * a kill condition generated from the metric that drives the lead tag
  * a pitch line from THEIR lineup and playoff odds, P(accept) and the reasons behind it
    (mega/trade_worth.py, the same function the live search uses)
  * our rank vs ECR per player — the "we disagree" wording only where the backtest earned it

Every constant marked (J) is a judgment default; the acceptance odds are estimates, tuned
on offers Chris labelled (docs/trade_logic_v2.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import trade_engine as te
from . import trade_worth as TW

ROOT = Path(__file__).resolve().parents[1]

TOP_SIM = 25                  # handoff §5: ΔTitle only for the top 25 (by expected ΔROS since 2026-10-06)
SHAPES = ["1-for-1", "2-for-1"]
SELL_HIGH_GAP = 4.0           # pts − xFP per game (existing site threshold)
SELL_HIGH_MIN_PTS = 8.0
SELL_HIGH_MIN_GAMES = 2
TRAJ_QUANTILE = 0.80          # top quintile of share slope at his position
FRAGILE_AGE = 28              # (J) a starter this old counts as fragile for CONTINGENCY
PLAYOFF_SOS = 1.10            # (J) weeks 15–17 opponents allow 10%+ above average
STACK_TEAM = 3                # handoff §6.2: 3+ starters from one offense
MAX_PER_PARTNER = 3           # (J) keep the list from being one manager's roster
MAX_PER_GIVE = 3              # (J) ...or one of our players' showcase: weeks 3-7 of 2026 the board
                              # offered Nabers in 12 of 25 ideas and his breakout sank half of it
SIM_PER_PARTNER = 5           # (J) ...and the 25 simulations from being one manager's either
SIM_PER_GIVE = 6              # (J) room for the title odds to re-rank before MAX_PER_GIVE cuts
FAIR_MAX = 1.30               # (J) they get at most 130% of what they give, by name (trade_worth
                              # ratio). Ranking by expected gain rewards paying
                              # more (the yes gets likelier) and nothing charged us for the value
                              # handed over: every card became a 150-180% overpay (Chris 2026-10-06)
# Lead-tag priority. LINEUP (our own fallback: the incoming player simply starts) outranks
# PORTFOLIO so "spreads your byes" rides as the second reason rather than the headline.
TAG_ORDER = ["SELL_HIGH", "BUY_LOW", "ROLE_EXPIRY", "TRAJECTORY", "CONTINGENCY",
             "PLAYOFF_SCHEDULE", "CONSOLIDATION", "LINEUP", "PORTFOLIO"]


# ======================================================================== player facts
def player_facts(season: int, now: int) -> dict:
    """gsis_id -> the numbers the tags and kill conditions are written from."""
    from . import proj_ros
    from . import season as S

    pw = S.player_week(season)
    agg = pw.groupby("gsis_id").agg(games=("week", "nunique"), pts=("half_ppr", "sum"),
                                    xfp=("half_ppr_exp", "sum"), team=("team", "last"), name=("player", "last"))
    agg["pts_pg"] = agg["pts"] / agg["games"].clip(lower=1)
    agg["xfp_pg"] = agg["xfp"] / agg["games"].clip(lower=1)
    agg["gap_pg"] = agg["pts_pg"] - agg["xfp_pg"]

    g = proj_ros.games(season)
    share = pd.DataFrame()
    if g is not None and not g.empty:
        t = g[[c for c in g.columns if c.startswith("t_")]].sum(axis=1)
        g = g.assign(share=np.where(t > 0, g["xfp"] / t.replace(0, np.nan), 0.0))
        rows = []
        for gid, x in g[g["week"] < now].sort_values("week").groupby("gsis_id"):
            s = x["share"].to_numpy(float)
            slope = float(np.polyfit(np.arange(len(s)), s, 1)[0]) if len(s) >= 2 else 0.0
            rows.append({"gsis_id": gid, "pos": x["pos"].iloc[-1], "share": float(s[-min(3, len(s)):].mean()),
                         # the games before the last three (two at least), for the role-risk flag
                         "share_early": float(s[:-3].mean()) if len(s) >= 5 else float("nan"),
                         "share_sd": float(s.std(ddof=1)) if len(s) >= 2 else float("nan"),
                         "slope": slope, "share_first": float(s[0]), "share_last": float(s[-1])})
        share = pd.DataFrame(rows)
        if not share.empty:
            share["slope_q"] = share.groupby("pos")["slope"].rank(pct=True)
    out = {gid: dict(r) for gid, r in agg.iterrows()}
    for r in share.to_dict("records"):
        out.setdefault(r["gsis_id"], {}).update(r)
    proj = (proj_ros.cached(season) or {}).get("players") or {}
    for gid, p in proj.items():
        d = out.setdefault(gid, {})
        d.update({"our_rank": p.get("our_rank"), "ecr_rank": p.get("ecr_rank"),
                  "usage_rank": p.get("usage_rank"), "ros_pg": p.get("ros_pg"),
                  "disagreement": p.get("disagreement") or {}, "proj_team": p.get("team"),
                  "byes": [int(w) for w, v in (p.get("weeks") or {}).items() if v.get("bye")]})
    return out


# ======================================================================== our-side flags
ROLE_DROP = 0.25        # (J) last 3 games' share of the offense this far below the earlier games'
ROLE_MIN_SHARE = 0.03   #     ...and at least this many points of share: the role is shrinking
REBOUND_GAP = 3.0       # (J) pts/g the projection sits above what he has scored so far...
REBOUND_REL = 0.25      #     ...and at least this share of it: the value is a bet on a rebound
REBOUND_MIN_GAMES = 3


def slot_typical(ctx) -> dict:
    """Median per-game projection of each starting slot (QB, RB1, RB2, ..., FLEX) across the
    league's lineups as they stand — what "thin" is measured against."""
    from statistics import median
    by = {}
    for t in ctx.teams.values():
        for slot, pid in _slots(ctx, ctx.base[t["id"]].ids):
            by.setdefault(slot, []).append(float(ctx.players[pid].get("ppg") or 0.0))
    return {k: median(v) for k, v in by.items()}


def _slots(ctx, ids) -> list:
    """[(slot label, pid)] for a roster's starters: RB1/RB2 and WR1/WR2 by order, then FLEX."""
    out, seen = [], {}
    for pid, pos, _v, slot in te.lineup(ids, ctx, None).starters:
        if slot == "FLEX":
            out.append(("FLEX", pid))
            continue
        seen[pos] = seen.get(pos, 0) + 1
        out.append((f"{pos}{seen[pos]}" if ctx.cfg["slots"].get(pos, 1) > 1 else pos, pid))
    return out


def thinnest(ctx, ids, typical: dict) -> dict | None:
    """The starter furthest below the league's typical player in that slot, as a share of it.
    A slot a free agent already beats is a waiver pickup, not a trade need, and is skipped
    (Mahomes on waivers over Bryce Young)."""
    best_fa = {}
    for i in ctx.fa:
        pos = ctx.players[i]["pos"]
        best_fa[pos] = max(best_fa.get(pos, 0.0), te.value_at(ctx.players[i], None))
    worst = None
    for slot, pid in _slots(ctx, ids):
        p = ctx.players[pid]
        ppg, typ = float(p.get("ppg") or 0.0), typical.get(slot)
        if not typ or best_fa.get(p["pos"], 0.0) >= ppg:
            continue
        gap = ppg / typ - 1.0
        if worst is None or gap < worst["gap"]:
            worst = {"slot": slot, "id": pid, "name": p["name"], "ppg": round(ppg, 1),
                     "typical": round(typ, 1), "gap": round(gap, 3)}
    return worst


def back_week(p: dict, now: int):
    """("out", week back | None for the season) for a player not playing now, else None."""
    if p.get("out"):
        return "out", p["out"].get("back")
    wk = p.get("weekly") or {}
    if wk and float(wk.get(now, wk.get(str(now), 1.0)) or 0.0) == 0.0 and p.get("bye") != now:
        nxt = sorted(int(w) for w, v in wk.items() if int(w) > now and float(v or 0) > 0)
        return "out", (nxt[0] if nxt else None)
    return None


def our_flags(ctx, give_ids, get_ids, ref_ids, after_ids, facts: dict, now: int, typical: dict) -> list[dict]:
    """What a card should warn me about before I send it (docs/trade_logic_v2.md step 4):
    a player I'd get whose role is shrinking, or whose value is a bet on a rebound; injured
    players and whether they're back for weeks 15–17; and my thinnest starting spot before
    and after the trade."""
    out = []
    name = lambda pid: ctx.players[pid]["name"]
    for pid in get_ids:
        f, p = facts.get(pid) or {}, ctx.players[pid]
        early, recent = f.get("share_early"), f.get("share")
        if (p.get("pos") in ("RB", "WR", "TE") and not back_week(p, now)   # hurt, not demoted
                and early == early and early is not None and recent is not None
                and early - recent >= max(ROLE_MIN_SHARE, ROLE_DROP * early)):
            out.append({"kind": "role_risk", "text": f"Role risk: {name(pid)}'s share of the {f.get('team') or p.get('nfl')} "
                        f"offense fell from {early:.0%} to {recent:.0%} over the last three games."})
        ros, pts, games = f.get("ros_pg"), f.get("pts_pg"), f.get("games") or 0
        if ros and pts is not None and games >= REBOUND_MIN_GAMES and ros - pts >= max(REBOUND_GAP, REBOUND_REL * pts):
            out.append({"kind": "rebound", "text": f"Rebound bet: projected at {ros:.1f}/g from here, against "
                        f"{pts:.1f}/g so far."})
    for side, ids in (("get", get_ids), ("give", give_ids)):
        for pid in ids:
            b = back_week(ctx.players[pid], now)
            if not b:
                continue
            wk = b[1]
            when = ("out for the season" if wk is None else f"out until week {wk}, back for the playoffs" if wk <= 14
                    else f"out until week {wk}, back for only part of the playoffs" if wk <= 17 else "out for the season")
            out.append({"kind": "injury", "text": (f"{name(pid)} is {when}." if side == "get"
                                                   else f"You'd give up {name(pid)}, {when}.")})
    before, after = thinnest(ctx, ref_ids, typical), thinnest(ctx, after_ids, typical)
    if before and after:
        line = lambda t: f"{t['slot']} ({t['name']}, {t['ppg']:.1f}/g vs a typical {t['slot']}'s {t['typical']:.1f})"
        if after["id"] == before["id"] and after["slot"] == before["slot"]:
            text = f"Your weakest starter stays {line(before)}."
        elif after["gap"] > before["gap"]:
            text = (f"Fixes your weakest starter, {line(before)}; "
                    + ("every starter is at or above typical after." if after["gap"] >= 0 else f"weakest after: {line(after)}."))
        else:
            text = f"Your weakest starter goes from {line(before)} to {line(after)}."
        out.append({"kind": "need", "text": text})
    return out


def defense_vs(season: int, now: int, pos: str) -> dict:
    """Defense -> points allowed to `pos` per game over league average, shrunk 4 games —
    the same read the backtest's forward-schedule test used."""
    from . import proj_ros
    from . import season as S
    g = proj_ros.games(season)
    if g is None or g.empty:
        return {}
    sc = S.schedules(season)
    sc = sc[sc["game_type"] == "REG"]
    opp = pd.concat([sc[["week", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opp"}),
                     sc[["week", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opp"})])
    opp["team"], opp["opp"] = opp["team"].map(proj_ros._canon), opp["opp"].map(proj_ros._canon)
    grp = g[(g["pos"] == pos) & (g["week"] < now)].groupby(["team", "week"])["pts"].sum().reset_index()
    grp = grp.merge(opp, on=["team", "week"], how="left")
    if grp.empty:
        return {}
    lg = grp["pts"].mean()
    a = grp.groupby("opp")["pts"].agg(["sum", "size"])
    return (((a["sum"] + 4 * lg) / (a["size"] + 4)) / lg).to_dict()


def opponents(season: int, team: str, weeks) -> list:
    from . import proj_ros
    from . import season as S
    sc = S.schedules(season)
    sc = sc[(sc["game_type"] == "REG") & sc["week"].isin(list(weeks))]
    out = []
    for r in sc.itertuples():
        h, a = proj_ros._canon(r.home_team), proj_ros._canon(r.away_team)
        if h == team:
            out.append(a)
        elif a == team:
            out.append(h)
    return out


# ======================================================================== one card
def _names(ctx, ids):
    return [ctx.players[i]["name"] for i in ids]


def tags_for(c: dict, ctx, facts: dict, env: dict) -> list[dict]:
    """Every thesis tag that fires for this trade, each with its thesis line and kill."""
    out = []
    give, get = c["giveIds"], c["getIds"]
    lens = env["schedule_lens"]
    for pid in give:
        f, name = facts.get(pid, {}), ctx.players[pid]["name"]
        er, orr = f.get("ecr_rank"), f.get("our_rank")
        if (f.get("gap_pg", 0) >= SELL_HIGH_GAP and f.get("pts_pg", 0) >= SELL_HIGH_MIN_PTS
                and f.get("games", 0) >= SELL_HIGH_MIN_GAMES and er and orr and er < orr):
            out.append({"tag": "SELL_HIGH", "player": name,
                        "thesis": f"Sell {name} before the touchdowns dry up: {f['gap_pg']:+.1f} pts/g over what "
                                  f"his usage earns, and consensus still ranks him {ctx.players[pid]['pos']}{er} "
                                  f"(we have him {ctx.players[pid]['pos']}{orr}).",
                        "kill": f"Wrong if his expected points rise to meet his scoring — xFP at or above "
                                f"{f['pts_pg'] - 2:.1f}/g over the next two games (now {f['xfp_pg']:.1f})."})
        tm = f.get("proj_team")
        for mate, a in env["avail"].items():
            mf = facts.get(mate, {})
            if (mate != pid and mf.get("proj_team") == tm and ctx.players.get(mate, {}).get("pos") == ctx.players[pid]["pos"]
                    and a["back"] <= env["last_week"] and (mf.get("ros_pg") or 0) >= 0.8 * (f.get("ros_pg") or 0)
                    and f.get("share_last", 0) > f.get("share_first", 0)):
                mname = mf.get("name") or ctx.players.get(mate, {}).get("name", mate)
                out.append({"tag": "ROLE_EXPIRY", "player": name,
                            "thesis": f"Sell {name}'s borrowed role: his share has run {f['share_last']:.0%} with "
                                      f"{mname} out, and {mname} is due back week {a['back']}.",
                            "kill": f"Wrong if {mname} isn't back by week {a['back']}, or {name} holds "
                                    f"{f['share_last']:.0%} of {tm}'s expected points after he is."})
                break
    for pid in get:
        f, name, pos = facts.get(pid, {}), ctx.players[pid]["name"], ctx.players[pid]["pos"]
        er, orr, ur = f.get("ecr_rank"), f.get("our_rank"), f.get("usage_rank")
        sd = f.get("share_sd")
        kill_share = (f"Wrong if his share of his team's expected points falls below "
                      f"{max(0.0, f.get('share', 0) - (sd if sd == sd and sd else 0.03)):.0%} "
                      f"(now {f.get('share', 0):.0%}, minus one sd).") if f.get("share") is not None else \
            "Wrong if his role shrinks over the next two games."
        if f.get("gap_pg", 0) <= -SELL_HIGH_GAP and er and ur and ur < er:
            out.append({"tag": "BUY_LOW", "player": name,
                        "thesis": f"Buy {name} low: {f['gap_pg']:+.1f} pts/g under what his usage earns, and his "
                                  f"usage ranks {pos}{ur} against consensus {pos}{er}.", "kill": kill_share})
        if f.get("slope_q", 0) >= TRAJ_QUANTILE and orr and er and orr < er:
            out.append({"tag": "TRAJECTORY", "player": name,
                        "thesis": f"{name}'s role is growing — share up from {f['share_first']:.0%} to "
                                  f"{f['share_last']:.0%}, top fifth at {pos} — and we rank him {pos}{orr} "
                                  f"to consensus {pos}{er}.", "kill": kill_share})
        cu = env["cuffs"].get(pid)
        if cu:
            s = cu.get("cuff_of")
            st = env["status"].get(s, "")
            age = env["ages"].get(s)
            if st or (age and age >= FRAGILE_AGE):
                why = f"is listed {st}" if st else f"is {age:.0f}"
                out.append({"tag": "CONTINGENCY", "player": name,
                            "thesis": f"{name} is next up behind {cu['cuff_of_name']}, who {why} — the job "
                                      f"comes with him if {cu['cuff_of_name']} misses time.",
                            "kill": f"Wrong if {cu['cuff_of_name']} stays healthy through week 14 — then "
                                    f"{name} is a bench spot you paid for."})
        if lens.get(pos) and f.get("proj_team"):
            ratings = env["def_vs"].get(pos) or {}
            opp = opponents(env["season"], f["proj_team"], (15, 16, 17))
            if opp and ratings:
                sos = float(np.mean([ratings.get(o, 1.0) for o in opp]))
                if sos >= PLAYOFF_SOS:
                    out.append({"tag": "PLAYOFF_SCHEDULE", "player": name,
                                "thesis": f"{name}'s weeks 15–17 opponents ({', '.join(opp)}) have allowed "
                                          f"{sos - 1:+.0%} to {pos}s — the only position where a schedule read "
                                          f"has predicted anything (backtest).",
                                "kill": f"Wrong if those defenses' {pos} numbers regress toward average "
                                        f"(now {sos:.2f}× league)."})
    if c.get("consolidation"):
        fa = c.get("fa_add") or "the best free agent"
        got = " + ".join(_names(ctx, get))
        out.append({"tag": "CONSOLIDATION", "player": ", ".join(_names(ctx, give)),
                    "thesis": f"Two for one: {c['d_week_me']:+.1f} this week, but {c['dMe']:+.2f} pts/wk over the "
                              f"weeks that count most — already net of picking up {fa}, which you could do anyway.",
                    "kill": f"Wrong if {got} misses time: this puts {' and '.join(_names(ctx, give))}'s "
                            f"points on one player."})
    port = c.get("portfolio")
    if port:
        out.append({"tag": "PORTFOLIO", "player": "", "thesis": port["thesis"], "kill": port["kill"]})
    incoming = [s for s in c["me_starters_in"] if s["id"] in get]
    if incoming and c["dMe"] > 0:
        a = incoming[0]
        outs = [s for s in c["me_starters_out"] if s["id"] not in give]
        b = (outs or [s for s in c["me_starters_out"]] or [None])[0]
        over = ""
        if b:
            over = f" over {b['name']}" + ("" if b["id"] in env["my_roster"] else
                                           " (the free-agent pickup you'd otherwise make)")
        out.append({"tag": "LINEUP", "player": a["name"],
                    "thesis": f"{a['name']} starts for you" + over
                              + f": {c['dMe']:+.2f} pts/wk to your lineup over the weeks that count most.",
                    "kill": (f"Wrong if {a['name']}'s projection falls below {b['name']}'s "
                             f"({a['ppg']:.1f} vs {b['ppg']:.1f} now)." if b else
                             f"Wrong if {a['name']}'s projection drops below your weakest starter's.")})
    seen, uniq = set(), []
    for t in sorted(out, key=lambda t: TAG_ORDER.index(t["tag"])):
        if t["tag"] not in seen:
            seen.add(t["tag"])
            uniq.append(t)
    return uniq


def portfolio(ctx, before: list, after: list, facts: dict, weeks) -> dict | None:
    """Fewer of our starters on bye in the same week, or fewer from one offense."""
    def starters(ids):
        return [e[0] for e in te.lineup(ids, ctx, None).starters]

    def worst_bye(ids):
        c = {}
        for pid in starters(ids):
            for w in facts.get(pid, {}).get("byes", []):
                if w in weeks:
                    c[w] = c.get(w, 0) + 1
        return max(c.items(), key=lambda kv: kv[1]) if c else (None, 0)

    def stack(ids):
        c = {}
        for pid in starters(ids):
            t = ctx.players[pid].get("nfl")
            if t:
                c[t] = c.get(t, 0) + 1
        return max(c.items(), key=lambda kv: kv[1]) if c else (None, 0)

    (wb, nb), (wa, na) = worst_bye(before), worst_bye(after)
    if nb >= 2 and na < nb:
        return {"thesis": f"Spreads your byes: {nb} starters off in week {wb} today, {na} at most after.",
                "kill": f"Wrong if you'd have covered week {wb} from the bench anyway — check that week's lineup."}
    (tb, sb), (ta, sa) = stack(before), stack(after)
    if sb >= STACK_TEAM and (ta != tb or sa < sb):
        return {"thesis": f"Cuts your exposure to one offense: {sb} starters from {tb} today, fewer after.",
                "kill": f"Wrong if {tb}'s offense is the reason you win — concentration cuts both ways."}
    return None


def free_swap(ctx, ids) -> tuple[list, float]:
    """The best single add/drop available without a trade (§6.5 "neutral free-swap").
    Returns the roster after it and its value — the roster as it stands if nothing helps."""
    best_ids, best_val = list(ids), te.team_value(list(ids), ctx)
    for fa in ctx.fa_candidates:
        for d in [p for p in ids if ctx.players[p]["pos"] in ctx.valued_pos]:
            trial = [p for p in ids if p != d] + [fa]
            v = te.team_value(trial, ctx)
            if v > best_val + te.EPS:
                best_ids, best_val = trial, v
    return best_ids, best_val


# ======================================================================== the whole list
def keep(cards: list[dict]) -> list[dict]:
    """Best first, with no manager in more than MAX_PER_PARTNER cards and no player of ours in
    more than MAX_PER_GIVE, so one projection miss can't sink most of the page. A card turned
    away for its player doesn't use up its manager's slots."""
    per, used, kept = {}, {}, []
    for k in cards:
        give = [g["name"] for g in k["give"]]
        if per.get(k["partner"], 0) >= MAX_PER_PARTNER or any(used.get(x, 0) >= MAX_PER_GIVE for x in give):
            continue
        per[k["partner"]] = per.get(k["partner"], 0) + 1
        for x in give:
            used[x] = used.get(x, 0) + 1
        kept.append(k)
    return kept


def run(season: int, now: int, yahoo_rosters=None) -> dict:
    from . import season as S
    from . import title_odds as T
    from . import trade_league as tl
    from .config import MY_TEAM
    from . import league_memory as LM
    from .waiver_value import _birthdays, _rosters_weekly, availability

    from .config import TRADE_DEADLINE, trades_open
    if not trades_open():
        return {"available": False, "closed": True, "deadline": TRADE_DEADLINE, "cards": []}
    lg = tl.build_league(season, yahoo_rosters)
    my_id = lg.get("report", {}).get("my_team_id")
    if not lg.get("teams") or my_id is None:
        return {"available": False, "cards": [], "meta": {"error": "no league"}}
    name_of = {t["id"]: t["name"] for t in lg["teams"]}
    id_of = {v: k for k, v in name_of.items()}
    bias_by_name = LM.bias()
    # the title model first: its seasons say how much each week counts for each team
    # (week_weights), which the engine then values players by — and the live service reads
    # the same weights from the file written here
    tm = T.build(season, now, yahoo_rosters, league=lg)
    base = tm.odds() if tm is not None else {}
    ww_payload = T.week_weights_payload(tm, season, now) if tm is not None else None
    weights = ww_payload["teams"] if ww_payload else {}
    cfg = tl.trade_config(season, weights)
    cfg["market"] = {"bias": {str(id_of[n]): b for n, b in bias_by_name.items() if n in id_of}}
    ctx = te.build_context(lg, cfg)
    profiles = tl.team_profiles(ctx)       # who needs what, shown on each card
    typical = slot_typical(ctx)            # the league's typical starter per slot (need view)

    # ---- §6.1 candidates from the engine, not from FantasyCalc
    me = ctx.teams[my_id]
    tradeable = [p for p in me["roster"] if ctx.players[p]["pos"] in ctx.valued_pos]
    seen, cands = set(), []
    for pid in tradeable:
        r = te.find_trades(ctx, my_id, [pid], {"shapes": SHAPES, "includeFlags": te.FLAG_ORDER,
                                               "minDeltaMe": 0.01, "topN": 400})
        for res in r["results"]:
            k = (res["partner"]["id"], tuple(sorted(res["giveIds"])), tuple(sorted(res["getIds"])))
            if k not in seen:
                seen.add(k)
                cands.append(res)
    evaluated = len(cands)
    # Every candidate is ranked on the chance they say yes times what it does for us
    # (mega/trade_worth.py, on week-weighted values), then at most SIM_PER_PARTNER per
    # manager go to the title simulation, so the 25 simulations cover the league rather
    # than one roster.
    board = TW.Board(ctx, season, {id_of[n]: b for n, b in bias_by_name.items() if n in id_of}, profiles,
                     weights)

    def counterparty(c) -> dict:
        """The chance they say yes, and why (mega/trade_worth.py)."""
        partner = name_of[c["partner"]["id"]]
        p_now = (base.get(partner) or {}).get("p_playoffs", 0.0)
        w = TW.score(board, c["partner"]["id"], c["giveIds"], c["getIds"], float(c["dMe"]), float(c["dThem"]),
                     p_now, LM.trade_offset(partner))
        return {**w, "p_now": p_now, "urg": TW.urgency(p_now), "sits": tl.sits_text(ctx, w["sits"])}

    # Chosen by EXPECTED gain — the chance they say yes times what it does for us — so the
    # page offers deals a manager would take, not the 2-for-1s that wreck his lineup (picked
    # with Chris 2026-10-06). An overpay by name is capped (FAIR_MAX), and a deal too small
    # for either side to bother with is dropped.
    cp = {}

    def cpty(r):
        k = (r["partner"]["id"], tuple(r["giveIds"]), tuple(r["getIds"]), round(r["dMe"], 6))
        if k not in cp:
            cp[k] = counterparty(r)
        return cp[k]

    exp_pre = lambda r: cpty(r)["p"] * r["dMe"]
    cands = sorted((r for r in cands if cpty(r)["ratio"] <= FAIR_MAX and not cpty(r)["small"]),
                   key=lambda r: -exp_pre(r))
    # ...and no player of ours in more than SIM_PER_GIVE of them, so the simulations aren't
    # spent on offers the page's MAX_PER_GIVE would turn away
    picked, per, used = [], {}, {}
    for r in cands:
        n = per.get(r["partner"]["id"], 0)
        if n >= SIM_PER_PARTNER or any(used.get(g, 0) >= SIM_PER_GIVE for g in r["giveIds"]):
            continue
        per[r["partner"]["id"]] = n + 1
        for g in r["giveIds"]:
            used[g] = used.get(g, 0) + 1
        picked.append(r)
    cands = picked

    # ---- the facts every card is written from
    facts = player_facts(season, now)
    avail = availability(S.injuries(season), _rosters_weekly(season), now)
    inj = S.injuries(season)
    status = {}
    if inj is not None and not inj.empty:
        cw = inj[pd.to_numeric(inj["week"], errors="coerce") == int(now)]
        status = {g: s for g, s in zip(cw["gsis_id"], cw["report_status"].fillna("")) if s}
    kick = pd.Timestamp(f"{season}-09-01")
    ages = {g: (kick - d).days / 365.25 for g, d in _birthdays().items() if pd.notna(d)}
    try:
        cuffs = S.cuff_lookup(season)
    except Exception:
        cuffs = {}
    from . import proj_ros
    lens = (proj_ros.load_params().get("schedule_lens") or {})
    allowed = (proj_ros.load_params().get("disagreement_allowed") or {})
    env = {"season": season, "avail": avail, "status": status, "ages": ages, "cuffs": cuffs,
           "schedule_lens": lens, "def_vs": {p: defense_vs(season, now, p) for p, on in lens.items() if on},
           "last_week": T.PLAYOFF_WEEKS[-1], "my_roster": set(ctx.teams[my_id]["roster"])}

    # ---- §6.5 the neutral free swap: the best add/drop we could make WITHOUT trading.
    # A 2-for-1 opens a spot the engine fills with the best free agent; that pickup was
    # always available, so the trade is credited only net of it.
    b_me = ctx.base[my_id]
    swap_ids, swap_val = free_swap(ctx, b_me.ids)

    def starters_diff(before, after):
        bs = {e[0]: e for e in te.lineup(before, ctx, None).starters}
        as_ = {e[0]: e for e in te.lineup(after, ctx, None).starters}
        # the per-game projection for the text, not the week-weighted value the lineup was built on
        info = lambda e: {"id": e[0], "name": ctx.players[e[0]]["name"], "pos": e[1],
                          "ppg": float(ctx.players[e[0]].get("ppg") or 0.0), "slot": e[3]}
        return ([info(e) for k, e in as_.items() if k not in bs], [info(e) for k, e in bs.items() if k not in as_])

    # ---- per candidate: this week, ROS, consolidation, portfolio
    weeks_h = list(range(int(now), T.PLAYOFF_WEEKS[-1] + 1))
    rows = []
    for res in cands:
        core = te.evaluate_core(ctx, my_id, res["partner"]["id"], res["giveIds"], res["getIds"])
        a_me, a_them = core["_after"]["me"], core["_after"]["them"]
        b_them = ctx.base[res["partner"]["id"]]
        two = res["shape"] == "2-for-1" and bool(a_me.added)
        ref_ids, ref_val = (swap_ids, swap_val) if two else (b_me.ids, b_me.value)
        d_me = a_me.value - ref_val
        if d_me <= 0.01:
            continue                                   # nothing left once the free swap is netted out
        dwk_me = te.lineup(a_me.ids, ctx, now).total - te.lineup(ref_ids, ctx, now).total
        dwk_them = te.lineup(a_them.ids, ctx, now).total - te.lineup(b_them.ids, ctx, now).total
        fa_add = ", ".join(_names(ctx, a_me.added)) or None
        consolidation = two and dwk_me < 0 < d_me
        s_in, s_out = starters_diff(ref_ids, a_me.ids)
        rows.append({**res, "dMe": d_me, "d_week_me": dwk_me, "d_week_them": dwk_them,
                     "consolidation": consolidation, "fa_add": fa_add, "two_for_one_netted": two,
                     "ref_me": list(ref_ids), "after_me": list(a_me.ids), "after_them": list(a_them.ids),
                     "me_starters_in": s_in, "me_starters_out": s_out,
                     "portfolio": portfolio(ctx, ref_ids, a_me.ids, facts, weeks_h)})
    rows.sort(key=lambda r: -exp_pre(r))          # dMe is netted by now

    # ---- §5 title odds for the top 25 by expected ΔROS (plus every consolidation among them)
    my_name = name_of[my_id]
    # one card per (partner, what we receive): the best version of each ask, not three
    best, seen_ask = [], set()
    for c in rows:
        k = (c["partner"]["id"], tuple(sorted(c["getIds"])))
        if k not in seen_ask:
            seen_ask.add(k)
            best.append(c)
    top = best[:TOP_SIM]
    cards = []
    for c in top:
        partner = c["partner"]["name"]
        ir_me = [i for i in (tm.rosters.get(my_name, []) if tm else []) if i not in ctx.teams[my_id]["roster"]]
        ir_them = [i for i in (tm.rosters.get(partner, []) if tm else []) if i not in ctx.teams[c["partner"]["id"]]["roster"]]
        dd = tm.delta_vs({my_name: c["ref_me"] + ir_me},
                         {my_name: c["after_me"] + ir_me, partner: c["after_them"] + ir_them},
                         (my_name, partner)) if tm else {}
        us, them = dd.get(my_name, {}), dd.get(partner, {})
        d_us, d_them = us.get("d_title", 0.0), them.get("d_title", 0.0)
        matchups = tm.matchups(my_name, c["ref_me"] + ir_me, c["after_me"] + ir_me, partner,
                               c["after_them"] + ir_them) if tm else []
        if not (c["consolidation"] or (d_us > 0 and not us.get("noise", True))):
            continue                                        # §6.1 hard filter
        tags = tags_for(c, ctx, facts, env)
        if not tags:
            continue
        w = cpty(c)
        pa, fair, urg, p_them_now = w["p"], w["ratio"], w["urg"], w["p_now"]
        # pitch: the slot our outgoing player fills for them
        weak = None
        for s in c["them"]["startersOut"]:
            weak = s
            break
        pitch = (f"They start {weak['name']} at {weak['slot']} ({weak['ppg']:.1f}/g)" if weak else
                 "It deepens their bench more than their lineup")
        pitch += f" and sit at {p_them_now:.0%} playoff odds" + (" — the range where managers deal." if urg == 1.0 else ".")

        def ranks(ids):
            out = []
            for i in ids:
                f = facts.get(i, {})
                dg = f.get("disagreement") or {}
                pos = ctx.players[i]["pos"]
                shown = bool(dg.get("tag")) and (allowed.get(pos) or {}).get(dg.get("driver"))
                out.append({"name": ctx.players[i]["name"], "pos": pos, "our_rank": f.get("our_rank"),
                            "ecr_rank": f.get("ecr_rank"), "driver": dg.get("driver"),
                            "view": (f"we're {dg['tag']} than consensus ({dg['driver']}-driven)" if shown
                                     else "consensus view")})
            return out

        cards.append({
            "partner": partner, "shape": c["shape"], "partner_profile": profiles.get(c["partner"]["id"]),
            "give": [{"name": ctx.players[i]["name"], "pos": ctx.players[i]["pos"], "ros_pg": round(ctx.players[i]["ppg"], 2)} for i in c["giveIds"]],
            "get": [{"name": ctx.players[i]["name"], "pos": ctx.players[i]["pos"], "ros_pg": round(ctx.players[i]["ppg"], 2)} for i in c["getIds"]],
            "tags": [t["tag"] for t in tags[:2]], "all_tags": [t["tag"] for t in tags],
            "thesis": tags[0]["thesis"], "kill": tags[0]["kill"],
            "second": {"tag": tags[1]["tag"], "thesis": tags[1]["thesis"]} if len(tags) > 1 else None,
            "us": {"d_week": round(c["d_week_me"], 2), "d_ros": round(c["dMe"], 2),
                   "d_title": round(d_us, 4), "se_title": round(us.get("se_title", 0.0), 4),
                   "title_noise": bool(us.get("noise", True)), "p_title": round(us.get("p_title", 0.0), 4),
                   "p_playoffs": round(us.get("p_playoffs", 0.0), 4)},
            # their own week weights, as they'd judge it (mega/trade_worth.py their_gain)
            "them": {"d_week": round(c["d_week_them"], 2), "d_ros": round(w["d_them"], 2),
                     "d_title": round(d_them, 4), "se_title": round(them.get("se_title", 0.0), 4),
                     "title_noise": bool(them.get("noise", True)), "p_playoffs_now": round(p_them_now, 4)},
            "i_would_start": [s["name"] for s in c["me_starters_in"]],
            "i_would_bench": [s["name"] for s in c["me_starters_out"] if s["id"] not in c["giveIds"]],
            "netted_free_swap": c["two_for_one_netted"],
            "they_would_start": [s["name"] for s in c["them"]["startersIn"]],
            "they_would_bench": [s["name"] for s in c["them"]["startersOut"] if s["id"] not in c["getIds"]],
            "fa_add": c["fa_add"] if c["consolidation"] else None,
            "pitch": pitch, "p_accept": round(pa, 3), "flag": TW.flag(pa, d_them), "why": w["why"],
            "exp_title": round(pa * (0.0 if us.get("noise", True) else d_us), 5),
            "exp_ros": round(pa * c["dMe"], 3),
            "flags": our_flags(ctx, c["giveIds"], c["getIds"], c["ref_me"], c["after_me"], facts, now, typical),
            "matchups": matchups,
            "fairness": round(fair, 2), "sits_for_them": w["sits"], "ranks": {"give": ranks(c["giveIds"]), "get": ranks(c["getIds"])},
        })
    cards.sort(key=lambda k: (-k["exp_title"], -k["exp_ros"]))
    kept = keep(cards)

    tx = 0
    try:
        from .config import DATA
        t = pd.read_csv(DATA / "yahoo_transactions.csv")
        tx = int(t.iloc[:, 0].astype(str).str.contains("Traded to").sum() // 2)
    except Exception:
        pass
    mine = base.get(my_name, {})
    return {
        "available": True, "cards": kept, "deadline": TRADE_DEADLINE,
        "_week_weights": ww_payload,          # tools/export_web.py writes it; not part of the page
        "meta": {"evaluated": evaluated, "simulated": len(top), "shown": len(kept),
                 "p_playoffs": mine.get("p_playoffs"), "p_title": mine.get("p_title"),
                 "posture": T.posture(mine.get("p_playoffs", 0.0)) if mine else None,
                 "seasons": tm.n if tm else 0, "priors": "estimated",
                 "league_trades": tx,
                 "bias_teams": sorted(bias_by_name)},
    }
