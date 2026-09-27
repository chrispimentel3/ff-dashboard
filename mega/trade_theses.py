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
  * a pitch line from THEIR lineup and playoff odds, P(accept) and FantasyCalc fairness
  * our rank vs ECR per player — the "we disagree" wording only where the backtest earned it

Every constant marked (J) is a judgment default; the acceptance priors live, labelled
"estimated", in config/acceptance_priors.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import trade_engine as te

ROOT = Path(__file__).resolve().parents[1]
PRIORS = ROOT / "config" / "acceptance_priors.json"

TOP_SIM = 25                  # handoff §5: ΔTitle only for the top 25 by ΔROS
SHAPES = ["1-for-1", "2-for-1"]
SELL_HIGH_GAP = 4.0           # pts − xFP per game (existing site threshold)
SELL_HIGH_MIN_PTS = 8.0
SELL_HIGH_MIN_GAMES = 2
TRAJ_QUANTILE = 0.80          # top quintile of share slope at his position
FRAGILE_AGE = 28              # (J) a starter this old counts as fragile for CONTINGENCY
PLAYOFF_SOS = 1.10            # (J) weeks 15–17 opponents allow 10%+ above average
STACK_TEAM = 3                # handoff §6.2: 3+ starters from one offense
MAX_PER_PARTNER = 3           # (J) keep the list from being one manager's roster
SIM_PER_PARTNER = 5           # (J) ...and the 25 simulations from being one manager's either
FAIR_MIN = 0.80               # (J) they get back at least 80% of what they give, by FantasyCalc
# Lead-tag priority. LINEUP (our own fallback: the incoming player simply starts) outranks
# PORTFOLIO so "spreads your byes" rides as the second reason rather than the headline.
TAG_ORDER = ["SELL_HIGH", "BUY_LOW", "ROLE_EXPIRY", "TRAJECTORY", "CONTINGENCY",
             "PLAYOFF_SCHEDULE", "CONSOLIDATION", "LINEUP", "PORTFOLIO"]


def load_priors() -> dict:
    try:
        return json.loads(PRIORS.read_text())
    except Exception:
        return {"status": "missing", "a0": -0.75, "a": 20.0, "b": 4.0, "c": 0.8, "d": 2.0, "e": 1.0,
                "flags": {"LIKELY": 0.5, "EXPLOIT": 0.3, "NEEDS_PITCH": 0.15}}


def urgency(p_playoffs: float) -> float:
    """§6.3: teams at 25–50% playoff odds are the most motivated to deal."""
    if 0.25 <= p_playoffs <= 0.50:
        return 1.0
    if 0.10 <= p_playoffs < 0.25 or 0.50 < p_playoffs <= 0.70:
        return 0.5
    return 0.0


def p_accept(pri: dict, d_title_them: float, fairness: float, need_fit: float, bias: float,
             urg: float) -> float:
    z = (pri["a0"] + pri["a"] * d_title_them + pri["b"] * (fairness - 1.0) + pri["c"] * need_fit
         + pri["d"] * bias + pri["e"] * urg)
    return 1.0 / (1.0 + math.exp(-z))


def flag(pri: dict, p: float, d_title_them: float) -> str:
    f = pri.get("flags") or {}
    if p >= f.get("LIKELY", 0.5):
        return "LIKELY"
    if p >= f.get("EXPLOIT", 0.3) and d_title_them < 0:
        return "EXPLOIT"
    if p >= f.get("NEEDS_PITCH", 0.15):
        return "NEEDS_PITCH"
    return "LONGSHOT"


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
                    "thesis": f"Two for one: {c['d_week_me']:+.1f} this week, but {c['dMe']:+.2f} pts/wk for the "
                              f"rest of the season — already net of picking up {fa}, which you could do anyway.",
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
                              + f": {c['dMe']:+.2f} pts/wk to your lineup for the rest of the season.",
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
def run(season: int, now: int, yahoo_rosters=None) -> dict:
    from . import season as S
    from . import title_odds as T
    from . import trade_league as tl
    from .config import MY_TEAM
    from .market import seed_bias
    from .sources import fantasycalc_values
    from .waiver_value import _birthdays, _rosters_weekly, availability

    lg = tl.build_league(season, yahoo_rosters)
    my_id = lg.get("report", {}).get("my_team_id")
    if not lg.get("teams") or my_id is None:
        return {"available": False, "cards": [], "meta": {"error": "no league"}}
    name_of = {t["id"]: t["name"] for t in lg["teams"]}
    id_of = {v: k for k, v in name_of.items()}
    bias_by_name = seed_bias()
    cfg = tl.engine_config()
    cfg["market"] = {"bias": {str(id_of[n]): b for n, b in bias_by_name.items() if n in id_of}}
    ctx = te.build_context(lg, cfg)

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
    # The engine's LONGSHOT fails both its lineup test for them and the market test on the
    # rankings they can see — simulating those just fills the top 25 with Gibbs-for-a-bench-
    # receiver offers nobody accepts. Then at most SIM_PER_PARTNER per manager, so the 25
    # simulations cover the league rather than one roster.
    # D2: FantasyCalc is the trade-price axis — what a manager will actually accept. An
    # offer they'd be paid less than FAIR_MIN of their player's value for isn't a
    # candidate however much it helps us; the engine's own market test (a perceived rank
    # two good box scores can inflate) let Bryce Young-for-Josh Allen through.
    fc = fantasycalc_values()
    fc_by_name = dict(zip(fc["name"], fc["value"])) if not fc.empty else {}

    def fc_fair(r):
        g = sum(float(fc_by_name.get(ctx.players[i]["name"], 0) or 0) for i in r["giveIds"])
        t = sum(float(fc_by_name.get(ctx.players[i]["name"], 0) or 0) for i in r["getIds"])
        return g / t if t > 0 else 0.0

    cands = sorted((r for r in cands if r["flag"] != "LONGSHOT" and fc_fair(r) >= FAIR_MIN),
                   key=lambda r: -r["dMe"])
    picked, per = [], {}
    for r in cands:
        n = per.get(r["partner"]["id"], 0)
        if n < SIM_PER_PARTNER:
            per[r["partner"]["id"]] = n + 1
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
        info = lambda e: {"id": e[0], "name": ctx.players[e[0]]["name"], "pos": e[1], "ppg": e[2], "slot": e[3]}
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
    rows.sort(key=lambda r: -r["dMe"])

    # ---- §5 title odds for the top 25 by ΔROS (plus every consolidation among them)
    tm = T.build(season, now, yahoo_rosters, league=lg)
    base = tm.odds() if tm is not None else {}
    my_name = name_of[my_id]
    # one card per (partner, what we receive): the best version of each ask, not three
    best, seen_ask = [], set()
    for c in rows:
        k = (c["partner"]["id"], tuple(sorted(c["getIds"])))
        if k not in seen_ask:
            seen_ask.add(k)
            best.append(c)
    top = best[:TOP_SIM]
    pri = load_priors()
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
        if not (c["consolidation"] or (d_us > 0 and not us.get("noise", True))):
            continue                                        # §6.1 hard filter
        tags = tags_for(c, ctx, facts, env)
        if not tags:
            continue
        # §6.3 counterparty
        give_val = sum(float(fc_by_name.get(n, 0) or 0) for n in _names(ctx, c["giveIds"]))
        get_val = sum(float(fc_by_name.get(n, 0) or 0) for n in _names(ctx, c["getIds"]))
        fair = give_val / get_val if get_val > 0 else 1.0
        need_fit = 1.0 if any(s["id"] in c["giveIds"] for s in c["them"]["startersIn"]) else 0.0
        pb = bias_by_name.get(partner) or {}
        bias = (np.mean([pb.get(ctx.players[i]["pos"], 1.0) for i in c["giveIds"]])
                - np.mean([pb.get(ctx.players[i]["pos"], 1.0) for i in c["getIds"]]))
        p_them_now = (base.get(partner) or {}).get("p_playoffs", 0.0)
        urg = urgency(p_them_now)
        pa = p_accept(pri, d_them, fair, need_fit, float(bias), urg)
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
            "partner": partner, "shape": c["shape"],
            "give": [{"name": ctx.players[i]["name"], "pos": ctx.players[i]["pos"], "ros_pg": round(ctx.players[i]["ppg"], 2)} for i in c["giveIds"]],
            "get": [{"name": ctx.players[i]["name"], "pos": ctx.players[i]["pos"], "ros_pg": round(ctx.players[i]["ppg"], 2)} for i in c["getIds"]],
            "tags": [t["tag"] for t in tags[:2]], "all_tags": [t["tag"] for t in tags],
            "thesis": tags[0]["thesis"], "kill": tags[0]["kill"],
            "second": {"tag": tags[1]["tag"], "thesis": tags[1]["thesis"]} if len(tags) > 1 else None,
            "us": {"d_week": round(c["d_week_me"], 2), "d_ros": round(c["dMe"], 2),
                   "d_title": round(d_us, 4), "se_title": round(us.get("se_title", 0.0), 4),
                   "title_noise": bool(us.get("noise", True)), "p_title": round(us.get("p_title", 0.0), 4),
                   "p_playoffs": round(us.get("p_playoffs", 0.0), 4)},
            "them": {"d_week": round(c["d_week_them"], 2), "d_ros": round(c["dThem"], 2),
                     "d_title": round(d_them, 4), "se_title": round(them.get("se_title", 0.0), 4),
                     "title_noise": bool(them.get("noise", True)), "p_playoffs_now": round(p_them_now, 4)},
            "i_would_start": [s["name"] for s in c["me_starters_in"]],
            "i_would_bench": [s["name"] for s in c["me_starters_out"] if s["id"] not in c["giveIds"]],
            "netted_free_swap": c["two_for_one_netted"],
            "they_would_start": [s["name"] for s in c["them"]["startersIn"]],
            "they_would_bench": [s["name"] for s in c["them"]["startersOut"] if s["id"] not in c["getIds"]],
            "fa_add": c["fa_add"] if c["consolidation"] else None,
            "pitch": pitch, "p_accept": round(pa, 3), "flag": flag(pri, pa, d_them),
            "fairness": round(fair, 2), "ranks": {"give": ranks(c["giveIds"]), "get": ranks(c["getIds"])},
        })
    cards.sort(key=lambda k: (-(0.0 if k["us"]["title_noise"] else k["us"]["d_title"]), -k["us"]["d_ros"]))
    # keep any one manager from filling the page
    per, kept = {}, []
    for k in cards:
        per[k["partner"]] = per.get(k["partner"], 0) + 1
        if per[k["partner"]] <= MAX_PER_PARTNER:
            kept.append(k)

    tx = 0
    try:
        from .config import DATA
        t = pd.read_csv(DATA / "yahoo_transactions.csv")
        tx = int(t.iloc[:, 0].astype(str).str.contains("Traded to").sum() // 2)
    except Exception:
        pass
    mine = base.get(my_name, {})
    return {
        "available": True, "cards": kept,
        "meta": {"evaluated": evaluated, "simulated": len(top), "shown": len(kept),
                 "p_playoffs": mine.get("p_playoffs"), "p_title": mine.get("p_title"),
                 "posture": T.posture(mine.get("p_playoffs", 0.0)) if mine else None,
                 "seasons": tm.n if tm else 0, "priors": pri.get("status", "estimated"),
                 "league_trades": tx, "refit_at": 5,
                 "bias_teams": sorted(bias_by_name)},
    }
