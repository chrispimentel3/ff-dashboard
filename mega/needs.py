"""What your roster actually needs, and what a given pickup is worth to it.

The old waiver board ranked free agents by talent and industry buzz and never once looked
at your team. That is how it kept offering tight ends to a roster that starts Kittle and
Kelce in a league whose flex is RB/WR only — a third tight end there cannot enter the
lineup in any week, under any injury, so his value to you is exactly zero however good he is.

So value a pickup the way a trade is valued, with the same engine: add him, force the roster
back to legal size, re-optimise, and measure the change in starters plus weighted bench
depth. `gain` is points per week, and it already accounts for who he displaces and who you
would have to cut for him.

Two deliberate choices:

  * **Replacement level uses the third-best free agent, not the best.** The engine's default
    is the maximum, which is the JS original's rule, but the maximum of a few hundred noisy
    per-game estimates is the single luckiest of them and is biased high — that is how a
    two-game cameo came to set the bar for a whole position. Third-best is both steadier and
    truer to how a waiver wire works: you do not reliably win the pool's best player.
  * **A zero is reported, not hidden.** Most of the pool genuinely cannot improve a set
    roster, and a board that manufactures separation between those players is lying. They
    are ranked instead by the old talent-and-trend score, under a label that says plainly
    they are speculative, and they are priced at a dollar or two.
"""
from __future__ import annotations

import functools
import math

import pandas as pd

from . import trade_engine as te
from . import trade_league as tl

REPLACEMENT_RANK = 3      # see module docstring
UPGRADE = 0.25            # pts/wk at which a pickup is a real lineup improvement
DEPTH = 0.02              # ...and at which it is worth anything at all

# HANDOFF §12.5: a role flag is worth more than a raw-talent score among the many players
# whose lineup gain is zero, because it says the offence has started treating him
# differently. ROLE+ is the strongest of them — he is producing like the rung above him.
ROLE_BONUS = {"ROLE+": 1.50, "TGT": 0.60, "AIR": 0.40, "SNAP": 0.40, "LEAD": 0.80, "GL": 0.60}
SUSTAINED = 1.0           # a flag that held across the window counts fully...
SPIKE = 0.35              # ...one big afternoon counts for much less (§4.2)
ROLE_MINUS = -1.00        # share at risk: a sell / do-not-add signal


def build(season: int, yahoo_rosters: pd.DataFrame | None = None) -> dict:
    """League, engine context and my team id, ready for the functions below."""
    lg = tl.build_league(season, yahoo_rosters)
    if not lg.get("teams"):
        return {"ok": False, "report": lg.get("report", {})}
    cfg = tl.engine_config()
    cfg["replacementRank"] = REPLACEMENT_RANK
    ctx = te.build_context(lg, cfg)
    my = lg["report"].get("my_team_id")
    if my is None or my not in ctx.teams:
        return {"ok": False, "report": lg["report"]}
    return {"ok": True, "league": lg, "ctx": ctx, "my_id": my,
            "base": te.team_value(ctx.teams[my]["roster"], ctx), "report": lg["report"]}


def add_value(ctx, my_id, pid: str, base: float | None = None) -> dict:
    """What adding one player does to my roster, in points per week."""
    roster = ctx.teams[my_id]["roster"]
    if base is None:
        base = te.team_value(roster, ctx)
    if pid in roster:
        return {"gain": 0.0, "drop": "", "reason": "already rostered"}
    s = te.settle(list(roster) + [pid], ctx, keep={pid})
    gain = te.team_value(s.ids, ctx) - base
    starts = pid in {e[0] for e in te.lineup(s.ids, ctx).starters}
    return {
        "gain": round(gain, 3),
        "drop": ", ".join(ctx.players[d]["name"] for d in s.dropped),
        "starts": starts,
    }


def position_needs(ctx, my_id) -> pd.DataFrame:
    """Where the roster is thin: what each starting slot is getting, against the wire."""
    roster = ctx.teams[my_id]["roster"]
    lu = te.lineup(roster, ctx)
    repl = ctx.repl[None]
    rows = []
    for pos in sorted(ctx.valued_pos):
        mine = sorted((ctx.players[i]["ppg"] for i in roster if ctx.players[i]["pos"] == pos),
                      reverse=True)
        # starters are (id, pos, pts, slot); a flexed RB still counts against RB here
        in_lineup = [e[2] for e in lu.starters if e[1] == pos]
        rows.append(dict(
            pos=pos,
            rostered=len(mine),
            starting=len(in_lineup),
            worst_starter=round(min(in_lineup), 2) if in_lineup else None,
            replacement=round(repl.get(pos, 0.0), 2),
            # How much the wire could plausibly beat your weakest starter at this slot.
            headroom=round(min(in_lineup) - repl.get(pos, 0.0), 2) if in_lineup else None,
        ))
    return pd.DataFrame(rows)


def label(gain: float) -> str:
    if gain >= UPGRADE:
        return "UPGRADE"
    if gain >= DEPTH:
        return "DEPTH"
    return "STASH"


def why_zero(ctx, my_id, pid: str) -> str:
    """Plain reason a player adds nothing — the bit the old board never said."""
    p = ctx.players[pid]
    pos = p["pos"]
    roster = ctx.teams[my_id]["roster"]
    lu = te.lineup(roster, ctx)
    at_pos = [ctx.players[e[0]] for e in lu.starters if e[1] == pos]
    flex_ok = pos in set(ctx.cfg["flexEligible"])
    worst = min((q["ppg"] for q in at_pos), default=None)
    if worst is not None and p["ppg"] < worst:
        blockers = ", ".join(q["name"] for q in sorted(at_pos, key=lambda q: -q["ppg"]))
        extra = "" if flex_ok else f"; {pos} can't fill the {ctx.cfg['flexCount']}-man RB/WR flex"
        return f"behind {blockers} ({p['ppg']:.1f} vs {worst:.1f}){extra}"
    if not flex_ok and len(at_pos) >= ctx.cfg["slots"].get(pos, 0):
        return f"{pos} slot is full and the flex is RB/WR only"
    return f"no better than the wire ({ctx.repl[None].get(pos, 0):.1f} replacement)"


def board(season: int, week: int, yahoo_rosters: pd.DataFrame | None = None,
          top: int = 25, aggression: float = 1.0) -> pd.DataFrame:
    """Free agents ranked by what they add to *this* roster, with a FAAB bid attached."""
    from . import faab as fb
    from .intel import waiver_signals

    st = build(season, yahoo_rosters)
    if not st.get("ok"):
        return pd.DataFrame()
    ctx, my, base = st["ctx"], st["my_id"], st["base"]

    # §12.5 role context. The league player id IS the gsis id wherever one resolved, so
    # this joins directly. A player who never resolved simply has no role — he is still
    # addable, he just gets no role bonus.
    try:
        from .season import role_lookup
        rl = role_lookup(season)
    except Exception:
        rl = {}

    bud = fb.cached_budgets()
    mine = bud[bud["team"] == ctx.teams[my]["name"]]["faab_left"]
    budget_left = int(mine.iloc[0]) if not mine.empty else fb.BUDGET

    rows = []
    for pid in st["league"]["freeAgents"]:
        p = ctx.players[pid]
        if p["pos"] not in ctx.valued_pos:
            continue
        av = add_value(ctx, my, pid, base)
        g = av["gain"]
        bid = fb.suggest(g, budget_left, week, aggression)
        rc = rl.get(pid) or {}
        rows.append(dict(
            # nfl_team, not nfl: ui.COLS maps it to TM, which ui.table turns into the logo.
            player=p["name"], pos=p["pos"], nfl_team=p["nfl"], ppg=p["ppg"],
            gain=g, fit=label(g), starts=av.get("starts", False),
            drop=av["drop"], bid=bid["bid"], max_bid=bid["max_worth"],
            season_pts=bid["season_pts"],
            role=_role_text(rc), role_flags=_flag_text(rc),
            role_score=role_score(rc),
            why=why_zero(ctx, my, pid) if g < DEPTH else
                ("steps straight into your lineup" if av.get("starts") else "real bench value"),
            norm=p["name"],
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Talent and trend still decide the order among the many players who move the lineup by
    # nothing — that is where a breakout is found before he is obviously a breakout.
    sig = waiver_signals(season)
    if not sig.empty:
        from .intel import _norm

        df["norm"] = df["player"].map(_norm)
        df = df.merge(sig, on="norm", how="left", suffixes=("", "_sig"))
    if "add_score" not in df.columns:
        df["add_score"] = 0.0
    # No signal row means he has not played a snap in the form window — injured, benched or
    # just called up. Rank him last rather than dropping him: he is still addable.
    df["add_score"] = df["add_score"].fillna(df["add_score"].min())
    if "upside" not in df.columns:
        df["upside"] = ""
    df["upside"] = df["upside"].fillna("no snaps in the form window")
    # Role signal rides on top of the talent score for the speculative tail. It never
    # reorders the players who actually improve the lineup — `gain` is a measurement and a
    # flag is an opinion, so the measurement sorts first.
    df["add_score"] = df["add_score"] + df["role_score"].fillna(0.0)
    return (df.sort_values(["gain", "add_score"], ascending=[False, False])
              .head(top).reset_index(drop=True))


def _role_text(rc: dict) -> str:
    """"Committee back" rather than "COMMITTEE"."""
    from .glossary import role_label

    return role_label((rc or {}).get("role")) if (rc or {}).get("role") else ""


def role_score(rc: dict) -> float:
    """§12.5 flags as a single number, discounted when a flag is only a spike."""
    if not rc:
        return 0.0
    tags = rc.get("tags") or {}
    total = 0.0
    for f in rc.get("flags") or []:
        if f == "ROLE-":
            total += ROLE_MINUS
            continue
        w = ROLE_BONUS.get(f)
        if w is None:
            continue
        total += w * (SPIKE if tags.get(f) == "spike" else SUSTAINED)
    return round(total, 3)


def _flag_text(rc: dict) -> str:
    """The flags in plain English: "target hog, goal line (1 game)".

    Worded by mega.glossary so the board, the tables and the player card cannot end up
    describing the same flag three different ways."""
    if not rc:
        return ""
    from .glossary import flag_label

    tags = rc.get("tags") or {}
    return ", ".join(flag_label(f, tags.get(f)) for f in (rc.get("flags") or []))


# ---------------------------------------------------------------- §5.3-5.5 FAAB
POOL_PER_WEEK = 11.0      # (J) weekly lineup gain the remaining budget expects to buy
HYPE_SCALE = 20.0         # points of % owned movement that doubles the hype term
HYPE_MAX = 0.5


def max_bid(gain: float, faab_left: float, pool_per_week: float = POOL_PER_WEEK) -> int:
    """§5.3 — the most a claim can be worth to you.

        rate   = FAAB_left / (poolPerWeek * H)
        maxBid = gain * H * rate = FAAB_left * gain / poolPerWeek

    The horizon cancels, which is the neat part: what a player is worth is his weekly gain
    as a fraction of the total weekly gain the rest of your budget can buy. $100 left and
    +3 pts/wk is $27.
    """
    if pool_per_week <= 0 or faab_left <= 0:
        return 0
    raw = float(faab_left) * max(0.0, float(gain)) / float(pool_per_week)
    return int(min(float(faab_left), round(raw)))


def rival_bid(gain_t: float, faab_t: float, pct_owned_delta: float = 0.0,
              pool_per_week: float = POOL_PER_WEEK,
              history_scale: float = 1.0) -> float:
    """§5.4 — what one rival would plausibly bid.

    `hype` is the Yahoo ownership jump: a player everybody is adding draws bids above what
    a model says he is worth, and ignoring that is how you lose a claim by a dollar.
    `history_scale` is the league's own correction once enough settled bids exist.
    """
    if faab_t <= 0 or gain_t <= 0:
        return 0.0
    hype = 1.0 + HYPE_MAX * min(max(float(pct_owned_delta) / HYPE_SCALE, 0.0), 1.0)
    raw = min(float(faab_t), float(faab_t) * float(gain_t) / float(pool_per_week))
    return raw * hype * float(history_scale)


def recommend(gain_mine: float, faab_mine: float, rivals: list[tuple[float, float]],
              pct_owned_delta: float = 0.0, pool_per_week: float = POOL_PER_WEEK,
              min_bid: int | None = None, history_scale: float = 1.0) -> dict:
    """§5.4 — beat the best rival by a dollar, or walk away.

    `rivals` is [(their gain from this player, their FAAB left)]. A claim nobody else
    wants costs the league minimum; one that would cost more than it is worth to you is
    reported as a pass WITH the number, because "don't bid" is only useful advice when it
    says what the player was going to go for.
    """
    from . import faab as fb

    floor = fb.MIN_BID if min_bid is None else int(min_bid)
    mine = max_bid(gain_mine, faab_mine, pool_per_week)
    bids = [rival_bid(g, f, pct_owned_delta, pool_per_week, history_scale) for g, f in rivals]
    top = max(bids) if bids else 0.0
    if top <= 0:
        rec = max(floor, 0)
        return {"bid": rec, "max_bid": mine, "top_rival": 0.0, "pass": False,
                "why": "nobody else gains from him — the league minimum wins it"}
    rec = int(math.ceil(top + 1))
    if rec > mine:
        return {"bid": 0, "max_bid": mine, "top_rival": round(top, 1), "pass": True,
                "why": f"likely outbid at about ${top:.0f}, which is over your ${mine} ceiling"}
    return {"bid": rec, "max_bid": mine, "top_rival": round(top, 1), "pass": False,
            "why": f"clears the best rival bid of about ${top:.0f}"}


def history_scale(settled: pd.DataFrame, modelled: dict, min_rows: int = 5,
                  clamp: tuple = (0.5, 2.0)) -> float:
    """§5.4 — what this league actually pays, against what the model said it would.

    The median ratio of winning bid to modelled bid, once there are enough settled claims
    to mean anything. Clamped, because five observations can produce any ratio at all.
    """
    if settled is None or settled.empty or not modelled:
        return 1.0
    rows = []
    for _, r in settled.iterrows():
        m = modelled.get(r.get("player"))
        bid = pd.to_numeric(pd.Series([r.get("bid")]), errors="coerce").iloc[0]
        if m and m > 0 and pd.notna(bid):
            rows.append(float(bid) / float(m))
    if len(rows) < min_rows:
        return 1.0
    return float(min(max(pd.Series(rows).median(), clamp[0]), clamp[1]))


def claim_plan(season: int, week: int, yahoo_rosters: pd.DataFrame | None = None,
               max_claims: int = 3, aggression: float = 1.0) -> pd.DataFrame:
    """§5.5 — a running order, not a wish list.

    Take the best claim, apply its add and its drop to a copy of the roster, spend its
    bid, then re-price everything against the roster you would then have. Two targets that
    both want the same player cut get resolved this way, and Yahoo processes claims by bid
    within a drop, so the order matters.
    """
    from . import faab as fb

    st = build(season, yahoo_rosters)
    if not st.get("ok"):
        return pd.DataFrame()
    ctx, my = st["ctx"], st["my_id"]
    roster = list(ctx.teams[my]["roster"])
    bud = fb.cached_budgets()
    mine_b = bud[bud["team"] == ctx.teams[my]["name"]]["faab_left"]
    budget = float(mine_b.iloc[0]) if not mine_b.empty else float(fb.BUDGET)

    pool = [p for p in st["league"]["freeAgents"] if ctx.players[p]["pos"] in ctx.valued_pos]
    out = []
    for _ in range(max(1, int(max_claims))):
        ctx.teams[my]["roster"] = roster
        base = te.team_value(roster, ctx)
        best, best_av = None, None
        for pid in pool:
            av = add_value(ctx, my, pid, base)
            if best_av is None or av["gain"] > best_av["gain"]:
                best, best_av = pid, av
        if best is None or best_av["gain"] < DEPTH:
            break
        bid = fb.suggest(best_av["gain"], budget, week, aggression)["bid"]
        out.append({"order": len(out) + 1, "player": ctx.players[best]["name"],
                    "pos": ctx.players[best]["pos"], "gain": round(best_av["gain"], 3),
                    "drop": best_av["drop"], "bid": bid, "budget_after": int(budget - bid)})
        roster = [i for i in roster if i != best_av.get("drop_id", best_av["drop"])] + [best]
        pool = [p for p in pool if p != best]
        budget = max(0.0, budget - bid)
    return pd.DataFrame(out)
