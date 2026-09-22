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

import pandas as pd

from . import trade_engine as te
from . import trade_league as tl

REPLACEMENT_RANK = 3      # see module docstring
UPGRADE = 0.25            # pts/wk at which a pickup is a real lineup improvement
DEPTH = 0.02              # ...and at which it is worth anything at all


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
        rows.append(dict(
            # nfl_team, not nfl: ui.COLS maps it to TM, which ui.table turns into the logo.
            player=p["name"], pos=p["pos"], nfl_team=p["nfl"], ppg=p["ppg"],
            gain=g, fit=label(g), starts=av.get("starts", False),
            drop=av["drop"], bid=bid["bid"], max_bid=bid["max_worth"],
            season_pts=bid["season_pts"],
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
    return (df.sort_values(["gain", "add_score"], ascending=[False, False])
              .head(top).reset_index(drop=True))
