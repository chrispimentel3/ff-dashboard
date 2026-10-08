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

# HANDOFF §12.5: a role flag is worth more than a raw-talent score among the many players
# whose lineup gain is zero, because it says the offence has started treating him
# differently. ROLE+ is the strongest of them — he is producing like the rung above him.
ROLE_BONUS = {"ROLE+": 1.50, "TGT": 0.60, "AIR": 0.40, "SNAP": 0.40, "LEAD": 0.80,
              "GL": 0.60, "CUFF": 0.70}
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
        from .season import cuff_lookup, role_lookup
        rl = role_lookup(season)
        cu = cuff_lookup(season)
    except Exception:
        rl, cu = {}, {}

    # §15.2 — the man who would inherit a starter's job carries an option on it. That is
    # worth something on a bench even while he is doing nothing, and it is the single
    # reason to hold a player whose current role prices at zero.
    for gid, c in (cu or {}).items():
        if gid in rl:
            rl[gid] = {**rl[gid], "flags": list(rl[gid].get("flags") or []) + ["CUFF"],
                       "cuff_of_name": c["cuff_of_name"], "clear_two": c["clear_two"]}

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
            role=_role_text(rc), role_flags=_flag_text(rc), behind=rc.get("cuff_of_name", ""),
            opp_score=rc.get("opp_score"),
            role_score=role_score(rc),
            why=(f"first man up behind {rc['cuff_of_name']}" if g < DEPTH and rc.get("cuff_of_name")
                 else why_zero(ctx, my, pid) if g < DEPTH else
                 ("steps straight into your lineup" if av.get("starts") else "real bench value")),
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

    # HANDOFF v1.3 Pass 1 — the season-long, mechanism-split value (mega/waiver_value.py)
    # decides the lanes, the drop and the bid. Trend and industry adds no longer rank
    # anyone: a player with no fit to this roster sits in no lane however hot he is.
    try:
        from . import waiver_value as wval
        v13 = wval.run(season, week, yahoo_rosters)
    except Exception as e:
        v13 = {"rows": pd.DataFrame(), "notes": [], "meta": {"error": f"{type(e).__name__}: {e}"}}
    rows13 = v13["rows"]
    if not rows13.empty:
        df = df.drop(columns=["drop", "bid", "max_bid"]).merge(
            rows13.drop(columns=["pid", "pos", "nfl_team", "ppg", "gain", "role"]).rename(
                columns={"fit": "fit_pts", "behind": "insures"}),
            on="player", how="left")
        df["lane"] = df["lane"].fillna("")
        df["why"] = [lane_why(r) if r["lane"] else r["why"] for _, r in df.iterrows()]
        df["_lane"] = df["lane"].map(LANE_ORDER).fillna(len(LANE_ORDER))
        # D1: rank by the change in title odds where the simulation can see one; a Δ inside
        # its own noise band ranks as zero and pts/wk (or the signal score) breaks the tie.
        dt = df["d_title"] if "d_title" in df.columns else pd.Series(0.0, index=df.index)
        nz = df["title_noise"] if "title_noise" in df.columns else pd.Series(True, index=df.index)
        df["_title"] = [0.0 if (n is None or n is True or pd.isna(d)) else float(d) for d, n in zip(dt, nz)]
        df["_key"] = [(r["signal_score"] if r["lane"] == "early_signal" else r["fit_pts"]) or 0.0
                      for _, r in df.iterrows()]
        df = df.sort_values(["_lane", "_title", "_key", "add_score"], ascending=[True, False, False, False])
        df = df.drop(columns=["_lane", "_key", "_title"]).head(top).reset_index(drop=True)
    else:
        df = df.sort_values(["gain", "add_score"], ascending=[False, False]).head(top).reset_index(drop=True)
    df.attrs["roster_notes"] = v13.get("notes") or []
    df.attrs["v13"] = v13.get("meta") or {}
    return df


# trade_chip (FLIP only, no fit) rides after the lanes so the top-N cut can't drop it
LANE_ORDER = {"bid_now": 0, "early_signal": 1, "stash": 2, "trade_chip": 3}


def lane_why(r) -> str:
    """One line saying which mechanism makes him worth a spot, in points per week."""
    st = r.get("out_status")
    if isinstance(st, str) and st and pd.notna(r.get("out_back")):
        if int(r["out_back"]) > 17:
            return "out for the season — " + _lane_why(r)
        out = {"IR": "on injured reserve", "Out": "ruled out", "Doubtful": "doubtful"}.get(st, st)
        return f"{out}, back week {int(r['out_back'])} — " + _lane_why(r)
    return _lane_why(r)


def _lane_why(r) -> str:
    lane, mech = r.get("lane"), r.get("mechanism")
    if lane == "bid_now":
        return (f"+{r['next3']:.1f} pts/wk over the next 3 weeks"
                + (" — covers a bye or injury" if mech == "COVER" else " — starts for you"))
    if lane == "early_signal":
        from .glossary import flag_label
        return (f"{flag_label(r['signal'], None)}: {r['p_expand']:.0%} chance his role grows within "
                f"3 weeks, worth +{r['gain_if_expands']:.1f} pts/wk if it does")
    if mech == "INSURE":
        return (f"insurance on {r['insures']}: +{r['insure']:.2f} pts/wk in expectation"
                + (" (your own starter)" if r.get("handcuff") else ""))
    if mech == "COVER":
        return f"bye and injury cover: +{r['cover']:.2f} pts/wk across the season"
    return f"+{r['start']:.2f} pts/wk upgrade across the season"


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
