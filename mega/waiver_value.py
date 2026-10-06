"""HANDOFF v1.3 Pass 1 — what a free agent is worth to THIS roster, and why.

The old board asked one question — does he start for you *this week*? — and ranked the
many players who didn't by trend and industry adds. That is how a fourth quarterback sat
at the top of "Speculative" for a roster already holding Young, Stroud and Daniels. This
module values every free agent across the rest of the season (weeks now..17, playoff weeks
weighted 1.5) and splits the value into the mechanism that produces it:

  * **START** — the upgrade if everyone were healthy: he beats a starter (or deepens the
    bench) week after week.
  * **COVER** — what absences add or take away: he fills a bye/injury hole, or dropping
    `d*` opens one. START + COVER is exactly the engine's horizon gain for the add; bad
    weeks count against it (settled with Chris 2026-09-26, test F).
  * **INSURE** — shown beside the total, not inside it: the value of the job he inherits
    if the starter ahead of him misses time, times the fitted chance of that
    (config/injury_hazard.json). `HANDCUFF` when that starter is ours.
  * **FLIP** — resale value (FantasyCalc) when at least two other teams would start him.
    Never ranks anything; a FLIP-only player is a trade chip, not a waiver claim.

`fit = START + COVER + INSURE − INSURE lost with the drop`. A player with no fit cannot
appear in any lane, however much the industry is adding him.

IR is modelled forward (§3.2): an IR player is off the roster until his expected return
week, then comes back and forces the cheapest drop — so "Stroud is your cheapest drop once
Daniels is back" falls out of the same arithmetic as every add.

Every constant below marked (J) is a judgment default; the rest are fitted or league rules.
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import contingency as cg
from . import trade_engine as te

# ---------------------------------------------------------------- horizon (league rules)
LAST_WEEK = 17                 # Mega Bowl's final
PLAYOFF_WEEKS = (15, 16, 17)   # 6-team bracket, weeks 15-17 (mega/sim.py)
PLAYOFF_WEIGHT = 1.5           # (J) a playoff week is worth half again a regular one (handoff §3.1)

# ---------------------------------------------------------------- lanes
TAU_BID = 1.0      # (J) pts/wk of START + COVER over the next BID_WINDOW weeks for "Bid now";
                   #     handoff §3.3 default, to be tuned after two weeks of use
BID_WINDOW = 3     # weeks
FIT_MIN = 0.02     # pts/wk below which a player has no fit — mega/needs.py's DEPTH floor.
                   # Bench-depth arithmetic produces crumbs like +0.005 for a fourth QB; a
                   # crumb is not a reason to roster anyone.
MARKET_ON = 30.0   # (J) Yahoo % rostered at which "the market is on him"
SIGNAL_K = 20.0    # (J) pseudo-count shrinking a flag's fitted expansion rate to its
                   #     position's base rate — several flag cells have single-digit samples
SIGNALS = ("TGT", "AIR", "SNAP", "LEAD", "GL", "ROLE+")
FLIP_TEAMS = 2     # handoff §3.1: resale value needs at least two buyers

# ---------------------------------------------------------------- return weeks (§3.2)
IR_MIN_WEEKS = cg.IR_WEEKS_DEFAULT   # NFL IR: placement week + 4 at the earliest
OUT_NO_PRACTICE = 2                  # (J) ruled Out without a full practice: back in 2 weeks
UNAVAILABLE_NOW = {"Out", "Doubtful"}

SIGNAL_FILE = Path(__file__).resolve().parents[1] / "config" / "signal_rates.json"


# ======================================================================== inputs
def horizon(now: int) -> list[tuple[int, float]]:
    return [(w, PLAYOFF_WEIGHT if w in PLAYOFF_WEEKS else 1.0) for w in range(int(now), LAST_WEEK + 1)]


def bye_weeks(sched: pd.DataFrame) -> dict[str, int]:
    """team -> bye week, read off the schedule: the regular-season week a team doesn't play."""
    from .ids import canon_team
    if sched is None or sched.empty:
        return {}
    s = sched[sched["game_type"].astype(str).str.upper() == "REG"] if "game_type" in sched.columns else sched
    weeks = sorted(pd.to_numeric(s["week"], errors="coerce").dropna().astype(int).unique())
    played: dict[str, set] = {}
    for side in ("home_team", "away_team"):
        for t, w in zip(s[side], pd.to_numeric(s["week"], errors="coerce")):
            played.setdefault(canon_team(t), set()).add(int(w))
    out = {}
    for t, ws in played.items():
        missing = [w for w in weeks if w not in ws and w <= LAST_WEEK]
        if missing:
            out[t] = missing[0]
    return out


def availability(inj: pd.DataFrame, rosters_weekly: pd.DataFrame, now: int) -> dict[str, dict]:
    """gsis_id -> {"status", "back"}: why he is out and the first week he is expected back.

      NFL IR (weekly roster status RES) -> placement week + 4, and never before next week
      Out, full practice by Friday      -> next week
      Out otherwise                     -> OUT_NO_PRACTICE weeks from now (J)
      Doubtful                          -> next week (misses this one)
    Questionable players are treated as playing: three in four of them do (AVAIL_NOW).
    """
    out: dict[str, dict] = {}
    if rosters_weekly is not None and not rosters_weekly.empty:
        rw = rosters_weekly.sort_values("week")
        last = rw.groupby("gsis_id").tail(1)
        for gid in last.loc[last["status"] == "RES", "gsis_id"].dropna():
            hist = rw[rw["gsis_id"] == gid]
            not_res = hist[hist["status"] != "RES"]["week"]
            placed = int(not_res.max()) + 1 if not not_res.empty else int(hist["week"].min())
            out[gid] = {"status": "IR", "back": max(placed + IR_MIN_WEEKS, int(now) + 1)}
    if inj is not None and not inj.empty:
        cur = inj[pd.to_numeric(inj["week"], errors="coerce") == int(now)]
        for r in cur.itertuples():
            gid, st = getattr(r, "gsis_id", None), str(getattr(r, "report_status", "") or "")
            if not gid or gid in out or st not in UNAVAILABLE_NOW:
                continue
            full = "full" in str(getattr(r, "practice_status", "") or "").lower()
            back = int(now) + 1 if (st == "Doubtful" or full) else int(now) + OUT_NO_PRACTICE
            out[gid] = {"status": st, "back": back}
    return out


# ======================================================================== the valuation
@dataclass
class Board:
    """One roster's horizon valuation state. `ctx` prices weeks as they will actually be
    (byes, injuries, IR); `ctx_h` prices the same weeks as if our own players were all
    healthy and never on bye — the difference between the two is COVER."""
    ctx: te.Ctx
    ctx_h: te.Ctx
    my_id: object
    roster: list
    ir: list                        # [(pid, back_week)], sorted by back_week
    weeks: list                     # [(week, weight)]
    memo: dict = field(default_factory=dict)
    # pid -> INSURE pts/wk for our own players. The lineup can't see a handcuff's value
    # (it's all in weeks that haven't gone wrong yet), so without this an IR return would
    # cut CMC's backup before a third quarterback who can never start.
    keep_value: dict = field(default_factory=dict)

    # ------------------------------------------------------------ roster over time
    def path(self, roster) -> dict:
        """week -> (roster ids, IR ids still out) as the season unfolds, plus the drops each
        IR return forces. The drop is the player whose loss costs the least over the weeks
        that remain — what a manager would actually cut the day his starter comes back."""
        key = ("path", frozenset(roster))
        if key in self.memo:
            return self.memo[key]
        cur, pend, by_week, drops = list(roster), list(self.ir), {}, []
        for w, _ in self.weeks:
            while pend and pend[0][1] <= w:
                pid, _back = pend.pop(0)
                cur = cur + [pid]
                if len(cur) > self.ctx.cfg["rosterSize"]:
                    d = self._cheapest(cur, [x for x, _ in self.weeks if x >= w])
                    if d is not None:
                        cur = [x for x in cur if x != d]
                        drops.append({"week": w, "returning": pid, "drop": d})
            by_week[w] = (tuple(cur), tuple(p for p, _ in pend))
        out = {"by_week": by_week, "drops": drops}
        self.memo[key] = out
        return out

    def _cheapest(self, ids, weeks) -> str | None:
        best, best_v, best_ppg = None, -math.inf, math.inf
        for pid in ids:
            p = self.ctx.players[pid]
            if p["pos"] not in self.ctx.valued_pos:
                continue
            rest = [x for x in ids if x != pid]
            v = (sum(self._total(rest, (), w, False) for w in weeks)
                 - self.keep_value.get(pid, 0.0) * len(weeks))
            ppg = p.get("ppg") or 0
            if v > best_v + te.EPS or (abs(v - best_v) <= te.EPS and ppg < best_ppg):
                best, best_v, best_ppg = pid, v, ppg
        return best

    # ------------------------------------------------------------ lineup totals
    def _total(self, ids, pending, w, healthy: bool, override: dict | None = None) -> float:
        """Optimal lineup + bench depth in week w. IR players still out ride along: worth
        nothing in the real world that week, their full value in the healthy one."""
        ctx = self.ctx_h if healthy else self.ctx
        key = ("t", frozenset(ids), frozenset(pending), w, healthy,
               tuple(sorted(override.items())) if override else None)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        saved = {}
        for pid, val in (override or {}).items():
            p = ctx.players[pid]
            saved[pid] = p
            ctx.players[pid] = {**p, "weekly": {**(p.get("weekly") or {}), w: val}, "bye": None}
        try:
            out = te.lineup(list(ids) + list(pending), ctx, w).total
        finally:
            ctx.players.update(saved)
        self.memo[key] = out
        return out

    def weekly(self, roster, healthy: bool = False, override_by_week: dict | None = None) -> dict:
        path = self.path(roster)["by_week"]
        return {w: self._total(*path[w], w, healthy, (override_by_week or {}).get(w))
                for w, _ in self.weeks}

    def average(self, by_week: dict, weeks=None) -> float:
        ws = self.weeks if weeks is None else [(w, wt) for w, wt in self.weeks if w in weeks]
        den = sum(wt for _, wt in ws)
        return sum(wt * by_week[w] for w, wt in ws) / den if den else 0.0

    def value(self, roster) -> float:
        return self.average(self.weekly(roster))


# ---------------------------------------------------------------- INSURE
@dataclass
class Cuff:
    starter: str                    # gsis_id of the player ahead of him
    boost: dict                     # week -> points he GAINS when the starter sits
    p_miss: dict                    # week -> chance the starter misses that week
    cap: float = math.inf           # the starter's own projection: a backup never out-
                                    # projects the man he replaces

    def promoted(self, normal: float, w: int) -> float:
        return min(normal + self.boost.get(w, 0.0), max(normal, self.cap))


def insure(b: Board, holder: str, cuff: Cuff, with_ids, without_ids) -> float:
    """Pts/wk that holding `holder` is worth over and above his normal role, in the weeks
    the starter ahead of him misses:

        INSURE = Σ_w wt_w · P(starter out, w) · max(0, Δ_w(starter out, holder promoted) − Δ_w)

    where Δ_w is the lineup difference between `with_ids` and `without_ids` that week.
    Measured as an increment over Δ_w so a backup who already starts for us isn't paid
    twice for the same points. If the starter is ours, his absence opens our slot too, and
    the formula sees that without a special case."""
    base_w = b.weekly(with_ids)
    base_wo = b.weekly(without_ids)
    path_w, path_wo = b.path(with_ids)["by_week"], b.path(without_ids)["by_week"]
    by_week = {}
    for w, _ in b.weeks:
        pm = cuff.p_miss.get(w, 0.0)
        if pm <= 0:
            by_week[w] = 0.0
            continue
        hp = b.ctx.players[holder]
        if hp.get("bye") == w:
            by_week[w] = 0.0              # his team's bye is the starter's bye too
            continue
        out_s = {cuff.starter: 0.0} if cuff.starter in b.ctx.players else {}
        up = cuff.promoted(te.value_at(hp, w), w)
        scn_w = b._total(*path_w[w], w, False, {**out_s, holder: up})
        scn_wo = b._total(*path_wo[w], w, False, out_s)
        by_week[w] = pm * max(0.0, (scn_w - scn_wo) - (base_w[w] - base_wo[w]))
    return b.average(by_week)


# ---------------------------------------------------------------- one free agent
def evaluate(b: Board, pid: str, insure_mine: dict, cuffs: dict) -> dict:
    """Best drop, the START/COVER split, INSURE, and the drop's own cost."""
    R = list(b.roster)
    base = b.value(R)
    droppable = [d for d in R if b.ctx.players[d]["pos"] in b.ctx.valued_pos]
    options = droppable if len(R) >= b.ctx.cfg["rosterSize"] else [None, *droppable]
    best = None
    for d in options:
        Rp = [x for x in R if x != d] + [pid]
        g = b.value(Rp) - base
        score = g - insure_mine.get(d, 0.0)
        if best is None or score > best[0] + te.EPS:
            best = (score, d, Rp, g)
    _, d, Rp, gain = best

    act_new, act_old = b.weekly(Rp), b.weekly(R)
    hl_new, hl_old = b.weekly(Rp, healthy=True), b.weekly(R, healthy=True)
    start_w = {w: hl_new[w] - hl_old[w] for w, _ in b.weeks}
    gain_w = {w: act_new[w] - act_old[w] for w, _ in b.weeks}
    cover_w = {w: gain_w[w] - start_w[w] for w, _ in b.weeks}
    first = [w for w, _ in b.weeks][:BID_WINDOW]

    c = cuffs.get(pid)
    # Against the roster as it stands, not R − d*: if the man we'd cut would himself fill
    # the hole when the starter sits, the add is only worth the difference.
    ins = insure(b, pid, c, Rp, R) if c else 0.0
    handcuff = bool(c and (c.starter in R or c.starter in {p for p, _ in b.ir}))
    if b.ctx.players[pid]["pos"] == "QB" and not handcuff:
        # (J) A quarterback is one slot and a roster already holds two or three. Insurance on
        # a starter we don't own is a lottery ticket on another team's injury, not cover for
        # ours — bye and injury cover for our own QBs is COVER and is already counted. Case
        # Keenum, behind Caleb Williams, was a $7 "insurance" bid for a manager with Young
        # and Daniels. A QB handcuff to OUR quarterback still counts.
        ins = 0.0
    lost = insure_mine.get(d, 0.0) if d else 0.0
    drop_lineup = (base - b.value([x for x in R if x != d])) if d else 0.0
    return {
        "gain": gain, "start": b.average(start_w), "cover": b.average(cover_w),
        "next3": b.average(gain_w, first), "insure": ins, "insure_lost": lost,
        "fit": b.average(start_w) + b.average(cover_w) + ins - lost,
        "drop_id": d, "drop_cost": drop_lineup + lost,
        "handcuff": handcuff,
        "_start_w": start_w, "_cover_w": cover_w,
    }


# ======================================================================== the whole board
def _signal_rates() -> dict:
    try:
        return json.loads(SIGNAL_FILE.read_text())
    except Exception:
        return {}


def signal_p(pos: str, flags, tags) -> tuple[float, str] | None:
    """P(role reaches the next rung within 3 weeks), best of the player's rising-usage flags,
    shrunk toward the position's no-flag base rate (config/signal_rates.json)."""
    rates = _signal_rates()
    base = (rates.get("base") or {}).get(pos, {}).get("rate")
    if base is None:
        return None
    best = None
    for f in flags or []:
        if f not in SIGNALS:
            continue
        persist = "window" if f == "ROLE+" else ("spike" if (tags or {}).get(f) == "spike" else "sustained")
        cell = ((rates.get("flags") or {}).get(f) or {}).get(pos, {}).get(persist)
        if not cell:
            continue
        p = (cell["rate"] * cell["n"] + base * SIGNAL_K) / (cell["n"] + SIGNAL_K)
        if best is None or p > best[0]:
            best = (p, f)
    return best


def _flip(ctx: te.Ctx, my_id, pid: str, fc_value: float | None) -> tuple[float, int]:
    """FantasyCalc value if at least two other teams would start him at his position."""
    p = ctx.players[pid]
    n = 0
    for tid, s in ctx.base.items():
        if tid == my_id:
            continue
        starters = [e[2] for e in te.lineup(s.ids, ctx, None).starters if e[1] == p["pos"]]
        if starters and (p.get("ppg") or 0) > min(starters):
            n += 1
    return (float(fc_value or 0.0) if n >= FLIP_TEAMS else 0.0), n


def build_board(league: dict, cfg: dict, my_id, now: int, avail: dict, byes: dict) -> Board:
    """Two engine contexts over the same horizon: as-it-will-be and all-healthy."""
    import copy
    from .ids import canon_team

    weeks = horizon(now)
    wset = [w for w, _ in weeks]
    lg = copy.deepcopy(league)
    lg_h = copy.deepcopy(league)
    me = next(t for t in lg["teams"] if t["id"] == my_id)
    mine = set(me["roster"]) | set(me.get("ir") or [])
    for pid, p in lg["players"].items():
        p["bye"] = byes.get(canon_team(p.get("nfl") or ""))
        a = avail.get(pid)
        if a:
            # on top of the projection's own weekly values, never instead of them
            p["weekly"] = {**(p.get("weekly") or {}), **{w: 0.0 for w in wset if w < a["back"]}}
        ph = lg_h["players"][pid]
        if pid in mine:
            ph["bye"], ph["weekly"] = None, None
        else:
            ph["bye"], ph["weekly"] = p["bye"], p.get("weekly")
    hz = {"horizon": {"weeks": wset, "weights": {w: wt for w, wt in weeks}}}
    ctx = te.build_context(lg, te.merge(cfg, hz))
    ctx_h = te.build_context(lg_h, te.merge(cfg, hz))
    ir = sorted(((pid, (avail.get(pid) or {}).get("back", int(now) + IR_MIN_WEEKS))
                 for pid in (me.get("ir") or [])), key=lambda x: x[1])
    return Board(ctx, ctx_h, my_id, list(me["roster"]), ir, weeks)


def make_cuffs(b: Board, cuff_rows: dict, xfp_pg: dict, vol_ratio: dict, avail: dict,
               ages: dict) -> dict:
    """pid -> Cuff for every player whose next-man-up starter we can price.

        promoted_w = min(normal_w + boost_w, max(normal_w, starter's projection))
        boost_w    = inherit[pos] × starter's xFP/g × (implied_w / avg)^GAMMA

    The boost is the fitted increment a next man up actually gains when the starter sits
    (contingency.inherit_fraction), in expected half-PPR points — usage at average
    efficiency, so a backup inherits the touches, not the starter's touchdown luck. It is
    added to the SAME projection the rest of the board uses for him, so INSURE is only the
    job opening and never a disagreement between two projections of his current role."""
    from .forward import GAMMA
    from .ids import canon_team

    now = b.weeks[0][0]
    out = {}
    for pid, c in cuff_rows.items():
        s = c.get("cuff_of")
        if pid not in b.ctx.players or not s or s not in xfp_pg:
            continue
        team = canon_team(b.ctx.players[pid].get("nfl") or "")
        sa = avail.get(s)
        pos = c.get("pos") or b.ctx.players[pid]["pos"]
        share = cg.inherit_fraction(pos)
        boost, p_miss = {}, {}
        for w, _ in b.weeks:
            boost[w] = share * xfp_pg[s] * (vol_ratio.get((team, w), 1.0) ** GAMMA)
            if sa and w < sa["back"]:
                p_miss[w] = 1.0
            else:
                p_miss[w] = cg.miss_hazard(pos, ages.get(s), w - now)
        cap = (b.ctx.players.get(s) or {}).get("ppg")
        out[pid] = Cuff(s, boost, p_miss, float(cap) if cap else math.inf)
    return out


def insure_mine(b: Board, cuffs: dict) -> dict:
    """INSURE for each of our own players, and fed back into the IR-return drop choice.

    Two passes: the first prices each handcuff against the roster path as it stands; the
    second re-runs the path with those values in hand, in case a return now cuts someone
    else. One more pass can't change anything the second didn't."""
    out: dict = {}
    for _ in range(2):
        b.keep_value, b.memo = dict(out), {}
        R = list(b.roster)
        out = {d: insure(b, d, cuffs[d], R, [x for x in R if x != d]) for d in R if d in cuffs}
    b.keep_value, b.memo = dict(out), {}
    return out


COVER_ONLY_START = 0.25     # pts/wk of START under which an add is judged as pure cover
COVER_REPLACEMENT = 3       # the k-th best cover-only add at a position is what a miss still gets


def cover_baseline(evs: dict, pos_of: dict, k: int = COVER_REPLACEMENT) -> dict:
    """(J) Cover is a hole in a week, and one body fills it: when a dozen free quarterbacks
    would each cover our QB's bye the same ~14 pts, none is worth that to us, because the
    next one is there for nothing. Returns pos -> fit of the k-th best cover-only add at
    that position, netted off every such add's fit (its price), not its lane: the hole is
    still a reason to claim one; 0 when fewer than k compete."""
    by_pos: dict = {}
    for pid, ev in evs.items():
        if ev["start"] < COVER_ONLY_START and ev["insure"] <= 0 and ev["cover"] > 0:
            by_pos.setdefault(pos_of[pid], []).append(ev)
    out = {}
    for pos, lst in by_pos.items():
        f = sorted((e["fit"] for e in lst), reverse=True)
        out[pos] = f[k - 1] if len(lst) >= k else 0.0
    return out


def assess(b: Board, pool: list, cuffs: dict, fc_by_name: dict, roles: dict, flat: dict,
           baselines: dict, pct_ros: dict, budget_left: int, now: int,
           avail: dict | None = None) -> pd.DataFrame:
    """Every free agent in `pool`, decomposed and placed in at most one lane."""
    from . import faab as fb
    from .roles import LADDER

    mine_ins = insure_mine(b, cuffs)
    share_mu = (baselines.get("role") or {}).get("xfp_share") or {}
    rows = []
    pool = [pid for pid in pool if b.ctx.players[pid]["pos"] in b.ctx.valued_pos]
    evs = {pid: evaluate(b, pid, mine_ins, cuffs) for pid in pool}
    base = cover_baseline(evs, {pid: b.ctx.players[pid]["pos"] for pid in pool})
    for pid in pool:
        p = b.ctx.players[pid]
        ev = evs[pid]
        if ev["start"] < COVER_ONLY_START and ev["insure"] <= 0 and ev["cover"] > 0:
            bf = base.get(p["pos"], 0.0)
            ev = {**ev, "cover": max(0.0, ev["cover"] - bf), "fit": ev["fit"] - bf}
        flip, n_buyers = _flip(b.ctx, b.my_id, pid, fc_by_name.get(p["name"]))

        rc = roles.get(pid) or {}
        sig = signal_p(p["pos"], rc.get("flags"), rc.get("tags")) if flat.get(pid) else None
        signal_score, gain_up = 0.0, 0.0
        if sig and ev["fit"] > FIT_MIN:
            lad = LADDER.get(p["pos"], ())
            role = rc.get("role")
            if role in lad and lad.index(role) > 0:
                up = lad[lad.index(role) - 1]
                mu_now, mu_up = share_mu.get(role), share_mu.get(up)
                if mu_now and mu_up:
                    gain_up = _gain_with_ppg(b, pid, (p.get("ppg") or 0) * mu_up / mu_now, mine_ins, cuffs)
                    signal_score = sig[0] * max(0.0, gain_up)

        fit = ev["fit"]
        if fit <= FIT_MIN:
            lane = "trade_chip" if flip > 0 else ""
        elif ev["next3"] > TAU_BID:
            lane = "bid_now"
        elif signal_score > 0:
            lane = "early_signal"
        else:
            lane = "stash"
        mech = {"START": ev["start"], "COVER": ev["cover"], "INSURE": ev["insure"]}
        own = pct_ros.get(p["name"])
        bid = fb.suggest(max(0.0, fit), budget_left, now) if lane in ("bid_now", "stash", "early_signal") else None
        d = ev["drop_id"]
        rows.append({
            "pid": pid, "player": p["name"], "pos": p["pos"], "nfl_team": p.get("nfl"),
            "ppg": p.get("ppg"), "lane": lane,
            "fit": round(fit, 3), "start": round(ev["start"], 3), "cover": round(ev["cover"], 3),
            "insure": round(ev["insure"], 3), "next3": round(ev["next3"], 3),
            "gain": round(ev["gain"], 3), "flip": round(flip, 1), "flip_buyers": n_buyers,
            "mechanism": max(mech, key=lambda k: mech[k]) if fit > FIT_MIN else "",
            "handcuff": ev["handcuff"],
            "behind": b.ctx.players[cuffs[pid].starter]["name"] if pid in cuffs and cuffs[pid].starter in b.ctx.players else "",
            "drop": b.ctx.players[d]["name"] if d else "", "drop_cost": round(ev["drop_cost"], 3),
            "drop_insure": round(ev["insure_lost"], 3),
            "drop_flip": round(_flip(b.ctx, b.my_id, d, fc_by_name.get(b.ctx.players[d]["name"]))[0], 1) if d else 0.0,
            "signal": sig[1] if sig else "", "p_expand": round(sig[0], 3) if sig else None,
            "gain_if_expands": round(gain_up, 3), "signal_score": round(signal_score, 3),
            "pct_ros": own, "market_on": bool(own is not None and own >= MARKET_ON),
            "bid": bid["bid"] if bid else 0, "max_bid": bid["max_worth"] if bid else 0,
            "role": rc.get("role") or "",
            # already priced: build_board zeroes his weeks before `back`. This only labels it.
            "out_status": (avail or {}).get(pid, {}).get("status", ""),
            "out_back": (avail or {}).get(pid, {}).get("back"),
        })
    return pd.DataFrame(rows)


RIVAL_ALTS = 6     # (J) free agents per position, best by ppg, a rival weighs a claim against
RIVAL_MIN = COVER_ONLY_START   # (J) pts/wk net a rival must gain to bother bidding: the league's
                               # settled claims drew 2-5 bidders, not the 6-9 a 0.02 floor implies


def bid_vs_rivals(top: int, who: str, n: int, fit: float, max_bid: int) -> tuple[int, str]:
    """(bid, why) once the likeliest top rival bid is known — shared with the K/DEF
    streamers (mega/kdef.py) so every claim on the board follows one rule."""
    if top <= 0:
        return (1 if fit > 0.05 else 0,
                "no other team gains enough from him to bid" if n == 0 else
                f"{n} other team{'s' if n > 1 else ''} would add him, none likely to pay")
    if top + 1 <= max_bid:
        return top + 1, f"beats {who}'s likely ${top} ({n} team{'s' if n > 1 else ''} interested)"
    if top <= max_bid:
        return int(max_bid), f"matches {who}'s likely ${top}; a tie goes to the worse record"
    return 0, f"likely outbid: {who} ~${top}, over your ${max_bid} ceiling"


def rival_bids(b: Board, rows: pd.DataFrame, faab_left: dict, now: int) -> pd.DataFrame:
    """Price each claim against the other eleven rosters (handoff §5.4: beat the likeliest
    top bid by a dollar, or walk away).

    A rival's interest is his own horizon gain from the add — his roster rebuilt with the
    player and forced back to size, on the same engine context that priced ours — and his
    bid is that gain through the same faab.suggest curve, capped by what he has left. One
    pricing rule for all twelve teams, so the comparison is like for like.

    His gain is net of what he could add for free instead — the COVER_REPLACEMENT-th best
    gain among the other free agents at that position, the rule cover_baseline applies to
    ours. Without it every thin bench "wants" every free agent for bye cover alone, and
    six to nine teams turned up interested in a backup quarterback.

      nobody gains (or nobody can pay) -> $1, the floor that still wins a tie
      top rival + $1 within my ceiling -> bid that
      top rival exactly at my ceiling  -> bid the ceiling (a tie goes by reverse standings)
      top rival + $1 over my ceiling   -> pass, with the number
    """
    from . import faab as fb
    ctx = b.ctx
    cols = {"bid": [], "rival_top": [], "rival_team": [], "rivals_n": [], "bid_note": []}
    gain_memo: dict = {}

    def gain(tid, pid):
        k = (tid, pid)
        if k not in gain_memo:
            t = ctx.teams[tid]
            gain_memo[k] = te.settle(list(t["roster"]) + [pid], ctx, {pid}).value - ctx.base[tid].value
        return gain_memo[k]

    by_pos: dict = {}            # the free agents a rival would consider instead, best first
    for pid in rows["pid"]:
        by_pos.setdefault(ctx.players[pid]["pos"], []).append(pid)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda i: -(ctx.players[i].get("ppg") or 0))
        by_pos[pos] = by_pos[pos][:RIVAL_ALTS]

    def free_alt(tid, pid):
        alts = sorted((gain(tid, a) for a in by_pos.get(ctx.players[pid]["pos"], []) if a != pid), reverse=True)
        return max(0.0, alts[COVER_REPLACEMENT - 1]) if len(alts) >= COVER_REPLACEMENT else 0.0

    for r in rows.itertuples():
        if r.lane not in ("bid_now", "early_signal", "stash"):
            for k, v in (("bid", r.bid), ("rival_top", 0), ("rival_team", ""), ("rivals_n", 0), ("bid_note", "")):
                cols[k].append(v)
            continue
        top, who, n = 0, "", 0
        for tid, t in ctx.teams.items():
            if tid == b.my_id:
                continue
            g = gain(tid, r.pid) - free_alt(tid, r.pid)
            if g < RIVAL_MIN:
                continue
            n += 1
            bid = fb.suggest(g, int(faab_left.get(t["name"], 0)), now)["bid"]
            if bid > top:
                top, who = bid, t["name"]
        bid, note = bid_vs_rivals(top, who, n, r.fit, r.max_bid)
        for k, v in (("bid", bid), ("rival_top", top), ("rival_team", who), ("rivals_n", n), ("bid_note", note)):
            cols[k].append(v)
    return rows.assign(**cols)


def _gain_with_ppg(b: Board, pid: str, ppg: float, insure_mine: dict, cuffs: dict) -> float:
    """START + COVER if his value were `ppg` — the "gain if his role expands" (§3.3)."""
    p = b.ctx.players[pid]
    ph = b.ctx_h.players[pid]
    saved_memo, saved = b.memo, (p, ph)
    b.memo = {}
    b.ctx.players[pid] = {**p, "ppg": ppg}
    b.ctx_h.players[pid] = {**ph, "ppg": ppg}
    try:
        ev = evaluate(b, pid, insure_mine, {})
        return ev["start"] + ev["cover"]
    finally:
        b.ctx.players[pid], b.ctx_h.players[pid] = saved
        b.memo = saved_memo


def roster_notes(b: Board, avail: dict) -> list[str]:
    """Plain-English consequences of the IR path, e.g. the drop a return will force."""
    names = lambda pid: b.ctx.players[pid]["name"]
    notes = []
    for pid, back in b.ir:
        a = avail.get(pid) or {}
        why = {"IR": "on injured reserve", "Out": "ruled out"}.get(a.get("status"), "on your IR slot")
        notes.append(f"{names(pid)} is {why}; expected back week {back}.")
    for d in b.path(b.roster)["drops"]:
        if d["drop"] == d["returning"]:
            pos = b.ctx.players[d["drop"]]["pos"]
            notes.append(f"When {names(d['returning'])} is back (week {d['week']}) he projects as your "
                         f"weakest {pos} — on these numbers he, not a healthy player, is the one to cut.")
        else:
            notes.append(f"{names(d['drop'])} is your cheapest drop once {names(d['returning'])} is back "
                         f"(week {d['week']}), so he is the default cut for any add from then on.")
    return notes


# ======================================================================== data wiring
@functools.lru_cache(maxsize=2)
def _rosters_weekly(season: int) -> pd.DataFrame:
    try:
        import nflreadpy as nfl
        return nfl.load_rosters_weekly(seasons=[season]).to_pandas()[["gsis_id", "week", "status"]]
    except Exception:
        return pd.DataFrame(columns=["gsis_id", "week", "status"])


@functools.lru_cache(maxsize=1)
def _birthdays() -> dict:
    try:
        import nflreadpy as nfl
        p = nfl.load_players().to_pandas()[["gsis_id", "birth_date"]].dropna()
        return dict(zip(p["gsis_id"], pd.to_datetime(p["birth_date"], errors="coerce")))
    except Exception:
        return {}


def run(season: int, now: int, yahoo_rosters: pd.DataFrame | None = None) -> dict:
    """The live board: {"rows": DataFrame, "notes": [...], "meta": {...}}."""
    from . import faab as fb
    from . import forward as fw
    from . import season as S
    from . import trade_league as tl
    from .config import MY_TEAM
    from .sources import fantasycalc_values

    lg = tl.build_league(season, yahoo_rosters)
    my_id = lg.get("report", {}).get("my_team_id")
    if not lg.get("teams") or my_id is None:
        return {"rows": pd.DataFrame(), "notes": [], "meta": {"error": "no league"}}
    cfg = tl.engine_config()
    cfg["replacementRank"] = 3          # mega/needs.py REPLACEMENT_RANK: the winner's-curse fix

    avail = availability(S.injuries(season), _rosters_weekly(season), now)
    from .ids import norm
    from .status import back_weeks
    held = back_weeks(now)           # data/player_status.csv: Chris's word on who is out
    for pid, p in lg["players"].items():
        b = held.get(norm(p["name"]))
        if b and b > (avail.get(pid) or {}).get("back", 0):
            avail[pid] = {"status": "Out", "back": b}
    sched = S.schedules(season)
    b = build_board(lg, cfg, my_id, now, avail, bye_weeks(sched))

    pw = S.player_week(season)
    recent = pw.sort_values("week").groupby("gsis_id").tail(4)      # roles.WINDOW_METRICS
    xfp_pg = recent.groupby("gsis_id")["half_ppr_exp"].mean().dropna().to_dict()
    last3 = pw.sort_values("week").groupby("gsis_id").tail(3)
    agg = last3.groupby("gsis_id")[["half_ppr", "half_ppr_exp"]].sum()
    flat = (agg["half_ppr"] <= agg["half_ppr_exp"]).to_dict()      # production hasn't caught up
    vol = fw.team_volume(S.ff_opportunity(season), sched, [w for w, _ in b.weeks])
    vol_ratio = {(r.team, int(r.week)): float(r.implied_ratio) for r in vol.itertuples()} if not vol.empty else {}
    bd = _birthdays()
    kick = pd.Timestamp(f"{season}-09-01")
    ages = {g: (kick - d).days / 365.25 for g, d in bd.items() if pd.notna(d)}

    try:
        cu = S.cuff_lookup(season)
    except Exception:
        cu = {}
    cuffs = make_cuffs(b, cu, xfp_pg, vol_ratio, avail, ages)
    try:
        roles = S.role_lookup(season)
        baselines = S.role_context(season).get("baselines") or {}
    except Exception:
        roles, baselines = {}, {}

    fc = fantasycalc_values()
    fc_by_name = dict(zip(fc["name"], fc["value"])) if not fc.empty else {}
    fa_csv = tl.DATA / "yahoo_free_agents.csv"
    pct_ros = {}
    if fa_csv.is_file():
        f = pd.read_csv(fa_csv, dtype=str)
        pct_ros = dict(zip(f["player"], pd.to_numeric(f.get("pct_ros"), errors="coerce")))
        pct_ros = {k: float(v) for k, v in pct_ros.items() if pd.notna(v)}

    bud = fb.cached_budgets()
    mine = bud[bud["team"] == MY_TEAM]["faab_left"] if not bud.empty else pd.Series(dtype=float)
    budget_left = int(mine.iloc[0]) if not mine.empty else fb.BUDGET

    rows = assess(b, lg["freeAgents"], cuffs, fc_by_name, roles, flat, baselines, pct_ros,
                  budget_left, now, avail)
    if not rows.empty and not bud.empty:
        rows = rival_bids(b, rows, dict(zip(bud["team"], pd.to_numeric(bud["faab_left"], errors="coerce").fillna(0))), now)

    # HANDOFF v1.3 §5 / D1: rank by the change in title odds. Only lane players are
    # simulated — the rest have no fit, so there is nothing to price.
    meta_title = {}
    try:
        from . import title_odds as T
        tm = T.build(season, now, yahoo_rosters, league=lg)
        if tm is not None and not rows.empty:
            my_name = next(t["name"] for t in lg["teams"] if t["id"] == my_id)
            base = tm.odds()[my_name]
            meta_title = {"p_playoffs": base["p_playoffs"], "p_title": base["p_title"],
                          "posture": T.posture(base["p_playoffs"]), "seasons": tm.n}
            mine = list(tm.rosters[my_name])
            d_title, se_title, noise = [], [], []
            for r in rows.itertuples():
                if r.lane not in ("bid_now", "early_signal", "stash"):
                    d_title.append(None), se_title.append(None), noise.append(None)
                    continue
                drop = next((pid for pid in mine if b.ctx.players.get(pid, {}).get("name") == r.drop), None)
                after = [p for p in mine if p != drop] + [r.pid]
                dd = tm.delta({my_name: after}, (my_name,))[my_name]
                d_title.append(round(dd["d_title"], 4)), se_title.append(round(dd["se_title"], 4))
                noise.append(bool(dd["noise"]))
            rows["d_title"], rows["se_title"], rows["title_noise"] = d_title, se_title, noise
    except Exception as e:
        meta_title = {"error": f"{type(e).__name__}: {e}"}
    return {"rows": rows, "notes": roster_notes(b, avail),
            "meta": {"weeks": [w for w, _ in b.weeks], "tau_bid": TAU_BID, "fit_min": FIT_MIN,
                     "budget_left": budget_left, "title": meta_title}}
