"""Trade value engine — a Python port of Chris's `trade_engine.js` (HANDOFF.md, v1.0.0).

A trade is never valued as "player A against player B". Both rosters are rebuilt, forced
back to legal size, re-optimised, and compared with where they started. What comes out is
the change in each side's **optimal lineup plus weighted bench depth**, in points per week.

The port is deliberately line-for-line with the JavaScript so the two can be diffed, and
`tests/test_trade_engine.py` asserts the same hand-checked numbers the JS suite does
(TaylorMade 99.475, Opponent A 95.2, and the 2-for-1 worked example).

One thing had to change: Python is far slower than V8 at the settle loop, which calls
team_value once per drop candidate. Every team value is memoised on the roster itself, so
the search reuses work across the thousands of trades it enumerates rather than recomputing
the same lineups. Semantics are unchanged; see `_ctx.memo`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

EPS = 1e-9

DEFAULTS: dict[str, Any] = {
    "slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1},
    "flexCount": 1,
    "flexEligible": ["RB", "WR", "TE"],
    "rosterSize": 15,
    "depthWeights": {"QB": [0.10], "RB": [0.15, 0.05], "WR": [0.15, 0.05], "TE": [0.10]},
    "horizon": None,
    "themTolerance": 1.0,
    "market": {"k": 50, "tolerance": 0.15, "unrankedEcr": 300},
    "faCandidatesPerPos": 3,
    # Which free agent sets replacement level: 1 = the best one, the JS engine's rule and
    # what the ported tests pin. The max over a large pool is biased high — it is the single
    # luckiest estimate of a few hundred — so a caller pricing waiver adds should raise this.
    # See mega/needs.py, which uses 3 (you rarely win a bid on the pool's very best player).
    "replacementRank": 1,
    "padTolerance": 0.05,
    "search": {
        "shapes": ["1-for-1", "2-for-1", "1-for-2"],
        "includeFlags": ["LIKELY", "EXPLOIT", "NEEDS_PITCH"],
        "minDeltaMe": 0.01,
        "topN": 50,
        "sortBy": "dMe",
        "protectMine": [],
        "untouchable": [],
        "receivePositions": None,
        "partners": None,
    },
}

FLAG_ORDER = ["LIKELY", "EXPLOIT", "NEEDS_PITCH", "LONGSHOT"]


def merge(a: dict, b: dict | None) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = merge(a[k], v) if isinstance(a.get(k), dict) and isinstance(v, dict) else v
    return out


def value_at(p: dict, week: int | None) -> float:
    if week is None:
        return p.get("ppg") or 0.0
    if p.get("bye") == week:
        return 0.0
    wk = (p.get("weekly") or {}).get(str(week), (p.get("weekly") or {}).get(week))
    return wk if wk is not None else (p.get("ppg") or 0.0)


def market_value(p: dict, cfg: dict) -> float:
    r = p.get("ecr")
    r = cfg["market"]["unrankedEcr"] if r is None else r
    return 100 * math.exp(-(r - 1) / cfg["market"]["k"])


@dataclass
class Ctx:
    cfg: dict
    players: dict
    teams: dict
    valued_pos: set
    pos_order: list
    fa: list
    weeks: list
    repl: dict
    fa_candidates: list
    base: dict = field(default_factory=dict)
    memo: dict = field(default_factory=dict)


@dataclass
class Lineup:
    starters: list
    bench: list
    starter_pts: float
    depth_pts: float
    total: float


def lineup(ids, ctx: Ctx, week: int | None = None) -> Lineup:
    """Optimal lineup for a set of ids in one week.

    Greedy is exact for this slot structure: dedicated slots take the top N at each
    position and FLEX takes the best leftover. The JS suite proves it against brute force
    over 300 random rosters; the Python suite repeats that test.
    """
    cfg, players = ctx.cfg, ctx.players
    by_pos: dict[str, list] = {}
    for pid in ids:
        p = players.get(pid)
        if not p or p["pos"] not in ctx.valued_pos:
            continue
        by_pos.setdefault(p["pos"], []).append((pid, p["pos"], value_at(p, week)))
    for lst in by_pos.values():
        lst.sort(key=lambda e: -e[2])

    starters, leftovers = [], []
    for pos in ctx.pos_order:
        n = cfg["slots"].get(pos, 0)
        for i, e in enumerate(by_pos.get(pos, [])):
            if i < n:
                starters.append((*e, pos))      # (id, pos, pts, slot)
            else:
                leftovers.append(e)

    flex = sorted((e for e in leftovers if e[1] in cfg["flexEligible"]),
                  key=lambda e: -e[2])[: cfg["flexCount"]]
    flex_ids = {e[0] for e in flex}
    starters.extend((*e, "FLEX") for e in flex)
    bench = [e for e in leftovers if e[0] not in flex_ids]   # grouped by position, desc within

    repl = ctx.repl.get(week, ctx.repl[None])
    seen: dict[str, int] = {}
    depth_pts = 0.0
    for pid, pos, v in bench:
        i = seen[pos] = 0 if pos not in seen else seen[pos] + 1
        weights = cfg["depthWeights"].get(pos) or []
        w = weights[i] if i < len(weights) else 0
        depth_pts += w * max(0.0, v - repl.get(pos, 0))
    starter_pts = sum(e[2] for e in starters)
    return Lineup(starters, bench, starter_pts, depth_pts, starter_pts + depth_pts)


def team_value(ids, ctx: Ctx) -> float:
    """Points per week: the lineup total, averaged across the horizon when one is set.

    Memoised on the roster — the settle loop asks for the same rosters over and over as the
    search walks thousands of trades, and in Python that repetition is the whole cost.
    """
    key = frozenset(ids)
    hit = ctx.memo.get(key)
    if hit is not None:
        return hit
    num = den = 0.0
    for week, weight in ctx.weeks:
        num += weight * lineup(ids, ctx, week).total
        den += weight
    out = num / den if den else 0.0
    ctx.memo[key] = out
    return out


@dataclass
class Settled:
    ids: list
    dropped: list
    added: list
    value: float


def settle(ids, ctx: Ctx, keep: set) -> Settled:
    """Force a roster to legal size. Over: drop whoever costs the least (ties -> lowest ppg).
    Under: add the free agent who adds the most (ties -> highest ppg). `keep` and K/DEF are
    never dropped."""
    cfg, players = ctx.cfg, ctx.players
    cur = list(ids)
    dropped, added = [], []

    while len(cur) > cfg["rosterSize"]:
        best, best_val, best_ppg = None, -math.inf, math.inf
        for pid in cur:
            if pid in keep or players[pid]["pos"] not in ctx.valued_pos:
                continue
            v = team_value([x for x in cur if x != pid], ctx)
            ppg = players[pid].get("ppg") or 0
            if v > best_val + EPS or (abs(v - best_val) <= EPS and ppg < best_ppg):
                best, best_val, best_ppg = pid, v, ppg
        if best is None:
            break
        cur = [x for x in cur if x != best]
        dropped.append(best)

    while len(cur) < cfg["rosterSize"]:
        have = set(cur)
        best, best_val, best_ppg = None, -math.inf, -math.inf
        for pid in ctx.fa_candidates:
            if pid in have:
                continue
            v = team_value(cur + [pid], ctx)
            ppg = players[pid].get("ppg") or 0
            if v > best_val + EPS or (abs(v - best_val) <= EPS and ppg > best_ppg):
                best, best_val, best_ppg = pid, v, ppg
        if best is None:
            break
        cur.append(best)
        added.append(best)

    return Settled(cur, dropped, added, team_value(cur, ctx))


def build_context(league: dict, config: dict | None = None) -> Ctx:
    cfg = merge(merge(DEFAULTS, league.get("settings")), config)
    players = league.get("players") or {}
    teams = {t["id"]: t for t in (league.get("teams") or [])}

    missing, rostered = [], set()
    for t in teams.values():
        for pid in t["roster"]:
            if pid not in players:
                missing.append(f"{t['id']}:{pid}")
            rostered.add(pid)
    if missing:
        raise ValueError("Rostered ids missing from players: " + ", ".join(missing))

    valued_pos = set(cfg["slots"]) | set(cfg["flexEligible"])
    pos_order = list(valued_pos)

    fa = [pid for pid in (league.get("freeAgents") or [])
          if pid in players and pid not in rostered and players[pid]["pos"] in valued_pos]

    hz = cfg.get("horizon")
    weeks = ([(w, (hz.get("weights") or {}).get(w, 1)) for w in hz["weeks"]]
             if hz and hz.get("weeks") else [(None, 1)])

    rank = max(1, int(cfg.get("replacementRank", 1)))
    repl: dict = {}
    for w in [None, *[w for w, _ in weeks if w is not None]]:
        by_pos: dict[str, list[float]] = {pos: [] for pos in valued_pos}
        for pid in fa:
            p = players[pid]
            by_pos[p["pos"]].append(value_at(p, w))
        r = {}
        for pos, vals in by_pos.items():
            vals.sort(reverse=True)
            r[pos] = vals[min(rank, len(vals)) - 1] if vals else 0.0
        repl[w] = r

    fa_candidates = []
    for pos in valued_pos:
        same = [pid for pid in fa if players[pid]["pos"] == pos]
        same.sort(key=lambda pid: -(players[pid].get("ppg") or 0))
        fa_candidates.extend(same[: cfg["faCandidatesPerPos"]])

    ctx = Ctx(cfg, players, teams, valued_pos, pos_order, fa, weeks, repl, fa_candidates)
    for t in teams.values():
        ctx.base[t["id"]] = settle(t["roster"], ctx, set())
    return ctx


def _info(ctx: Ctx, pid: str) -> dict:
    p = ctx.players[pid]
    return {"id": pid, "name": p.get("name") or pid, "pos": p["pos"], "nfl": p.get("nfl"),
            "ppg": p.get("ppg") or 0, "ecr": p.get("ecr")}


def flag_for(lineup_ok: bool, market_ok: bool) -> str:
    if lineup_ok and market_ok:
        return "LIKELY"
    if market_ok:
        return "EXPLOIT"          # looks fair on rankings, hurts their lineup — the edge
    if lineup_ok:
        return "NEEDS_PITCH"
    return "LONGSHOT"


def evaluate_core(ctx: Ctx, my_id, their_id, give_ids, get_ids) -> dict:
    cfg, players = ctx.cfg, ctx.players
    me, them = ctx.teams[my_id], ctx.teams[their_id]
    give_set, get_set = set(give_ids), set(get_ids)

    a_me = settle([i for i in me["roster"] if i not in give_set] + list(get_ids), ctx, get_set)
    a_them = settle([i for i in them["roster"] if i not in get_set] + list(give_ids), ctx, give_set)
    b_me, b_them = ctx.base[my_id], ctx.base[their_id]

    d_me = a_me.value - b_me.value
    d_them = a_them.value - b_them.value

    mv_give = sum(market_value(players[i], cfg) for i in give_ids)   # what they receive
    mv_get = sum(market_value(players[i], cfg) for i in get_ids)     # what they give up
    ratio = mv_give / mv_get if mv_get > 0 else math.inf

    lineup_ok = d_them >= -cfg["themTolerance"] - EPS
    market_ok = ratio >= 1 - cfg["market"]["tolerance"] - EPS

    return {"partner": {"id": their_id, "name": them.get("name") or str(their_id)},
            "shape": f"{len(give_ids)}-for-{len(get_ids)}",
            "giveIds": list(give_ids), "getIds": list(get_ids),
            "dMe": d_me, "dThem": d_them,
            "lineupOK": lineup_ok, "marketOK": market_ok,
            "flag": flag_for(lineup_ok, market_ok),
            "market": {"mvYouGive": mv_give, "mvYouGet": mv_get, "ratio": ratio},
            "_after": {"me": a_me, "them": a_them}}


def _snapshot(ctx: Ctx, ids) -> dict:
    l = lineup(ids, ctx, None)
    return {"starters": [{"slot": e[3], **_info(ctx, e[0])} for e in l.starters],
            "starterPts": l.starter_pts, "depthPts": l.depth_pts}


def _side(ctx: Ctx, base: Settled, after: Settled) -> dict:
    before, aft = _snapshot(ctx, base.ids), _snapshot(ctx, after.ids)
    b_ids = {s["id"] for s in before["starters"]}
    a_ids = {s["id"] for s in aft["starters"]}
    return {"valueBefore": base.value, "valueAfter": after.value,
            "before": before, "after": aft,
            "startersIn": [s for s in aft["starters"] if s["id"] not in b_ids],
            "startersOut": [s for s in before["starters"] if s["id"] not in a_ids],
            "drops": [_info(ctx, i) for i in after.dropped],
            "adds": [_info(ctx, i) for i in after.added]}


def with_detail(ctx: Ctx, my_id, core: dict) -> dict:
    """Core result plus both sides' before/after. Never mutates `core` — the search caches
    and reuses these, and a popped key would break the second read."""
    after = core.get("_after")
    r = {k: v for k, v in core.items() if k != "_after"}
    r["give"] = [_info(ctx, i) for i in core["giveIds"]]
    r["get"] = [_info(ctx, i) for i in core["getIds"]]
    if after:
        r["me"] = _side(ctx, ctx.base[my_id], after["me"])
        r["them"] = _side(ctx, ctx.base[core["partner"]["id"]], after["them"])
    return r


def find_trades(ctx: Ctx, my_id, give_ids, opts: dict | None = None) -> dict:
    o = merge(ctx.cfg["search"], opts)
    players = ctx.players
    me = ctx.teams.get(my_id)
    if not me:
        raise ValueError(f"Unknown team id {my_id}")
    if not isinstance(give_ids, (list, tuple)) or not 1 <= len(give_ids) <= 2:
        raise ValueError("give_ids must hold 1 or 2 player ids")
    def tradeable(pid):
        return players[pid]["pos"] in ctx.valued_pos
    for pid in give_ids:
        if pid not in me["roster"]:
            raise ValueError(f"{pid} is not on team {my_id}")
        if not tradeable(pid):
            raise ValueError(f"{pid} ({players[pid]['pos']}) is not tradeable")

    give_ids = list(give_ids)
    protect, untouchable = set(o["protectMine"] or []), set(o["untouchable"] or [])
    add_ons = [i for i in me["roster"] if tradeable(i) and i not in give_ids and i not in protect]
    give_sets = [give_ids] + [give_ids + [y] for y in add_ons] if len(give_ids) == 1 else [give_ids]
    want_pairs = any(s.endswith("-for-2") for s in o["shapes"])
    recv = set(o["receivePositions"]) if o["receivePositions"] else None

    cache: dict = {}
    def cached(their_id, g, r):
        k = (their_id, tuple(sorted(g)), tuple(sorted(r)))
        if k not in cache:
            cache[k] = evaluate_core(ctx, my_id, their_id, g, r)
        return cache[k]

    def sub_trades(g, r):
        out = []
        if len(g) > len(give_ids):
            out.append(([x for x in g if x in give_ids], r))
        if len(r) > 1:
            out.extend((g, [y for y in r if y != x]) for x in r)
        return out

    pad = ctx.cfg["padTolerance"]
    results, evaluated, padded = [], 0, 0
    for t in ctx.teams.values():
        if t["id"] == my_id:
            continue
        if o["partners"] and t["id"] not in o["partners"]:
            continue
        theirs = [i for i in t["roster"] if tradeable(i) and i not in untouchable]
        get_sets = [[x] for x in theirs] + ([list(c) for c in combinations(theirs, 2)] if want_pairs else [])
        for g in give_sets:
            for r in get_sets:
                if f"{len(g)}-for-{len(r)}" not in o["shapes"]:
                    continue
                if recv and not any(players[i]["pos"] in recv for i in r):
                    continue
                evaluated += 1
                res = cached(t["id"], g, r)
                if res["dMe"] < o["minDeltaMe"] or res["flag"] not in o["includeFlags"]:
                    continue
                if any(cached(t["id"], sg, sr)["dMe"] >= res["dMe"] - pad
                       and cached(t["id"], sg, sr)["dThem"] >= res["dThem"] - pad
                       for sg, sr in sub_trades(g, r)):
                    padded += 1
                    continue
                results.append(res)

    key = (lambda r: r["dMe"] - r["dThem"]) if o["sortBy"] == "edge" else (lambda r: r["dMe"])
    results.sort(key=lambda r: (-key(r), FLAG_ORDER.index(r["flag"])))

    return {"team": {"id": my_id, "name": me.get("name"), "value": ctx.base[my_id].value},
            "evaluated": evaluated, "padded": padded, "matched": len(results),
            "results": [with_detail(ctx, my_id, r) for r in results[: o["topN"]]]}


def explain(r: dict) -> str:
    f = lambda n: ("+" if n >= 0 else "") + f"{n:.2f}"
    names = lambda lst: " + ".join(f"{p['name']} ({p['pos']}, {p['ppg']:.1f})" for p in lst) or "none"
    return "\n".join([
        f"[{r['flag']}] {r['shape']} with {r['partner']['name']}",
        f"  You give: {names(r['give'])}",
        f"  You get:  {names(r['get'])}",
        f"  You:  {f(r['dMe'])} pts/wk  | starters in: {names(r['me']['startersIn'])} | out: {names(r['me']['startersOut'])}"
        + (f" | add FA: {names(r['me']['adds'])}" if r["me"]["adds"] else "")
        + (f" | drop: {names(r['me']['drops'])}" if r["me"]["drops"] else ""),
        f"  Them: {f(r['dThem'])} pts/wk | starters in: {names(r['them']['startersIn'])} | out: {names(r['them']['startersOut'])}"
        + (f" | add FA: {names(r['them']['adds'])}" if r["them"]["adds"] else "")
        + (f" | drop: {names(r['them']['drops'])}" if r["them"]["drops"] else ""),
        f"  Market: they receive {r['market']['mvYouGive']:.1f} vs give {r['market']['mvYouGet']:.1f} "
        f"(ratio {r['market']['ratio']:.2f})",
    ])


class Engine:
    def __init__(self, league: dict, config: dict | None = None):
        self.ctx = build_context(league, config)
        self.config = self.ctx.cfg

    def team_value(self, team_id):
        return self.ctx.base[team_id].value

    def lineup(self, ids, week=None):
        return lineup(ids, self.ctx, week)

    def evaluate_trade(self, my_id, their_id, give_ids, get_ids):
        return with_detail(self.ctx, my_id, evaluate_core(self.ctx, my_id, their_id, give_ids, get_ids))

    def find_trades(self, my_id, give_ids, opts=None):
        return find_trades(self.ctx, my_id, give_ids, opts)

    explain = staticmethod(explain)


def create_engine(league: dict, config: dict | None = None) -> Engine:
    return Engine(league, config)
