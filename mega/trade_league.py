"""Assemble the `league` dict that mega.trade_engine consumes, from data the app already has.

This is the handoff's §8 data pass, with the sources swapped for ones that are already
wired up here: rosters come from the Yahoo scrape (resolved through mega.ids), rest-of-season
points per game from `projections.nflverse_estimate`, and the market rank from FantasyCalc's
overall rank rather than FantasyPros ECR.

Two league specifics the engine's defaults get wrong and this module fixes:

  * **The flex is RB/WR only.** Mega Bowl can't start a TE there (config.FLEX_ELIGIBLE),
    and the engine ships with the usual RB/WR/TE. Left alone it would value every roster as
    if a second tight end could start, which is the same bug that once put Kittle in the
    flex on the Start/Sit tab.
  * **IR doesn't count.** Yahoo roster rows include IR, the engine wants the active roster,
    and leaving them in would make a 15-man roster look 16 deep and drop someone real.
"""
from __future__ import annotations

import pandas as pd

from .config import DATA, FLEX_ELIGIBLE, IR_SLOTS, LINEUP, MY_TEAM

ROSTER_SIZE = 15          # 9 starters (incl. K/DEF) + 6 bench, once IR is excluded


def _s(v) -> str:
    """pandas NA is neither falsy nor stringifiable safely — normalise once, here."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("nan", "none", "<na>") else t


def _pid(row) -> str:
    """Stable id: the nflverse id where the resolver found one, else the normalized name."""
    return _s(row.get("gsis_id")) or "n:" + _s(row.get("norm"))


def engine_config() -> dict:
    """League rules, from config.py rather than the engine's generic defaults."""
    return {
        "slots": {p: n for p, n in LINEUP.items() if p in ("QB", "RB", "WR", "TE")},
        "flexCount": LINEUP.get("W/R", 1),
        "flexEligible": sorted(FLEX_ELIGIBLE),
        "rosterSize": ROSTER_SIZE,
        "irSlots": IR_SLOTS,
    }


def build_league(season: int, yahoo_rosters: pd.DataFrame | None = None) -> dict:
    """{teams, players, freeAgents} plus a `report` of what couldn't be priced."""
    from .intel import current_rosters
    from .projections import fp_ros, nflverse_estimate
    from .sources import fantasycalc_values

    ros = current_rosters(yahoo_rosters)
    if ros.empty:
        return {"teams": [], "players": {}, "freeAgents": [], "report": {"error": "no rosters"}}
    # IR players hold no roster spot, so the engine never sees them in `roster`. They are
    # still priced and listed per team under `ir`: mega/waiver_value.py puts them back on
    # the roster at their expected return week (HANDOFF v1.3 §3.2). The engine ignores it.
    ir_rows = pd.DataFrame()
    if "slot" in ros.columns:
        on_ir = ros["slot"].astype(str).str.upper() == "IR"
        ir_rows, ros = ros[on_ir], ros[~on_ir]

    # Points per game is forward-looking: a trade is decided on what a player will score
    # from here, not what he already banked. FantasyPros' rest-of-season number is the best
    # answer where it exists, but the free API tier returns only each position's top 10 —
    # about 40 players — so nflverse prices the rest from the weeks actually played.
    #
    # Mixing two projection sources is only safe if they're on the same scale, so it was
    # checked rather than assumed: across the 40 players both cover, the position medians
    # agree to within 3% (QB .98, RB .98, TE 1.01, WR .97). No calibration applied.
    fp = fp_ros(season)
    fp_by_gsis = dict(zip(fp.get("gsis_id", []), fp.get("fp_ros_pg", []))) if not fp.empty else {}
    fp_by_norm = dict(zip(fp.get("norm", []), fp.get("fp_ros_pg", []))) if not fp.empty else {}

    proj = nflverse_estimate(season)
    ppg_by_gsis = dict(zip(proj["gsis_id"], proj["nfl_est"]))
    ppg_by_norm = dict(zip(proj["norm"], proj["nfl_est"]))
    fc = fantasycalc_values()
    ecr_by_norm = dict(zip(fc["norm"], fc["overall_rank"]))

    # HANDOFF §6.2 — a league-mate values a player on what he can SEE: a consensus ranking
    # and the box scores. Not expected points, not target share. Pricing his side of a
    # trade on our model and calling the result fair is how a deal that is plainly good for
    # us reads as plainly good for him too. `perceived_rank` blends the rest-of-season
    # consensus with how the player's actual points have read, leaning on the box score as
    # the season gives it more to say. It replaces `ecr` outright (§7 says to, so the v1
    # code paths keep working), and falls back to FantasyCalc wherever it cannot be built.
    perceived_by_gsis: dict = {}
    try:
        from . import ffa, market
        from .season import player_week

        perceived = market.perceived(ffa.ecr_ros(season), market.box_rank(player_week(season)))
        perceived_by_gsis = dict(zip(perceived["gsis_id"], perceived["perceived_rank"]))
    except Exception:
        perceived_by_gsis = {}

    # HANDOFF v1.3 Pass 3: the value input is our blended rest-of-season projection
    # (mega/proj_ros.py — ours and FantasyPros ECR in the proportion the 2021–25 backtest
    # earned). `ppg` is per game played; `weekly` is each remaining week's EXPECTED points
    # (0 on a bye, discounted by the chance he plays), which the horizon engines read. This
    # week keeps the Vegas props number (§5: "keep this week's props as the week-current
    # value"). FantasyPros ROS / nflverse remain the fallback for anyone it doesn't cover.
    from . import proj_ros
    proj = (proj_ros.cached(season) or {}).get("players") or {}
    proj_week = int(((proj_ros.cached(season) or {}).get("meta") or {}).get("week") or 0)
    vegas_now = {}
    try:
        vp = DATA / "build" / f"vegas_{season}_wk{proj_week:02d}.csv"
        if proj_week and vp.is_file():
            v = pd.read_csv(vp)
            vegas_now = dict(zip(v["gsis_id"], pd.to_numeric(v["vegas"], errors="coerce")))
    except Exception:
        vegas_now = {}

    players: dict[str, dict] = {}
    unpriced: list[str] = []
    sources = {"proj_ros": 0, "fantasypros_ros": 0, "nflverse": 0, "none": 0}

    def add(row) -> str | None:
        pos = _s(row.get("pos")).upper()
        if not pos:
            return None
        pid = _pid(row)
        if pid in players:
            return pid
        gid, nrm = row.get("gsis_id"), row.get("norm")
        weekly, avail = None, None
        pr = proj.get(gid) if isinstance(gid, str) else None
        if pr and pos in ("QB", "RB", "WR", "TE"):
            ppg, src = pr["ros_pg"], "proj_ros"
            if pr.get("backup_of"):
                # a backup QB is on the field only when the starter isn't: his number per
                # game PLAYED is not what a roster spot on him earns (proj_ros._backup_qbs)
                ppg = ppg * float(pr.get("start_p") or 0.0)
            weekly = {int(w): float(v["expected"]) for w, v in (pr.get("weeks") or {}).items()}
            # share of his remaining (non-bye) games he's projected to play — what an
            # injured-reserve player is worth relative to a healthy one (trade_engine.ir_value)
            left = [v for w, v in (pr.get("weeks") or {}).items()
                    if int(w) >= proj_week and not v.get("bye")]
            avail = (sum(float(v.get("p_active", 1.0)) for v in left) / len(left)) if left else None
            vn = vegas_now.get(gid)
            wk_now = (pr.get("weeks") or {}).get(str(proj_week)) or {}
            if vn is not None and pd.notna(vn) and proj_week and not wk_now.get("bye"):
                weekly[proj_week] = round(float(vn) * float(wk_now.get("p_active", 1.0)), 2)
        else:
            ppg, src = fp_by_gsis.get(gid), "fantasypros_ros"
        if ppg is None or pd.isna(ppg):
            ppg = fp_by_norm.get(nrm)
        if ppg is None or pd.isna(ppg):
            ppg, src = ppg_by_gsis.get(gid), "nflverse"
            if ppg is None or pd.isna(ppg):
                ppg = ppg_by_norm.get(nrm)
        if ppg is None or pd.isna(ppg):
            src = "none"
            if pos in ("QB", "RB", "WR", "TE"):
                unpriced.append(_s(row.get("player")))
        if pos in ("QB", "RB", "WR", "TE"):
            sources[src] = sources.get(src, 0) + 1
        ecr = perceived_by_gsis.get(gid)
        if ecr is None or pd.isna(ecr):
            ecr = ecr_by_norm.get(nrm)
        players[pid] = {
            "id": pid, "name": _s(row.get("player")), "pos": pos,
            "nfl": _s(row.get("nfl_team")) or None,
            "ppg": 0.0 if ppg is None or pd.isna(ppg) else round(float(ppg), 3),
            "ecr": None if ecr is None or pd.isna(ecr) else int(ecr),
            "ppg_src": src,
        }
        if weekly:
            players[pid]["weekly"] = weekly
        if weekly is not None and avail is not None:
            players[pid]["avail"] = round(avail, 3)
        return pid

    teams, seat_of = [], {}
    for team, g in ros.groupby("team", sort=False):
        roster = [pid for pid in (add(r) for _, r in g.iterrows()) if pid]
        seat = pd.to_numeric(g["seat"], errors="coerce").dropna()
        tid = int(seat.iloc[0]) if not seat.empty else len(teams) + 100
        seat_of[str(team)] = tid
        ir = ir_rows[ir_rows["team"] == team] if not ir_rows.empty else ir_rows
        ir_ids = [pid for pid in (add(r) for _, r in ir.iterrows()) if pid]
        teams.append({"id": tid, "name": str(team), "roster": roster, "ir": ir_ids})

    free_agents: list[str] = []
    fa_path = DATA / "yahoo_free_agents.csv"
    if fa_path.is_file():
        from .ids import resolve

        fa, _ = resolve(pd.read_csv(fa_path, dtype=str).fillna(""), name_col="player")
        for _, r in fa.iterrows():
            pid = add(r)
            if pid and pid not in {p for t in teams for p in t["roster"] + t["ir"]}:
                free_agents.append(pid)

    return {
        "teams": teams, "players": players, "freeAgents": free_agents,
        "report": {
            "teams": len(teams), "rostered": sum(len(t["roster"]) for t in teams),
            "free_agents": len(free_agents), "unpriced": unpriced,
            "ppg_sources": sources, "fp_coverage": fp.attrs.get("fp_count") if not fp.empty else {},
            "my_team_id": seat_of.get(MY_TEAM),
        },
    }


# ────────────────────────────────────────────────────────── search, both directions
def _rows(engine, results: list[dict]) -> "pd.DataFrame":
    """Engine results as a table the UI can render."""
    names = lambda lst: " + ".join(p["name"] for p in lst)
    return pd.DataFrame([{
        "partner": r["partner"]["name"], "shape": r["shape"],
        "give": names(r["give"]), "get": names(r["get"]),
        "d_me": r["dMe"], "d_them": r["dThem"],
        "mkt_ratio": r["market"]["ratio"], "flag": r["flag"].replace("_", " ").title(),
    } for r in results])


def find_from_my_player(engine, my_id: int, give_ids: list[str], **opts) -> dict:
    """What comes back for one (or two) of mine — the engine's own search."""
    out = engine.find_trades(my_id, give_ids, opts or None)
    return {"results": out["results"], "table": _rows(engine, out["results"]),
            "evaluated": out["evaluated"], "padded": out["padded"], "matched": out["matched"]}


def drop_padding(results: list, pad: float) -> tuple[list, int]:
    """A pair is padding when either of its players alone, for the same return, already does
    as well for both sides. The engine checks that only against the player it searched from,
    and a 2-for-1 restated by net_free_swap changes its numbers — so a search that merges many
    starting points, or nets, runs this over the whole list."""
    singles = {(r["giveIds"][0], frozenset(r["getIds"])): r for r in results if len(r["giveIds"]) == 1}
    kept = [r for r in results if not (len(r["giveIds"]) == 2 and any(
        (s := singles.get((g, frozenset(r["getIds"])))) and s["dMe"] >= r["dMe"] - pad
        and s["dThem"] >= r["dThem"] - pad for g in r["giveIds"]))]
    return kept, len(results) - len(kept)


def find_with_team(engine, my_id: int, partner_id: int, top_n: int = 40,
                   include_flags: tuple[str, ...] = ("LIKELY", "EXPLOIT", "NEEDS_PITCH"),
                   shapes: tuple[str, ...] = ("1-for-1", "2-for-1")) -> dict:
    """Every offer to one named team: the engine's own search, once per player of mine with
    `partners` pinned, merged (a pair found from either of its players is one offer) and ranked
    like the single-player search."""
    ctx = engine.ctx
    empty = {"results": [], "evaluated": 0, "padded": 0, "matched": 0}
    if partner_id == my_id or partner_id not in ctx.teams:
        return empty
    mine = [i for i in [*ctx.teams[my_id]["roster"], *ctx.ir.get(my_id, [])]
            if ctx.players[i]["pos"] in ctx.valued_pos]
    opts = {"partners": [partner_id], "shapes": list(shapes), "includeFlags": list(include_flags),
            "topN": 10**6}
    seen, evaluated, padded = {}, 0, 0
    for pid in mine:
        out = engine.find_trades(my_id, [pid], opts)
        evaluated += out["evaluated"]
        padded += out["padded"]
        for r in out["results"]:
            seen.setdefault((frozenset(r["giveIds"]), frozenset(r["getIds"])), r)
    kept, n = drop_padding(list(seen.values()), ctx.cfg["padTolerance"])
    padded += n
    order = {"LIKELY": 0, "EXPLOIT": 1, "NEEDS_PITCH": 2, "LONGSHOT": 3}
    results = sorted(kept, key=lambda r: (-r["dMe"], order.get(r["flag"], 9)))
    return {"results": results[:top_n], "evaluated": evaluated, "padded": padded,
            "matched": len(results)}


def targets_of(results: list) -> list[dict]:
    """Their players, best change to my lineup first. Run on offers already restated by
    net_free_swap — before that, any trade that opens a roster spot is credited with the free
    agent who fills it, and every player it touches looks equally good."""
    targets: dict = {}
    for r in results:
        for p in r["get"]:
            t = targets.setdefault(p["id"], {"name": p["name"], "pos": p.get("pos"),
                                             "best_d_me": r["dMe"], "offers": 0})
            t["best_d_me"] = max(t["best_d_me"], r["dMe"])
            t["offers"] += 1
    return sorted(targets.values(), key=lambda t: -t["best_d_me"])


def net_free_swap(engine, my_id: int, results: list, min_delta_me: float = 0.01) -> tuple[list, list]:
    """Re-state 2-for-1s against the best add/drop available WITHOUT trading — the same
    netting the offer cards do (mega/trade_theses.py, handoff §6.5).

    A 2-for-1 opens a roster spot the engine fills with the best free agent, and credits
    the trade with him. That pickup was always available, so the trade only earns what it
    adds beyond it. Rows with nothing left afterwards are dropped. Returns the rows and
    the free-swap roster they are now measured against."""
    from . import trade_engine as te
    from .trade_theses import free_swap

    ctx = engine.ctx
    swap_ids, swap_val = free_swap(ctx, ctx.base[my_id].ids)
    swap_val_ir = None
    swap = None
    out = []
    for r in results:
        core = te.evaluate_core(ctx, my_id, r["partner"]["id"], r["giveIds"], r["getIds"])
        a_me = core["_after"]["me"]
        if r["shape"] == "2-for-1" and a_me.added:
            ref = swap_val
            if "_after_ir" in core:          # an IR trade is valued counting IR players
                if swap_val_ir is None:
                    swap_val_ir = te.ir_value(list(swap_ids), ctx.ir.get(my_id, []), ctx)
                ref = swap_val_ir
            d = a_me.value - ref
            if d < min_delta_me:
                continue
            swap = swap or te.settle(list(swap_ids), ctx, set())
            r = {**r, "dMe": d, "me": te._side(ctx, swap, a_me), "netted": True,
                 "fa_add": [ctx.players[i]["name"] for i in a_me.added]}
        out.append(r)
    return out, list(swap_ids)


def find_for_their_player(engine, my_id: int, target_id: str, top_n: int = 50,
                          include_flags: tuple[str, ...] = ("LIKELY", "EXPLOIT", "NEEDS_PITCH"),
                          min_delta_me: float = 0.01, shapes: tuple[str, ...] = ("1-for-1", "2-for-1")) -> dict:
    """What package of mine gets a named player — the handoff's "target mode".

    The engine searches outward from my roster, so chasing a specific player means walking
    my own side instead: every tradeable player and every pair, each evaluated against him.
    Same valuation, same flags, roughly ninety evaluations rather than three thousand.
    """
    ctx = engine.ctx
    owner = next((t for t in ctx.teams.values()
                  if target_id in t["roster"] or target_id in ctx.ir.get(t["id"], [])), None)
    if owner is None or owner["id"] == my_id:
        return {"results": [], "table": pd.DataFrame(), "evaluated": 0, "padded": 0, "matched": 0}
    mine = [i for i in ctx.teams[my_id]["roster"] if ctx.players[i]["pos"] in ctx.valued_pos]
    from itertools import combinations

    packages = [[i] for i in mine]
    if any(s.startswith("2-for") for s in shapes):
        packages += [list(c) for c in combinations(mine, 2)]

    results, evaluated = [], 0
    for give in packages:
        if f"{len(give)}-for-1" not in shapes:
            continue
        evaluated += 1
        r = engine.evaluate_trade(my_id, owner["id"], give, [target_id])
        if r["dMe"] < min_delta_me or r["flag"] not in include_flags:
            continue
        results.append(r)
    # A two-piece package is padding when one of the pieces alone already does as well for
    # both sides — same rule the engine applies in its own search.
    singles = {r["giveIds"][0]: r for r in results if len(r["giveIds"]) == 1}
    pad = ctx.cfg["padTolerance"]
    kept, padded = [], 0
    for r in results:
        if len(r["giveIds"]) == 2 and any(
            (s := singles.get(g)) and s["dMe"] >= r["dMe"] - pad and s["dThem"] >= r["dThem"] - pad
            for g in r["giveIds"]
        ):
            padded += 1
            continue
        kept.append(r)
    order = {"LIKELY": 0, "EXPLOIT": 1, "NEEDS_PITCH": 2, "LONGSHOT": 3}
    kept.sort(key=lambda r: (-r["dMe"], order.get(r["flag"], 9)))
    kept = kept[:top_n]
    return {"results": kept, "table": _rows(engine, kept), "owner": owner["name"],
            "evaluated": evaluated, "padded": padded, "matched": len(kept)}
