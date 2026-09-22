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

from .config import DATA, FLEX_ELIGIBLE, LINEUP, MY_TEAM

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
    }


def build_league(season: int, yahoo_rosters: pd.DataFrame | None = None) -> dict:
    """{teams, players, freeAgents} plus a `report` of what couldn't be priced."""
    from .intel import current_rosters
    from .projections import nflverse_estimate
    from .sources import fantasycalc_values

    ros = current_rosters(yahoo_rosters)
    if ros.empty:
        return {"teams": [], "players": {}, "freeAgents": [], "report": {"error": "no rosters"}}
    if "slot" in ros.columns:
        ros = ros[ros["slot"].astype(str).str.upper() != "IR"]

    proj = nflverse_estimate(season)
    ppg_by_gsis = dict(zip(proj["gsis_id"], proj["nfl_est"]))
    ppg_by_norm = dict(zip(proj["norm"], proj["nfl_est"]))
    fc = fantasycalc_values()
    ecr_by_norm = dict(zip(fc["norm"], fc["overall_rank"]))

    players: dict[str, dict] = {}
    unpriced: list[str] = []

    def add(row) -> str | None:
        pos = _s(row.get("pos")).upper()
        if not pos:
            return None
        pid = _pid(row)
        if pid in players:
            return pid
        ppg = ppg_by_gsis.get(row.get("gsis_id"))
        if ppg is None or pd.isna(ppg):
            ppg = ppg_by_norm.get(row.get("norm"))
        ecr = ecr_by_norm.get(row.get("norm"))
        if (ppg is None or pd.isna(ppg)) and pos in ("QB", "RB", "WR", "TE"):
            unpriced.append(_s(row.get("player")))
        players[pid] = {
            "id": pid, "name": str(row.get("player")), "pos": pos,
            "nfl": str(row.get("nfl_team") or "") or None,
            "ppg": 0.0 if ppg is None or pd.isna(ppg) else round(float(ppg), 3),
            "ecr": None if ecr is None or pd.isna(ecr) else int(ecr),
        }
        return pid

    teams, seat_of = [], {}
    for team, g in ros.groupby("team", sort=False):
        roster = [pid for pid in (add(r) for _, r in g.iterrows()) if pid]
        seat = pd.to_numeric(g["seat"], errors="coerce").dropna()
        tid = int(seat.iloc[0]) if not seat.empty else len(teams) + 100
        seat_of[str(team)] = tid
        teams.append({"id": tid, "name": str(team), "roster": roster})

    free_agents: list[str] = []
    fa_path = DATA / "yahoo_free_agents.csv"
    if fa_path.is_file():
        from .ids import resolve

        fa, _ = resolve(pd.read_csv(fa_path, dtype=str).fillna(""), name_col="player")
        for _, r in fa.iterrows():
            pid = add(r)
            if pid and pid not in {p for t in teams for p in t["roster"]}:
                free_agents.append(pid)

    return {
        "teams": teams, "players": players, "freeAgents": free_agents,
        "report": {
            "teams": len(teams), "rostered": sum(len(t["roster"]) for t in teams),
            "free_agents": len(free_agents), "unpriced": unpriced,
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


def find_for_their_player(engine, my_id: int, target_id: str, top_n: int = 50,
                          include_flags: tuple[str, ...] = ("LIKELY", "EXPLOIT", "NEEDS_PITCH"),
                          min_delta_me: float = 0.01, shapes: tuple[str, ...] = ("1-for-1", "2-for-1")) -> dict:
    """What package of mine gets a named player — the handoff's "target mode".

    The engine searches outward from my roster, so chasing a specific player means walking
    my own side instead: every tradeable player and every pair, each evaluated against him.
    Same valuation, same flags, roughly ninety evaluations rather than three thousand.
    """
    ctx = engine.ctx
    owner = next((t for t in ctx.teams.values() if target_id in t["roster"]), None)
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
