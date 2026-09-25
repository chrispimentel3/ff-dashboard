"""Pure-data Trades: the league-wide "offers the league is set up for" list, grouped by manager.

This is deliberately only the static half of the Streamlit Trades tab. "Trade around one
player" — the interactive search that reruns `mega.trade_engine`/`mega.trade_league` per pick,
including a 1,500-season playoff-odds simulation for the top candidates — needs live
computation per query. mega-bowl-web is a statically-generated site with no backend by design
(see the project's grilling-session decisions), so that search tool isn't portable here without
adding a live API, which is a bigger architectural change than this export. This module only
covers what's already a fixed, precomputed list.

Each offer is additionally run through `mega.trade_engine` — the same roster-rebuild-and-
reoptimize valuation the live search uses — so "fair on value" and "actually helps your
lineup" are both answered, not just the first one. `trade_finder`'s ideas are always 1-for-1,
so `evaluate_trade` is called with exactly one player each side.
"""
from __future__ import annotations

import json

import pandas as pd

from .config import MY_TEAM

COLS = ["give", "give_val", "get", "get_val", "addresses", "fairness", "edge"]


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _build_engine(season: int):
    """Same engine, same data source, as app.py's own `_trade_engine()` — see that
    function's docstring for why `cached_rosters()` rather than the live Yahoo API here."""
    from .trade_engine import create_engine
    from .trade_league import build_league, engine_config
    from .yahoo import cached_rosters

    league = build_league(season, cached_rosters())
    if not league["teams"]:
        return None
    engine = create_engine(league, engine_config())
    engine.name_to_pid = {p["name"]: pid for pid, p in league["players"].items()}
    engine.team_id_by_name = {t["name"]: t["id"] for t in league["teams"]}
    return engine


def _impact(engine, their_team: str, give_name: str, get_name: str) -> dict | None:
    """Roster impact of a specific 1-for-1 trade, or None if any side can't be resolved to
    a priced player — a name mismatch shouldn't blank the whole row, just this one field."""
    my_id = engine.team_id_by_name.get(MY_TEAM)
    their_id = engine.team_id_by_name.get(their_team)
    give_pid = engine.name_to_pid.get(give_name)
    get_pid = engine.name_to_pid.get(get_name)
    if my_id is None or their_id is None or give_pid is None or get_pid is None:
        return None
    try:
        r = engine.evaluate_trade(my_id, their_id, [give_pid], [get_pid])
    except Exception:
        return None
    me, them = r["me"], r["them"]
    return {
        "my_lineup_delta": round(r["dMe"], 2),
        "their_lineup_delta": round(r["dThem"], 2),
        "i_would_start": [s["name"] for s in me["startersIn"]],
        "i_would_bench": [s["name"] for s in me["startersOut"]],
        "they_would_start": [s["name"] for s in them["startersIn"]],
        "they_would_bench": [s["name"] for s in them["startersOut"]],
    }


def build(trades: pd.DataFrame | None, roster_src: str | None, season: int | None = None) -> dict:
    """`trades` is IB["trades"] (raw — give_pos/get_pos not yet folded into give/get)."""
    if trades is None or trades.empty:
        return {"available": False, "groups": [], "roster_src": roster_src}

    engine = None
    if season is not None:
        try:
            engine = _build_engine(season)
        except Exception:
            engine = None

    impacts = (
        [
            _impact(engine, row.partner, row.give, row.get)
            for row in trades.itertuples()
        ]
        if engine is not None
        else [None] * len(trades)
    )

    tt = trades.copy()
    tt["give"] = tt["give"] + " (" + tt["give_pos"] + ")"
    tt["get"] = tt["get"] + " (" + tt["get_pos"] + ")"
    tt["addresses"] = tt["addresses"].str.replace(r"^my (\S+) need$", r"\1", regex=True)
    tt["impact"] = impacts

    groups = []
    for partner, grp in tt.groupby("partner", sort=False):
        need = grp["they_need"].iloc[0] if "they_need" in grp.columns else None
        offers = _records(grp.reindex(columns=COLS))
        for offer, impact in zip(offers, grp["impact"]):
            offer["impact"] = impact
        groups.append({
            "partner": partner,
            "they_need": need if need and need != "—" else None,
            "offers": offers,
        })

    return {"available": True, "groups": groups, "roster_src": roster_src,
            "impact_available": engine is not None}
