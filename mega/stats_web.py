"""The league-wide season stat table behind the site's chart builder: every QB/RB/WR/TE with
a game this season or last, one row per player-season, every number the player cards rank
on — the same `mega.lookup.season_table` frame, so a point on the chart and the stat on
that player's card can't disagree.

`METRICS` is the menu: what each column is called, how to format it and which positions it
means something for. The page builds its dropdowns from this rather than keeping its own
list, so a stat added here shows up there.

Columnar (`columns` + `rows` of arrays) because ~1,200 player-seasons × ~45 numbers as
objects would repeat every key 1,200 times.
"""
from __future__ import annotations

import math

import pandas as pd

ALL = ("QB", "RB", "WR", "TE")
REC = ("RB", "WR", "TE")
WRTE = ("WR", "TE")

# (column, label, kind, positions, what it means) — kind is "pct" (0.25 -> 25%), "num",
# "int" or "signed"
METRICS = [
    ("pts_pg", "Fantasy pts/game", "num", ALL, "Half-PPR points per game."),
    ("xfp_pg", "Expected pts/game", "num", ALL, "What his volume should score (nflverse expected points)."),
    ("vs_exp_pg", "Pts over expected/game", "signed", ALL, "Scoring above (+) or below (−) his opportunity."),
    ("pts", "Fantasy pts (total)", "num", ALL, ""),
    ("games", "Games", "int", ALL, ""),
    ("snap_pct", "Snap share", "pct", ALL, "Share of his offense's plays he was on the field for."),
    ("routes_pg", "Routes/game (est.)", "num", WRTE, "Snap share × team dropbacks — nflverse has no charted routes."),
    ("routes", "Routes (total, est.)", "int", WRTE, ""),
    ("tgt_pg", "Targets/game", "num", REC, ""),
    ("targets", "Targets (total)", "int", REC, ""),
    ("tgt_share", "Target share", "pct", REC, "His share of his team's targets."),
    ("tprr", "Targets per route (est.)", "pct", WRTE, "How often he's thrown to when he runs a route."),
    ("yprr", "Yards per route (est.)", "num", WRTE, ""),
    ("fd_rr", "1st downs per route (est.)", "pct", WRTE, ""),
    ("ay_share", "Air-yards share", "pct", WRTE, "Share of the team's downfield targets."),
    ("wopr", "WOPR", "num", WRTE, "1.5 × target share + 0.7 × air-yards share."),
    ("adot", "aDOT", "num", WRTE, "Average depth of target, in yards."),
    ("catch_rate", "Catch rate", "pct", REC, ""),
    ("rec_pg", "Receptions/game", "num", REC, ""),
    ("receptions", "Receptions (total)", "int", REC, ""),
    ("rec_ypg", "Rec yds/game", "num", REC, ""),
    ("receiving_yards", "Rec yds (total)", "int", REC, ""),
    ("receiving_tds", "Rec TD", "int", REC, ""),
    ("ypr", "Yds/reception", "num", REC, ""),
    ("yac_pr", "YAC/reception", "num", REC, ""),
    ("rec_epa_tgt", "EPA/target", "signed", REC, ""),
    ("rush_share", "Carry share", "pct", ("QB", "RB"), "His share of his team's carries."),
    ("car_pg", "Carries/game", "num", ("QB", "RB"), ""),
    ("carries", "Carries (total)", "int", ("QB", "RB"), ""),
    ("rush_ypg", "Rush yds/game", "num", ("QB", "RB"), ""),
    ("rushing_yards", "Rush yds (total)", "int", ("QB", "RB"), ""),
    ("rushing_tds", "Rush TD", "int", ("QB", "RB"), ""),
    ("ypc", "Yds/carry", "num", ("QB", "RB"), ""),
    ("rush_epa_car", "Rush EPA/carry", "signed", ("QB", "RB"), ""),
    ("att_pg", "Pass att/game", "num", ("QB",), ""),
    ("pass_ypg", "Pass yds/game", "num", ("QB",), ""),
    ("passing_yards", "Pass yds (total)", "int", ("QB",), ""),
    ("passing_tds", "Pass TD", "int", ("QB",), ""),
    ("passing_interceptions", "INT", "int", ("QB",), ""),
    ("cmp_pct", "Completion %", "pct", ("QB",), ""),
    ("cpoe", "CPOE", "signed", ("QB",), "Completion % over expected."),
    ("ypa", "Yds/attempt", "num", ("QB",), ""),
    ("pass_epa_db", "EPA/dropback", "signed", ("QB",), ""),
]

ID_COLS = ["gsis_id", "name", "pos", "team", "season", "owner", "mine"]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 4)


def build(seasons: dict[int, dict], owners: dict[str, dict]) -> dict:
    """`seasons` is player_web's `{season: {"table": season_table_df, ...}}`; `owners` maps
    gsis_id -> player_web._ownership_info(...) ({kind, team, ...})."""
    cols = [m[0] for m in METRICS]
    rows = []
    for season in sorted(seasons, reverse=True):
        t = seasons[season].get("table")
        if t is None or t.empty:
            continue
        t = t[t["pos"].isin(ALL) & (pd.to_numeric(t["games"], errors="coerce") > 0)]
        for _, r in t.iterrows():
            gid = r["gsis_id"]
            o = owners.get(gid) or {}
            kind = o.get("kind")
            owner = o.get("team") if kind in ("mine", "other_team") else ("FA" if kind in ("free_agent", "waivers") else None)
            rows.append([gid, r.get("name"), r.get("pos"), r.get("team"), int(season), owner, kind == "mine"]
                        + [_num(r.get(c)) for c in cols])
    return {
        "available": bool(rows),
        "metrics": [{"key": k, "label": label, "kind": kind, "positions": list(pos), "means": means}
                    for k, label, kind, pos, means in METRICS],
        "columns": ID_COLS + cols,
        "rows": rows,
    }
