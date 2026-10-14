"""The league's past seasons for the site: one standings table and one set of final rosters
per completed year, the champions, the single-season records, and an all-time table by
manager.

Reads data/history/seasons/<year>.json, which tools/pull_history.py writes one season at a
time (Yahoo rate-limits a bulk pull). Whatever is on disk is exported, so the page grows as
seasons are added and says which are still missing.

Teams are renamed between years and the pulled standings carry no manager, so a name is tied
to a draft seat only when it is EXACTLY a name config knows (current or former) — never by
fuzzy matching, which would quietly credit one manager with another's seasons. All-time rows
are one per seat where known and one per name otherwise, and the page says renamed teams
can appear twice.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import FORMER_NAMES, MY_TEAM, SEAT_BY_TEAM, TEAM_BY_SEAT

SEASONS_DIR = Path(__file__).resolve().parents[1] / "data" / "history" / "seasons"


def _norm(name: str) -> str:
    """Spacing, case and apostrophe style only — "Bills Mafia" and "BillsMafia" are one team,
    "Steve" and "Steven" are not."""
    return "".join(ch for ch in (name or "").replace("’", "'").casefold() if not ch.isspace())


def _seat(team: str) -> int | None:
    """Draft seat for a team name, exact up to spacing, case and apostrophe style."""
    flat = _norm(team)
    for known in (SEAT_BY_TEAM, FORMER_NAMES):
        for name, seat in known.items():
            if _norm(name) == flat:
                return seat
    return None


def _who(year: int, team: str) -> tuple[object, str, int | None]:
    """(key, name today, seat) for a season's team. The manager behind it (pulled per season,
    mega/league_memory.py) is exact across renames; without that file it falls back to the
    name-only match, which can't follow a rename it hasn't been told about."""
    from .league_memory import _name
    by_team, now = _owners()
    k = by_team.get((int(year), _name(team)))
    if k is not None:
        cur = now.get(k)
        return (SEAT_BY_TEAM.get(cur, k) if cur else k), (cur or team), SEAT_BY_TEAM.get(cur) if cur else None
    seat = _seat(team)
    return (seat if seat is not None else team), (TEAM_BY_SEAT.get(seat, team) if seat is not None else team), seat


def _owners():
    from .league_memory import owners
    return owners()


def _load(directory: Path = SEASONS_DIR) -> list[dict]:
    out = []
    for f in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return sorted(out, key=lambda s: s["year"], reverse=True)


def _record(r: dict) -> str:
    t = int(r.get("ties") or 0)
    return f"{int(r['wins'])}-{int(r['losses'])}" + (f"-{t}" if t else "")


def _win_pct(r: dict) -> float:
    g = int(r["wins"]) + int(r["losses"]) + int(r.get("ties") or 0)
    return (int(r["wins"]) + 0.5 * int(r.get("ties") or 0)) / g if g else 0.0


def season_view(s: dict) -> dict:
    rows = []
    for r in s["standings"]:
        rows.append({
            "rank": int(r["rank"]), "team": r["team"], "record": _record(r),
            "wins": int(r["wins"]), "losses": int(r["losses"]), "ties": int(r.get("ties") or 0),
            "pf": float(r["points_for"]), "pa": float(r["points_against"]),
            "made_playoffs": bool(r.get("made_playoffs")), "seat": _who(s["year"], r["team"])[2],
            "mine": r["team"] == MY_TEAM,
        })
    rosters = {team: [{"slot": p.get("slot"), "player": p["player"], "pos": p.get("pos"),
                       "nfl_team": p.get("nfl_team")} for p in ps]
               for team, ps in (s.get("rosters") or {}).items()}
    return {"year": s["year"], "league_name": s.get("league_name"), "champion": s.get("champion"),
            "champion_seat": _who(s["year"], s.get("champion") or "")[2], "standings": rows, "rosters": rosters}


def records(seasons: list[dict]) -> list[dict]:
    """The single-season extremes, with the team and year they belong to."""
    rows = [{**r, "year": s["year"]} for s in seasons for r in s["standings"]]
    if not rows:
        return []

    def top(label, key, high=True, fmt=lambda r: ""):
        r = (max if high else min)(rows, key=key)
        return {"label": label, "team": r["team"], "year": r["year"], "value": fmt(r)}

    pf = lambda r: f"{float(r['points_for']):,.1f}"
    return [
        top("Most points in a season", lambda r: float(r["points_for"]), True, pf),
        top("Fewest points in a season", lambda r: float(r["points_for"]), False, pf),
        top("Best record", lambda r: (_win_pct(r), float(r["points_for"])), True, _record),
        top("Worst record", lambda r: (_win_pct(r), -float(r["points_for"])), False, _record),
        top("Most points allowed", lambda r: float(r["points_against"]), True,
            lambda r: f"{float(r['points_against']):,.1f}"),
    ]


def all_time(seasons: list[dict]) -> dict:
    by_key: dict[object, dict] = {}
    for s in seasons:
        champ_key = _who(s["year"], s.get("champion") or "")[0]
        for r in s["standings"]:
            key, name, seat = _who(s["year"], r["team"])
            m = by_key.setdefault(key, {"seat": seat, "team": name, "seasons": 0, "titles": 0, "playoffs": 0,
                                        "wins": 0, "losses": 0, "ties": 0, "pf": 0.0, "years_won": []})
            m["seasons"] += 1
            m["wins"] += int(r["wins"])
            m["losses"] += int(r["losses"])
            m["ties"] += int(r.get("ties") or 0)
            m["pf"] += float(r["points_for"])
            m["playoffs"] += 1 if r.get("made_playoffs") else 0
            if key == champ_key:
                m["titles"] += 1
                m["years_won"].append(s["year"])
    rows = []
    for m in by_key.values():
        g = m["wins"] + m["losses"] + m["ties"]
        rows.append({**m, "pf": round(m["pf"], 1), "win_pct": round((m["wins"] + 0.5 * m["ties"]) / g, 3) if g else 0.0,
                     "record": f"{m['wins']}-{m['losses']}" + (f"-{m['ties']}" if m["ties"] else ""),
                     "years_won": sorted(m["years_won"]), "mine": m["team"] == MY_TEAM})
    rows.sort(key=lambda r: (-r["titles"], -r["win_pct"], -r["pf"]))
    return {"rows": rows, "named_only": sum(1 for r in rows if r["seat"] is None)}


def build(expected_years: list[int] | None = None, directory: Path = SEASONS_DIR) -> dict:
    seasons = _load(directory)
    if not seasons:
        return {"available": False, "seasons": [], "records": [], "all_time": {"rows": [], "named_only": 0},
                "missing": sorted(expected_years or [], reverse=True)}
    have = {s["year"] for s in seasons}
    return {
        "available": True,
        "seasons": [season_view(s) for s in seasons],
        "records": records(seasons),
        "all_time": all_time(seasons),
        "habits": habits(),
        "missing": sorted(set(expected_years or []) - have, reverse=True),
    }


HABIT_MIN_BIDS = 5


def habits() -> list[dict]:
    """How each manager plays, fitted from the transaction history (mega/league_memory.py):
    what the waiver and trade models now assume about him, shown so it can be checked."""
    from .league_memory import fit
    f = fit()
    lean = f.get("_bias") or {}
    rows = []
    for team, m in f.items():
        if team.startswith("_"):
            continue
        tilt = sorted(((p, v) for p, v in (lean.get(team) or {}).items() if v >= 1.05), key=lambda pv: -pv[1])
        rows.append({"team": team, "manager": m["manager"], "seat": SEAT_BY_TEAM.get(team), "mine": team == MY_TEAM,
                     "seasons": m["seasons"], "trades": m["trades"],
                     "trades_per_season": round(m["trades"] / m["seasons"], 1) if m["seasons"] else None,
                     "trade_offset": m["trade_offset"],
                     # the model uses it shrunk toward 1.0 at any count; shown once there's enough to read
                     "faab_aggression": m["faab_aggression"] if m["contested_bids"] >= HABIT_MIN_BIDS else None,
                     "contested_bids": m["contested_bids"],
                     "lean": [{"pos": p, "mult": round(v, 2)} for p, v in tilt[:2]]})
    rows.sort(key=lambda r: (r["faab_aggression"] is None, -(r["faab_aggression"] or 0)))
    return rows
