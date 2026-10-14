"""The weekly scorecard: the site's own calls, graded against what happened.

Every publish commits data/web and data/build, so git already holds each call exactly as it
was made. A week is graded from the last commit before its first Sunday kickoff (1pm ET as a
rule) — the lineup you would have set from — and nothing published after kickoff can leak in.

  * **Lineup calls**: the published Start/Sit starters (QB/RB/WR/TE) against the best lineup
    that roster could have fielded in hindsight, and every close call ("≈ X") — did the
    player we started outscore the one we sat?
  * **Projections**: our weekly expected points (data/proj_ros_<season>.json, the number the
    waiver and trade pages price with) against the ffanalytics consensus and Vegas props,
    on the same fantasy-relevant players (consensus >= MIN_PROJ), mean absolute error.
  * **Waiver starts**: every pickup the board said would START for us, with the player it said
    to drop — did the add outscore the drop from that week on? Insurance and bye-cover picks
    aren't graded on points; their payoff is the week something goes wrong.

Kickers and defenses are left out: nflverse player stats don't score them.
"""
from __future__ import annotations

import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import FLEX_ELIGIBLE, LINEUP

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
SKILL = ("QB", "RB", "WR", "TE")
MIN_PROJ = 5.0            # fantasy-relevant: the consensus projects him for 5+ points


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def snapshot(path: str, before: datetime) -> str | None:
    """A tracked file as it stood at `before`, or None if it didn't exist yet."""
    sha = _git("log", "-1", f"--before={before.isoformat()}", "--format=%H", "--", path).strip()
    return _git("show", f"{sha}:{path}") if sha else None


def schedule(season: int) -> dict[int, dict]:
    """Week -> {"kick": its first Sunday kickoff (Eastern), "teams": every team with a game}."""
    import nflreadpy as nfl
    s = nfl.load_schedules(seasons=[season]).to_pandas()
    s = s[s["game_type"] == "REG"]
    out = {}
    for w, g in s.groupby("week"):
        sun = g[g["weekday"] == "Sunday"]
        if sun.empty:
            continue
        t = min(datetime.strptime(f"{d} {h}", "%Y-%m-%d %H:%M") for d, h in zip(sun["gameday"], sun["gametime"]))
        out[int(w)] = {"kick": t.replace(tzinfo=ET), "teams": set(g["home_team"]) | set(g["away_team"])}
    return out


def actuals(season: int) -> pd.DataFrame:
    from .intel import weekly
    w = weekly(season)
    w = w[w["season_type"].astype(str).str.upper().eq("REG") & w["pos"].isin(SKILL)]
    return w[["gsis_id", "player", "norm", "pos", "team", "week", "half_ppr"]]


def _pts(act: pd.DataFrame, week: int) -> tuple[dict, dict]:
    a = act[act["week"] == week]
    return dict(zip(a["gsis_id"], a["half_ppr"])), dict(zip(a["norm"], a["half_ppr"]))


def best_lineup(rows: list[dict]) -> list[dict]:
    """The legal lineup with the most points, by each row's `pts` (greedy is exact here)."""
    left = sorted(rows, key=lambda r: -r["pts"])
    out = []
    for pos in SKILL:
        take = [r for r in left if r["pos"] == pos][:LINEUP[pos]]
        out += take
        left = [r for r in left if r not in take]
    out += [r for r in left if r["pos"] in FLEX_ELIGIBLE][:LINEUP.get("W/R", 0)]
    return out


def grade_lineup(ss: dict, by_norm: dict, playing: set | None = None, ours_exp: dict | None = None) -> dict | None:
    """`playing` is the teams with a game that week: a close call against a player on bye
    wasn't a call at all (it was the bye bug week 5 found), so it's set aside, not scored.

    `ours_exp` (name -> our expected points that week) prices the shadow lineups: the one our
    projections would have set and the one the consensus would have, each scored on what
    happened. Since week 6 the published lineup IS ours and each row carries `consensus`;
    before that the published number was the consensus."""
    from .intel import _norm
    rows = [{"player": r["player"], "pos": r["pos"], "slot": r["lineup"], "proj": r.get("proj_adj"),
             "consensus": r.get("consensus"), "source": r.get("proj_source"),
             "bye": bool(playing) and r.get("nfl_team") not in playing,
             "close": r.get("close_call") or "", "pts": float(by_norm.get(_norm(r["player"]), 0.0))}
            for r in (ss.get("starters") or []) + (ss.get("bench") or []) if r.get("pos") in SKILL]
    if not rows:
        return None
    ours = [r for r in rows if r["slot"] != "BENCH"]
    best = best_lineup(rows)
    pts = {r["player"]: r["pts"] for r in rows}
    calls, byes = [], []
    for r in rows:
        if r["slot"] == "BENCH" and r["close"].startswith("≈ ") and r["bye"]:
            byes.append(r["player"])
        elif r["slot"] == "BENCH" and r["close"].startswith("≈ "):
            over = r["close"][2:].strip()
            if over in pts:
                calls.append({"started": over, "sat": r["player"], "started_pts": round(pts[over], 1),
                              "sat_pts": round(r["pts"], 1), "right": pts[over] >= r["pts"]})
    def shadow(key):
        cand = [{**r, "pts_key": 0.0 if r["bye"] or not (r["proj"] or 0) else key(r)} for r in rows]
        lu = best_lineup([{**r, "pts": r["pts_key"]} for r in cand])
        return round(sum(pts[r["player"]] for r in lu), 1)

    ours_map = ours_exp or {}
    shadows = {
        "ours": shadow(lambda r: float(r["proj"]) if r["source"] == "ours"
                       else float(ours_map.get(r["player"], r["proj"] or 0.0))),
        "consensus": shadow(lambda r: float(r["consensus"] if r["consensus"] is not None else r["proj"] or 0.0)),
    }
    return {"ours": round(sum(r["pts"] for r in ours), 1), "best": round(sum(r["pts"] for r in best), 1),
            "shadow": shadows,
            "left": round(sum(r["pts"] for r in best) - sum(r["pts"] for r in ours), 1),
            "should_have_started": [r["player"] for r in best if r["slot"] == "BENCH"],
            "calls": calls, "bye_calls": byes}


def _ours_by_name(season: int, week: int, before: datetime, ss: dict) -> dict:
    """Our expected points that week for each Start/Sit row, by name, from the projection
    file as it stood at kickoff."""
    from . import ids
    from .proj_ros import week_expected
    txt = snapshot(f"data/proj_ros_{season}.json", before)
    exp = week_expected(season, week, json.loads(txt)) if txt else {}
    rows = pd.DataFrame([{"player": r["player"]} for r in (ss.get("starters") or []) + (ss.get("bench") or [])])
    if not exp or rows.empty:
        return {}
    res, _ = ids.resolve(rows, name_col="player")
    return {n: exp[g] for n, g in zip(res["player"], res["gsis_id"]) if isinstance(g, str) and g in exp}


def _consensus(text: str) -> pd.DataFrame:
    from . import ffa
    return ffa.from_frame(pd.read_csv(io.StringIO(text)))


def grade_projections(season: int, week: int, before: datetime, by_gsis: dict) -> dict | None:
    ours_txt = snapshot(f"data/proj_ros_{season}.json", before)
    cons_txt = snapshot(f"data/build/proj_{season}_wk{week:02d}.csv", before)
    if not ours_txt or not cons_txt:
        return None
    ours_j = json.loads(ours_txt)
    if int((ours_j.get("meta") or {}).get("week") or 0) != week:
        return None                     # the snapshot was built for another week
    ours = {g: (p.get("weeks") or {}).get(str(week), {}).get("expected") for g, p in ours_j["players"].items()}
    cons = _consensus(cons_txt)
    vegas_txt = snapshot(f"data/build/vegas_{season}_wk{week:02d}.csv", before)
    vegas = dict(pd.read_csv(io.StringIO(vegas_txt))[["gsis_id", "vegas"]].dropna().values) if vegas_txt else {}
    rows = []
    for g, c, pos in zip(cons["gsis_id"], cons["proj"], cons["pos"]):
        if pos in SKILL and c is not None and float(c) >= MIN_PROJ and ours.get(g) is not None:
            rows.append({"gsis_id": g, "pos": pos, "ours": float(ours[g]), "consensus": float(c),
                         "vegas": vegas.get(g), "actual": float(by_gsis.get(g, 0.0))})
    return {"rows": rows} if rows else None


def summarize_projections(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    d = pd.DataFrame(rows)
    err = lambda col, sub: float((sub[col] - sub["actual"]).abs().mean())
    out = {"n": len(d), "mae_ours": round(err("ours", d), 2), "mae_consensus": round(err("consensus", d), 2),
           "ours_closer": round(float(((d["ours"] - d["actual"]).abs() < (d["consensus"] - d["actual"]).abs()).mean()), 3),
           "by_pos": {p: {"n": len(g), "mae_ours": round(err("ours", g), 2), "mae_consensus": round(err("consensus", g), 2)}
                      for p, g in d.groupby("pos")}}
    v = d.dropna(subset=["vegas"])
    if len(v):
        out["vegas"] = {"n": len(v), "mae_vegas": round(err("vegas", v), 2), "mae_ours": round(err("ours", v), 2),
                        "mae_consensus": round(err("consensus", v), 2)}
    return out


def grade_waivers(snaps: dict, act: pd.DataFrame, through: int) -> list[dict]:
    """Each START pickup the first week the board called it, against the drop it named."""
    from .intel import _norm
    seen, out = set(), []
    for week, wv in sorted(snaps.items()):
        for lane in (wv.get("lanes") or {}).values():
            for r in lane or []:
                add, drop = r.get("player"), r.get("drop")
                if r.get("mechanism") != "START" or not add or not drop or (add, drop) in seen:
                    continue
                seen.add((add, drop))
                span = act[(act["week"] >= week) & (act["week"] <= through)]
                tot = lambda n: float(span.loc[span["norm"] == _norm(n), "half_ppr"].sum())
                out.append({"week": week, "add": add, "drop": drop, "add_pts": round(tot(add), 1),
                            "drop_pts": round(tot(drop), 1), "weeks": through - week + 1,
                            "right": tot(add) > tot(drop)})
    return out


def build(season: int, now_week: int) -> dict:
    """Every finished week with a pre-kickoff snapshot, graded."""
    act = actuals(season)
    played = set(act["week"].unique())
    sched = schedule(season)
    weeks, proj_rows, wv_snaps = [], [], {}
    for w in sorted(sched):
        if w >= now_week or w not in played:
            continue
        before = sched[w]["kick"]
        by_gsis, by_norm = _pts(act, w)
        ss_txt = snapshot("data/web/start_sit.json", before)
        ss = json.loads(ss_txt) if ss_txt else {}
        lineup = (grade_lineup(ss, by_norm, sched[w]["teams"], _ours_by_name(season, w, before, ss))
                  if int(ss.get("next_week") or 0) == w else None)
        proj = grade_projections(season, w, before, by_gsis)
        wv_txt = snapshot("data/web/waivers.json", before)
        if wv_txt:
            wv_snaps[w] = json.loads(wv_txt)
        if lineup is None and proj is None:
            continue
        if proj:
            proj_rows += [{**r, "week": w} for r in proj["rows"]]
        weeks.append({"week": w, "snapshot": before.strftime("%a %b %-d, %-I:%M %p ET"), "lineup": lineup,
                      "projections": summarize_projections(proj["rows"]) if proj else None})
    graded = [x["lineup"] for x in weeks if x["lineup"]]
    calls = [c for g in graded for c in g["calls"]]
    through = max((x["week"] for x in weeks), default=0)
    waivers = grade_waivers(wv_snaps, act, through) if through else []
    return {
        "available": bool(weeks), "season": season, "through_week": through,
        "weeks": sorted(weeks, key=lambda x: -x["week"]),
        "totals": {
            "lineup_weeks": len(graded),
            "points_left": round(sum(g["left"] for g in graded), 1),
            "perfect_weeks": sum(1 for g in graded if g["left"] <= 0.05),
            "shadow": {k: round(sum(g["shadow"][k] for g in graded), 1) for k in ("ours", "consensus")},
            "calls": len(calls), "calls_right": sum(c["right"] for c in calls),
            "projections": summarize_projections(proj_rows),
            "waivers": len(waivers), "waivers_right": sum(w["right"] for w in waivers),
        },
        "waivers": waivers,
    }
