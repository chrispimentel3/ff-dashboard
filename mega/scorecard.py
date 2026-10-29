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
  * **Trade ideas**: each offer the home board showed, replayed on our roster every week since
    — our best lineup with the deal against without it, on what was scored.
  * **Win odds**: each league game's win chance from the season sim at kickoff against who won;
    the weekly **playoff odds** against who made it, once the regular season is over.

Kickers and defenses are left out: nflverse player stats don't score them.
"""
from __future__ import annotations

import io
import json
import re
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
LIVE_BLEND = ROOT / "data" / "build" / "live_blend.json"
K_LIVE = 150              # (J) player-weeks of "ours only" in every position's live weight
MIN_TESTED = 4            # walk-forward weeks the live blend has to win before Start/Sit uses it
W_GRID = [i / 20 for i in range(21)]


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


def grade_movers(snaps: dict, act: pd.DataFrame) -> list[dict]:
    """Each week's prop movers (the books raised his role) against what he then scored, and
    against his average before that week — did the market's move show up on the field?"""
    out = []
    for week, wv in sorted(snaps.items()):
        for m in wv.get("prop_movers") or []:
            g = act[act["gsis_id"] == m["gsis_id"]]
            now = g.loc[g["week"] == week, "half_ppr"]
            before = g.loc[g["week"] < week, "half_ppr"]
            out.append({"week": week, "player": m["player"], "pos": m["pos"], "kind": m["kind"],
                        "pts": round(float(now.sum()), 1) if len(now) else 0.0,
                        "before": round(float(before.mean()), 1) if len(before) else None})
    return out


def fit_blend(rows: list[dict]) -> dict:
    """pos -> weight on ours (the rest on the consensus) that missed least on these graded
    player-weeks, shrunk toward 1.0 (ours alone, what Start/Sit uses) by K_LIVE."""
    out = {}
    d = pd.DataFrame(rows)
    for pos, g in d.groupby("pos"):
        mae = {w: float((w * g["ours"] + (1 - w) * g["consensus"] - g["actual"]).abs().mean()) for w in W_GRID}
        best = min(mae, key=mae.get)
        out[pos] = round((len(g) * best + K_LIVE * 1.0) / (len(g) + K_LIVE), 3)
    return out


def live_blend(rows: list[dict]) -> dict:
    """Walk forward: fit on the weeks before, grade on the next. The blend switches on for
    Start/Sit only once it has beaten ours-alone over MIN_TESTED tested weeks."""
    if not rows:
        return {"active": False, "weights": {}, "tested": 0}
    d = pd.DataFrame(rows)
    weeks = sorted(d["week"].unique())
    tested, ours_err, blend_err = 0, 0.0, 0.0
    for w in weeks[1:]:
        train, test = d[d["week"] < w], d[d["week"] == w]
        wt = fit_blend(train.to_dict("records"))
        blended = [x * r.ours + (1 - x) * r.consensus for r, x in zip(test.itertuples(), (wt.get(p, 1.0) for p in test["pos"]))]
        blend_err += float((pd.Series(blended, index=test.index) - test["actual"]).abs().sum())
        ours_err += float((test["ours"] - test["actual"]).abs().sum())
        tested += 1
    n = len(d[d["week"] > weeks[0]]) or 1
    gain = (ours_err - blend_err) / n
    return {"weights": fit_blend(rows), "tested": tested, "mae_gain": round(gain, 3),
            "active": tested >= MIN_TESTED and gain > 0}


MATCHUP_EDGE = 5.0      # matchup tags at +5% or better / -5% or worse are the ones the page calls out


def grade_matchups(snaps: dict, act: pd.DataFrame) -> dict:
    """The published matchup % for every ranked player, against what he then scored relative
    to his own average before that week: did good matchups beat it and tough ones fall short?"""
    rows = []
    for week, rk in snaps.items():
        for r in rk.get("rows") or []:
            pct, g = r.get("matchup_pct"), r.get("gsis_id")
            if pct is None or not g:
                continue
            mine = act[act["gsis_id"] == g]
            before = mine.loc[mine["week"] < week, "half_ppr"]
            now = mine.loc[mine["week"] == week, "half_ppr"]
            if len(before) < 2 or not len(now) or before.mean() < MIN_PROJ:
                continue
            rows.append({"bucket": "good" if pct >= MATCHUP_EDGE else "tough" if pct <= -MATCHUP_EDGE else "neutral",
                         "pts": float(now.iloc[0]), "avg": float(before.mean())})
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    return {b: {"n": len(g), "vs_avg_pct": round(float(100 * (g["pts"].sum() / g["avg"].sum() - 1)), 1)}
            for b, g in d.groupby("bucket")}


def grade_trades(snaps: dict, rosters: dict, pts: dict) -> dict:
    """Every trade idea the home board showed (best first, one per manager), from the week it
    first appeared: our best lineup with the deal against our best lineup without it, each week
    since, scored on what happened. Hindsight-best on both sides, so a bench piece only counts in
    a week he'd have started. A week stops counting once a player we'd have given is gone from
    our roster — the deal can't be replayed on a roster that no longer has him.

    `snaps` week -> the home board's trades (action_board.json) at that week's kickoff, read as
    published, so a later change to how the board picks can't rewrite what it showed;
    `rosters` week -> [{player, pos}] from that week's Start/Sit; `pts` week -> normalized
    name -> half-PPR scored."""
    from .intel import _norm
    rows, seen = [], set()
    for w in sorted(snaps):
        for b in snaps[w]:
            give, get = _pieces(b.get("give")), _pieces(b.get("get"))
            if not give or not get:
                continue
            key = (tuple(n for n, _ in give), tuple(n for n, _ in get))
            if key in seen:
                continue
            seen.add(key)
            gone = {_norm(n) for n, _ in give}
            diffs = []
            for x in sorted(rosters):
                if x < w or x not in pts or not gone <= {_norm(r["player"]) for r in rosters[x]}:
                    continue
                p = pts[x]
                now = [{**r, "pts": p.get(_norm(r["player"]), 0.0)} for r in rosters[x]]
                after = [r for r in now if _norm(r["player"]) not in gone] + \
                        [{"player": n, "pos": pos, "pts": p.get(_norm(n), 0.0)} for n, pos in get]
                diffs.append(sum(r["pts"] for r in best_lineup(after)) - sum(r["pts"] for r in best_lineup(now)))
            if not diffs:
                continue
            per = sum(diffs) / len(diffs)
            said = b.get("d_me")
            if said is None:      # boards before d_me was published still state it in the thesis
                m = re.search(r"([+-]\d+(?:\.\d+)?) pts/wk", b.get("thesis") or "")
                said = float(m.group(1)) if m else None
            rows.append({"week": w, "partner": b["partner"], "give": list(key[0]), "get": list(key[1]),
                         "said": said, "weeks": len(diffs), "per_week": round(per, 1),
                         "right": None if abs(per) < 0.05 else per > 0})
    graded = [r for r in rows if r["right"] is not None]
    return {"n": len(rows), "helped": sum(r["right"] for r in graded), "hurt": sum(not r["right"] for r in graded),
            "per_week": round(sum(r["per_week"] for r in rows) / len(rows), 1) if rows else None,
            "rows": sorted(rows, key=lambda r: (-r["week"], -r["per_week"]))}


def _pieces(s: str | None) -> list[tuple[str, str]]:
    """'Malik Nabers (WR) + George Kittle (TE)' -> [(name, pos), ...], as the board prints them."""
    return [(m.group(1), m.group(2)) for part in (s or "").split(" + ")
            if (m := re.match(r"^(.+) \((\w+)\)$", part.strip()))]


def _calibration(d: pd.DataFrame, bins: list[float], labels: list[str]) -> list[dict]:
    """Rows of (p, hit) bucketed by p: what we said against how often it happened."""
    d = d.assign(bucket=pd.cut(d["p"], bins, labels=labels, include_lowest=True))
    return [{"bucket": str(b), "n": len(g), "said": round(float(g["p"].mean()), 3),
             "happened": round(float(g["hit"].mean()), 3)}
            for b, g in d.groupby("bucket", observed=True) if len(g)]


def grade_odds(snaps: dict, results: pd.DataFrame, my_team: str) -> dict:
    """Each league game's win chance as the season sim had it at kickoff (the rooting guide on
    the playoffs page, saved every publish since week 6), against who won. Ties don't count."""
    rows = []
    for w, ro in sorted(snaps.items()):
        games = [(my_team, ro["mine"]["opp"], ro["mine"]["p_win"])] if ro.get("mine") else []
        games += [(g["root_for"], g["against"], g["p_root_wins"]) for g in ro.get("games") or []]
        for a, b, p in games:
            r = results[(results["week"] == w) & (results["team"] == a) & (results["opponent"] == b)]
            if r.empty or float(r["points"].iloc[0]) == float(r["opp_points"].iloc[0]):
                continue
            a_won = float(r["points"].iloc[0]) > float(r["opp_points"].iloc[0])
            fav, dog, pf, won = (a, b, p, a_won) if p >= 0.5 else (b, a, 1 - p, not a_won)
            rows.append({"week": int(w), "favorite": fav, "underdog": dog, "p": round(float(pf), 3), "won": bool(won)})
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    return {"n": len(d), "favorites_won": int(d["won"].sum()), "said": round(float(d["p"].mean()), 3),
            "buckets": _calibration(d.assign(hit=d["won"]), [0.5, 0.6, 0.75, 1.0], ["50–60%", "60–75%", "75%+"]),
            "games": sorted(rows, key=lambda r: (-r["week"], -r["p"]))}


def grade_playoff_odds(hist: pd.DataFrame, standings: pd.DataFrame, results: pd.DataFrame) -> dict:
    """The weekly playoff odds (data/history/playoff_odds_weekly.csv, player-level sim only)
    against who made it — which isn't known until the regular season is over."""
    from .sim import PLAYOFF_TEAMS
    from .title_odds import REG_LAST
    played = int(results["week"].max()) if not results.empty else 0
    if played < REG_LAST or standings.empty or hist.empty:
        return {"available": False, "after_week": REG_LAST}
    made = set(standings.loc[standings["rank"] <= PLAYOFF_TEAMS, "team"])
    h = hist[hist["model"].astype(str).eq("player")]
    d = pd.DataFrame({"p": h["p_playoffs"].astype(float), "hit": h["team"].isin(made)})
    return {"available": True, "n": len(d), "weeks": sorted(int(x) for x in h["week"].unique()),
            "buckets": _calibration(d, [0.0, 0.25, 0.75, 1.0], ["under 25%", "25–75%", "75%+"])}


def build(season: int, now_week: int) -> dict:
    """Every finished week with a pre-kickoff snapshot, graded."""
    act = actuals(season)
    played = set(act["week"].unique())
    sched = schedule(season)
    weeks, proj_rows, wv_snaps, rk_snaps = [], [], {}, {}
    tr_snaps, odds_snaps, rosters, pts_by_week = {}, {}, {}, {}
    for w in sorted(sched):
        if w >= now_week or w not in played:
            continue
        before = sched[w]["kick"]
        by_gsis, by_norm = _pts(act, w)
        pts_by_week[w] = by_norm
        ss_txt = snapshot("data/web/start_sit.json", before)
        ss = json.loads(ss_txt) if ss_txt else {}
        if int(ss.get("next_week") or 0) == w:
            rosters[w] = [{"player": r["player"], "pos": r["pos"]}
                          for r in (ss.get("starters") or []) + (ss.get("bench") or []) if r.get("pos") in SKILL]
        ab_txt = snapshot("data/web/action_board.json", before)
        if ab_txt:
            tr_snaps[w] = json.loads(ab_txt).get("trades") or []
        po_txt = snapshot("data/web/playoffs.json", before)
        ro = (json.loads(po_txt).get("rooting") or {}) if po_txt else {}
        if int(ro.get("week") or 0) == w:
            odds_snaps[w] = ro
        lineup = (grade_lineup(ss, by_norm, sched[w]["teams"], _ours_by_name(season, w, before, ss))
                  if int(ss.get("next_week") or 0) == w else None)
        proj = grade_projections(season, w, before, by_gsis)
        rk_txt = snapshot("data/web/rankings.json", before)
        if rk_txt and int(json.loads(rk_txt).get("next_week") or 0) == w:
            rk_snaps[w] = json.loads(rk_txt)
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
    lb = live_blend(proj_rows)
    try:
        LIVE_BLEND.write_text(json.dumps({"season": season, "through_week": through, **lb}, indent=1))
    except Exception:
        pass
    movers = grade_movers({w: s for w, s in wv_snaps.items() if w <= through}, act) if through else []
    from .config import MY_TEAM
    results, standings, hist = (_csv(ROOT / "data" / f) for f in
                                ("yahoo_scores.csv", "yahoo_standings.csv", "history/playoff_odds_weekly.csv"))
    hist = hist[hist["season"] == season] if not hist.empty else hist
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
        "movers": movers,
        "live_blend": lb,
        "matchups": grade_matchups(rk_snaps, act),
        "trades": grade_trades(tr_snaps, rosters, pts_by_week),
        "odds": grade_odds(odds_snaps, results, MY_TEAM) if not results.empty else {},
        "playoff_odds": grade_playoff_odds(hist, standings, results),
    }


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
