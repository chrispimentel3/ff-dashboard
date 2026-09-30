"""Rebuild the website's data (data/web/) and check it before it is committed.

mega-bowl-web reads the JSON in data/web/ from this repo (refreshed hourly), so the site
only moves when these files do: new game scores and standings, the matchup week turning
over after Monday night, waiver results. tools/export_web.py builds them; this wraps it
for the scheduled refresh tasks with the checks a human would do by eye, and first takes
the week's trend snapshot (data/history/) if it hasn't been taken yet.

    PYTHONPATH=. .venv/bin/python tools/publish_web.py

Exit codes (the scheduled tasks read these):
  0  rebuilt and checked — stage the paths printed on the STAGE line
  3  a check failed — nothing should be committed from data/web/ this run
  4  the export itself crashed
Free to run: it reads nflverse, DynastyProcess and the local Yahoo CSVs; it never calls
the paid odds API (that sweep lives in tools/refresh.py, on its own budget).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# what the export writes and the site (or the Streamlit app) reads, beyond data/web/
EXTRA = ["data/build/", "data/yahoo_fixtures.csv", "data/history/"]


def main() -> int:
    from mega import data

    # Weekly trend snapshots first, so the export's trend charts include this week. Only a
    # week not yet captured is taken (mega.history.snapshot_new) — safe to run every day.
    season0 = data.SEASON_DEFAULT
    try:
        import nflreadpy as nfl
        season0 = int(nfl.get_current_season())
    except Exception:
        pass
    try:
        from mega import history
        wk = data.current_week(season0, 0)
        if wk:
            took = history.snapshot_new(season0, wk)
            print(f"[web] trend snapshot week {wk}: " + (", ".join(took) if took else "already taken"))
    except Exception as e:
        print(f"[web] trend snapshot skipped: {type(e).__name__}: {e}")

    run = subprocess.run([sys.executable, str(ROOT / "tools" / "export_web.py")], cwd=ROOT,
                         capture_output=True, text=True)
    out = run.stdout + run.stderr
    if run.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-15:])
        print(f"[web] FAILED: export_web.py exited {run.returncode}\n{tail}")
        return 4

    season = season0
    problems = []
    if f"wrote data/proj_ros_{season}.json" not in out:
        problems.append(f"the projection (data/proj_ros_{season}.json) was not rebuilt")
    bad = []
    for f in (ROOT / "data" / "web").rglob("*.json"):
        try:
            json.loads(f.read_text())
        except Exception:
            bad.append(str(f.relative_to(ROOT)))
    if bad:
        problems.append(f"{len(bad)} file(s) are not valid JSON: {', '.join(bad[:5])}")
    try:
        ab = json.loads((ROOT / "data" / "web" / "action_board.json").read_text())
        want = data.current_week(season, 0)
        if want and int(ab.get("next_week") or 0) != want:
            problems.append(f"the site would show matchup week {ab.get('next_week')} "
                            f"but the NFL schedule says week {want}")
        print(f"[web] season {ab.get('season')} · through week {ab.get('week')} · "
              f"matchup week {ab.get('next_week')}")
    except Exception as e:
        problems.append(f"action_board.json unreadable: {e}")

    if problems:
        for p in problems:
            print(f"[web] CHECK FAILED: {p}")
        return 3
    print("STAGE data/web/ data/proj_ros_%d.json %s" % (season, " ".join(EXTRA)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
