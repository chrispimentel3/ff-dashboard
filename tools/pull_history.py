"""Pull ONE completed season of Mega Bowl history from Yahoo and save it to
data/history/seasons/<year>.json. Resumable: a season already on disk is skipped, so run it
once per season (or `--next` for the newest one still missing) with a rest between runs.

    PYTHONPATH=. .venv/bin/python tools/pull_history.py 2025
    PYTHONPATH=. .venv/bin/python tools/pull_history.py --next

Why one at a time: pulling all six seasons back-to-back made Yahoo answer "Request denied"
to every page, including the live league. A season is ~14 page loads at REQUEST_DELAY apart;
the rest between seasons is the caller's job. On a rate-limit or an expired session this
stops at once and writes nothing for that season — it never retries and never logs in.

Exit codes: 0 saved (or already had it) · 5 rate-limited · 6 session expired · 7 nothing found
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mega import yahoo_history as yh  # noqa: E402
from mega.yahoo import AuthExpired, LeagueSession  # noqa: E402

OUT = ROOT / "data" / "history" / "seasons"
DELAY = 8.0     # seconds between page loads; the module's own 4s is what got us blocked
RETRIES = 3     # for a dropped connection only; "Request denied" and an expired session never retry
RETRY_WAIT = 30.0


def _patient(fn):
    """Retry a page load that failed on OUR network (timeout, network changed), waiting between
    tries. A rate-limit or auth error is not a network error and passes straight through."""
    import time

    def wrapped(*a, **k):
        for attempt in range(1, RETRIES + 1):
            try:
                return fn(*a, **k)
            except (yh.RateLimited, AuthExpired):
                raise
            except Exception as e:
                if "net::ERR_" not in str(e) or attempt == RETRIES:
                    raise
                print(f"[history] network hiccup ({str(e).splitlines()[0][-60:]}); retry {attempt}/{RETRIES - 1} in {RETRY_WAIT:.0f}s")
                time.sleep(RETRY_WAIT)
    return wrapped


def path(year: int) -> Path:
    return OUT / f"{year}.json"


def _records(df) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def main(argv: list[str]) -> int:
    missing = [y for y in sorted(yh.SEASON_LEAGUE_IDS, reverse=True) if not path(y).exists()]
    if argv and argv[0] == "--next":
        if not missing:
            print("[history] every season is already saved")
            return 0
        year = missing[0]
    elif argv:
        year = int(argv[0])
        if year not in yh.SEASON_LEAGUE_IDS:
            print(f"[history] no league id for {year}")
            return 7
        if path(year).exists():
            print(f"[history] {year} already saved ({path(year).name})")
            return 0
    else:
        print(__doc__)
        return 7

    yh.REQUEST_DELAY = DELAY
    yh.roster, yh.standings = _patient(yh.roster), _patient(yh.standings)
    print(f"[history] pulling {year} (league {yh.SEASON_LEAGUE_IDS[year]}), {DELAY:.0f}s between pages")
    try:
        with LeagueSession() as s:
            res = yh.pull_season(s, year)
    except yh.RateLimited as e:
        print(f"[history] RATE LIMITED on {year}: {e}\n[history] nothing saved; wait before the next run")
        return 5
    except AuthExpired as e:
        print(f"[history] session expired: {e}")
        return 6

    if res["standings"].empty or not res["rosters"]:
        print(f"[history] {year}: standings {len(res['standings'])} rows, {len(res['rosters'])} rosters — not saving a partial season")
        return 7
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "year": year, "league_id": res["league_id"], "league_name": res["league_name"],
        "champion": res["champion"],
        "standings": _records(res["standings"]),
        "rosters": {team: _records(df) for team, df in res["rosters"].items()},
    }
    path(year).write_text(json.dumps(payload, indent=1, default=str))
    print(f"[history] saved {year}: champion {res['champion']}, {len(res['standings'])} teams, "
          f"{len(res['rosters'])} rosters -> {path(year).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
