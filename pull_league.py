"""Pull Yahoo Fantasy league 173489 to JSON via the official API (YFPY).

ONE-TIME SETUP
  1. Create a Yahoo app:  https://developer.yahoo.com/apps/create/
       Application Type : Installed Application
       Redirect URI     : https://localhost:8080
       API Permissions  : Fantasy Sports  (Read)
  2. Copy .env.example -> .env  and add:
       YAHOO_CONSUMER_KEY=<your Client ID>
       YAHOO_CONSUMER_SECRET=<your Client Secret>
  3. uv run python pull_league.py
       First run prints a Yahoo URL -> open it, approve, paste the code back.
       The token is cached into .env, so later runs are non-interactive.

Output: data/yahoo_api/*.json  (consumed by mega/yahoo_api.py -> the dashboard + tuesday.py)
"""
from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from yfpy.query import YahooFantasySportsQuery
from yfpy.utils import jsonify_data

LEAGUE_ID = "173489"
GAME_CODE = "nfl"
SEASON = None  # None = current season
WEEK = None    # None = league's current week

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "yahoo_api"


def save(name: str, data) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(jsonify_data(data), encoding="utf-8")
    print(f"  saved data/yahoo_api/{name}.json")


def text(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def main() -> None:
    q = YahooFantasySportsQuery(
        league_id=LEAGUE_ID,
        game_code=GAME_CODE,
        env_file_location=HERE,
        save_token_data_to_env_file=True,
    )
    q.league_key = q.get_league_key(SEASON)
    print(f"League key: {q.league_key}")

    meta = q.get_league_metadata()
    week = WEEK or getattr(meta, "current_week", None) or 1
    print(f"League: {text(getattr(meta, 'name', 'league'))} | week {week}")

    save("league_metadata", meta)
    save("league_settings", q.get_league_settings())
    save("league_standings", q.get_league_standings())

    teams = q.get_league_teams()
    save("league_teams", teams)
    save("draft_results", q.get_league_draft_results())
    save("transactions", q.get_league_transactions())
    save(f"scoreboard_week_{week}", q.get_league_scoreboard_by_week(week))

    rosters = {}
    for team in teams:
        tid = getattr(team, "team_id", None)
        nm = text(getattr(team, "name", tid))
        print(f"  roster: {nm}")
        rosters[nm] = q.get_team_roster_by_week(tid, week)
    save(f"rosters_week_{week}", rosters)
    save("_week", {"week": week})

    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
