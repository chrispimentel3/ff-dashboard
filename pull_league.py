"""Pull Yahoo Fantasy league 173489 to JSON via the official API (YFPY).

ONE-TIME SETUP
  1. Create a Yahoo app:  https://developer.yahoo.com/apps/create/
       Application Type : Installed Application
       Redirect URI     : https://localhost:8080
       API Permissions  : Fantasy Sports  (Read)
  2. Put the credentials in .env:
       YAHOO_CONSUMER_KEY=<your Client ID>
       YAHOO_CONSUMER_SECRET=<your Client Secret>
  3. .venv/bin/python pull_league.py
       First run prints a Yahoo URL -> open it, approve, paste the code back.
       The token is cached into .env, so later runs are non-interactive.

USAGE
  python pull_league.py                    # scoreboards for every week so far
  python pull_league.py --through-week 4   # weeks 1..4
  python pull_league.py --weeks 1 2 3      # specific weeks
  python pull_league.py --dry-run          # talk to Yahoo, write nothing

GITHUB ACTIONS
  After a successful local run, cache the token so CI never logs in:
      base64 -i .env | tr -d '\\n'     # macOS
  Store that as the YAHOO_TOKEN_B64 repo secret alongside YAHOO_CONSUMER_KEY
  and YAHOO_CONSUMER_SECRET. Refresh tokens do expire — when CI starts failing
  auth, rerun locally once and update the secret.

Output: data/yahoo_api/*.json  (consumed by mega/yahoo_api.py -> the dashboard + tuesday.py)
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Any

LEAGUE_ID = "173489"
GAME_CODE = "nfl"
SEASON = None  # None = current season

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data" / "yahoo_api"


# ---------------------------------------------------------------- credentials
def load_env_file(path: Path) -> None:
    """Minimal .env loader. Existing env vars win, so CI secrets are never
    clobbered by a stale file."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def materialize_ci_token(work_dir: Path) -> None:
    """In CI, decode YAHOO_TOKEN_B64 back into a .env file yfpy can read."""
    blob = os.environ.get("YAHOO_TOKEN_B64")
    if not blob:
        return
    try:
        decoded = base64.b64decode(blob)
    except Exception as exc:
        sys.exit(f"YAHOO_TOKEN_B64 is not valid base64: {exc}")
    target = work_dir / ".env"
    target.write_bytes(decoded)
    load_env_file(target)
    print("Restored cached Yahoo token from YAHOO_TOKEN_B64")


TOKEN_KEYS = ("YAHOO_ACCESS_TOKEN", "YAHOO_REFRESH_TOKEN", "YAHOO_TOKEN_TIME",
              "YAHOO_TOKEN_TYPE", "YAHOO_GUID")


def clear_cached_token(path: Path) -> None:
    """Drop the cached token so the next run re-runs the consent flow.

    Needed whenever the token's scope is wrong (Yahoo answers
    `additional_authorization_required`) or the refresh token has expired —
    neither of which yfpy recovers from on its own.
    """
    if not path.is_file():
        return
    kept = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.partition("=")[0].strip() not in TOKEN_KEYS]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    for k in TOKEN_KEYS:
        os.environ.pop(k, None)
    print("Cleared cached Yahoo token — you'll be asked to approve again.")


def check_token() -> bool:
    """Probe what the cached token can actually reach.

    As of 2026-09 Yahoo no longer offers Fantasy Sports in the app permissions
    console, so tokens authenticate (OpenID returns your profile) but every
    fantasy endpoint 401s with `additional_authorization_required` — including
    public ones. Run this to find out whether that has changed.
    """
    import json
    import urllib.error
    import urllib.request

    tok = os.environ.get("YAHOO_ACCESS_TOKEN", "")
    try:
        tok = json.loads(tok).get("access_token", tok)
    except Exception:
        pass
    if not tok:
        print("No cached token. Run without --check first.")
        return False

    probes = {
        "openid profile": "https://api.login.yahoo.com/openid/v1/userinfo",
        "fantasy public": "https://fantasysports.yahooapis.com/fantasy/v2/game/nfl?format=json",
        "fantasy private": "https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games?format=json",
    }
    ok = {}
    for name, url in probes.items():
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        try:
            urllib.request.urlopen(req, timeout=20)
            ok[name] = True
            print(f"  {name:16s} OK")
        except urllib.error.HTTPError as e:
            ok[name] = False
            print(f"  {name:16s} {e.code} {e.read()[:120].decode('utf-8', 'replace')}")
        except Exception as e:
            ok[name] = False
            print(f"  {name:16s} ERR {e}")

    if ok.get("fantasy private"):
        print("\nFantasy scope is live — run without --check to pull.")
        return True
    if ok.get("openid profile"):
        print("\nToken is valid but has no Fantasy Sports scope — Yahoo still hasn't\n"
              "restored that permission in the app console. Use the browser scrape:\n"
              "  .venv/bin/python -m mega.yahoo login && .venv/bin/python -m mega.yahoo pull")
    else:
        print("\nToken looks dead. Re-run with --reauth.")
    return False


def require_credentials() -> tuple[str, str]:
    key = os.environ.get("YAHOO_CONSUMER_KEY") or os.environ.get("YAHOO_CLIENT_ID")
    secret = os.environ.get("YAHOO_CONSUMER_SECRET") or os.environ.get("YAHOO_CLIENT_SECRET")
    if not key or not secret:
        sys.exit(
            "Missing credentials. Set YAHOO_CONSUMER_KEY and YAHOO_CONSUMER_SECRET "
            "in .env (local) or as repo secrets (CI)."
        )
    return key, secret


# ---------------------------------------------------------------- accessors
# yfpy's object shapes drift between releases: names come back as bytes in some
# versions and str in others, points live on .points or .team_points. These
# absorb that so a yfpy upgrade can't silently empty the season file.
def text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        found = getattr(obj, name, None)
        if found is not None:
            return found
    return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(text(value) or default)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- output
def save(name: str, data) -> None:
    from yfpy.utils import jsonify_data

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(jsonify_data(data), encoding="utf-8")
    print(f"  saved data/yahoo_api/{name}.json")


def resolve_weeks(args, start: int, current: int) -> list[int]:
    if args.weeks:
        return sorted(set(args.weeks))
    if args.through_week:
        return list(range(start, args.through_week + 1))
    return list(range(start, current + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull Yahoo league data for the dashboard")
    ap.add_argument("--weeks", type=int, nargs="+", help="specific weeks to pull")
    ap.add_argument("--through-week", type=int, help="pull weeks 1..N")
    ap.add_argument("--dry-run", action="store_true", help="fetch but write nothing")
    ap.add_argument("--reauth", action="store_true",
                    help="drop the cached token and redo the Yahoo consent flow")
    ap.add_argument("--check", action="store_true",
                    help="probe what the cached token can reach, then exit")
    args = ap.parse_args()

    if args.reauth:
        clear_cached_token(HERE / ".env")
    load_env_file(HERE / ".env")

    if args.check:
        check_token()
        return
    materialize_ci_token(HERE)
    key, secret = require_credentials()

    try:
        from yfpy.query import YahooFantasySportsQuery
    except ImportError:
        sys.exit("yfpy is not installed. Run: .venv/bin/pip install -r requirements-local.txt")

    q = YahooFantasySportsQuery(
        league_id=LEAGUE_ID,
        game_code=GAME_CODE,
        yahoo_consumer_key=key,
        yahoo_consumer_secret=secret,
        env_file_location=HERE,
        save_token_data_to_env_file=True,
    )
    q.league_key = q.get_league_key(SEASON)
    print(f"League key: {q.league_key}")

    meta = q.get_league_metadata()
    start = as_int(first_attr(meta, "start_week"), 1)
    current = as_int(first_attr(meta, "current_week"), start)
    print(f"League: {text(first_attr(meta, 'name', default='league'))} | current week {current}")

    weeks = resolve_weeks(args, start, current)
    print(f"Weeks to pull: {', '.join(str(w) for w in weeks)}")

    teams = q.get_league_teams()
    print(f"  {len(teams)} teams")

    scoreboards = {}
    for w in weeks:
        try:
            scoreboards[w] = q.get_league_scoreboard_by_week(w)
        except Exception as exc:
            print(f"  week {w} scoreboard failed: {exc}")

    # Rosters only for the current week — ownership is a "right now" question,
    # and one file per week would break the newest-file glob in mega/yahoo_api.py.
    rosters = {}
    for team in teams:
        tid = first_attr(team, "team_id")
        nm = text(first_attr(team, "name", default=tid))
        try:
            rosters[nm] = q.get_team_roster_by_week(tid, current)
            print(f"  roster: {nm}")
        except Exception as exc:
            print(f"  roster {nm} failed: {exc}")

    if args.dry_run:
        print(f"\n--dry-run: {len(scoreboards)} scoreboards, {len(rosters)} rosters, nothing written")
        return

    save("league_metadata", meta)
    save("league_settings", q.get_league_settings())
    save("league_standings", q.get_league_standings())
    save("league_teams", teams)
    save("draft_results", q.get_league_draft_results())
    save("transactions", q.get_league_transactions())
    for w, sb in scoreboards.items():
        save(f"scoreboard_week_{w}", sb)
    save(f"rosters_week_{current}", rosters)
    save("_week", {"week": current, "weeks_pulled": sorted(scoreboards)})

    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
