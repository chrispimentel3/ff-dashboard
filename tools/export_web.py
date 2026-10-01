"""Export dashboard data as JSON for the mega-bowl-web frontend.

Runs app.py headlessly via Streamlit's own AppTest harness (the same one used for the
whole-app exception check) with default widget values — i.e. exactly what a visitor sees
on load — and reads back whatever the app itself stashed in `st.session_state` under an
`_export_*` key. This avoids re-deriving season/week/roll defaults by hand: the export can
never drift from what the live dashboard computes, because it IS the live dashboard.

Writes to data/web/<name>.json. That directory is committed and public (same as the rest
of data/), so mega-bowl-web's Next.js build fetches these files straight off
raw.githubusercontent.com — no live backend, no second copy of any analysis logic.

Usage:
    .venv/bin/python tools/export_web.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "web"

EXPORTS = {
    "_export_action_board": "action_board.json",
    "_export_start_sit": "start_sit.json",
    "_export_matchups": "matchups.json",
    "_export_waivers": "waivers.json",
    "_export_trades": "trades.json",
    "_export_wopr": "wopr.json",
    "_export_archetypes": "archetypes.json",
    "_export_roster": "roster.json",
    "_export_routes": "routes.json",
    "_export_axe": "axe.json",
    "_export_usage": "usage.json",
    "_export_league": "league.json",
    "_export_glossary": "glossary.json",
    "_export_news": "news.json",
    "_export_downloads": "downloads.json",
}

# _export_players is handled separately (see _write_players below): one big JSON with a
# full detail payload per player runs to several MB (~1,000 players × ~5KB), too large for
# a single fetch. Split into a small search index plus one file per player, fetched only
# for whichever player the frontend's dynamic /players/[gsisId] route is asked to render.
PLAYERS_KEY = "_export_players"


def _write_players(payload: dict) -> list[Path]:
    written = []
    index_path = OUT_DIR / "players_index.json"
    index_path.write_text(json.dumps(
        {"available": payload.get("available", False), "index": payload.get("index", [])},
        indent=2, default=str))
    written.append(index_path)

    players_dir = OUT_DIR / "players"
    players_dir.mkdir(parents=True, exist_ok=True)
    for gid, detail in payload.get("players", {}).items():
        path = players_dir / f"{gid}.json"
        path.write_text(json.dumps(detail, indent=2, default=str))
        written.append(path)
    return written


def main() -> None:
    # app.py checks this to call every exported tab's function directly instead of going
    # through st.navigation — see the MEGA_EXPORT_WEB branch at the bottom of app.py.
    os.environ["MEGA_EXPORT_WEB"] = "1"
    # HANDOFF v1.3: rebuild our rest-of-season projection first — the trade and waiver
    # engines inside the app run read data/proj_ros_<season>.json.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import nflreadpy as _nfl0
        from mega import data as _d0
        from mega import proj_ros as _pr
        _s0 = int(_nfl0.get_current_season())
        _pr.build(_s0, min(_d0.current_week(_s0, 1), 17))
        print(f"wrote data/proj_ros_{_s0}.json")
    except Exception as e:
        print(f"[export_web] projection build failed, engines use the last file: {e}", file=sys.stderr)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()

    if at.exception:
        for e in at.exception:
            print(f"[export_web] app.py raised during run: {e}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for key, filename in EXPORTS.items():
        if key not in at.session_state:
            print(f"[export_web] WARNING: {key} not found in session_state — tab may not "
                  f"have run, or the export key was renamed.", file=sys.stderr)
            continue
        payload = dict(at.session_state[key])
        path = OUT_DIR / filename
        path.write_text(json.dumps(payload, indent=2, default=str))
        written.append(path)

    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")

    if PLAYERS_KEY not in at.session_state:
        print(f"[export_web] WARNING: {PLAYERS_KEY} not found in session_state — tab may not "
              f"have run, or the export key was renamed.", file=sys.stderr)
    else:
        player_files = _write_players(dict(at.session_state[PLAYERS_KEY]))
        written.extend(player_files)
        print(f"wrote data/web/players_index.json + {len(player_files) - 1} data/web/players/<gsis_id>.json files")

    # Trend charts (WOPR, archetype fit, trade value) read mega/history.py's weekly
    # snapshot CSVs directly — plain files, no Streamlit run context needed, so this
    # runs outside the AppTest harness above rather than through session_state.
    sys.path.insert(0, str(ROOT))
    from mega import data as _data
    from mega import trend_web as _trend_web

    norms = _data.my_norms(_data.my_roster()[0])
    trends = _trend_web.build(norms)
    for key, filename in (("wopr", "wopr_trend.json"), ("archetype", "archetype_trend.json"),
                          ("trade_value", "trade_value_trend.json")):
        path = OUT_DIR / filename
        path.write_text(json.dumps(trends[key], indent=2, default=str))
        written.append(path)
        print(f"wrote {path.relative_to(ROOT)}")

    # Logic (methodology) content is static and Streamlit-free too — see mega/logic.py.
    from mega import logic_web as _logic_web

    logic_path = OUT_DIR / "logic.json"
    logic_path.write_text(json.dumps(_logic_web.build(), indent=2, default=str))
    written.append(logic_path)
    print(f"wrote {logic_path.relative_to(ROOT)}")

    # Player headshots, name-keyed — see mega/headshots_web.py.
    from mega import headshots_web as _headshots_web

    try:
        import nflreadpy as _nfl
        _season = int(_nfl.get_current_season())
    except Exception:
        _season = _data.SEASON_DEFAULT
    headshots_path = OUT_DIR / "headshots.json"
    headshots_path.write_text(json.dumps(_headshots_web.build(_data.load_rosters(_season)),
                                         indent=2, default=str))
    written.append(headshots_path)
    print(f"wrote {headshots_path.relative_to(ROOT)}")

    # Weekly rankings — league-wide, position-ranked by projection. Streamlit-free (same
    # tier as logic/headshots/trend); see mega/rankings_web.py.
    from mega import rankings_web as _rankings_web

    _next_week = min(_data.current_week(_season, 1), 18)
    _gsis_list, _ = _data.my_gsis_ids(_data.my_roster()[0])
    rankings_path = OUT_DIR / "rankings.json"
    rankings_path.write_text(json.dumps(
        _rankings_web.build(_season, _next_week, set(_gsis_list)), indent=2, default=str))
    written.append(rankings_path)
    print(f"wrote {rankings_path.relative_to(ROOT)}")

    # The weekly digest — the week just played and the one coming, in sentences. Built
    # from the league files and the league.json written above; see mega/digest_web.py.
    try:
        from mega import digest_web as _digest_web
        from mega.config import MY_TEAM as _MY_TEAM
        from mega.yahoo import cached_fixtures as _cfx, cached_scores as _csc

        _league = json.loads((OUT_DIR / "league.json").read_text())
        _through = int((_league.get("power_xwins") or {}).get("through_week") or 0)
        if _digest_web.snapshot_odds(_season, _through, _league.get("playoff_odds") or []):
            print(f"wrote {_digest_web.ODDS_CSV.relative_to(ROOT)} (week {_through} odds)")
        _ab = json.loads((OUT_DIR / "action_board.json").read_text())
        digest = _digest_web.build(_season, _MY_TEAM, _csc(), _cfx(), _league, _ab,
                                   _digest_web.odds_history(_season))
        digest_path = OUT_DIR / "digest.json"
        digest_path.write_text(json.dumps(digest, indent=2, default=str))
        written.append(digest_path)
        print(f"wrote {digest_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"[export_web] digest failed: {type(e).__name__}: {e}", file=sys.stderr)

    # Starters, NFL teams and projections for the site's live scoreboard, which scores
    # the week from ESPN box scores as it's played — see mega/live_web.py.
    try:
        from mega import live_web as _live_web
        from mega.config import MY_TEAM as _MY_TEAM
        from mega.yahoo import cached_fixtures as _cfx, cached_rosters as _cros

        _rk = json.loads(rankings_path.read_text()).get("rows") or []
        setup = _live_web.build(_season, _next_week, _cros(), _cfx(), _rk,
                                _data.load_rosters(_season), _MY_TEAM)
        live_path = OUT_DIR / "live_setup.json"
        live_path.write_text(json.dumps(setup, indent=2, default=str))
        written.append(live_path)
        print(f"wrote {live_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"[export_web] live setup failed: {type(e).__name__}: {e}", file=sys.stderr)

    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
