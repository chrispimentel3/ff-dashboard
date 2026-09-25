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
    "_export_draft": "draft.json",
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

    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
