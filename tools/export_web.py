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
}


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

    if not written:
        sys.exit(1)

    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
