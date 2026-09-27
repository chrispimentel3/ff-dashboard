"""Pre-build the Ask player-week table for every finished season (data/ask_history/).

A finished season never changes, and building its table live means reading a season of
play-by-play — the biggest memory spike the Render service had. Run this once a season
(after the Super Bowl), or whenever tests/test_ask_history.py says the stamp is stale
because the code that builds the table changed.

    .venv/bin/python tools/build_ask_history.py            # 2015 .. last season
    .venv/bin/python tools/build_ask_history.py 2024 2025  # just these
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mega import data  # noqa: E402

FIRST = 2015     # the oldest season the Ask page lets you pick


def main(argv: list[str]) -> None:
    try:
        import nflreadpy as nfl
        current = int(nfl.get_current_season())
    except Exception:
        current = data.SEASON_DEFAULT + 1
    seasons = [int(a) for a in argv] or list(range(FIRST, current))
    seasons = [s for s in seasons if s < current]          # never freeze a season in progress
    out = data.ASK_HISTORY
    out.mkdir(parents=True, exist_ok=True)
    man_f = out / "manifest.json"
    stamp = data.ask_pw_stamp()
    man = json.loads(man_f.read_text()) if man_f.is_file() else {}
    have = set(man.get("seasons", [])) if man.get("stamp") == stamp else set()
    for s in seasons:
        pw = data.ask_player_week(s, prebuilt=False)
        if pw.empty:
            print(f"{s}: empty, skipped")
            continue
        pw.to_parquet(out / f"player_week_{s}.parquet", index=False)
        have.add(s)
        print(f"{s}: {pw.shape[0]} rows × {pw.shape[1]} cols")
    man_f.write_text(json.dumps({"stamp": stamp, "seasons": sorted(have)}, indent=1) + "\n")
    print(f"stamp {stamp}, seasons {sorted(have)}")


if __name__ == "__main__":
    main(sys.argv[1:])
