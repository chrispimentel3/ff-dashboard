"""Catalogue every nflverse table and column, so the question box can find any of them.

    PYTHONPATH=. .venv/bin/python tools/build_schema.py [season]

Writes data/nflverse_schema.json: for each loader, its columns, their dtypes, the keys it
joins on and a couple of example values. Built once and committed, so the app can resolve
"list WRs by <anything>" without loading a 100 MB play-by-play file just to learn that a
column exists.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nflverse_schema.json"

# Loaders with a required or useful variant argument. Each variant becomes its own table.
VARIANTS = {
    "load_nextgen_stats": [("passing",), ("receiving",), ("rushing",)],
    "load_pfr_advstats": [("pass",), ("rec",), ("rush",), ("def",)],
    "load_ff_opportunity": [("weekly",)],
    "load_player_stats": [(None,)],
}
VARIANT_KW = {"load_nextgen_stats": "stat_type", "load_pfr_advstats": "stat_type",
              "load_ff_opportunity": "stat_type"}

# Columns that identify a player or a team, in the order we prefer to join on them.
PLAYER_KEYS = ("player_id", "gsis_id", "pfr_player_id", "pfr_id", "esb_id", "nfl_id",
               "player_gsis_id", "receiver_player_id", "rusher_player_id", "passer_player_id")
NAME_KEYS = ("player_display_name", "player_name", "full_name", "football_name", "player",
             "display_name", "pfr_player_name")
TEAM_KEYS = ("team", "recent_team", "posteam", "team_abbr", "club_code")
WEEK_KEYS = ("week", "game_week")


def _first(cols, cands):
    return next((c for c in cands if c in cols), None)


def describe(name: str, fn, season: int, variant=None) -> dict | None:
    kw = {}
    sig = inspect.signature(fn)
    if "seasons" in sig.parameters:
        kw["seasons"] = [season]
    if variant is not None and name in VARIANT_KW:
        kw[VARIANT_KW[name]] = variant
    t0 = time.time()
    try:
        df = fn(**kw)
    except Exception as e:
        print(f"  {name}{'/' + str(variant) if variant else '':<12} SKIPPED — {type(e).__name__}: {str(e)[:90]}")
        return None
    cols = list(df.columns)
    dtypes = {c: str(df.schema[c]) for c in cols} if hasattr(df, "schema") else {}
    # a couple of real values per column, for the "what does this hold" panel
    samples = {}
    try:
        head = df.head(200)
        for c in cols:
            vals = [v for v in head[c].to_list() if v is not None][:3]
            if vals:
                samples[c] = [str(v)[:40] for v in vals]
    except Exception:
        pass
    numeric = [c for c in cols if any(k in dtypes.get(c, "").lower()
                                      for k in ("int", "float", "decimal"))]
    rec = dict(
        table=name.replace("load_", "") + (f"_{variant}" if variant else ""),
        loader=name, variant=variant, rows=int(df.height if hasattr(df, "height") else len(df)),
        seconds=round(time.time() - t0, 1), n_cols=len(cols), columns=cols, dtypes=dtypes,
        numeric=numeric, samples=samples,
        player_key=_first(cols, PLAYER_KEYS), name_key=_first(cols, NAME_KEYS),
        team_key=_first(cols, TEAM_KEYS), week_key=_first(cols, WEEK_KEYS),
        has_season="season" in cols,
    )
    print(f"  {rec['table']:<28} {rec['rows']:>7} rows  {len(cols):>3} cols  "
          f"{rec['seconds']:>5.1f}s  key={rec['player_key'] or rec['team_key'] or '-'}")
    return rec


def main(season: int = 2026) -> int:
    import nflreadpy as nfl

    # `dir(nfl)` also turns up submodules named load_* (load_ffverse is a module, not a
    # function), so callability is the filter, not the prefix.
    names = sorted(n for n in dir(nfl)
                   if n.startswith("load_") and callable(getattr(nfl, n, None)))
    out = {}
    print(f"cataloguing {len(names)} loaders for {season} …")
    for n in names:
        fn = getattr(nfl, n)
        for variant in [v[0] for v in VARIANTS.get(n, [(None,)])]:
            rec = describe(n, fn, season, variant)
            if rec:
                out[rec["table"]] = rec
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    total = sum(r["n_cols"] for r in out.values())
    print(f"\n{len(out)} tables, {total} columns -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2026))
