# Render service memory (service/main.py)

The Ask / trade-search API runs on Render's free instance: **512 MB**, restarted when it
goes over. Measured 2026-09-27 on a Mac (resident memory after a realistic session;
Linux differs a little, and the Linux-only fixes below don't show up here at all).

| Step | Before | After |
|---|---|---|
| Libraries loaded (pandas, polars, pyarrow, fastapi) | 135 MB | 135 MB |
| `/meta` (this season's Ask table) | 270 MB | 212 MB |
| `/ask` about 2025 | **587 MB** | 247 MB |
| `/ask` about 2023–2025 | **1,022 MB** | 247 MB |
| 8 trade searches, title odds off | 492 MB | 473 MB |
| 8 trade searches, title odds on | 617 MB | 494 MB |

## What was using it

Almost none of it was data the service keeps (~55 MB of frames). It was spikes that the
allocator never handed back:

* **Past-season Ask questions** rebuilt that season's table from nflverse, including a
  full season of play-by-play (372 columns, ~130 MB parsed) read twice. Each past season
  asked about left ~300 MB behind.
* **Play-by-play for the current season** was parsed in full for the ~6 columns used.
  It grows all season (5,700 plays at week 3, ~48,000 by week 17).
* **nflreadpy's in-memory cache** held every table it loaded for 24 hours.
* **The title-odds model stored its dice** — 2,300 arrays of 3,000 draws — ~55 MB, plus
  up to 1,500 cached team-week arrays in float64.

## What changed

1. **Finished seasons are pre-built** — `data/ask_history/player_week_<season>.parquet`
   (2015–2025, 3 MB total, from `tools/build_ask_history.py`). They carry a fingerprint of
   the code that builds them; if that code changes they are ignored (never served stale)
   and `tests/test_service_memory.py` fails until they are rebuilt. Rebuild once a year
   after the Super Bowl to add the season just finished.
2. **Play-by-play is read only as wide as asked** (`mega/pbp.py`): streamed to disk, then
   just the requested columns read from the parquet. Catalogue questions in Ask ask for
   just the stat and its keys (`catalog.needed`).
3. **nflreadpy caches on disk** in the service; at most 4 Ask seasons are held.
4. **The title-odds model regenerates dice from the seed** instead of storing them (same
   key, same roll — results are identical), keeps team-week totals in float32, and keeps
   the current rosters' totals apart from trade variants, which are capped at 400.
   Cost: ~21 MB instead of ~125 MB. `TITLE_ODDS_LIVE` is now **on** by default
   (`TITLE_ODDS_LIVE=0` falls back to the team-level playoff odds).
5. **Linux only:** after each request the service runs `gc.collect()` and
   `malloc_trim(0)`, and `render.yaml` sets `MALLOC_ARENA_MAX=2` and
   `ARROW_DEFAULT_MEMORY_POOL=system` so pyarrow's buffers go through the same malloc.
   These return the trade-search spike (~200 MB, mostly building the league) to the OS
   instead of keeping it. Env vars in `render.yaml` apply only if the service is synced
   from the blueprint; otherwise add them in the Render dashboard.

## If it still runs out

The remaining peak is the trade search building the league (`trade_league.build_league`,
chiefly `projections.nflverse_estimate` reading two seasons of full-width player stats).
Trimming those loads to the columns used is the next lever. A paid Render tier only helps
if it adds memory.
