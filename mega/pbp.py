"""Play-by-play, loaded once and cached on disk.

Lives in its own module because two callers need it — mega/routes.py for team dropbacks
and mega/redzone.py for scoring-position work — and a second copy of the cache dance
below is exactly the kind of duplication that drifts.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
FRESH_S = 24 * 3600      # the current season's file is re-fetched after a day, like nflreadpy's


def _file(season: int) -> Path:
    """The season's parquet on local disk, streamed there without passing through memory.

    nflreadpy reads the whole ~30 MB download into a bytes object and then parses every
    one of the 372 columns, ~130 MB resident for a full season — the biggest single
    spike in the Render service, which then keeps the memory. Every caller here wants
    under ten columns, and parquet can hand back just those from a file on disk.
    """
    import requests

    d = Path(os.environ.get("MEGA_PBP_DIR") or Path(tempfile.gettempdir()) / "mega_pbp")
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"play_by_play_{season}.parquet"
    try:
        from nflreadpy import get_current_season
        current = int(season) >= int(get_current_season())
    except Exception:
        current = True
    if f.is_file() and not (current and time.time() - f.stat().st_mtime > FRESH_S):
        return f
    tmp = f.with_suffix(".part")
    with requests.get(URL.format(season=season), stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as out:
            for chunk in r.iter_content(chunk_size=1 << 20):
                out.write(chunk)
    tmp.replace(f)
    return f


def load(season: int, columns: list[str] | None = None):
    """A season of play-by-play as a polars frame — only `columns` when given."""
    import polars as pl

    try:
        f = _file(season)
        if columns:
            have = set(pl.read_parquet_schema(f))
            return pl.read_parquet(f, columns=[c for c in columns if c in have])
        return pl.read_parquet(f)
    except Exception:
        pass
    # fallback: nflreadpy's own loader, kept off its in-memory cache
    import nflreadpy as nfl

    prev = None
    try:
        from nflreadpy import config as _cfg

        prev = _cfg.get_config().cache_mode
        _cfg.update_config(cache_mode="filesystem")
    except Exception:
        prev = None
    try:
        df = nfl.load_pbp([season])
    finally:
        if prev is not None:
            try:
                from nflreadpy import config as _cfg

                _cfg.update_config(cache_mode=prev)
            except Exception:
                pass
    if columns:
        df = df.select([c for c in columns if c in df.columns])
    return df
