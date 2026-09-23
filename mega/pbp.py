"""Play-by-play, loaded once and cached on disk.

Lives in its own module because two callers need it — mega/routes.py for team dropbacks
and mega/redzone.py for scoring-position work — and a second copy of the cache dance
below is exactly the kind of duplication that drifts.
"""
from __future__ import annotations


def load(season: int, columns: list[str] | None = None):
    """A season of play-by-play as a polars frame, filesystem-cached.

    nflreadpy caches in memory by default. Play-by-play is ~100 MB a season and every
    caller here reduces it to a few hundred rows, so holding all of it for the life of
    the process is the wrong trade — especially on Streamlit Cloud, where the app and
    its cached frames share one small container.
    """
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
