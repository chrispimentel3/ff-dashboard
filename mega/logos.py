"""NFL team logos for table cells, served from the app itself.

Source: `nfl_teamlogos.csv` from Michael Lopez's BlogPosts repo
(github.com/statsbylopez/BlogPosts) — the list nflverse's `team_logo_wikipedia`
column descends from. Vendored to data/nfl_teamlogos.csv with fixes, because the
file as published renders a broken image for every team today:

  * Wikimedia now rejects thumbnail widths outside its standard steps (HTTP 400,
    "use thumbnail sizes listed on w.wiki/GHai") — 33 of the 35 rows. Rewritten to 120px.
  * WAS still pointed at the 2020 "Washington Football Team" logo -> Commanders.
  * TEN's file was replaced for the 2026 rebrand -> the current file, via nflverse.
  * KC and LA 404 at every width (the files are gone; nflverse's copy is stale too)
    -> ESPN's CDN.
  * OAK / STL / SD dropped; ids.canon_team maps them to LV / LA / LAC.

The images themselves live in static/logos/ and Streamlit serves them
(server.enableStaticServing). Hotlinking Wikimedia is not an option: it returned
HTTP 429 at one request every 1.5 s during the one-time download, so every viewer's
browser pulling 32 logos would leave gaps in the tables.

    python -m mega.logos --check   # every team has a local file
    python -m mega.logos --fetch   # download any missing ones, politely
"""
from __future__ import annotations

import functools

import pandas as pd

from .config import DATA, ROOT
from .ids import canon_team

LOGOS_CSV = DATA / "nfl_teamlogos.csv"
STATIC_DIR = ROOT / "static" / "logos"
# Streamlit's static route. Relative, so it resolves against wherever the app is hosted.
URL_PREFIX = "app/static/logos"


@functools.lru_cache(maxsize=1)
def _sources() -> dict[str, str]:
    """canonical team code -> original image URL (provenance, and what --fetch pulls)."""
    if not LOGOS_CSV.is_file():
        return {}
    df = pd.read_csv(LOGOS_CSV, dtype=str).fillna("")
    return {canon_team(c): u for c, u in zip(df["team_code"], df["url"]) if u}


@functools.lru_cache(maxsize=1)
def _available() -> frozenset[str]:
    return frozenset(p.stem for p in STATIC_DIR.glob("*.png")) if STATIC_DIR.is_dir() else frozenset()


def logo_url(team: object) -> str | None:
    """Served logo path for any abbreviation (LAR, JAC, GBP ... all resolve). None if unknown."""
    code = canon_team(team)
    return f"{URL_PREFIX}/{code}.png" if code in _available() else None


def check() -> list[str]:
    """Teams with a source URL but no local image."""
    return sorted(set(_sources()) - _available())


def fetch() -> list[str]:
    """Download any missing logos, honouring Wikimedia's rate limit. Returns failures."""
    import time

    import httpx

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    with httpx.Client(headers={"User-Agent": "mega-bowl-dashboard/0.1 (one-time logo fetch)"},
                      timeout=20, follow_redirects=True) as c:
        for code in check():
            for attempt in range(6):
                r = c.get(_sources()[code])
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                    (STATIC_DIR / f"{code}.png").write_bytes(r.content)
                    break
                time.sleep(int(r.headers.get("retry-after", 0) or 0) or 20 * (attempt + 1))
            else:
                failed.append(code)
            time.sleep(1.5)
    _available.cache_clear()
    return failed


if __name__ == "__main__":
    import sys

    if "--fetch" in sys.argv:
        bad = fetch()
        print("all logos present" if not bad else "failed: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    missing = check()
    print(f"{len(_sources())} teams, {len(_available())} local logos"
          + (f", missing: {', '.join(missing)}" if missing else ", none missing"))
    sys.exit(1 if missing else 0)
