"""Player headshot URLs, keyed by display name.

Every export in this app already shows a player's name (nflverse's own `full_name`, or the
id crosswalk's resolved name) — a name-keyed lookup is a small, un-invasive companion to
every export here, rather than threading a gsis_id column through a dozen files just to
carry a decorative image. A rare name collision means a wrong or missing headshot, not a
wrong number, so the risk is worth the simplicity.

Hotlinked straight from the URL nflverse's roster file already gives (NFL's own CDN) — no
vendoring needed. The Streamlit player card already hotlinks the same field directly
(app.py's `_tab_lookup`) with no reliability issues, unlike the Wikimedia-sourced team logos
mega/logos.py had to vendor after Wikimedia started rejecting the request pattern.
"""
from __future__ import annotations

import pandas as pd

POS = ("QB", "RB", "WR", "TE")


def build(rosters: pd.DataFrame) -> dict[str, str]:
    if rosters is None or rosters.empty:
        return {}
    r = rosters.dropna(subset=["gsis_id"])
    if "position" in r.columns:
        r = r[r["position"].isin(POS)]
    out: dict[str, str] = {}
    for name, url in zip(r.get("full_name", pd.Series(dtype=str)), r.get("headshot_url", pd.Series(dtype=str))):
        if isinstance(name, str) and name and isinstance(url, str) and url.startswith("http"):
            out.setdefault(name, url)
    return out
