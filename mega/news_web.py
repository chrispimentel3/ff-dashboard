"""Pure-data packaging for the News tab.

`news_for_players` is just `news_items()` filtered by title match, so the export fetches the
full feed once and tags each item with whether it mentions a roster player — the Streamlit
tab's "scope" radio (My roster / Watchlist + roster / All NFL) becomes a client-side filter
over this one list. The free-text watchlist box is a simplification: the frontend does a
plain case-insensitive substring match on the title rather than replicating the name-
normalization/last-name logic `news_for_players` uses for roster names, since that's a
convenience filter, not a correctness-critical one.
"""
from __future__ import annotations

import json

from mega.sources import news_items, news_for_players

COLS = ["source", "title", "link", "summary", "published"]


def build(my_names: list[str]) -> dict:
    try:
        items = news_items()
    except Exception as e:
        return {"available": False, "items": [], "error": str(e)}

    if items.empty:
        return {"available": True, "items": []}

    mine = news_for_players(my_names)
    mine_links = set(mine["link"]) if not mine.empty else set()

    items = items.copy()
    items["mentions_mine"] = items["link"].isin(mine_links)
    items = items.reindex(columns=COLS + ["mentions_mine"]).head(200)
    return {"available": True, "items": json.loads(items.to_json(orient="records"))}
