"""Pure-data packaging for the Glossary tab. `mega.glossary.frame()` is already pure and
completely static (doesn't depend on season/week) — this just groups it the way the
Streamlit tab does, with each group's lede text carried along.
"""
from __future__ import annotations

import json

from mega import glossary as GL

GROUPS = [
    ("Role", "Roles — the job he has",
     "One per player. Worked out from his last three games, not from where he was "
     "drafted, so it changes during the season when his usage does."),
    ("Flag", "Flags — what he is doing well",
     "A player can carry several. These are measured against others in the SAME "
     "role, so a third receiver is judged against other third receivers."),
    ("How long", "How long it has held",
     "The difference between a pattern and a good afternoon."),
    ("Vegas", "Vegas — what the betting market says", GL.VEGAS_HEADLINE),
]


def build() -> dict:
    g = GL.frame()
    groups = []
    for key, title, lede in GROUPS:
        sub = g[g["group"] == key][["tag", "what it means", "why it matters"]]
        groups.append({
            "key": key,
            "title": title,
            "lede": lede,
            "rows": json.loads(sub.to_json(orient="records")),
        })
    return {"headline": GL.HEADLINE, "groups": groups}
