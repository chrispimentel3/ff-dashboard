"""Packaging-only export of the Logic tab — mega.logic.frame() is already pure static
content, so there's nothing to derive here. Exported directly by tools/export_web.py
rather than through a Streamlit tab, same as mega/trend_web.py."""
from __future__ import annotations

from . import logic as LOGIC


def build() -> dict:
    return {"topics": LOGIC.frame()}
