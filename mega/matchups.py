"""Pure-data Matchups: team game environment, Vegas player projections, defense-vs-position
difficulty.

Unlike action_board.py/start_sit.py, this one doesn't re-derive any of the three tables —
app.py's `_tab_match` already builds `mt`, `_vb`, and `show` as plain DataFrames before
rendering them, so this module is packaging only: shape those three frames (already computed
once, by the same code Streamlit renders) into JSON-safe records. The "my roster only" /
"minimum markets" filtering that the Streamlit tab does with a checkbox and a slider is
deliberately NOT done here — the full Vegas player list ships to the frontend, which filters
client-side, the same way the interactive widgets do.
"""
from __future__ import annotations

import json
import re

import pandas as pd

_VERDICT_EMOJI = re.compile(r"^[^\w]*\s*")


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _clean_difficulty(show: pd.DataFrame | None) -> list[dict]:
    """Same rows the Streamlit tab shows, minus the emoji prefix on `verdict` (e.g.
    "🟢 great" -> "great") — the frontend renders its own badge for the tier."""
    if show is None or show.empty:
        return []
    df = show.copy()
    if "verdict" in df.columns:
        df["verdict"] = df["verdict"].astype(str).map(lambda v: _VERDICT_EMOJI.sub("", v))
    return _records(df)


def build(
    team_env: pd.DataFrame | None,
    vegas_players: pd.DataFrame | None,
    player_difficulty: pd.DataFrame | None,
    next_week: int,
    my_team: str,
) -> dict:
    """Full Matchups payload as JSON-safe plain Python — no Streamlit calls.

    `my_team` is `mega.config.MY_TEAM` — the frontend needs it to offer the same "my roster
    only" filter the Streamlit checkbox does, since `vegas_players[].owner` is a team name or
    "FA", not a boolean, and nothing else in this payload says which name is Chris's.
    """
    return {
        "available": team_env is not None and not team_env.empty,
        "next_week": next_week,
        "my_team": my_team,
        "team_environment": _records(team_env),
        "vegas_players": _records(vegas_players),
        "player_difficulty": _clean_difficulty(player_difficulty),
    }
