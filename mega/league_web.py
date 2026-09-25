"""Pure-data packaging for "The league": playoff odds, standings, power rankings,
transactions, weekly results / points-by-week.

Packaging only — every table here (`_odds()`, `cached_standings()`, `power_table()`,
`_power()`, `_matchup_log()`) is already a plain, pure DataFrame by the time `_tab_league`
builds it. The KPI cards the frontend shows for "your row" are derived client-side by
matching `my_team` against these tables, rather than duplicating that lookup here for the
several different shapes "me" takes across playoff odds / standings / power rankings.
"""
from __future__ import annotations

import json

import pandas as pd

ODDS_COLS = ["team", "p_playoffs", "p_bye", "p_title", "mean_seed"]
STANDINGS_COLS = ["rank", "team", "manager", "wins", "losses", "ties",
                  "points_for", "points_against", "streak", "faab_balance", "moves", "trades"]
XWINS_COLS = ["power_rank", "team", "xwins", "power", "wins", "luck_w", "pf", "ppg", "cv"]
ROSTER_STRENGTH_COLS = ["power_rank", "team", "starters_pg", "bench_pg", "rank", "luck", "matched"]


def _records(df: pd.DataFrame | None, cols: list[str] | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    if cols is not None:
        df = df.reindex(columns=[c for c in cols if c in df.columns])
    return json.loads(df.to_json(orient="records"))


def build(
    my_team: str,
    odds_df: pd.DataFrame | None,
    standings: pd.DataFrame | None,
    standings_source: str | None,
    xwins: pd.DataFrame | None,
    xwins_through_week: int | None,
    roster_strength: pd.DataFrame | None,
    transactions: pd.DataFrame | None,
    weekly_results: pd.DataFrame | None,
    weekly_results_source: str | None,
) -> dict:
    return {
        "my_team": my_team,
        "playoff_odds": _records(odds_df, ODDS_COLS),
        "standings": {
            "available": standings is not None and not standings.empty,
            "source": standings_source,
            "rows": _records(standings, STANDINGS_COLS),
        },
        "power_xwins": {
            "available": xwins is not None and not xwins.empty,
            "through_week": xwins_through_week,
            "rows": _records(xwins, XWINS_COLS),
        },
        "power_roster_strength": _records(roster_strength, ROSTER_STRENGTH_COLS),
        "transactions": _records(transactions.head(40) if transactions is not None else None),
        "weekly_results": {
            "available": weekly_results is not None and not weekly_results.empty,
            "source": weekly_results_source,
            "rows": _records(weekly_results),
        },
    }
