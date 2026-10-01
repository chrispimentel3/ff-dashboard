"""mega/stats_web.py — the chart builder's table."""
from __future__ import annotations

import pandas as pd

from mega import lookup, stats_web


def test_every_metric_is_a_season_table_column_and_rows_line_up_with_columns():
    t = pd.DataFrame([{"gsis_id": "a", "name": "A", "pos": "WR", "team": "MIN", "games": 3,
                       "pts_pg": 12.3456789, "tgt_share": float("nan")},
                      {"gsis_id": "b", "name": "B", "pos": "K", "team": "MIN", "games": 3}])
    out = stats_web.build({2026: {"table": t}}, {"a": {"kind": "mine", "team": "TaylorMade"}})
    assert out["columns"][:7] == stats_web.ID_COLS and len(out["rows"]) == 1   # kickers left out
    row = dict(zip(out["columns"], out["rows"][0]))
    assert row["pts_pg"] == 12.3457 and row["tgt_share"] is None and row["mine"] and row["owner"] == "TaylorMade"
    assert {m["key"] for m in out["metrics"]} == {m[0] for m in stats_web.METRICS}


def test_the_metrics_exist_on_the_real_season_table():
    # every card stat is charted, and every charted column is one season_table makes
    card_cols = {c for rows in lookup.CARD.values() for c, *_ in rows}
    keys = {m[0] for m in stats_web.METRICS}
    assert card_cols <= keys | {"routes_pg"}


def test_matchup_score_ranks_the_week_ten_is_best():
    from mega.player_web import _matchup_score
    week = pd.Series([12.0, 5.0, 0.0, -4.0, -10.0])
    assert _matchup_score(week, 12.0) == {"matchup_rank": 1, "matchup_n": 5, "matchup_score": 10.0}
    assert _matchup_score(week, -10.0)["matchup_score"] == 1.0
    assert _matchup_score(week, 0.0)["matchup_score"] == 5.5
    assert _matchup_score(week, None) == {}
