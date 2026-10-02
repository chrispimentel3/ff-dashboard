"""mega/trend_web.py: every snapshotted player is exported; movers are last-vs-previous week."""
from __future__ import annotations

import pandas as pd

from mega import trend_web as T


def _df(rows):
    return pd.DataFrame(rows, columns=["week", "norm", "player", "pos", "value"])


def test_movers_rank_the_change_between_the_last_two_snapshots():
    df = _df([
        (3, "a", "A", "WR", 1000), (4, "a", "A", "WR", 1600),
        (3, "b", "B", "RB", 5000), (4, "b", "B", "RB", 4000),
        (3, "c", "C", "TE", 2000), (4, "c", "C", "TE", 2050),
    ])
    m = T.movers(df, "value", {"b"})
    assert m["weeks"] == [3, 4]
    assert [r["player"] for r in m["up"]] == ["A", "C"]
    assert [r["player"] for r in m["down"]] == ["B"]
    assert m["up"][0]["delta"] == 600 and m["down"][0]["delta"] == -1000
    assert m["down"][0]["mine"] is True and m["up"][0]["mine"] is False


def test_one_snapshot_has_no_movers():
    m = T.movers(_df([(4, "a", "A", "WR", 1000)]), "value", set())
    assert m == {"weeks": [], "up": [], "down": []}


def test_a_player_missing_a_week_is_not_a_mover_and_noise_values_are_skipped():
    df = _df([(3, "a", "A", "WR", 1000), (4, "b", "B", "WR", 900),
              (3, "c", "C", "WR", 100), (4, "c", "C", "WR", 300)])
    m = T.movers(df, "value", set())
    assert m["up"] == [] and m["down"] == []


def test_series_exports_everyone_with_position_and_mine_flag(tmp_path):
    p = tmp_path / "w.csv"
    pd.DataFrame({"season": 2026, "week": [3, 4, 4], "norm": ["a", "a", "z"], "player": ["A", "A", "Z"],
                  "pos": ["WR", "WR", "TE"], "wopr_anchored": [0.3, 0.35, 0.2]}).to_csv(p, index=False)
    out = T._series(p, "wopr_anchored", {"a"})
    assert out["players"] == ["A", "Z"]
    assert out["info"]["A"] == {"pos": "WR", "mine": True} and out["info"]["Z"]["mine"] is False
    assert len(out["rows"]) == 3
    assert out["movers"]["up"][0]["player"] == "A"


def test_movers_can_be_limited_to_players_who_matter():
    df = _df([(3, "a", "A", "WR", 1000), (4, "a", "A", "WR", 2000),
              (3, "z", "Z", "WR", 1000), (4, "z", "Z", "WR", 5000)])
    assert [r["player"] for r in T.movers(df, "value", set(), only={"a"})["up"]] == ["A"]
    assert [r["player"] for r in T.movers(df, "value", set(), only=None)["up"]] == ["Z", "A"]
    # but your own players are always eligible
    assert [r["player"] for r in T.movers(df, "value", {"z"}, only={"a"})["up"]] == ["Z", "A"]
