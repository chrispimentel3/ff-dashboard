"""§20.2 assembly: the fill, the published build file and the edge column."""
from __future__ import annotations

import pandas as pd
import pytest

from mega import odds as O
from mega import vegas as V


def _lines():
    return pd.DataFrame([
        {"gsis_id": "00-A", "player": "A", "team": "MIN", "pos": "WR",
         "market": "player_reception_yds", "line": 68.5, "price_over": -115,
         "price_under": -105},
        {"gsis_id": "00-A", "player": "A", "team": "MIN", "pos": "WR",
         "market": "player_receptions", "line": 4.5, "price_over": -120,
         "price_under": 100},
    ])


def test_the_week_reads_the_published_build_file_when_raw_lines_are_absent(monkeypatch, tmp_path):
    """Streamlit Cloud has no Odds API key: it must still show the column."""
    monkeypatch.setattr(O, "cached", lambda s, w: pd.DataFrame())
    f = tmp_path / "vegas.csv"
    pd.DataFrame([{"gsis_id": "00-A", "player": "A", "team": "MIN", "vegas": 12.3,
                   "vegas_parts": 3, "vegas_filled": 0, "vegas_complete": True}]).to_csv(f, index=False)
    monkeypatch.setattr(V, "build_csv", lambda s, w: f)
    out = V.week(2026, 5)
    assert len(out) == 1 and out["vegas"].iloc[0] == pytest.approx(12.3)


def test_no_props_anywhere_is_an_empty_frame_with_the_right_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "cached", lambda s, w: pd.DataFrame())
    monkeypatch.setattr(V, "build_csv", lambda s, w: tmp_path / "nope.csv")
    out = V.week(2026, 5)
    assert out.empty and list(out.columns) == V.COLS


def test_raw_lines_win_over_a_stale_build_file(monkeypatch, tmp_path):
    """A fresh sweep on this machine must not be shadowed by last week's published file."""
    monkeypatch.setattr(O, "cached", lambda s, w: _lines())
    monkeypatch.setattr(V, "fill_frame", lambda s: pd.DataFrame())
    f = tmp_path / "vegas.csv"
    pd.DataFrame([{"gsis_id": "00-A", "player": "A", "team": "MIN", "vegas": 999.0,
                   "vegas_parts": 3, "vegas_filled": 0, "vegas_complete": True}]).to_csv(f, index=False)
    monkeypatch.setattr(V, "build_csv", lambda s, w: f)
    assert V.week(2026, 5)["vegas"].iloc[0] < 100


def test_the_fill_is_a_per_game_rate_not_a_season_total(monkeypatch):
    """Filling a missing market with a season total would hand a player ten times his week."""
    ffo = pd.DataFrame([
        {"player_id": "00-A", "week": 1, "rec_touchdown_exp": 0.4, "rush_touchdown_exp": 0.1},
        {"player_id": "00-A", "week": 2, "rec_touchdown_exp": 0.6, "rush_touchdown_exp": 0.1},
    ])
    import mega.season as S
    monkeypatch.setattr(S, "ff_opportunity", lambda s: ffo)
    f = V.fill_frame(2026).set_index("gsis_id")
    assert f.loc["00-A", "anytime_td"] == pytest.approx(0.6)     # (0.5+0.7)/2, not 1.2


def test_a_player_with_no_history_gets_no_fill_rather_than_an_average(monkeypatch):
    """A positional average he has no claim to is worse than an honest gap."""
    import mega.season as S
    monkeypatch.setattr(S, "ff_opportunity", lambda s: pd.DataFrame(
        [{"player_id": "00-B", "week": 1, "rec_touchdown_exp": 0.4}]))
    f = V.fill_frame(2026)
    assert "00-A" not in set(f["gsis_id"])


def test_edge_is_vegas_minus_the_model():
    df = pd.DataFrame([{"vegas": 14.0, "proj": 11.5}, {"vegas": 8.0, "proj": 12.0}])
    e = V.edge(df)["vegas_edge"]
    assert e.iloc[0] == pytest.approx(2.5) and e.iloc[1] == pytest.approx(-4.0)


def test_edge_is_blank_rather_than_zero_when_there_is_no_vegas_number():
    """Zero would read as 'the market agrees exactly', which is a claim we cannot make."""
    out = V.edge(pd.DataFrame([{"proj": 11.5}]))
    assert out["vegas_edge"].isna().all()


def test_attach_leaves_a_frame_usable_when_no_props_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "cached", lambda s, w: pd.DataFrame())
    monkeypatch.setattr(V, "build_csv", lambda s, w: tmp_path / "nope.csv")
    df = pd.DataFrame([{"gsis_id": "00-A", "proj": 10.0}])
    out = V.attach(df, 2026, 5)
    assert "vegas" in out.columns and len(out) == 1 and out["proj"].iloc[0] == 10.0
