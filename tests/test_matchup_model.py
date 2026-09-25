"""Matchup impact model (Sept 2026 handoff): opponent-adjusted defense rating,
shrunk toward a fitted last-season prior, combined with the Vegas implied team
total via fitted exponents. Exercises the two pure cores directly (no network
fetch) with a small synthetic 4-team league, the same style tests/test_roles.py
uses for `mega.roles`.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega import matchup_model as mm

# One position, four teams, simple round numbers throughout.
C = {
    "QB": {"k": 10.0, "rho": 0.5, "b_def": 0.5, "b_impl": 0.5, "b_def_only": 0.6, "gbar": 100.0},
    "RB": {"k": 10.0, "rho": 0.5, "b_def": 0.5, "b_impl": 0.5, "b_def_only": 0.6, "gbar": 100.0},
    "WR": {"k": 10.0, "rho": 0.5, "b_def": 0.5, "b_impl": 0.5, "b_def_only": 0.6, "gbar": 100.0},
    "TE": {"k": 10.0, "rho": 0.5, "b_def": 0.5, "b_impl": 0.5, "b_def_only": 0.6, "gbar": 100.0},
    "_prior_def": {"AAA|QB": 1.0, "BBB|QB": 1.0, "CCC|QB": 1.0, "DDD|QB": 1.0,
                   "AAA|RB": 1.0, "BBB|RB": 1.0, "CCC|RB": 1.0, "DDD|RB": 1.0,
                   "AAA|WR": 1.0, "BBB|WR": 1.0, "CCC|WR": 1.0, "DDD|WR": 1.0,
                   "AAA|TE": 1.0, "BBB|TE": 1.0, "CCC|TE": 1.0, "DDD|TE": 1.0},
}
TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _agg(rows):
    """rows: {(team, pos): (A, B)} -> the (opponent_team, pos)-indexed frame
    `_defense_ratings_core` expects."""
    idx = pd.MultiIndex.from_tuples(list(rows.keys()), names=["opponent_team", "pos"])
    return pd.DataFrame({"A": [v[0] for v in rows.values()], "B": [v[1] for v in rows.values()]}, index=idx)


def test_defense_ratings_center_on_one_per_position():
    """League-average defense = 1.00 exactly, by construction (D / D.mean())."""
    agg = _agg({("AAA", "QB"): (150.0, 100.0), ("BBB", "QB"): (80.0, 100.0),
                ("CCC", "QB"): (100.0, 100.0), ("DDD", "QB"): (100.0, 100.0)})
    out = mm._defense_ratings_core(agg, C, TEAMS)
    assert out[out["pos"] == "QB"]["rating"].mean() == pytest.approx(1.0, abs=1e-3)


def test_a_defense_that_allowed_more_than_baseline_rates_tougher_to_face():
    """AAA allowed 150 vs a 100 baseline (rough +50% game); with no prior gap and a
    real in-season sample outweighing the flat prior, AAA should rate above 1.0 and
    above every team that only allowed exactly its baseline."""
    agg = _agg({("AAA", "QB"): (150.0, 100.0), ("BBB", "QB"): (100.0, 100.0),
                ("CCC", "QB"): (100.0, 100.0), ("DDD", "QB"): (100.0, 100.0)})
    out = mm._defense_ratings_core(agg, C, TEAMS)
    out = out[out["pos"] == "QB"].set_index("defense")
    assert out.loc["AAA", "rating"] > out.loc["BBB", "rating"]
    assert out.loc["AAA", "rank"] == 1   # rank 1 = toughest (allows the most)


def test_no_in_season_sample_falls_back_fully_to_the_prior():
    """Zero games played (A=B=0 for everyone): weight_this_season is 0 and every
    team's rating is driven entirely by its prior — here a flat prior, so every
    team lands back on exactly 1.00."""
    agg = _agg({("AAA", "QB"): (0.0, 0.0), ("BBB", "QB"): (0.0, 0.0),
                ("CCC", "QB"): (0.0, 0.0), ("DDD", "QB"): (0.0, 0.0)})
    out = mm._defense_ratings_core(agg, C, TEAMS)
    assert (out[out["pos"] == "QB"]["weight_this_season"] == 0).all()
    assert out[out["pos"] == "QB"]["rating"].apply(lambda v: v == pytest.approx(1.0)).all()


def test_combine_multiplies_fitted_rating_and_vegas_powers():
    """mult = rating^b_def * impl_rel^b_impl when a line exists — hand-computed."""
    D = pd.DataFrame([{"defense": "BBB", "pos": "RB", "rating": 1.21, "rank": 3}])
    tg = pd.DataFrame([{"week": 4, "team": "AAA", "opp": "BBB", "impl_rel": 1.0816, "played": False}])
    out = mm._week_matchups_core(D, tg, C, from_week=4)
    row = out.iloc[0]
    expected_mult = 1.21 ** 0.5 * 1.0816 ** 0.5
    assert row["mult"] == pytest.approx(round(expected_mult, 3))
    assert row["basis"] == "defense+vegas"
    assert row["vegas_pct"] == pytest.approx(round((1.0816 ** 0.5 - 1) * 100, 1))


def test_missing_line_falls_back_to_defense_only():
    """A future week with no posted Vegas line yet (impl_rel is NaN) uses
    rating**b_def_only instead of guessing a Vegas component."""
    D = pd.DataFrame([{"defense": "CCC", "pos": "WR", "rating": 0.9, "rank": 20}])
    tg = pd.DataFrame([{"week": 9, "team": "DDD", "opp": "CCC", "impl_rel": float("nan"), "played": False}])
    out = mm._week_matchups_core(D, tg, C, from_week=9)
    row = out.iloc[0]
    assert row["basis"] == "defense_only"
    assert row["vegas_pct"] is None
    assert row["mult"] == pytest.approx(round(0.9 ** 0.6, 3))


def test_already_played_games_are_excluded():
    D = pd.DataFrame([{"defense": "AAA", "pos": "TE", "rating": 1.0, "rank": 16}])
    tg = pd.DataFrame([{"week": 2, "team": "BBB", "opp": "AAA", "impl_rel": 1.0, "played": True},
                       {"week": 5, "team": "BBB", "opp": "AAA", "impl_rel": 1.0, "played": False}])
    out = mm._week_matchups_core(D, tg, C, from_week=2)
    assert list(out["week"]) == [5]
