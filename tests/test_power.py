import itertools

import pandas as pd
import pytest

from mega import power


def _league():
    """Lines generated from known ratings: base 22, home field 2, every pairing over 8 weeks."""
    off = {"A": 4.0, "B": 1.0, "C": -1.0, "D": -4.0}
    dfn = {"A": 2.0, "B": -1.0, "C": 0.0, "D": -1.0}
    rows = []
    for wk, (h, a) in enumerate(itertools.permutations(off, 2)):
        hi = 22 + 1 + off[h] - dfn[a]
        ai = 22 - 1 + off[a] - dfn[h]
        rows.append({"home_team": h, "away_team": a, "week": 1 + wk % 8, "game_type": "REG",
                     "total_line": hi + ai, "spread_line": hi - ai})
    return pd.DataFrame(rows), off, dfn


def test_ratings_recover_the_lines_they_came_from():
    g, off, dfn = _league()
    m = power.fit(g, half_life=float("inf"), lam=1e-6)
    assert m["hfa"] == pytest.approx(2.0, abs=1e-3)
    tot, sp = power.predict(m, "A", "D")
    assert tot == pytest.approx((23 + 4 + 1) + (21 - 4 - 2), abs=1e-3)
    assert sp == pytest.approx((23 + 4 + 1) - (21 - 4 - 2), abs=1e-3)


def test_fill_prices_only_the_games_without_a_line():
    g, _, _ = _league()
    future = pd.DataFrame([{"home_team": "D", "away_team": "A", "week": 15, "game_type": "REG",
                            "total_line": None, "spread_line": None}])
    out = power.fill(pd.concat([g, future], ignore_index=True))
    assert (out["line_source"] == "posted").sum() == len(g)
    last = out.iloc[-1]
    assert last["line_source"] == "model" and last["spread_line"] < 0      # A is better on the road
    assert out.iloc[:len(g)]["total_line"].tolist() == g["total_line"].tolist()   # posted lines untouched
