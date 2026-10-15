import pandas as pd

from mega import prop_moves as PM


def _sweep(rows):
    return pd.DataFrame([{"gsis_id": g, "market": m, "line": l, "player": g, "pos": "RB", "team": "X"} for g, m, l in rows])


def test_a_role_change_is_a_real_jump_or_a_first_volume_line(monkeypatch):
    weeks = {
        5: _sweep([("lead", "player_rush_yds", 40.0), ("steady", "player_rush_yds", 60.0),
                   ("backup", "player_anytime_td", None), ("starter", "player_rush_yds", 70.0)]),
        6: _sweep([("lead", "player_rush_yds", 70.0),          # +75%, +30 yds: UP
                   ("steady", "player_rush_yds", 64.0),        # +7%: nothing
                   ("backup", "player_rush_yds", 55.0),        # priced before, no volume line: NEW
                   ("starter", "player_receptions", 3.5),      # had a volume line already: coverage
                   ("ghost", "player_rush_yds", 80.0)]),       # never covered before: not a change
    }
    monkeypatch.setattr(PM, "_raw", lambda s, w: weeks.get(w, pd.DataFrame()))
    m = {x["gsis_id"]: x["kind"] for x in PM.movers(2026, 6)}
    assert m == {"lead": "UP", "backup": "NEW"}
