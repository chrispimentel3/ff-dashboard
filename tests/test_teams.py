"""Team renames (mega/teams.py). On 2026-10-01 two teams were renamed between the week-3
scores and that morning's standings, and joining by name split each in two."""
from __future__ import annotations

import pandas as pd

from mega import teams, yahoo


def test_a_renamed_team_is_pinned_by_its_manager_not_its_new_name():
    # "Saquon Enjoyer" reads like the old "Barkley's Balls Deep"; it is Alex's old
    # "A Place in the Hampton" (seat 7), which the roster overlap confirmed
    assert teams.seat_for("Saquon Enjoyer", "Alex") == 7
    assert teams.seat_for("Some Brand New Name", "Andrew") == 6
    assert teams.seat_for("A Place in the Hampton") == 7
    assert teams.seat_for("Barkley's Balls Deep") == teams.seat_for("Barkley’s Balls Deep") == 6


def test_old_names_in_older_files_map_to_the_current_ones(monkeypatch):
    monkeypatch.setattr(teams, "_standings_seats", lambda: {"Brand New": 7})
    s = pd.Series(["A Place in the Hampton", "TaylorMade", "Nobody We Know"])
    assert list(teams.canonical(s)) == ["Brand New", "TaylorMade", "Nobody We Know"]


def test_standings_learn_seats_from_managers_for_the_roster_pages_after_them():
    page = "Totally New Name\nAndrew\n2 - 1 - 0 | 1st\n"
    st = yahoo.parse_standings(page)
    assert int(st.loc[0, "seat"]) == 6
    assert teams.seat_for("Totally New Name") == 6


def test_the_scores_and_standings_on_disk_name_the_same_twelve_teams():
    sc, st = yahoo.cached_scores(), yahoo.cached_standings()
    if sc.empty or st.empty:
        return
    assert set(sc["team"]) == set(st["team"]) == set(sc["opponent"])
