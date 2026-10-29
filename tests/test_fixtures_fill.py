import pandas as pd

from mega.yahoo import _complete_weeks


def test_a_week_one_game_short_gets_its_last_game_from_the_two_teams_left():
    full = [(1, "A", "B"), (1, "C", "D"), (1, "E", "F")]
    short = [(2, "A", "C"), (2, "B", "E")]                    # D and F are the game that didn't parse
    very_short = [(3, "A", "D")]                               # four teams left: can't tell who plays whom
    df = pd.DataFrame(full + short + very_short, columns=["week", "home", "away"])
    out = _complete_weeks(df)
    assert {frozenset(p) for p in out[out["week"] == 2][["home", "away"]].values} == \
        {frozenset(("A", "C")), frozenset(("B", "E")), frozenset(("D", "F"))}
    assert len(out[out["week"] == 3]) == 1 and len(out[out["week"] == 1]) == 3
