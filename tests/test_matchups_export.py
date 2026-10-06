"""The matchup export must survive a player with no verdict (a bye week)."""
import pandas as pd

from mega import matchups


def test_a_missing_verdict_is_blank_not_a_crash_even_in_an_arrow_backed_column():
    df = pd.DataFrame({"player": ["A", "B"], "verdict": pd.array(["🟢 great", None], dtype="string[pyarrow]")})
    assert [r["verdict"] for r in matchups._clean_difficulty(df)] == ["great", ""]
