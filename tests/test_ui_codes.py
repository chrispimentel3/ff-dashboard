"""The column vocabulary: one code, one meaning, one wording.

`mega/ui.py` maps raw column names to short codes, codes to headers, and codes to the
tooltip that doubles as the legend. A duplicate literal key in any of those dicts is
invisible at runtime — Python keeps the last one and says nothing — so the only way to
catch it is to read the source.

It had gone wrong three times. `opp_score`, `opp` and `opponent` all mapped to `OPP`, and
the opportunity-score entry sat last, so the next-opponent column on the Roster, Start/Sit
and Matchups tables rendered under the header **"Opportunity"** with a tooltip about
earning your role, above values like "vs ATL". `arch_fit` and `fit` collided on `FIT` the
same way, which put "0-100" help on a column whose values are UPGRADE / DEPTH / STASH.
"""
from __future__ import annotations

import ast
import collections
import pathlib

import pytest

from mega import ui

UI = pathlib.Path(__file__).resolve().parent.parent / "mega" / "ui.py"
DICTS = ("COLS", "LABELS", "GLOSS", "_WIDTH", "POS_COLORS")


def _literal_keys(name: str) -> list[str]:
    """Keys as WRITTEN, before Python collapses the duplicates."""
    tree = ast.parse(UI.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    raise AssertionError(f"{name} not found as a module-level dict literal")


@pytest.mark.parametrize("name", DICTS)
def test_no_key_is_written_twice(name):
    keys = _literal_keys(name)
    dupes = {k: n for k, n in collections.Counter(keys).items() if n > 1}
    assert not dupes, (
        f"{name} defines these keys more than once, and the last one silently wins: "
        + ", ".join(sorted(dupes))
    )


# Raw columns that legitimately share a code: the SAME quantity arriving under different
# names from different sources. That is what COLS exists for. The list is exhaustive on
# purpose — a code that starts carrying a second, different quantity fails the test below
# until someone adds it here deliberately, which is the review step `opp_score` never got.
SYNONYMS = {
    "1D": {"fd", "receiving_first_downs"},
    "CAR": {"carries_pg", "carry_pg"},
    "FAAB": {"faab_balance", "faab_left"},
    "G": {"games", "gms"},
    "IMP": {"implied", "implied_pts"},
    "PLAYER": {"name", "player"},
    "POS": {"pos", "give_pos", "get_pos"},        # two sides of one trade row
    "PPG": {"half_ppr_pg", "pg_recent", "ppg"},
    "PROJ": {"proj", "proj_ppg"},
    "PTS": {"half_ppr", "points"},
    "ST": {"nfl_status", "report_status"},
    "TGT%": {"tgt_pct", "tgt_share"},
    "TM": {"nfl_team", "team", "team_2026_nfl"},
    "xFP": {"expected", "xfp", "xfp_tot"},
    "xFP±": {"diff", "vs_exp", "xfp_diff"},
    "xFP±/G": {"diff_pg", "per_g"},
}


def test_no_code_quietly_acquires_a_second_meaning():
    """Two raw columns may share a code only when they are the same quantity.

    The three that broke this were an opportunity score, an NFL opponent and a fantasy
    opponent, all mapped to OPP. Whichever LABELS/GLOSS entry is written last wins, so the
    other two are mislabelled on screen with nothing to indicate it.
    """
    by_code = collections.defaultdict(set)
    for raw, code in ui.COLS.items():
        by_code[code].add(raw)
    for code, raws in sorted(by_code.items()):
        if len(raws) > 1:
            assert raws == SYNONYMS.get(code), (
                f"{code} maps from {sorted(raws)}. If those are the same quantity, add them "
                f"to SYNONYMS. If they are not, give one of them its own code — otherwise "
                f"the header and tooltip will be wrong for all but one of them."
            )


def test_the_three_codes_that_collided_are_now_distinct():
    """Pinning the actual bug, by its symptom."""
    assert ui.LABELS["OPP"] == "Next opp"
    assert "opponent" in ui.GLOSS["OPP"]
    assert ui.LABELS["OPP SCORE"] == "Opportunity"
    assert ui.COLS["opp_score"] == "OPP SCORE"
    assert ui.COLS["opp"] == "OPP"
    assert ui.COLS["opponent"] == "VS"
    assert ui.COLS["arch_fit"] == "ARCH FIT"
    assert ui.COLS["fit"] == "FIT"


def test_the_two_luck_columns_no_longer_read_the_same():
    """They sit in the same tab, a few inches apart, and mean different things: one is
    wins minus expected wins, the other is power rank minus standing. Both said 'Luck'."""
    assert ui.LABELS["LUCK W"] != ui.LABELS["LUCK"]
    assert "luck" not in ui.LABELS["LUCK W"].lower()
    assert "luck" not in ui.LABELS["LUCK"].lower()


def test_every_glossed_code_has_a_header():
    missing = sorted(c for c in ui.GLOSS if c not in ui.LABELS)
    assert not missing, f"codes with a tooltip but no header: {missing}"


def test_current_slot_and_recommended_slot_are_not_the_same_word():
    """`slot` is where Yahoo has him; `lineup` is where the optimiser says to play him.
    Both rendered as "Slot", so on Start/Sit a recommendation read as current state."""
    assert ui.COLS["slot"] == "SLOT" and ui.COLS["lineup"] == "PLAY AS"
    assert ui.LABELS["SLOT"] != ui.LABELS["PLAY AS"]
    assert "recommendation" in ui.GLOSS["PLAY AS"]
    assert "Yahoo" in ui.GLOSS["SLOT"]


def test_text_columns_are_not_right_aligned():
    """A categorical column right-aligned among numbers reads as a figure."""
    for code in ("FIT", "VS", "OPP"):
        assert code in ui._LEFT, f"{code} holds text and must be left-aligned"
