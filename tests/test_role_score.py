"""Role score and flag wording (mega/needs.py)."""
from __future__ import annotations

import pytest

from mega import needs


# ---------------------------------------------------------------- role scoring
def test_role_score_discounts_a_spike():
    sustained = needs.role_score({"flags": ["TGT"], "tags": {"TGT": "sustained"}})
    spike = needs.role_score({"flags": ["TGT"], "tags": {"TGT": "spike"}})
    assert sustained == pytest.approx(needs.ROLE_BONUS["TGT"])
    assert spike == pytest.approx(needs.ROLE_BONUS["TGT"] * needs.SPIKE)
    assert spike < sustained


def test_role_minus_is_a_penalty():
    assert needs.role_score({"flags": ["ROLE-"], "tags": {}}) < 0


def test_role_plus_outweighs_any_single_usage_flag():
    plus = needs.role_score({"flags": ["ROLE+"], "tags": {}})
    assert all(plus > needs.ROLE_BONUS[f] for f in ("TGT", "AIR", "SNAP", "GL"))


def test_no_role_context_scores_nothing_rather_than_failing():
    assert needs.role_score({}) == 0.0
    assert needs.role_score(None) == 0.0
    assert needs._flag_text(None) == ""


def test_flag_text_is_plain_english_and_marks_the_one_off():
    """The board, the tables and the player card all word flags through mega.glossary,
    so "TGT" never reaches a human. A one-game flag says so; anything else held."""
    txt = needs._flag_text({"flags": ["TGT", "GL"],
                            "tags": {"TGT": "sustained", "GL": "spike"}})
    assert txt == "target hog, goal line (1 game)"


def test_role_text_is_plain_english_too():
    assert needs._role_text({"role": "COMMITTEE"}) == "Committee back"
    assert needs._role_text({}) == ""
