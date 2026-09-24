"""§5.3-5.5 FAAB maximum, recommended bid and claim plan.

Carries the spec's hand-checkable cases §8 #3 and #4.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega import needs


# ---------------------------------------------------------------- §8 #3
def test_max_bid():
    """FAAB 100, gain 3, pool 11 → $27."""
    assert needs.max_bid(3.0, 100) == 27
    assert needs.max_bid(3.0, 5) == 1        # capped at what is left
    assert needs.max_bid(0.0, 100) == 0
    assert needs.max_bid(3.0, 0) == 0


def test_max_bid_is_a_share_of_what_the_budget_can_buy():
    """The horizon cancels: doubling the budget doubles the ceiling."""
    assert needs.max_bid(2.0, 100) == 2 * needs.max_bid(2.0, 50)


# ---------------------------------------------------------------- §8 #4
def test_recommended_bid_beats_the_best_rival_by_a_dollar():
    """One rival, gain 2, FAAB 60, no ownership movement → bid_t 10.9 → $12."""
    assert needs.rival_bid(2.0, 60) == pytest.approx(10.909, abs=1e-3)
    out = needs.recommend(gain_mine=3.0, faab_mine=100, rivals=[(2.0, 60)])
    assert out["bid"] == 12 and not out["pass"]


def test_a_claim_over_your_ceiling_is_reported_as_a_pass_with_the_number():
    """"Don't bid" is only useful when it says what he was going to go for."""
    out = needs.recommend(gain_mine=0.99, faab_mine=100, rivals=[(2.0, 60)])
    assert out["pass"] and out["bid"] == 0
    assert out["max_bid"] == 9
    assert "$11" in out["why"] or "$10" in out["why"]


def test_an_uncontested_claim_costs_the_league_minimum():
    """Chris confirmed $0 claims are legal, so a player nobody wants is free."""
    out = needs.recommend(gain_mine=3.0, faab_mine=100, rivals=[])
    assert out["bid"] == 0 and not out["pass"]
    assert needs.recommend(3.0, 100, [], min_bid=1)["bid"] == 1


def test_a_rival_with_no_budget_cannot_bid():
    assert needs.rival_bid(5.0, 0) == 0.0
    assert needs.recommend(3.0, 100, [(5.0, 0)])["top_rival"] == 0.0


def test_hype_raises_a_rival_bid_when_everyone_is_adding_him():
    flat = needs.rival_bid(2.0, 60, pct_owned_delta=0)
    hyped = needs.rival_bid(2.0, 60, pct_owned_delta=20)
    assert hyped == pytest.approx(flat * 1.5)
    assert needs.rival_bid(2.0, 60, pct_owned_delta=100) == pytest.approx(flat * 1.5)   # capped


def test_the_top_rival_sets_the_price_not_the_sum_of_them():
    one = needs.recommend(3.0, 100, [(2.0, 60)])["bid"]
    many = needs.recommend(3.0, 100, [(2.0, 60), (1.0, 60), (0.5, 60)])["bid"]
    assert one == many


# ---------------------------------------------------------------- history correction
def test_history_scale_needs_enough_settled_claims():
    settled = pd.DataFrame([{"player": "A", "bid": 20}, {"player": "B", "bid": 30}])
    assert needs.history_scale(settled, {"A": 10, "B": 10}) == 1.0     # only 2 rows


def test_history_scale_learns_what_the_league_actually_pays():
    settled = pd.DataFrame([{"player": p, "bid": 20} for p in "ABCDE"])
    assert needs.history_scale(settled, {p: 10 for p in "ABCDE"}) == pytest.approx(2.0)


def test_history_scale_is_clamped():
    settled = pd.DataFrame([{"player": p, "bid": 200} for p in "ABCDE"])
    assert needs.history_scale(settled, {p: 1 for p in "ABCDE"}) == 2.0


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
