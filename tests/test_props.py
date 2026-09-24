"""§20 Vegas props → fantasy points.

The arithmetic here is hand-checkable, and the three corrections it makes (median→mean,
anytime-TD→expected-TD, Poisson for counts) are each pinned separately so a regression
says which one broke.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mega import props as P


# ---------------------------------------------------------------- prices
@pytest.mark.parametrize("odds,expect", [(-110, 0.5238), (100, 0.5000), (150, 0.4000),
                                         (-200, 0.6667), (250, 0.2857)])
def test_american_prices_convert(odds, expect):
    assert P.american_to_prob(odds) == pytest.approx(expect, abs=1e-4)


def test_devig_splits_the_hold_proportionally():
    """-110 both ways is a 4.76% hold and a fair coin, not 0.524."""
    p = P.devig(P.american_to_prob(-110), P.american_to_prob(-110))
    assert p == pytest.approx(0.5, abs=1e-12)


def test_devig_keeps_a_real_lean():
    """-140/+110 is genuinely a favourite — de-vigging must not flatten it to 0.5."""
    p = P.devig(P.american_to_prob(-140), P.american_to_prob(110))
    assert 0.54 < p < 0.57
    assert p < P.american_to_prob(-140)      # but below the vigged number


def test_a_one_sided_price_is_haircut_not_trusted():
    raw = P.american_to_prob(-120)
    assert P.devig(raw, float("nan")) == pytest.approx(raw - P.ONE_SIDED_HOLD)


def test_no_price_at_all_is_nan_not_a_guess():
    assert math.isnan(P.devig(float("nan"), float("nan")))


# ---------------------------------------------------------------- counts (Poisson)
def test_count_inversion_round_trips():
    """over 4.5 at a fair 0.55 → the lambda whose P(X>=5) really is 0.55."""
    lam = P.mean_from_count(4.5, 0.55)
    assert P._pois_sf(4, lam) == pytest.approx(0.55, abs=1e-6)


def test_a_coin_flip_line_lands_near_the_line():
    """Receptions o/u 4.5 at even money → mean just above 4.5, because the Poisson median
    sits a shade under its mean."""
    lam = P.mean_from_count(4.5, 0.5)
    assert 4.5 < lam < 5.2


def test_count_markets_skip_the_skew_correction():
    """Receptions are Poisson, not skewed. Applying the yardage ratio on top would inflate
    every reception line by 3-14% and double-count the same effect."""
    lam = P.mean_from_count(4.5, 0.5)
    assert lam == pytest.approx(P.mean_from_count(4.5, 0.5))
    assert lam < 4.5 * 1.06            # nowhere near the receptions skew ratio


def test_pass_td_line_inverts_to_a_sane_lambda():
    """o/u 1.5 pass TDs at -115/-105 → about 1.6, which is a normal starting QB."""
    p = P.devig(P.american_to_prob(-115), P.american_to_prob(-105))
    lam = P.mean_from_count(1.5, p)
    assert 1.4 < lam < 1.8


# ---------------------------------------------------------------- yards (skew)
def test_a_low_receiving_line_projects_well_above_itself():
    """30.5 yards at even money is a ~39-yard expectation: the line is the median and the
    distribution has a long right tail. Reading the line as the projection is the single
    biggest error this module exists to fix."""
    m = P.mean_from_yards(30.5, 0.5, "rec_yds")
    assert m == pytest.approx(30.5 * 1.293, rel=0.02)
    assert m > 38.0


def test_the_skew_shrinks_as_the_line_rises():
    lo = P.mean_from_yards(30.0, 0.5, "rec_yds") / 30.0
    hi = P.mean_from_yards(82.0, 0.5, "rec_yds") / 82.0
    assert lo > hi > 1.0


def test_passing_yards_are_not_skewed():
    """QB yardage is symmetric at every level measured, so the line IS the projection."""
    assert P.mean_from_yards(245.5, 0.5, "pass_yds") == pytest.approx(245.5, rel=1e-6)


def test_a_lopsided_price_moves_the_centre_the_right_way():
    """Over juiced to -140 means the true middle is above the posted number."""
    p_hi = P.devig(P.american_to_prob(-140), P.american_to_prob(110))
    p_lo = P.devig(P.american_to_prob(110), P.american_to_prob(-140))
    assert P.mean_from_yards(50.5, p_hi, "rec_yds") > P.mean_from_yards(50.5, 0.5, "rec_yds")
    assert P.mean_from_yards(50.5, p_lo, "rec_yds") < P.mean_from_yards(50.5, 0.5, "rec_yds")


def test_interp_is_held_flat_beyond_the_fitted_range():
    """Extrapolating the skew slope past the last anchor sends the ratio under 1, which
    would project a 150-yard line below its own number."""
    assert P._interp(500.0, P.SKEW["rec_yds"]) == P.SKEW["rec_yds"][-1][1]
    assert P._interp(1.0, P.SKEW["rec_yds"]) == P.SKEW["rec_yds"][0][1]
    assert P.mean_from_yards(150.0, 0.5, "rec_yds") > 150.0


# ---------------------------------------------------------------- touchdowns
def test_expected_td_uses_the_measured_multiplier():
    """p=0.2983 → 0.2983 x 1.1531 measured-at-that-level."""
    assert P.expected_td(0.2983) == pytest.approx(0.2983 * 1.1531, rel=1e-6)


def test_poisson_would_overstate_the_best_scorers():
    """The reason TD_MULT is fitted rather than assumed: at the top of the range Poisson
    hands half a point to exactly the players already starting."""
    p = 0.5574
    assert P.poisson_td(p) > P.expected_td(p)
    assert P.poisson_td(p) / p == pytest.approx(1.462, abs=0.01)   # Poisson's multiplier
    assert P.expected_td(p) / p == pytest.approx(1.3056, abs=1e-3)  # the measured one
    gap = (P.poisson_td(p) - P.expected_td(p)) * 6.0
    assert 0.4 < gap < 0.7              # fantasy points of pure model error


def test_anytime_td_beats_the_naive_reading():
    """6 x p ignores two-touchdown games entirely."""
    p = 0.40
    assert P.expected_td(p) > p


def test_expected_td_is_monotone_and_bounded():
    xs = [P.expected_td(p) for p in np.linspace(0.01, 0.95, 40)]
    assert all(b >= a for a, b in zip(xs, xs[1:]))
    assert P.expected_td(0.0) == 0.0


# ---------------------------------------------------------------- assembly
def _lines(pos: str = "WR", **kw) -> pd.DataFrame:
    rows = []
    for market, (line, po, pu) in kw.items():
        rows.append({"gsis_id": "00-X", "player": "Test Guy", "team": "MIN", "pos": pos,
                     "market": market, "line": line, "price_over": po, "price_under": pu})
    return pd.DataFrame(rows)


def test_a_full_receiver_line_scores_end_to_end():
    lines = _lines("WR", player_reception_yds=(68.5, -115, -105),
                   player_receptions=(4.5, -120, 100),
                   player_anytime_td=(None, 150, -190))
    means = P.to_means(lines)
    out = P.project(means).iloc[0]
    by = means.set_index("component")["mean"]
    want = (0.10 * by["rec_yds"] + 0.5 * by["rec"] + 6.0 * by["anytime_td"])
    assert out["vegas"] == pytest.approx(round(want, 2), abs=0.01)
    assert out["vegas_parts"] == 3 and bool(out["vegas_complete"])


def test_books_are_reduced_by_median_not_by_first_seen():
    """One stale book must not drag a projection."""
    rows = [{"gsis_id": "00-X", "player": "A", "team": "MIN",
             "market": "player_reception_yds", "line": L, "price_over": -110,
             "price_under": -110} for L in (60.5, 61.5, 95.5)]
    m = P.to_means(pd.DataFrame(rows)).iloc[0]
    assert m["line"] == pytest.approx(61.5)       # median, not 72.5 and not 60.5
    assert m["books"] == 3


def test_a_missing_market_is_filled_and_declared():
    """A receiver with yards and catches but no touchdown market: scoring his TDs as zero
    would dock him ~2.5 points and still look precise."""
    lines = _lines(player_reception_yds=(68.5, -115, -105),
                   player_receptions=(4.5, -120, 100))
    means = P.to_means(lines)
    bare = P.project(means).iloc[0]
    fill = pd.DataFrame([{"gsis_id": "00-X", "anytime_td": 0.42}])
    full = P.project(means, fill=fill).iloc[0]
    assert full["vegas"] - bare["vegas"] == pytest.approx(6.0 * 0.42, abs=0.01)
    assert full["vegas_filled"] == 1 and not bool(full["vegas_complete"])
    assert bool(bare["vegas_complete"]) is False or bare["vegas_parts"] == 2


def test_fill_never_overrides_a_real_posted_line():
    lines = _lines(player_anytime_td=(None, 150, -190))
    means = P.to_means(lines)
    a = P.project(means).iloc[0]["vegas"]
    b = P.project(means, fill=pd.DataFrame([{"gsis_id": "00-X", "anytime_td": 99.0}])).iloc[0]["vegas"]
    assert a == pytest.approx(b)


def test_a_quarterback_scores_on_the_passing_weights():
    lines = _lines("QB", player_pass_yds=(245.5, -110, -110),
                   player_pass_tds=(1.5, -115, -105),
                   player_pass_interceptions=(0.5, 105, -130),
                   player_rush_yds=(22.5, -110, -110))
    means = P.to_means(lines).set_index("component")["mean"]
    out = P.project(P.to_means(lines)).iloc[0]
    want = (0.04 * means["pass_yds"] + 4.0 * means["pass_tds"]
            - 1.0 * means["pass_int"] + 0.10 * means["rush_yds"])
    assert out["vegas"] == pytest.approx(round(want, 2), abs=0.01)
    assert means["pass_yds"] == pytest.approx(245.5, rel=1e-6)   # no skew on pass yards


def test_empty_input_is_an_empty_frame_not_a_crash():
    assert P.to_means(pd.DataFrame()).empty
    assert P.project(pd.DataFrame()).empty
    assert list(P.project(pd.DataFrame()).columns)[:2] == ["gsis_id", "player"]


def test_unknown_markets_are_dropped_rather_than_scored():
    lines = pd.DataFrame([{"gsis_id": "00-X", "player": "A", "team": "MIN",
                           "market": "player_kicking_points", "line": 7.5,
                           "price_over": -110, "price_under": -110}])
    assert P.to_means(lines).empty


# ---------------------------------------------------------------- position routing
def test_a_quarterbacks_rushing_line_is_not_read_as_a_running_backs():
    """Scrambles arrive more steadily than committee carries, so QB rushing yards are far
    less skewed. Routing a rushing QB through the RB curve invented about a point at a
    24.5-yard line — on the exact position where the rushing floor is why he starts."""
    qb = P.mean_from_yards(24.5, 0.5, "rush_yds", "QB")
    rb = P.mean_from_yards(24.5, 0.5, "rush_yds", "RB")
    assert rb > qb
    assert qb == pytest.approx(24.5 * 1.174, rel=0.03)
    assert (rb - qb) * 0.10 > 0.5          # fantasy points of pure position error


def test_an_unknown_position_falls_back_to_the_general_curve():
    assert P.mean_from_yards(40.0, 0.5, "rush_yds", "") == P.mean_from_yards(40.0, 0.5, "rush_yds")


def test_only_rushing_yards_are_position_split():
    """Receiving yards were checked for running backs and match the receiver curve (1.31
    measured against 1.29 fitted), so a second curve there would be noise, not signal."""
    for m in ("rec_yds", "pass_yds"):
        assert P.mean_from_yards(60.0, 0.5, m, "QB") == P.mean_from_yards(60.0, 0.5, m, "RB")


def test_position_reaches_the_maths_through_to_means():
    rows = [{"gsis_id": "00-Q", "player": "Rushing QB", "team": "BAL", "pos": "QB",
             "market": "player_rush_yds", "line": 24.5, "price_over": -110,
             "price_under": -110},
            {"gsis_id": "00-R", "player": "Some RB", "team": "BAL", "pos": "RB",
             "market": "player_rush_yds", "line": 24.5, "price_over": -110,
             "price_under": -110}]
    m = P.to_means(pd.DataFrame(rows)).set_index("gsis_id")["mean"]
    assert m["00-Q"] < m["00-R"]


# ---------------------------------------------------------------- completeness
def test_a_quarterback_is_complete_without_a_receptions_market():
    """Books never post receptions for a QB. Counting that absence as a gap marked every
    quarterback patched — on the first real sweep it flagged 332 of 420 players as filled
    in when almost none were, which made the flag worth nothing."""
    lines = _lines("QB", player_pass_yds=(245.5, -110, -110),
                   player_pass_tds=(1.5, -115, -105),
                   player_pass_interceptions=(0.5, 105, -130),
                   player_rush_yds=(33.5, -110, -110),
                   player_anytime_td=(None, 260, -340))
    out = P.project(P.to_means(lines)).iloc[0]
    assert bool(out["vegas_complete"]) and out["vegas_filled"] == 0


def test_a_back_missing_his_receiving_markets_is_not_complete():
    """The real case that must still be caught: a receiving back priced on carries alone."""
    lines = _lines("RB", player_rush_yds=(55.5, -110, -110),
                   player_anytime_td=(None, -155, 130))
    out = P.project(P.to_means(lines)).iloc[0]
    assert not bool(out["vegas_complete"])


def test_an_unknown_position_expects_everything():
    """Conservative on purpose: if we cannot say what he plays, we cannot say his card
    is full."""
    lines = _lines("", player_reception_yds=(68.5, -115, -105))
    assert not bool(P.project(P.to_means(lines)).iloc[0]["vegas_complete"])


def test_a_market_outside_his_position_is_still_scored():
    """Receivers carry rushing lines on jet sweeps. Relevance decides what gets filled and
    what counts as complete — never what counts."""
    lines = _lines("WR", player_reception_yds=(68.5, -115, -105),
                   player_receptions=(4.5, -120, 100),
                   player_anytime_td=(None, 150, -190),
                   player_rush_yds=(12.5, -110, -110))
    m = P.to_means(lines)
    out = P.project(m).iloc[0]
    rush = m.set_index("component")["mean"]["rush_yds"]
    bare = P.project(m[m["component"] != "rush_yds"]).iloc[0]
    assert out["vegas"] - bare["vegas"] == pytest.approx(0.10 * rush, abs=0.01)
    assert bool(out["vegas_complete"])


def test_relevant_markets_are_defined_for_every_scoring_position():
    for pos in ("QB", "RB", "WR", "TE"):
        assert P.relevant_for(pos)
        assert set(P.relevant_for(pos)) <= set(P.WEIGHTS)
