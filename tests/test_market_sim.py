"""§6.2 perceived market, §17 manager bias, §18 win probability.

Carries the spec's hand-checkable cases §19.3 #19, #22, #23 and §8 #10.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mega import market as mk
from mega import sim
from mega import trade_engine as te


# ---------------------------------------------------------------- §6.2
def test_perceived_rank_blends_consensus_with_the_box_score():
    """wb = 0.40 * min(g,3)/3. At g=3: 0.6*ecr + 0.4*box."""
    ecr = pd.DataFrame([{"gsis_id": "A", "ecr_ros": 10.0}])
    box = pd.DataFrame([{"gsis_id": "A", "box_rank": 60.0, "games": 3}])
    got = mk.perceived(ecr, box)["perceived_rank"].iloc[0]
    assert got == pytest.approx(0.6 * 10 + 0.4 * 60)


def test_early_in_a_season_the_consensus_is_all_anyone_has():
    ecr = pd.DataFrame([{"gsis_id": "A", "ecr_ros": 10.0}])
    box = pd.DataFrame([{"gsis_id": "A", "box_rank": 60.0, "games": 0}])
    assert mk.perceived(ecr, box)["perceived_rank"].iloc[0] == pytest.approx(10.0)


def test_a_missing_side_hands_the_whole_weight_to_the_other():
    ecr = pd.DataFrame([{"gsis_id": "A", "ecr_ros": np.nan}])
    box = pd.DataFrame([{"gsis_id": "A", "box_rank": 44.0, "games": 3}])
    assert mk.perceived(ecr, box)["perceived_rank"].iloc[0] == pytest.approx(44.0)


def test_a_player_nobody_ranks_is_unranked_not_free():
    ecr = pd.DataFrame([{"gsis_id": "A", "ecr_ros": np.nan}])
    box = pd.DataFrame([{"gsis_id": "A", "box_rank": np.nan, "games": 0}])
    assert mk.perceived(ecr, box)["perceived_rank"].iloc[0] == mk.UNRANKED


def test_box_rank_needs_two_games():
    pw = pd.DataFrame([{"gsis_id": "A", "pos": "WR", "week": 1, "half_ppr": 30.0},
                       {"gsis_id": "B", "pos": "WR", "week": 1, "half_ppr": 5.0},
                       {"gsis_id": "B", "pos": "WR", "week": 2, "half_ppr": 5.0}])
    got = set(mk.box_rank(pw)["gsis_id"])
    assert got == {"B"}          # A's one huge game does not earn him a rank


# ---------------------------------------------------------------- §19.3 #19
def test_manager_bias_flips_a_market_flag():
    """Give RB mv 35, get WR mv 45, tolerance 0.15. Neutral 35/45 = 0.778 fails;
    with Jamie's RB multiplier of 1.15 it is 40.25/45 = 0.894, which passes."""
    cfg = te.merge(te.DEFAULTS, {"market": {"tolerance": 0.15}})
    rb = {"pos": "RB", "ecr": 1}
    neutral = te.market_value(rb, cfg)
    scale = 35.0 / neutral
    give = {"pos": "RB", "ecr": 1}

    def mv(p, bias=None):
        return te.market_value(p, cfg, bias) * scale

    assert mv(give) == pytest.approx(35.0)
    assert mv(give, {"RB": 1.15}) == pytest.approx(40.25)
    assert 35.0 / 45.0 == pytest.approx(0.778, abs=1e-3)
    assert 40.25 / 45.0 == pytest.approx(0.894, abs=1e-3)
    assert (35.0 / 45.0) < 1 - 0.15 <= (40.25 / 45.0)


def test_no_bias_is_exactly_the_v1_function():
    """The eleven ported JS tests depend on this staying true."""
    cfg = te.DEFAULTS
    p = {"pos": "RB", "ecr": 12}
    assert te.market_value(p, cfg) == te.market_value(p, cfg, {})
    assert te.market_value(p, cfg) == te.market_value(p, cfg, None)


def test_seeds_do_nothing_until_managers_are_mapped():
    """Guessing which manager owns which team would put a silent thumb on every trade."""
    assert mk.seed_bias({}) == {}
    assert mk.seed_bias({"Jamie": "deez nuts"}) == {"deez nuts": {"RB": 1.15}}


def test_draft_reach_becomes_a_multiplier():
    """Taking a back 8 picks before the field did: 1 + 8/40 = 1.20."""
    picks = pd.DataFrame([{"team": "T", "pos": "RB", "round": 2, "pick": 14, "adp": 22},
                          {"team": "T", "pos": "RB", "round": 3, "pick": 26, "adp": 34}])
    assert mk.draft_bias(picks)["T"]["RB"] == pytest.approx(1.20)


def test_bias_multipliers_are_clamped():
    picks = pd.DataFrame([{"team": "T", "pos": "RB", "round": 1, "pick": 1, "adp": 200}])
    assert mk.draft_bias(picks)["T"]["RB"] == pytest.approx(mk.BIAS_CLAMP[1])


def test_faab_bias_ignores_a_team_that_has_barely_spent():
    bids = pd.DataFrame([{"team": "T", "pos": "RB", "bid": 3},
                         {"team": "U", "pos": "RB", "bid": 40},
                         {"team": "U", "pos": "WR", "bid": 10}])
    out = mk.faab_bias(bids)
    assert "T" not in out and "U" in out


def test_fitted_evidence_replaces_the_seed_rather_than_averaging_with_it():
    picks = pd.DataFrame([{"team": "deez nuts", "pos": "RB", "round": 1, "pick": 1, "adp": 5}])
    out = mk.bias(picks=picks, managers={"Jamie": "deez nuts"})
    assert out["deez nuts"]["RB"] == pytest.approx(1.10)      # 1 + 4/40, not blended to 1.15


# ---------------------------------------------------------------- §19.3 #23
def test_normal_model_agrees_with_the_dashboard_logistic():
    """Two different answers to the same question on two screens is worse than one
    imperfect answer, so this has to hold or we use the dashboard's function.

    The 1.7 exists precisely to make a logistic approximate a normal CDF — see the
    mega/xwins.py docstring — so agreement is the expected result, not a coincidence."""
    assert sim.win_prob_normal(0, 25.0) == pytest.approx(0.5)
    assert sim.win_prob_logistic(0, 25.0) == pytest.approx(0.5)
    for sigma in (18.0, 25.0, 30.0, 40.0):
        spread = sim.diff_spread(sigma)
        bad = sim.check_against(spread, gaps=(0, 5, 10, 20, 30))
        assert not bad, (sigma, bad)


def test_the_17_divides_the_spread_not_the_gap():
    """Dividing the gap instead makes a five-point win a 95% certainty. That was my first
    version of this, and §18.1's cross-check is what caught it."""
    spread = sim.diff_spread(25.0)
    assert sim.win_prob_logistic(5.0, spread) < 0.60
    assert sim.win_prob_logistic(spread, spread) == pytest.approx(
        1 / (1 + math.exp(-1.7)), abs=1e-9)


def test_check_against_reports_disagreement_rather_than_hiding_it():
    """A wrong divisor must fail loudly rather than pass quietly."""
    spread = sim.diff_spread(25.0)
    assert sim.check_against(spread, gaps=(10, 20), divisor=0.4)


# ---------------------------------------------------------------- §19.3 #22
def _season() -> sim.Season:
    teams = [f"t{i}" for i in range(1, 7)]
    sched = [{"week": w, "home": teams[i], "away": teams[-1 - i]}
             for w in (10, 11, 12) for i in range(3)]
    means = {(t, w): 100.0 + 5 * i for i, t in enumerate(teams) for w in (10, 11, 12)}
    return sim.Season(teams=teams, means=means, schedule=sched,
                      wins={t: 5 for t in teams}, points_for={t: 900.0 for t in teams},
                      sigma=25.0, playoff_teams=3, byes=1)


def test_identical_rosters_give_exactly_zero():
    """Common random numbers: the baseline and the candidate share every draw, so a
    change that changes nothing must come back as 0 and not as sampling noise."""
    s = _season()
    d = sim.delta(s, dict(s.means), n=500)
    assert all(v["d_title"] == 0.0 and v["d_playoffs"] == 0.0 for v in d.values())


def test_a_better_team_gets_better_odds():
    s = _season()
    after = dict(s.means)
    for w in (10, 11, 12):
        after[("t1", w)] = s.means[("t1", w)] + 40.0
    d = sim.delta(s, after, n=2000)
    assert d["t1"]["d_playoffs"] > 0.05


def test_early_season_uncertainty_softens_overconfident_odds():
    """§18 real-world bug report: playoff odds read as 99%/0% in week 3, which is the
    simulation trusting a 3-game team_ppw estimate as if it were a known constant. Over
    many remaining weeks a modest mean edge compounds to near-certainty through score
    noise alone (CLT) regardless of how shaky that estimate actually is — mean_se is
    supposed to prevent that by drawing each team's true mean ONCE per simulated season
    (not per week) around its estimate, widening it when few games back that estimate."""
    teams = [f"t{i}" for i in range(1, 7)]
    weeks = list(range(1, 15))          # a long remaining season, where CLT bites hardest
    sched = [{"week": w, "home": teams[i], "away": teams[-1 - i]}
             for w in weeks for i in range(3)]
    # t1 carries a modest, real edge (+16 pts/wk over the field) on both sides of the test.
    means = {(t, w): 100.0 + (16.0 if t == "t1" else 0.0) for t in teams for w in weeks}
    base = dict(teams=teams, means=means, schedule=sched,
               wins={t: 0 for t in teams}, points_for={t: 0.0 for t in teams},
               sigma=25.0, playoff_teams=3, byes=1)

    certain = sim.simulate(sim.Season(**base, mean_se={}), n=4000)
    uncertain = sim.simulate(
        sim.Season(**base, mean_se={t: 25.0 / math.sqrt(3) for t in teams}), n=4000)

    p_certain = certain.set_index("team").loc["t1", "p_playoffs"]
    p_uncertain = uncertain.set_index("team").loc["t1", "p_playoffs"]
    assert p_certain > 0.9                       # the old bug: a small edge reads as a lock
    assert p_uncertain < p_certain                # 3-game uncertainty pulls it back down
    assert p_uncertain > 0.5                      # a real edge should still show, just softer


def test_the_simulation_is_reproducible():
    s = _season()
    a = sim.simulate(s, n=800, seed=7)
    b = sim.simulate(s, n=800, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_odds_are_probabilities_and_the_field_sums_to_the_bracket_size():
    s = _season()
    out = sim.simulate(s, n=2000)
    assert ((out["p_playoffs"] >= 0) & (out["p_playoffs"] <= 1)).all()
    assert out["p_playoffs"].sum() == pytest.approx(s.playoff_teams, abs=0.01)


def test_partner_tags():
    assert sim.partner_tag(0.75) == "CONTENDER"
    assert sim.partner_tag(0.40) == "BUBBLE"
    assert sim.partner_tag(0.05) == "OUT"


def test_arms_rival_only_fires_when_both_teams_are_live():
    assert sim.arms_rival(0.02, 0.06, 0.5, 0.5)
    assert not sim.arms_rival(0.02, 0.06, 0.5, 0.95)   # partner already safe
    assert not sim.arms_rival(0.02, 0.01, 0.5, 0.5)    # partner barely helped


def test_weekly_sigma_comes_from_this_leagues_own_scores():
    scores = pd.DataFrame({"points": [100, 120, 80, 110, 90]})
    assert sim.weekly_sigma(scores) == pytest.approx(float(np.std([100, 120, 80, 110, 90], ddof=1)))
    assert sim.weekly_sigma(pd.DataFrame()) == 0.0


# ---------------------------------------------------------------- real fixtures
def _standings(teams):
    return pd.DataFrame({"team": teams, "wins": [1] * len(teams)})


def _scores(teams):
    return pd.DataFrame([{"team": t, "week": 1, "points": 100.0 + i}
                         for i, t in enumerate(teams)])


def test_real_fixtures_are_used_where_they_exist():
    teams = [f"t{i}" for i in range(4)]
    fx = pd.DataFrame([{"week": 5, "home": "t0", "away": "t3"},
                       {"week": 5, "home": "t1", "away": "t2"}])
    s = sim.from_league(_scores(teams), _standings(teams), {t: 100.0 for t in teams},
                        [5], fixtures=fx)
    assert s.approx_weeks == ()
    assert {(g["home"], g["away"]) for g in s.schedule} == {("t0", "t3"), ("t1", "t2")}
    assert "real remaining fixtures" in sim.schedule_note(s)


def test_weeks_without_fixtures_fall_back_and_are_named():
    """The caveat has to say WHICH weeks are invented, not just that some are."""
    teams = [f"t{i}" for i in range(4)]
    fx = pd.DataFrame([{"week": 5, "home": "t0", "away": "t3"},
                       {"week": 5, "home": "t1", "away": "t2"}])
    s = sim.from_league(_scores(teams), _standings(teams), {t: 100.0 for t in teams},
                        [5, 6, 7], fixtures=fx)
    assert s.approx_weeks == (6, 7)
    note = sim.schedule_note(s)
    assert "through week 5" in note and "6-7" in note


def test_no_fixtures_at_all_says_so_plainly():
    teams = [f"t{i}" for i in range(4)]
    s = sim.from_league(_scores(teams), _standings(teams), {t: 100.0 for t in teams},
                        [5, 6], fixtures=pd.DataFrame())
    assert s.approx_weeks == (5, 6)
    assert "stand-in" in sim.schedule_note(s)


def test_a_partial_week_of_fixtures_is_not_trusted():
    """Half a week's fixtures would leave teams with no game, which silently changes
    everyone's record. Better to approximate the whole week."""
    teams = [f"t{i}" for i in range(6)]
    fx = pd.DataFrame([{"week": 5, "home": "t0", "away": "t1"}])   # 1 of 3
    s = sim.from_league(_scores(teams), _standings(teams), {t: 100.0 for t in teams},
                        [5], fixtures=fx)
    assert s.approx_weeks == (5,)


def test_fixtures_naming_an_unknown_team_are_ignored():
    teams = [f"t{i}" for i in range(4)]
    fx = pd.DataFrame([{"week": 5, "home": "t0", "away": "GHOST"},
                       {"week": 5, "home": "t1", "away": "t2"}])
    s = sim.from_league(_scores(teams), _standings(teams), {t: 100.0 for t in teams},
                        [5], fixtures=fx)
    assert s.approx_weeks == (5,)
