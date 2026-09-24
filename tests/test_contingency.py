"""§15 injury expectation and handcuffs, §16 replacement level.

Carries the spec's hand-checkable cases §19.3 #17 and #18, and §8 #7.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega import contingency as cg


# ---------------------------------------------------------------- §19.3 #17
def test_availability_ramp_for_a_running_back():
    """P_out = 0.06 at now+1, 0.18 at now+3, and flat at 0.18 thereafter."""
    assert cg.p_out("RB", 1) == pytest.approx(0.06)
    assert cg.p_out("RB", 3) == pytest.approx(0.18)
    assert cg.p_out("RB", 6) == pytest.approx(0.18)
    assert cg.p_out("RB", 0) == 0.0
    assert cg.steady_state("RB") == pytest.approx(0.18)


def test_positions_carry_different_injury_rates():
    assert cg.steady_state("RB") > cg.steady_state("WR") >= cg.steady_state("QB")


def test_known_status_governs_this_week_only():
    for status, avail in (("Out", 0.0), ("Doubtful", 0.25), ("Questionable", 0.85), ("", 1.0)):
        assert cg.availability(status, "WR", week=5, now=5) == pytest.approx(avail)
    # next week is the curve again, not the status
    assert cg.availability("Out", "WR", week=6, now=5) == pytest.approx(1 - cg.p_out("WR", 1))


# ---------------------------------------------------------------- §8 #7
def test_ir_is_zero_until_the_return_week_then_full():
    """A player with irReturnWeek = now+2 contributes nothing for two weeks."""
    now = 5
    assert cg.availability("IR", "WR", week=5, now=now, ir_return=7) == 0.0
    assert cg.availability("IR", "WR", week=6, now=now, ir_return=7) == 0.0
    assert cg.availability("IR", "WR", week=7, now=now, ir_return=7) > 0.7
    assert cg.availability("IR", "WR", week=8, now=now, ir_return=7) > 0.7


def test_ir_without_an_override_assumes_the_default_wait():
    now = 5
    back = now + cg.IR_WEEKS_DEFAULT
    assert cg.availability("IR", "RB", week=back - 1, now=now) == 0.0
    assert cg.availability("IR", "RB", week=back, now=now) > 0.0


def test_a_healthy_starter_loses_a_sliver_of_future_value():
    """The point of §15.1: this is what lets a backup gain value without double counting."""
    assert cg.availability("", "RB", week=9, now=3) == pytest.approx(0.82)


# ---------------------------------------------------------------- §19.3 #18
def test_contingent_value_of_a_handcuff():
    """Promoted value 14, h 0.85 → 11.9; backup's own week 3; P_out 0.18 → +1.602."""
    cuffs = pd.DataFrame([{"gsis_id": "B", "cuff_of": "S", "team": "LA", "pos": "RB",
                           "share": 0.40}])                       # committee → h = 0.85
    base = {"role": {"xfp_share": {"LEAD": 0.35}}}
    team_xfp = pd.DataFrame([{"team": "LA", "week": 6, "xfp_rush": 40.0}])
    # 0.35 * 40 * 0.85 = 11.9 promoted; own 3.0; P_out(RB, 3) = 0.18
    out = cg.contingent(cuffs, base, team_xfp, {("B", 6): 3.0}, now=3, weeks=[6])
    assert out["contingent"].iloc[0] == pytest.approx(1.602, abs=1e-3)


def test_a_clear_number_two_takes_no_haircut():
    cuffs = pd.DataFrame([{"gsis_id": "B", "cuff_of": "S", "team": "LA", "pos": "RB",
                           "share": 0.80}])                       # clear #2 → h = 1.00
    base = {"role": {"xfp_share": {"LEAD": 0.35}}}
    team_xfp = pd.DataFrame([{"team": "LA", "week": 6, "xfp_rush": 40.0}])
    out = cg.contingent(cuffs, base, team_xfp, {("B", 6): 3.0}, now=3, weeks=[6])
    assert out["contingent"].iloc[0] == pytest.approx(0.18 * (14.0 - 3.0), abs=1e-3)


def test_contingency_never_goes_negative():
    """A backup already worth more than the job would be gains nothing from it."""
    cuffs = pd.DataFrame([{"gsis_id": "B", "cuff_of": "S", "team": "LA", "pos": "RB",
                           "share": 0.80}])
    base = {"role": {"xfp_share": {"LEAD": 0.35}}}
    team_xfp = pd.DataFrame([{"team": "LA", "week": 6, "xfp_rush": 40.0}])
    out = cg.contingent(cuffs, base, team_xfp, {("B", 6): 99.0}, now=3, weeks=[6])
    assert out["contingent"].iloc[0] == 0.0


def test_promoted_value_uses_the_role_baseline_not_the_starters_own_share():
    """A backup who takes over inherits the touches, not the player."""
    cuffs = pd.DataFrame([{"gsis_id": "B", "cuff_of": "S", "team": "LA", "pos": "RB",
                           "share": 0.80}])
    team_xfp = pd.DataFrame([{"team": "LA", "week": 6, "xfp_rush": 40.0}])
    lo = cg.contingent(cuffs, {"role": {"xfp_share": {"LEAD": 0.20}}}, team_xfp,
                       {}, now=3, weeks=[6])["contingent"].iloc[0]
    hi = cg.contingent(cuffs, {"role": {"xfp_share": {"LEAD": 0.40}}}, team_xfp,
                       {}, now=3, weeks=[6])["contingent"].iloc[0]
    assert hi == pytest.approx(2 * lo)


def test_only_one_backup_per_starter_gets_the_option():
    """A job can only be done by one man; spreading it across the room double counts it."""
    pw = pd.DataFrame([{"gsis_id": g, "week": 1, "offense_snaps": s, "carries": c}
                       for g, s, c in (("S", 50, 20), ("B", 25, 8), ("C", 10, 3))])
    tab = pd.DataFrame([{"gsis_id": "S", "team": "LA", "pos": "RB", "role": "LEAD"},
                        {"gsis_id": "B", "team": "LA", "pos": "RB", "role": "COMMITTEE"},
                        {"gsis_id": "C", "team": "LA", "pos": "RB", "role": "BACKUP"}])
    out = cg.next_man_up(pw, tab)
    assert list(out["gsis_id"]) == ["B"]
    assert out["cuff_of"].iloc[0] == "S"


# ---------------------------------------------------------------- §16
def test_replacement_is_the_mean_of_the_top_three():
    """One lucky free agent must not set the bar for a whole position."""
    vals = pd.Series([20.0, 12.0, 10.0, 9.0, 1.0])
    assert cg.replacement(vals) == pytest.approx(14.0)     # (20+12+10)/3
    assert cg.replacement(vals, top_n=1) == pytest.approx(20.0)


def test_replacement_handles_a_thin_pool():
    assert cg.replacement(pd.Series([7.0])) == pytest.approx(7.0)
    assert cg.replacement(pd.Series([], dtype=float)) == 0.0


def test_a_refilling_position_has_a_rising_replacement_level():
    """Where the wire keeps producing, giving a player away costs less than he scores."""
    flat = cg.replacement_curve(8.0, supply=0.0, delta=2.0, week=10, now=4)
    rising = cg.replacement_curve(8.0, supply=3.0, delta=2.0, week=10, now=4)
    assert flat == pytest.approx(8.0)
    assert rising == pytest.approx(8.0 + 3.0 * 2.0 * 6 / 12)
