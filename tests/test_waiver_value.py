"""HANDOFF v1.3 Pass 1 acceptance tests (§3.6) for mega/waiver_value.py.

A–F are the handoff's own cases. F is the reworded version settled with Chris 2026-09-26:
START + COVER — good and bad weeks both — equals the engine's horizon gain exactly, and
INSURE is reported beside it rather than inside it.

Everything but A runs on a small synthetic league so the numbers can be checked by hand.
A runs on the live week-3 data and is marked slow.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega import trade_engine as te
from mega import waiver_value as wv

CFG = {"slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flexCount": 1,
       "flexEligible": ["RB", "WR"], "rosterSize": 8, "replacementRank": 1}
NOW = 3


def _p(pid, pos, nfl, ppg):
    return {"id": pid, "name": pid, "pos": pos, "nfl": nfl, "ppg": ppg, "ecr": None}


def league(extra_fa=(), ir=(), mine_extra=(), roster_size=8):
    players = {}

    def add(*ps):
        for p in ps:
            players[p["id"]] = p
        return [p["id"] for p in ps]

    mine = add(_p("q1", "QB", "KC", 20), _p("r1", "RB", "SF", 18), _p("r2", "RB", "DAL", 12),
               _p("r3", "RB", "NYG", 4), _p("w1", "WR", "MIA", 15), _p("w2", "WR", "DET", 10),
               _p("w3", "WR", "CHI", 9), _p("t1", "TE", "BUF", 12), *mine_extra)
    ir_ids = add(*ir)

    def other(tag):
        return add(_p(f"{tag}q", "QB", "LAR", 15), _p(f"{tag}r1", "RB", "ATL", 10),
                   _p(f"{tag}r2", "RB", "CAR", 9), _p(f"{tag}r3", "RB", "TEN", 3),
                   _p(f"{tag}w1", "WR", "SEA", 11), _p(f"{tag}w2", "WR", "HOU", 10),
                   _p(f"{tag}w3", "WR", "JAX", 8), _p(f"{tag}t", "TE", "IND", 8))

    fas = add(_p("fa_te", "TE", "NE", 9), _p("fa_rb", "RB", "SF", 5),
              _p("fa_wr_low", "WR", "ARI", 2), *extra_fa)
    teams = [{"id": 1, "name": "Mine", "roster": mine, "ir": ir_ids},
             {"id": 2, "name": "B", "roster": other("b"), "ir": []},
             {"id": 3, "name": "C", "roster": other("c"), "ir": []}]
    cfg = {**CFG, "rosterSize": roster_size}
    return {"teams": teams, "players": players, "freeAgents": fas}, cfg


def board(extra_fa=(), byes=None, avail=None, ir=(), mine_extra=(), roster_size=8):
    lg, cfg = league(extra_fa, ir, mine_extra, roster_size)
    return wv.build_board(lg, cfg, 1, NOW, avail or {}, byes or {})


# ---------------------------------------------------------------- C
def test_c_a_wr_who_outprojects_our_flex_in_two_of_three_weeks_is_bid_now():
    b = board(extra_fa=[_p("fa_wr", "WR", "PHI", 13)], byes={"PHI": 4})
    rows = wv.assess(b, ["fa_wr"], {}, {}, {}, {}, {}, {}, 100, NOW)
    r = rows.iloc[0]
    assert r["lane"] == "bid_now"
    assert r["mechanism"] == "START"
    # +4 in weeks 3 and 5 (13 replaces the 9 in the flex chain), 0 on his own week-4 bye
    assert r["next3"] == pytest.approx(8 / 3, abs=1e-3)
    assert r["start"] > 0 and abs(r["cover"]) < 1e-9


# ---------------------------------------------------------------- D
def test_d_a_bye_with_no_bench_cover_is_cover_not_start():
    b = board(byes={"BUF": 7})                      # our only TE's bye
    ev = wv.evaluate(b, "fa_te", {}, {})
    assert ev["start"] == pytest.approx(0.0, abs=1e-9)
    assert ev["cover"] > 0
    # all of it is week 7: 9 points in one of fifteen weeks, three of them weighted 1.5
    den = 12 + 3 * 1.5
    assert ev["cover"] == pytest.approx(9 / den, abs=1e-6)
    assert ev["_cover_w"][7] == pytest.approx(9.0)


# ---------------------------------------------------------------- B
def test_b_an_fa_behind_our_own_rb1_gets_insure_and_the_handcuff_tag():
    b = board()
    cuffs = wv.make_cuffs(b, {"fa_rb": {"cuff_of": "r1", "clear_two": True, "pos": "RB"}},
                          xfp_pg={"r1": 18.0}, vol_ratio={}, avail={}, ages={})
    ev = wv.evaluate(b, "fa_rb", {}, cuffs)
    assert ev["insure"] > 0
    assert ev["handcuff"] is True
    # INSURE is shown beside the lineup gain, never inside START + COVER
    assert ev["start"] + ev["cover"] == pytest.approx(ev["gain"], abs=1e-9)


def test_insure_is_measured_against_the_roster_as_it_stands():
    """If the player we'd cut would fill the hole himself, the add is worth only the
    difference — not the whole promoted value."""
    b = board()
    cuffs = wv.make_cuffs(b, {"fa_rb": {"cuff_of": "r1", "clear_two": True, "pos": "RB"}},
                          xfp_pg={"r1": 18.0}, vol_ratio={}, avail={}, ages={})
    ev = wv.evaluate(b, "fa_rb", {}, cuffs)
    from mega import contingency as cg
    promoted = min(5 + cg.inherit_fraction("RB") * 18.0, 18.0)
    # when r1 sits, fa_rb (promoted) replaces r3 (4) in the RB2 slot, not an empty slot
    per_week = {w: cuffs["fa_rb"].p_miss[w] * (promoted - 4.0 - 0.0) for w, _ in b.weeks}
    assert ev["insure"] <= b.average(per_week) + 1e-6


# ---------------------------------------------------------------- E
def test_e_a_flip_only_player_never_appears_in_a_lane():
    # an 11-point TE who would start for both other teams (their TEs score 8), on our TE's
    # own NFL team so he can't even cover the bye — and TE can't flex here
    b = board(extra_fa=[_p("fa_te2", "TE", "BUF", 11)], byes={"BUF": 7})
    rows = wv.assess(b, ["fa_te2"], {}, {"fa_te2": 500.0}, {}, {}, {}, {}, 100, NOW)
    r = rows.iloc[0]
    assert r["fit"] <= wv.FIT_MIN
    assert r["flip"] == 500.0 and r["flip_buyers"] == 2
    assert r["lane"] == "trade_chip"


# ---------------------------------------------------------------- F
def test_f_mechanisms_sum_to_the_engines_own_horizon_gain():
    b = board(extra_fa=[_p("fa_wr", "WR", "PHI", 13)], byes={"PHI": 4, "BUF": 7, "MIA": 9})
    ev = wv.evaluate(b, "fa_wr", {}, {})
    assert ev["start"] + ev["cover"] == pytest.approx(ev["gain"], abs=1e-9)
    # ...and that gain is the engine's, not a parallel calculation that could drift
    R = list(b.roster)
    Rp = [x for x in R if x != ev["drop_id"]] + ["fa_wr"]
    engine = te.team_value(Rp, b.ctx) - te.team_value(R, b.ctx)
    assert ev["gain"] == pytest.approx(engine, abs=1e-9)


def test_f_holds_with_an_ir_return_in_the_horizon():
    b = board(extra_fa=[_p("fa_wr", "WR", "PHI", 13)], byes={"PHI": 4},
              ir=[_p("q9", "QB", "WAS", 22)], avail={"q9": {"status": "Out", "back": 6}})
    ev = wv.evaluate(b, "fa_wr", {}, {})
    assert ev["start"] + ev["cover"] == pytest.approx(ev["gain"], abs=1e-9)


# ---------------------------------------------------------------- §3.2 IR path
def test_an_ir_return_cuts_the_useless_backup_not_the_handcuff():
    """The Stroud case: when a third QB comes back, the QB who can no longer start goes —
    not the cheaper-looking back who insures our RB1."""
    b = board(ir=[_p("q9", "QB", "WAS", 22)], avail={"q9": {"status": "Out", "back": 5}},
              mine_extra=[_p("q2", "QB", "HOU", 14)], roster_size=9)
    cuffs = wv.make_cuffs(b, {"r3": {"cuff_of": "r1", "clear_two": True, "pos": "RB"}},
                          xfp_pg={"r1": 18.0}, vol_ratio={}, avail={}, ages={})
    mine = wv.insure_mine(b, cuffs)
    assert mine["r3"] > 0
    drops = b.path(b.roster)["drops"]
    assert [d["drop"] for d in drops] == ["q2"]
    notes = wv.roster_notes(b, {"q9": {"status": "Out", "back": 5}})
    assert any("q2 is your cheapest drop once q9 is back (week 5)" in n for n in notes)


def test_a_returning_player_who_projects_worst_is_named_as_the_cut_plainly():
    b = board(ir=[_p("q9", "QB", "WAS", 3)], avail={"q9": {"status": "Out", "back": 5}},
              mine_extra=[_p("q2", "QB", "HOU", 14)], roster_size=9)
    notes = wv.roster_notes(b, {"q9": {"status": "Out", "back": 5}})
    assert any("When q9 is back (week 5) he projects as your weakest QB" in n for n in notes)
    assert not any("q9 is your cheapest drop once q9" in n for n in notes)


def test_before_the_return_the_ir_player_is_worth_nothing_and_after_it_everything():
    b = board(ir=[_p("q9", "QB", "WAS", 22)], avail={"q9": {"status": "Out", "back": 6}},
              mine_extra=[_p("q2", "QB", "HOU", 14)], roster_size=9)
    wk = b.weekly(b.roster)
    assert wk[5] < wk[6]                            # q9 (22) replaces q1 (20) from week 6


# ---------------------------------------------------------------- inputs
def test_return_week_rules():
    inj = pd.DataFrame([
        {"gsis_id": "a", "week": 3, "report_status": "Out", "practice_status": "Did Not Participate In Practice"},
        {"gsis_id": "b", "week": 3, "report_status": "Out", "practice_status": "Full Participation in Practice"},
        {"gsis_id": "c", "week": 3, "report_status": "Doubtful", "practice_status": "Limited"},
        {"gsis_id": "d", "week": 3, "report_status": "Questionable", "practice_status": "Limited"},
    ])
    rw = pd.DataFrame([{"gsis_id": "e", "week": 1, "status": "ACT"},
                       {"gsis_id": "e", "week": 2, "status": "RES"},
                       {"gsis_id": "e", "week": 3, "status": "RES"}])
    a = wv.availability(inj, rw, 3)
    assert a["a"]["back"] == 3 + wv.OUT_NO_PRACTICE
    assert a["b"]["back"] == 4
    assert a["c"]["back"] == 4
    assert "d" not in a                              # questionable players play
    assert a["e"] == {"status": "IR", "back": 2 + wv.IR_MIN_WEEKS}


def test_byes_are_read_off_the_schedule():
    s = pd.DataFrame([{"game_type": "REG", "week": w, "home_team": "KC", "away_team": "BUF"}
                      for w in (1, 2, 4)] + [{"game_type": "REG", "week": 3, "home_team": "BUF",
                                               "away_team": "MIA"}])
    assert wv.bye_weeks(s)["KC"] == 3


def test_a_flag_rate_is_shrunk_toward_its_base_rate():
    rates = wv._signal_rates()
    if not rates:
        pytest.skip("config/signal_rates.json not fitted")
    p, flag = wv.signal_p("WR", ["TGT"], {"TGT": "sustained"})
    cell = rates["flags"]["TGT"]["WR"]["sustained"]
    base = rates["base"]["WR"]["rate"]
    assert flag == "TGT"
    assert min(base, cell["rate"]) <= p <= max(base, cell["rate"])
    assert wv.signal_p("WR", ["ROLE-"], {}) is None  # a falling-role flag is not a signal


# ---------------------------------------------------------------- A (live data)
@pytest.mark.slow
def test_a_the_extra_quarterbacks_have_no_fit_on_the_live_week3_roster():
    from mega.yahoo import cached_rosters
    ros = cached_rosters()
    if ros.empty:
        pytest.skip("no cached Yahoo rosters")
    mine = set(ros.loc[ros["team"] == "TaylorMade", "player"])
    if not {"Bryce Young", "C.J. Stroud", "Jayden Daniels"} <= mine:
        # pinned to the week-3 three-QB roster; with two QBs a third has real bye-week fit
        pytest.skip("TaylorMade no longer carries the week-3 three quarterbacks")
    out = wv.run(2026, 3, ros)
    r = out["rows"].set_index("player")
    for qb in ("Daniel Jones", "Jacoby Brissett", "Kirk Cousins", "Deshaun Watson"):
        if qb in r.index:
            assert r.loc[qb, "fit"] <= wv.FIT_MIN, qb
            assert r.loc[qb, "lane"] in ("", "trade_chip"), qb


def test_a_backup_qb_behind_someone_elses_starter_has_no_insure():
    """Keenum behind Caleb Williams was a $7 'insurance' bid for a manager holding two QBs.
    The starter isn't ours, so there is nothing to insure."""
    b = board(extra_fa=[_p("fa_qb", "QB", "CHI", 15)])
    cuffs = wv.make_cuffs(b, {"fa_qb": {"cuff_of": "bq", "clear_two": True, "pos": "QB"}},
                          xfp_pg={"bq": 20.0}, vol_ratio={}, avail={}, ages={})
    # make the starter someone on another team that the engine knows
    b.ctx.players["bq"]["pos"] = "QB"
    ev = wv.evaluate(b, "fa_qb", {}, cuffs)
    assert ev["insure"] == 0.0 and ev["handcuff"] is False


def test_a_backup_qb_behind_our_own_qb_still_gets_insure():
    b = board(extra_fa=[_p("fa_qb", "QB", "KC", 15)])
    cuffs = wv.make_cuffs(b, {"fa_qb": {"cuff_of": "q1", "clear_two": True, "pos": "QB"}},
                          xfp_pg={"q1": 20.0}, vol_ratio={}, avail={}, ages={})
    ev = wv.evaluate(b, "fa_qb", {}, cuffs)
    assert ev["handcuff"] is True and ev["insure"] > 0


def test_a_dozen_interchangeable_cover_adds_are_each_worth_the_gap_not_the_hole():
    ev = lambda f: {"start": 0.0, "insure": 0.0, "cover": f, "fit": f, "next3": f}
    evs = {f"q{i}": ev(v) for i, v in enumerate([1.3, 1.1, 0.95, 0.9, 0.9, 0.85])}
    evs["rb"] = {"start": 1.4, "insure": 0.0, "cover": 0.1, "fit": 1.5, "next3": 1.5}   # a starter is untouched
    base = wv.cover_baseline(evs, {k: ("RB" if k == "rb" else "QB") for k in evs})
    assert base == {"QB": 0.95}                      # the third-best of the QBs
    two = wv.cover_baseline({k: evs[k] for k in ("q0", "q1")}, {"q0": "QB", "q1": "QB"})
    assert two == {"QB": 0.0}                         # with only two there is no free alternative


# ---------------------------------------------------------------- rival bids (§5.4)
def test_a_claim_is_priced_against_what_the_other_rosters_would_pay():
    b = board(extra_fa=[_p("fa_wr", "WR", "PHI", 13)])
    rows = wv.assess(b, ["fa_wr"], {}, {}, {}, {}, {}, {}, 100, NOW)
    broke = wv.rival_bids(b, rows, {"B": 0, "C": 0}, NOW).iloc[0]
    assert broke["rival_top"] == 0 and broke["bid"] == 1          # nobody can pay: the $1 floor
    rich = wv.rival_bids(b, rows, {"B": 100, "C": 100}, NOW).iloc[0]
    assert rich["rival_top"] > 0 and rich["rival_team"] in ("B", "C") and rich["rivals_n"] == 2
    top, ceiling = rich["rival_top"], rows.iloc[0]["max_bid"]
    if top + 1 <= ceiling:
        assert rich["bid"] == top + 1
    elif top <= ceiling:
        assert rich["bid"] == ceiling
    else:
        assert rich["bid"] == 0 and "outbid" in rich["bid_note"]
