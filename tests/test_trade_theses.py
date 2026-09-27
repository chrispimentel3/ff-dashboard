"""HANDOFF v1.3 Pass 4 acceptance (§6.5) for mega/trade_theses.py.

The structural cases run on synthetic data; the three that are about THIS league's week-3
board (Hall -> Bowers gone, no negative-ΔTitle card, every card complete) run on the live
cached data and are marked slow.
"""
from __future__ import annotations

import pytest

from mega import trade_engine as te
from mega import trade_theses as TT

CFG = {"slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flexCount": 1,
       "flexEligible": ["RB", "WR"], "rosterSize": 8}


def _ctx():
    P = lambda pid, pos, ppg: {"id": pid, "name": pid, "pos": pos, "nfl": "KC", "ppg": ppg}
    players = {p["id"]: p for p in [
        P("q", "QB", 18), P("r1", "RB", 15), P("r2", "RB", 12), P("r3", "RB", 4),
        P("w1", "WR", 14), P("w2", "WR", 11), P("w3", "WR", 9), P("t", "TE", 8),
        P("oq", "QB", 16), P("or1", "RB", 20), P("or2", "RB", 9), P("or3", "RB", 3),
        P("ow1", "WR", 12), P("ow2", "WR", 10), P("ow3", "WR", 7), P("ot", "TE", 7),
        P("fa_wr", "WR", 13),
    ]}
    teams = [{"id": 1, "name": "Me", "roster": ["q", "r1", "r2", "r3", "w1", "w2", "w3", "t"]},
             {"id": 2, "name": "Them", "roster": ["oq", "or1", "or2", "or3", "ow1", "ow2", "ow3", "ot"]}]
    return te.build_context({"teams": teams, "players": players, "freeAgents": ["fa_wr"]}, CFG)


def test_a_two_for_one_is_credited_only_net_of_the_free_swap():
    """The spot a 2-for-1 opens gets the best free agent — who was available anyway. The
    trade must not be credited with a pickup we could make without it."""
    ctx = _ctx()
    raw = te.evaluate_core(ctx, 1, 2, ["r2", "w3"], ["or1"])
    after = raw["_after"]["me"]
    assert "fa_wr" in after.added                   # the engine filled the open spot
    swap_ids, swap_val = TT.free_swap(ctx, ctx.base[1].ids)
    assert "fa_wr" in swap_ids                      # ...with a player we could add today
    net = after.value - swap_val
    assert net < raw["dMe"]                         # netting removes the free pickup
    fa_alone = swap_val - ctx.base[1].value
    assert raw["dMe"] - net == pytest.approx(fa_alone, abs=1e-9)


def test_urgency_peaks_on_the_bubble():
    assert TT.urgency(0.35) == 1.0
    assert TT.urgency(0.15) == TT.urgency(0.60) == 0.5
    assert TT.urgency(0.90) == TT.urgency(0.02) == 0.0


def test_acceptance_moves_the_right_way():
    pri = TT.load_priors()
    base = TT.p_accept(pri, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert TT.p_accept(pri, 0.03, 1.0, 0.0, 0.0, 0.0) > base        # helps their title odds
    assert TT.p_accept(pri, 0.0, 0.8, 0.0, 0.0, 0.0) < base         # worse on FantasyCalc
    assert TT.p_accept(pri, 0.0, 1.0, 1.0, 0.0, 0.0) > base         # fills their lineup
    assert TT.p_accept(pri, 0.0, 1.0, 0.0, 0.15, 0.0) > base        # their positional bias
    assert TT.p_accept(pri, 0.0, 1.0, 0.0, 0.0, 1.0) > base         # on the bubble


def test_flags_come_from_p_accept():
    pri = TT.load_priors()
    assert TT.flag(pri, 0.6, 0.0) == "LIKELY"
    assert TT.flag(pri, 0.35, -0.01) == "EXPLOIT"
    assert TT.flag(pri, 0.35, 0.01) == "NEEDS_PITCH"
    assert TT.flag(pri, 0.05, 0.0) == "LONGSHOT"


def _card(**kw):
    c = {"giveIds": ["w1"], "getIds": ["or1"], "dMe": 2.0, "d_week_me": 1.0, "consolidation": False,
         "me_starters_in": [], "me_starters_out": [], "portfolio": None, "fa_add": None}
    c.update(kw)
    return c


def _env():
    return {"season": 2030, "avail": {}, "status": {}, "ages": {}, "cuffs": {}, "schedule_lens": {},
            "def_vs": {}, "last_week": 17, "my_roster": {"q", "r1", "r2", "r3", "w1", "w2", "w3", "t"}}


def test_sell_high_needs_luck_and_a_consensus_that_still_believes():
    ctx = _ctx()
    lucky = {"w1": {"gap_pg": 6.0, "pts_pg": 18.0, "xfp_pg": 12.0, "games": 3, "ecr_rank": 8, "our_rank": 20}}
    tags = [t["tag"] for t in TT.tags_for(_card(), ctx, lucky, _env())]
    assert "SELL_HIGH" in tags
    doubted = {"w1": {**lucky["w1"], "ecr_rank": 25}}                  # consensus already sold
    assert "SELL_HIGH" not in [t["tag"] for t in TT.tags_for(_card(), ctx, doubted, _env())]


def test_every_tag_carries_a_thesis_and_a_kill_condition():
    ctx = _ctx()
    facts = {"w1": {"gap_pg": 6.0, "pts_pg": 18.0, "xfp_pg": 12.0, "games": 3, "ecr_rank": 8, "our_rank": 20},
             "or1": {"gap_pg": -5.0, "pts_pg": 9.0, "xfp_pg": 14.0, "games": 3, "ecr_rank": 15, "our_rank": 9,
                     "usage_rank": 7, "share": 0.25, "share_sd": 0.04, "slope_q": 0.9,
                     "share_first": 0.18, "share_last": 0.30}}
    c = _card(me_starters_in=[{"id": "or1", "name": "or1", "ppg": 20.0}],
              me_starters_out=[{"id": "r2", "name": "r2", "ppg": 12.0}])
    tags = TT.tags_for(c, ctx, facts, _env())
    assert {t["tag"] for t in tags} >= {"SELL_HIGH", "BUY_LOW", "TRAJECTORY", "LINEUP"}
    assert all(t["thesis"] and t["kill"] for t in tags)
    assert tags[0]["tag"] == "SELL_HIGH"                                # lead tag by priority


# ---------------------------------------------------------------- live week-3 board
@pytest.fixture(scope="module")
def live():
    from mega.yahoo import cached_rosters
    ros = cached_rosters()
    if ros.empty:
        pytest.skip("no cached Yahoo rosters")
    return TT.run(2026, 3, ros)


@pytest.mark.slow
def test_hall_for_bowers_is_gone(live):
    for c in live["cards"]:
        gives = {g["name"] for g in c["give"]}
        gets = {g["name"] for g in c["get"]}
        assert not ("Breece Hall" in gives and "Brock Bowers" in gets)


@pytest.mark.slow
def test_no_card_lowers_our_title_odds_without_consolidation(live):
    for c in live["cards"]:
        assert "CONSOLIDATION" in c["all_tags"] or (c["us"]["d_title"] > 0 and not c["us"]["title_noise"])


@pytest.mark.slow
def test_every_card_is_complete(live):
    assert live["cards"], "expected at least one offer on the week-3 board"
    for c in live["cards"]:
        assert c["tags"] and c["thesis"] and c["kill"] and c["pitch"]
        for side in ("us", "them"):
            assert {"d_week", "d_ros", "d_title"} <= set(c[side])
        assert 0.0 <= c["p_accept"] <= 1.0
