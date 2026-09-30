"""Trading players on injured reserve (mega/trade_engine.py ir_value / _after_ir).

An IR player is valued for the share of the remaining games he's projected to play
(`avail`) times what he adds to the roster once back, after the cut his return forces.
Only trades that move an IR player use this; every other trade keeps the v1 numbers.
"""
from __future__ import annotations

import pytest

from mega import trade_engine as te

CFG = {"slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flexCount": 1,
       "flexEligible": ["RB", "WR"], "rosterSize": 8, "irSlots": 1}


def _league(my_ir=("iq",), their_ir=(), iq_avail=0.5):
    P = lambda pid, pos, ppg, **kw: {"id": pid, "name": pid, "pos": pos, "nfl": "KC", "ppg": ppg, **kw}
    players = {p["id"]: p for p in [
        P("q", "QB", 18), P("r1", "RB", 15), P("r2", "RB", 12), P("r3", "RB", 4),
        P("w1", "WR", 14), P("w2", "WR", 11), P("w3", "WR", 9), P("t", "TE", 8),
        P("iq", "QB", 25, avail=iq_avail),
        P("oq", "QB", 16), P("or1", "RB", 20), P("or2", "RB", 9), P("or3", "RB", 3),
        P("ow1", "WR", 12), P("ow2", "WR", 10), P("ow3", "WR", 7), P("ot", "TE", 7),
        P("ir2", "WR", 16, avail=0.8), P("fa_wr", "WR", 5),
    ]}
    teams = [{"id": 1, "name": "Me", "roster": ["q", "r1", "r2", "r3", "w1", "w2", "w3", "t"], "ir": list(my_ir)},
             {"id": 2, "name": "Them", "roster": ["oq", "or1", "or2", "or3", "ow1", "ow2", "ow3", "ot"],
              "ir": list(their_ir)}]
    return {"teams": teams, "players": players, "freeAgents": ["fa_wr"]}


def test_an_ir_player_counts_for_the_share_of_the_season_he_plays():
    ctx = te.build_context(_league(), CFG)
    base = ctx.base[1].value
    back = te.settle(ctx.base[1].ids + ["iq"], ctx, set()).value
    assert back > base                                   # a 25-ppg QB over an 18
    assert te._base_ir(ctx, 1) == pytest.approx(base + 0.5 * (back - base))


def test_an_ir_player_who_never_plays_again_is_worth_nothing():
    ctx = te.build_context(_league(iq_avail=0.0), CFG)
    assert te._base_ir(ctx, 1) == pytest.approx(ctx.base[1].value)


def test_trading_away_an_ir_player_costs_his_expected_return():
    ctx = te.build_context(_league(), CFG)
    r = te.evaluate_core(ctx, 1, 2, ["iq"], ["ow3"])      # him for their WR3
    ir_term = te._base_ir(ctx, 1) - ctx.base[1].value
    after_active = te.settle([*ctx.teams[1]["roster"], "ow3"], ctx, {"ow3"}).value
    assert r["dMe"] == pytest.approx(after_active - ctx.base[1].value - ir_term)
    assert r["_after_ir"]["me"] == [] and r["_after_ir"]["them"] == ["iq"]


def test_receiving_one_with_the_ir_slot_full_costs_a_roster_spot():
    ctx = te.build_context(_league(their_ir=("ir2",)), CFG)
    r = te.evaluate_core(ctx, 1, 2, ["w3"], ["ir2"])      # my IR slot already holds iq
    a = r["_after"]["me"]
    assert len(a.ids) == CFG["rosterSize"] - 1            # ir2 sits on the bench, holding a spot
    assert r["_after_ir"]["me"] == ["iq", "ir2"]


def test_with_a_free_ir_slot_the_open_spot_is_filled():
    ctx = te.build_context(_league(my_ir=(), their_ir=("ir2",)), CFG)
    r = te.evaluate_core(ctx, 1, 2, ["w3"], ["ir2"])
    assert len(r["_after"]["me"].ids) == CFG["rosterSize"]   # ir2 on IR, w3's spot refilled


def test_trades_without_an_ir_player_are_valued_exactly_as_before():
    with_ir = te.build_context(_league(), CFG)
    plain = te.build_context({**_league(my_ir=()), "players": _league()["players"]}, CFG)
    a = te.evaluate_core(with_ir, 1, 2, ["w2"], ["or2"])
    b = te.evaluate_core(plain, 1, 2, ["w2"], ["or2"])
    assert a["dMe"] == pytest.approx(b["dMe"]) and a["dThem"] == pytest.approx(b["dThem"])
    assert "_after_ir" not in a


def test_the_search_can_start_from_my_ir_player():
    ctx = te.build_context(_league(), CFG)
    out = te.find_trades(ctx, 1, ["iq"], {"includeFlags": te.FLAG_ORDER, "minDeltaMe": -99,
                                          "shapes": ["1-for-1"]})
    assert out["evaluated"] > 0 and all(r["giveIds"] == ["iq"] for r in out["results"])
