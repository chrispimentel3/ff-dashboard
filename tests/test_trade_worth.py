"""mega/trade_worth.py — the chance another manager says yes, on a league frozen 2026-10-22.

The calibration test is the agreement in docs/trade_logic_v2.md: Chris labelled ten offers
yes / maybe / no from the other manager's side, and the model has to match at least nine.
"""
from __future__ import annotations

import json
import pathlib
from unittest import mock

import pytest

from mega import trade_engine as te
from mega import trade_league as tl
from mega import trade_worth as W

FX = json.loads((pathlib.Path(__file__).parent / "fixtures" / "trade_worth_calibration.json").read_text())
BAND = {"Y": (0.5, 1.01), "M": (0.2, 0.5), "N": (0.0, 0.2)}


@pytest.fixture(scope="module")
def world():
    ctx = te.build_context(FX["league"], tl.engine_config())
    tid = {t["name"]: t["id"] for t in ctx.teams.values()}
    with mock.patch.object(W, "track_record", lambda season: (frozenset(FX["established"]), frozenset(FX["stars"]))):
        board = W.Board(ctx, FX["season"], {tid[n]: b for n, b in FX["bias"].items()})
    by_name = {p["name"]: pid for pid, p in ctx.players.items()}
    return ctx, board, tid, by_name


def score(world, n: str) -> dict:
    ctx, board, tid, by_name = world
    o = next(o for o in FX["offers"] if o["n"] == n)
    give, get = [by_name[x] for x in o["give"]], [by_name[x] for x in o["get"]]
    c = te.evaluate_core(ctx, tid[FX["my_team"]], tid[o["partner"]], give, get)
    return W.score(board, tid[o["partner"]], give, get, c["dMe"], c["dThem"],
                   FX["p_playoffs"][o["partner"]], FX["habit"][o["partner"]])


def test_calibration_matches_chris_on_nine_of_ten(world):
    offers = [o for o in FX["offers"] if o["n"] != "7"]          # 7 is an our-side check
    got = {o["n"]: score(world, o["n"])["p"] for o in offers}
    hits = [o["n"] for o in offers if BAND[o["label"]][0] <= got[o["n"]] < BAND[o["label"]][1]]
    assert len(hits) >= 9, {o["n"]: (o["label"], round(got[o["n"]], 2)) for o in offers}


def test_elite_tiers_end_at_the_first_big_drop(world):
    ctx, board, _, _ = world
    assert {ctx.players[i]["name"] for i in board.elite} == {"Brock Bowers", "Trey McBride", "Josh Allen"}


def test_a_throw_in_they_must_cut_for_lowers_the_odds(world):
    plain, padded = score(world, "5a"), score(world, "5b")         # + Jayden Daniels to Don R.I.C.O
    assert padded["clog"] == 1 and padded["p"] < plain["p"]
    assert any("throw-in" in w for w in padded["why"])


def test_a_fan_holds_their_own_teams_players(world):
    _, board, tid, by_name = world
    brown, colin = by_name["Chase Brown"], tid["Crabcakes and Football"]     # CIN; Colin roots for CIN
    fan, notes = board.name_value(brown, colin)
    assert "fan" in notes
    board.fan[colin] = None
    try:
        neutral, _ = board.name_value(brown, colin)
    finally:
        board.fan[colin] = "CIN"
    assert fan == pytest.approx(neutral * W.FAN_MULT)


def test_a_rookie_is_weak_currency(world):
    _, board, tid, by_name = world
    boston, team = by_name["Denzel Boston"], tid["deez nuts"]
    v, notes = board.name_value(boston, team)
    assert "unproven" in notes
    est = board.est
    board.est = est | {boston}
    try:
        proven, _ = board.name_value(boston, team)
    finally:
        board.est = est
    assert v == pytest.approx(proven * W.PERCEIVED_FLOOR)


def test_out_of_race_teams_play_for_wins_now(world):
    s = score(world, "8")                                           # Allen Moore connection, 1%
    assert s["out_of_race"] and "playing for wins now" in s["why"][0]


def test_urgency_peaks_on_the_bubble():
    assert W.urgency(0.35) == 1.0
    assert W.urgency(0.15) == W.urgency(0.60) == 0.5
    assert W.urgency(0.90) == W.urgency(0.02) == 0.0


def test_flags_come_from_the_odds():
    assert W.flag(0.6, 0.0) == "LIKELY"
    assert W.flag(0.35, -0.01) == "EXPLOIT"
    assert W.flag(0.35, 0.01) == "NEEDS_PITCH"
    assert W.flag(0.05, 0.0) == "LONGSHOT"


def test_offer_7_is_a_loss_for_us_once_the_free_pickup_is_netted(world):
    """Chris wouldn't send Rachaad White + Egbuka for Davante Adams (out until week 12). Its
    raw gain is the free agent the open spot picks up, which needs no trade."""
    from mega import trade_theses as TT
    ctx, _, tid, by_name = world
    me = tid[FX["my_team"]]
    core = te.evaluate_core(ctx, me, tid["BillsMafia"], [by_name["Rachaad White"], by_name["Emeka Egbuka"]],
                            [by_name["Davante Adams"]])
    _, swap_val = TT.free_swap(ctx, ctx.base[me].ids)
    assert core["dMe"] > 0 and core["_after"]["me"].value - swap_val <= 0


def test_week_weights_follow_the_title_odds():
    import numpy as np
    from mega import title_odds as T
    n = 4000
    rng = np.random.default_rng(1)
    won7 = rng.random(n) < 0.5
    won16 = rng.random(n) < 0.5
    title = won7 & won16                                         # A wins it all by winning both
    m = T.Model.__new__(T.Model)
    m._base_run = {"teams": ["A", "B"], "made": np.array([np.ones(n, bool), np.zeros(n, bool)]),
                   "title": np.array([title, np.zeros(n, bool)]),
                   "won": {(0, 7): won7, (1, 7): ~won7, (0, 8): won7, (1, 8): ~won7},
                   "games": [(16, np.zeros(n, int), np.ones(n, int), np.where(won16, 0, 1))]}
    w = m.week_weights()
    assert w["A"][7] == pytest.approx(0.5, abs=0.03)             # P(title | won 7) - P(title | lost 7)
    assert w["A"][16] == pytest.approx(0.5, abs=0.03)            # plays every season, P(title | won 16)
    assert w["B"] == {7: 1.0, 8: 1.0}                           # out of the race: the next weeks, evenly


# ---- our-side flags (mega/trade_theses.py our_flags), on the same frozen league
def _flags(world, give, get, facts):
    from mega import trade_theses as TT
    ctx, _, tid, by_name = world
    me = tid[FX["my_team"]]
    g, t = [by_name[n] for n in give], [by_name[n] for n in get]
    partner = next(o["partner"] for o in FX["offers"] if o["get"] == get)
    after = te.evaluate_core(ctx, me, tid[partner], g, t)["_after"]["me"].ids
    return TT.our_flags(ctx, g, t, ctx.base[me].ids, after, {by_name[k]: v for k, v in facts.items()}, 7,
                        TT.slot_typical(ctx))


def test_a_shrinking_role_and_a_rebound_bet_are_flagged(world):
    facts = {"Chase Brown": {"share_early": 0.27, "share": 0.18, "ros_pg": 14.8, "pts_pg": 10.6, "games": 5,
                             "team": "CIN"}}
    kinds = {f["kind"]: f["text"] for f in _flags(world, ["Christian Watson"], ["Chase Brown"], facts)}
    assert "from 27% to 18%" in kinds["role_risk"] and "14.8/g from here" in kinds["rebound"]
    steady = {"Chase Brown": {"share_early": 0.20, "share": 0.19, "ros_pg": 11.0, "pts_pg": 10.6, "games": 5}}
    assert not {"role_risk", "rebound"} & {f["kind"] for f in _flags(world, ["Christian Watson"], ["Chase Brown"], steady)}


def test_an_injured_player_is_flagged_for_his_return_not_his_role(world):
    facts = {"Davante Adams": {"share_early": 0.16, "share": 0.08, "games": 6}}       # hurt mid-game
    flags = _flags(world, ["Rachaad White", "Emeka Egbuka"], ["Davante Adams"], facts)
    assert "Davante Adams is out until week 12, back for the playoffs." in [f["text"] for f in flags]
    assert "role_risk" not in {f["kind"] for f in flags}


def test_the_thinnest_spot_skips_what_waivers_already_fix(world):
    from mega import trade_theses as TT
    ctx, _, tid, by_name = world
    me = ctx.base[tid[FX["my_team"]]].ids
    assert by_name["Patrick Mahomes"] in ctx.fa                   # beats Bryce Young, so QB is a pickup
    assert TT.thinnest(ctx, me, TT.slot_typical(ctx))["slot"] != "QB"
