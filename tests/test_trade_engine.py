"""The JS suite's 11 tests, ported alongside the engine.

Rule from the handoff: any engine change must keep these green, and any new behaviour
needs a hand-checked case added here. Expected numbers are the ones worked out by hand in
`test_trade_engine.js` — they are the contract, not a snapshot of what this code happens
to produce.

    .venv/bin/python -m pytest tests/test_trade_engine.py -q
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
from itertools import combinations

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mega.trade_engine import create_engine  # noqa: E402

LEAGUE = json.loads((pathlib.Path(__file__).parent / "fixtures" / "sample_league.json").read_text())
ALL_FLAGS = ["LIKELY", "EXPLOIT", "NEEDS_PITCH", "LONGSHOT"]
WIDE = {"includeFlags": ALL_FLAGS, "minDeltaMe": -float("inf"), "topN": 10**6}


@pytest.fixture(scope="module")
def eng():
    return create_engine(LEAGUE)


# Replacement levels (best FA): QB 11, RB 7, WR 7.5, TE 5.5
def test_baseline_mine(eng):
    # Starters: QB 18 | RB 16,13 | WR 17,14 | TE 10 | FLEX WR 11 -> 99
    # Depth: QB2 .10*(12-11)=.1 | RB3 .15*(9-7)=.3 | WR4 .15*(8-7.5)=.075 -> .475
    assert eng.team_value(1) == pytest.approx(99.475)


def test_baseline_theirs(eng):
    assert eng.team_value(2) == pytest.approx(95.2)


def test_hand_checked_two_for_one(eng):
    r = eng.evaluate_trade(1, 2, ["m_wr2", "m_rb3"], ["a_wr1"])
    assert r["me"]["valueAfter"] == pytest.approx(104.175)
    assert r["dMe"] == pytest.approx(4.7)
    assert [p["id"] for p in r["me"]["adds"]] == ["fa_qb"]
    assert r["them"]["valueAfter"] == pytest.approx(90.4)
    assert r["dThem"] == pytest.approx(-4.8)
    assert [p["id"] for p in r["them"]["drops"]] == ["a_wr5"]
    assert r["market"]["ratio"] == pytest.approx(0.63593, abs=1e-4)
    assert r["flag"] == "LONGSHOT"
    assert r["shape"] == "2-for-1"


def test_lineup_matches_brute_force():
    """Greedy fill is exact for 1QB/2RB/2WR/1TE/1FLEX — proven, not assumed."""
    seed = 42

    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) % 2147483648
        return seed / 2147483648

    def brute(vals):
        best = -float("inf")
        for chosen in combinations(vals, 7):
            c = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
            for pos, _ in chosen:
                c[pos] += 1
            if c["QB"] == 1 and c["RB"] >= 2 and c["WR"] >= 2 and c["TE"] >= 1:
                best = max(best, sum(v for _, v in chosen))
        return best

    for t in range(300):
        players, roster = {}, []
        counts = {"QB": 2, "RB": 3 + int(rnd() * 3), "WR": 3 + int(rnd() * 3), "TE": 1 + int(rnd() * 3)}
        for pos, n in counts.items():
            for i in range(n):
                pid = f"{pos}{i}"
                players[pid] = {"id": pid, "pos": pos, "ppg": round(rnd() * 250) / 10}
                roster.append(pid)
        e = create_engine({"settings": {"rosterSize": len(roster)},
                           "teams": [{"id": 1, "roster": roster}], "players": players, "freeAgents": []})
        got = e.lineup(roster).starter_pts
        assert got == pytest.approx(brute([(players[i]["pos"], players[i]["ppg"]) for i in roster])), f"roster {t}"


def test_search_invariants(eng):
    out = eng.find_trades(1, ["m_wr2"], WIDE)
    assert out["evaluated"] > 0
    assert out["matched"] + out["padded"] == out["evaluated"]
    for r in out["results"]:
        assert "m_wr2" in r["giveIds"]
        for p in r["give"] + r["get"]:
            assert p["pos"] not in ("K", "DEF")
        assert len(r["me"]["before"]["starters"]) == 7
        assert len(r["them"]["before"]["starters"]) == 7
        assert 15 - len(r["give"]) + len(r["get"]) - len(r["me"]["drops"]) + len(r["me"]["adds"]) == 15
        assert 15 - len(r["get"]) + len(r["give"]) - len(r["them"]["drops"]) + len(r["them"]["adds"]) == 15
    for a, b in zip(out["results"], out["results"][1:]):
        assert a["dMe"] >= b["dMe"] - 1e-12
    # 13 tradeable on each of 2 opponents: 1-for-1 = 26 · 2-for-1 = 12 add-ons × 26 · 1-for-2 = 2 × C(13,2)
    assert out["evaluated"] == 26 + 312 + 156


def test_default_filters(eng):
    for r in eng.find_trades(1, ["m_wr2"])["results"]:
        assert r["flag"] != "LONGSHOT"
        assert r["dMe"] > 0


def test_padding_pruned_sweeteners_kept(eng):
    out = eng.find_trades(1, ["m_wr2"], WIDE)
    keys = {",".join(sorted(r["giveIds"])) + ">" + ",".join(sorted(r["getIds"])) for r in out["results"]}
    assert "m_wr2>b_wr3" in keys                     # the plain 1-for-1
    assert "m_wr2>b_qb2,b_wr3" not in keys           # QB2 would just be dropped: padding
    assert "m_rb3,m_wr2>b_wr2" in keys               # RB3 starts for RB-thin B: a real sweetener
    assert out["padded"] > 0


def test_receive_and_partner_filters(eng):
    out = eng.find_trades(1, ["m_wr2"], {**WIDE, "receivePositions": ["TE"], "partners": [3]})
    for r in out["results"]:
        assert r["partner"]["id"] == 3
        assert any(p["pos"] == "TE" for p in r["get"])


def test_horizon_is_bye_aware():
    l2 = copy.deepcopy(LEAGUE)
    l2["players"]["m_qb1"]["bye"] = 5
    e = create_engine(l2, {"horizon": {"weeks": [5, 6]}})
    # Week 5 the QB1 is out, QB2 (12) starts -> 93.375; week 6 is the usual 99.475; mean 96.425
    assert e.team_value(1) == pytest.approx(96.425)


def test_two_players_runs_two_for_x_only(eng):
    out = eng.find_trades(1, ["m_wr2", "m_rb3"],
                          {**WIDE, "shapes": ["2-for-1", "2-for-2"]})
    assert out["evaluated"] == 26 + 156


def test_bad_input_rejected(eng):
    with pytest.raises(ValueError, match="not on team"):
        eng.find_trades(1, ["a_wr1"])
    with pytest.raises(ValueError, match="not tradeable"):
        eng.find_trades(1, ["m_k"])
    with pytest.raises(ValueError, match="missing"):
        create_engine({"teams": [{"id": 1, "roster": ["ghost"]}], "players": {}})


def test_a_team_search_is_every_offer_to_that_team_once(eng):
    from mega import trade_league as TL
    out = TL.find_with_team(eng, 1, 3, top_n=10**6, include_flags=tuple(ALL_FLAGS))
    rs = out["results"]
    assert rs and all(r["partner"]["id"] == 3 for r in rs)
    keys = [(frozenset(r["giveIds"]), frozenset(r["getIds"])) for r in rs]
    assert len(keys) == len(set(keys))                                  # a pair found from either side is one offer
    assert [r["dMe"] for r in rs] == sorted((r["dMe"] for r in rs), reverse=True)
    assert (frozenset(["m_wr2"]), frozenset(["b_wr3"])) in keys          # the plain 1-for-1
    assert (frozenset(["m_rb3", "m_wr2"]), frozenset(["b_wr2"])) in keys  # a sweetener found from m_rb3 or m_wr2
    pad = eng.ctx.cfg["padTolerance"]                                    # no pair is a single offer plus a sweetener nobody needs
    one = {(r["giveIds"][0], frozenset(r["getIds"])): r for r in rs if len(r["giveIds"]) == 1}
    for r in rs:
        for g in r["giveIds"] if len(r["giveIds"]) == 2 else []:
            s = one.get((g, frozenset(r["getIds"])))
            assert not (s and s["dMe"] >= r["dMe"] - pad and s["dThem"] >= r["dThem"] - pad)
    t = TL.targets_of(rs)
    assert [x["best_d_me"] for x in t] == sorted((x["best_d_me"] for x in t), reverse=True)
    assert {x["name"] for x in t} <= {p["name"] for p in LEAGUE["players"].values()}
    assert TL.find_with_team(eng, 1, 1)["results"] == []                # not a trade with yourself


def test_padding_is_any_piece_that_changes_nothing():
    from mega import trade_league as TL
    o = lambda g, r, me, them: {"giveIds": g, "getIds": r, "dMe": me, "dThem": them}
    single, pair = o(["a"], ["x"], 5.0, -1.0), o(["a", "b"], ["x"], 5.02, -1.0)
    triple = o(["a", "b", "c"], ["x"], 5.02, -1.0)        # a on its own already does it
    real = o(["a", "d"], ["x"], 6.5, -1.0)                 # d adds 1.5 pts for me: a real sweetener
    extra = o(["a"], ["x", "y"], 5.0, -1.0)                # taking y too changes nothing
    kept, n = TL.drop_padding([single, pair, triple, real, extra], 0.05)
    assert kept == [single, real] and n == 3


def test_a_big_search_adds_2_for_2_3_for_1_and_3_for_2_and_nets_every_spot_it_opens(eng):
    from mega import trade_league as TL
    wide = tuple(ALL_FLAGS)
    two = TL.find_with_team(eng, 1, 2, top_n=10**6, include_flags=wide, size=2)
    big = TL.find_with_team(eng, 1, 2, top_n=10**6, include_flags=wide, size=3)
    shapes = lambda out: {r["shape"] for r in out["results"]}
    assert shapes(two) <= {"1-for-1", "2-for-1", "1-for-2"}
    assert shapes(big) - shapes(two) <= {"2-for-2", "3-for-1", "3-for-2"} and shapes(big) >= shapes(two)
    assert TL.find_with_team(eng, 1, 2, include_flags=wide, size=1)["results"] and \
        shapes(TL.find_with_team(eng, 1, 2, top_n=10**6, include_flags=wide, size=1)) == {"1-for-1"}
    rows, _ = TL.net_free_swap(eng, 1, big["results"])
    assert all(r.get("netted") for r in rows if len(r["giveIds"]) > len(r["getIds"]) and r.get("fa_add"))


def test_a_teams_needs_and_spare_players():
    from mega import trade_league as TL
    e = create_engine(LEAGUE)
    for tid in (2, 3):
        p = TL.team_profile(e, tid)
        assert all(n["gap"] >= TL.NEED_GAP for n in p["needs"])
        assert [n["gap"] for n in p["needs"]] == sorted((n["gap"] for n in p["needs"]), reverse=True)
        assert len(p["spare"]) <= 3 and all(s["ppg"] > 0 for s in p["spare"])
    assert "RB" in {n["pos"] for n in TL.team_profile(e, 3)["needs"]}      # "RB-thin B"
    assert TL.team_profile(e, 99) == {"needs": [], "spare": []}


def test_lineup_detail_can_wait_until_the_list_is_cut(eng):
    """The big search leaves rows without lineup detail and nets them lazily; ensure_detail
    must give the same rows, numbers included, as netting with the detail up front."""
    from mega import trade_league as TL
    big = TL.find_with_team(eng, 1, 2, top_n=10**6, include_flags=tuple(ALL_FLAGS), size=3)
    eager = TL.ensure_detail(eng, 1, TL.net_free_swap(eng, 1, big["results"])[0])
    lazy, _ = TL.net_free_swap(eng, 1, big["results"], lite=True)
    full = TL.ensure_detail(eng, 1, lazy)
    key = lambda r: (tuple(sorted(r["giveIds"])), tuple(sorted(r["getIds"])))
    a, b = {key(r): r for r in eager}, {key(r): r for r in full}
    assert a.keys() == b.keys() and a
    for k, r in b.items():
        assert r["dMe"] == pytest.approx(a[k]["dMe"]) and "me" in r and "them" in r and "give" in r
        assert r["me"]["valueAfter"] == pytest.approx(a[k]["me"]["valueAfter"])


def test_near_duplicate_offers_fold_into_one_headline(eng):
    from mega import trade_league as TL
    rs = TL.find_with_team(eng, 1, 2, top_n=10**6, include_flags=tuple(ALL_FLAGS), size=3)["results"]
    rows, _ = TL.net_free_swap(eng, 1, rs)
    rows, _ = TL.drop_padding(rows, eng.ctx.cfg["padTolerance"])
    out = TL.collapse_variants(eng.ctx, rows)
    mv = lambda i: __import__("mega.trade_engine", fromlist=["x"]).market_value(eng.ctx.players[i], eng.ctx.cfg)
    keys = [(r["shape"], max(r["giveIds"], key=mv), max(r["getIds"], key=mv)) for r in out]
    assert len(keys) == len(set(keys)) and len(out) < len(rows)           # one per headline
    assert sum(1 + r["n_variants"] for r in out) == len(rows)             # nothing lost, only folded
    assert all(len(r["variants"]) == min(r["n_variants"], TL.VARIANTS_SHOWN) for r in out)
    firsts = {k: i for i, k in reversed(list(enumerate(
        (r["shape"], max(r["giveIds"], key=mv), max(r["getIds"], key=mv)) for r in rows)))}
    assert [firsts[k] for k in keys] == sorted(firsts[k] for k in keys)   # order of first appearance kept


def test_a_player_ruled_out_is_priced_for_the_games_he_plays():
    # QB1 (18) is ruled out of a quarter of his remaining games: priced at 13.5, so the
    # team loses about 4.5 pts/wk (a little less where QB2's depth weight shifts)
    l2 = copy.deepcopy(LEAGUE)
    l2["players"]["m_qb1"]["out"] = {"share": 0.25, "back": 8}
    e = create_engine(l2)
    assert e.team_value(1) < create_engine(LEAGUE).team_value(1) - 4.0
    assert e.ctx.players["m_qb1"]["out"]["back"] == 8
    assert create_engine(LEAGUE).team_value(1) == pytest.approx(99.475)   # absent = untouched


def test_lane_why_says_when_he_is_out():
    from mega.needs import lane_why
    r = {"lane": "stash", "mechanism": "START", "start": 0.5, "out_status": "IR", "out_back": 8.0}
    assert lane_why(r).startswith("on injured reserve, back week 8 — ")
    assert not lane_why({**r, "out_status": float("nan"), "out_back": float("nan")}).startswith("on injured")


def test_lane_why_season_out():
    from mega.needs import lane_why
    r = {"lane": "stash", "mechanism": "START", "start": 0.5, "out_status": "Out", "out_back": 18.0}
    assert lane_why(r).startswith("out for the season — ")
