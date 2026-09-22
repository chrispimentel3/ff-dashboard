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
