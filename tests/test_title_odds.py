"""HANDOFF v1.3 Pass 3: player-level title odds on common random numbers (mega/title_odds.py)."""
from __future__ import annotations

import pytest

from mega import sim as SIM
from mega import title_odds as T
from mega import trade_engine as te

CFG = {"slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flexCount": 1,
       "flexEligible": ["RB", "WR"], "rosterSize": 8}
TEAMS = [f"T{i}" for i in range(8)]


def _model(strength=None, n=3000):
    strength = strength or {t: 1.0 for t in TEAMS}
    players, teams = {}, []
    for i, t in enumerate(TEAMS):
        ids = []
        for pos, base in (("QB", 18), ("RB", 12), ("RB", 10), ("WR", 12), ("WR", 10), ("WR", 8), ("TE", 8), ("RB", 5)):
            pid = f"{t}_{pos}{len(ids)}"
            players[pid] = {"id": pid, "name": pid, "pos": pos, "nfl": "KC", "ppg": base * strength[t]}
            ids.append(pid)
        teams.append({"id": i, "name": t, "roster": ids})
    ctx = te.build_context({"teams": teams, "players": players, "freeAgents": []}, CFG)
    sched = SIM._round_robin(TEAMS, list(range(3, 15)))
    season = SIM.Season(teams=TEAMS, means={}, schedule=sched, playoff_teams=6, byes=2)
    return T.Model(season, ctx, {}, {"QB": 3.0, "RB": 2.6, "WR": 2.1, "TE": 1.6},
                   {t["name"]: list(t["roster"]) for t in teams}, n=n)


def test_odds_add_up():
    o = _model().odds()
    assert sum(v["p_playoffs"] for v in o.values()) == pytest.approx(6.0)
    assert sum(v["p_title"] for v in o.values()) == pytest.approx(1.0)


def test_a_no_op_trade_moves_nothing():
    """§5 acceptance: title odds for a no-op trade = baseline (±0.2pp); on CRN it is exact."""
    m = _model()
    d = m.delta({"T0": list(m.rosters["T0"])}, ("T0",))["T0"]
    assert d["d_title"] == 0.0 and d["d_playoffs"] == 0.0 and d["noise"]


def test_players_carry_their_dice_to_a_new_team():
    """Swap two identical-value players between teams: the lineups are unchanged in value,
    and because each player's draws are keyed to him, not to the slot, the swap moves the
    points he scores with him — the delta is the real effect, not reshuffled noise."""
    m = _model()
    a, b = m.rosters["T0"], m.rosters["T1"]
    d = m.delta({"T0": [x for x in a if not x.endswith("TE6")] + ["T1_TE6"],
                 "T1": [x for x in b if not x.endswith("TE6")] + ["T0_TE6"]}, ("T0", "T1"))
    assert abs(d["T0"]["d_title"]) < 0.05 and abs(d["T1"]["d_title"]) < 0.05


def test_a_stronger_roster_wins_more_titles():
    o = _model({**{t: 1.0 for t in TEAMS}, "T0": 1.3}).odds()
    assert o["T0"]["p_title"] == max(v["p_title"] for v in o.values())


def test_an_upgrade_raises_title_odds_beyond_the_noise_band():
    m = _model(n=4000)
    ids = m.rosters["T0"]
    m.ctx.players["FA_WR"] = {"id": "FA_WR", "name": "FA_WR", "pos": "WR", "nfl": "KC", "ppg": 22.0}
    d = m.delta({"T0": [x for x in ids if not x.endswith("RB7")] + ["FA_WR"]}, ("T0",))["T0"]
    assert d["d_title"] > 0 and not d["noise"]


@pytest.mark.parametrize("p,want", [(0.9, "protect"), (0.5, "balanced"), (0.2, "swing")])
def test_posture(p, want):
    assert T.posture(p) == want
