"""The rules the waiver board has to obey, pinned to the case that exposed them.

Chris runs Kittle and Kelce at tight end in a league whose flex is RB/WR only. The old
board kept offering him a third one, because it ranked free agents on talent and never
looked at his lineup. These tests exist so it cannot happen again.
"""
from __future__ import annotations

import pytest

from mega import needs
from mega import trade_engine as te


def _league():
    """A 2-team league shaped like the real problem: two strong TEs, RB/WR-only flex."""
    players = {}

    def add(pid, pos, ppg, ecr=None):
        players[pid] = {"id": pid, "name": pid, "pos": pos, "ppg": ppg, "ecr": ecr}
        return pid

    mine = [
        add("my_qb", "QB", 18), add("my_qb2", "QB", 12),
        add("my_rb1", "RB", 16), add("my_rb2", "RB", 13), add("my_rb3", "RB", 9),
        add("my_wr1", "WR", 17), add("my_wr2", "WR", 14), add("my_wr3", "WR", 11),
        add("my_te1", "TE", 13),          # Kittle
        add("my_te2", "TE", 12),          # Kelce
    ]
    theirs = [add(f"th_{i}", p, v) for i, (p, v) in enumerate(
        [("QB", 17), ("RB", 15), ("RB", 12), ("WR", 16), ("WR", 12),
         ("TE", 9), ("RB", 8), ("WR", 9), ("QB", 11), ("TE", 6)])]
    fa = [
        add("fa_te_stud", "TE", 11),      # better than most TEs, still useless to me
        add("fa_te_mid", "TE", 8),
        add("fa_te_low", "TE", 6),
        add("fa_wr", "WR", 12),
        add("fa_wr2", "WR", 8),
        add("fa_wr3", "WR", 7),
        add("fa_rb", "RB", 10),
        add("fa_rb2", "RB", 7),
        add("fa_rb3", "RB", 6),
        add("fa_qb", "QB", 19),           # a genuine upgrade on my starter
        add("fa_qb2", "QB", 13),
        add("fa_qb3", "QB", 11),
    ]
    return {
        "settings": {"rosterSize": 10},
        "teams": [{"id": 1, "name": "Mine", "roster": mine},
                  {"id": 2, "name": "Theirs", "roster": theirs}],
        "players": players,
        "freeAgents": fa,
    }


@pytest.fixture
def ctx():
    cfg = {"slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flexCount": 1,
           "flexEligible": ["RB", "WR"], "rosterSize": 10,
           "replacementRank": needs.REPLACEMENT_RANK}
    return te.build_context(_league(), cfg)


def test_third_te_can_never_help(ctx):
    """The bug, stated as a test: with two TEs and no TE flex, a third one never helps.

    Not merely zero — negative for the best of them, because he cannot enter the lineup and
    the roster still has to cut someone to fit him, and the cheapest cut is a real player.
    """
    base = te.team_value(ctx.teams[1]["roster"], ctx)
    gains = {pid: needs.add_value(ctx, 1, pid, base)["gain"]
             for pid in ("fa_te_stud", "fa_te_mid", "fa_te_low")}
    assert all(g <= 0 for g in gains.values()), gains
    assert all(needs.label(g) == "STASH" for g in gains.values()), gains
    # and none of them gets a dollar
    from mega import faab
    assert all(faab.suggest(g, 90, 3)["bid"] == 0 for g in gains.values()), gains


def test_a_real_upgrade_still_scores(ctx):
    """The board must not simply return zero for everything."""
    base = te.team_value(ctx.teams[1]["roster"], ctx)
    av = needs.add_value(ctx, 1, "fa_qb", base)
    assert av["gain"] > 0
    assert av["starts"] is True
    assert av["drop"]                      # something had to be cut to fit him


def test_reason_names_the_blockers(ctx):
    why = needs.why_zero(ctx, 1, "fa_te_stud")
    assert "my_te1" in why
    assert "flex" in why.lower()


def test_replacement_rank_is_robust_to_one_hot_estimate():
    """A single inflated free agent must not set the bar for a whole position."""
    lg = _league()
    lg["players"]["fa_qb_fluke"] = {"id": "fa_qb_fluke", "name": "fluke", "pos": "QB",
                                    "ppg": 40.0, "ecr": None}
    lg["freeAgents"].append("fa_qb_fluke")
    base_cfg = {"slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flexCount": 1,
                "flexEligible": ["RB", "WR"], "rosterSize": 10}

    top = te.build_context(lg, {**base_cfg, "replacementRank": 1})
    robust = te.build_context(lg, {**base_cfg, "replacementRank": 3})
    assert top.repl[None]["QB"] == 40.0           # the engine's default takes the max
    assert robust.repl[None]["QB"] < 20.0         # ours ignores the outlier


def test_labels_follow_the_thresholds():
    assert needs.label(1.0) == "UPGRADE"
    assert needs.label(needs.UPGRADE) == "UPGRADE"
    assert needs.label(0.1) == "DEPTH"
    assert needs.label(0.0) == "STASH"
