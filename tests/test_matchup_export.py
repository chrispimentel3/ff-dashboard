"""tools/export_web.py: the Matchup page's opponent lookup and the roster guard behind it."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd

from mega import data


def _export():
    p = pathlib.Path(__file__).resolve().parent.parent / "tools" / "export_web.py"
    spec = importlib.util.spec_from_file_location("export_web", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_opponent_is_whoever_shares_my_fixture(monkeypatch):
    from mega import config, yahoo
    E = _export()
    me = config.MY_TEAM
    fx = pd.DataFrame([{"week": 10, "home": "A", "away": me}, {"week": 10, "home": "B", "away": "C"},
                       {"week": 11, "home": me, "away": "B"}])
    monkeypatch.setattr(yahoo, "cached_fixtures", lambda: fx)
    assert E.opponent(10) == "A" and E.opponent(11) == "B" and E.opponent(12) is None


def test_another_teams_roster_never_falls_back_to_mine(monkeypatch):
    from mega import yahoo
    monkeypatch.setattr(yahoo, "cached_rosters", lambda: pd.DataFrame(columns=["team", "player", "slot"]))
    r, src = data.my_roster("Nobody")
    assert r.empty and "not in" in src
