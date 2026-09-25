"""mega/ids.py's resolve(): the real bug that shipped a Ravens QB's Vegas props under a
Panthers CB's gsis_id — two real players sharing a normalized name ("Lamar Jackson") were
genuinely ambiguous, but resolve() picked the first crosswalk row anyway instead of
honouring its own ambiguity check.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega import ids


@pytest.fixture(autouse=True)
def synthetic_crosswalk(monkeypatch):
    """Two "Lamar Jackson"s (QB/BAL and CB/CAR) plus one unambiguous player, standing in
    for the real crosswalk so this test doesn't depend on live nflverse data."""
    rows = [
        {"name": "Lamar Jackson", "gsis_id": "00-0034796", "pfr_id": "JackLa00",
         "yahoo_id": "31017", "norm": "lamar jackson", "norm_merge": "lamar jackson",
         "team_key": "BAL", "pos_key": "QB", "team": "BAL"},
        {"name": "Lamar Jackson", "gsis_id": "00-0036152", "pfr_id": "JackLa01",
         "yahoo_id": "", "norm": "lamar jackson", "norm_merge": "lamar jackson",
         "team_key": "CAR", "pos_key": "CB", "team": "CAR"},
        {"name": "Christian McCaffrey", "gsis_id": "00-0033280", "pfr_id": "McCaCh01",
         "yahoo_id": "28592", "norm": "christian mccaffrey", "norm_merge": "christian mccaffrey",
         "team_key": "SF", "pos_key": "RB", "team": "SF"},
    ]
    cw = pd.DataFrame(rows)
    monkeypatch.setattr(ids, "crosswalk", lambda: cw)
    monkeypatch.setattr(ids, "overrides", lambda: pd.DataFrame(columns=["yahoo_id", "name", "gsis_id", "note"]))
    monkeypatch.setattr(ids, "team_defenses", lambda: {})


def test_an_ambiguous_name_with_no_team_or_position_hint_stays_unresolved():
    """The exact real-world bug: a name-only lookup ("Lamar Jackson" from a props sweep
    with no team attached) must not silently attach either candidate's gsis_id."""
    df = pd.DataFrame([{"player": "Lamar Jackson", "pos": "", "nfl_team": ""}])
    out, report = ids.resolve(df, name_col="player")
    row = out.iloc[0]
    assert pd.isna(row["gsis_id"]) or row["gsis_id"] is None
    assert bool(row["resolved"]) is False
    assert row["match_method"] == "unresolved"
    assert report["unresolved"] == 1


def test_a_team_hint_disambiguates_an_otherwise_ambiguous_name():
    """The same ambiguous name resolves correctly once a team narrows it — this is what
    lets mega/odds.py recover the real player once it passes the game's home/away teams."""
    df = pd.DataFrame([
        {"player": "Lamar Jackson", "pos": "", "nfl_team": "BAL"},
        {"player": "Lamar Jackson", "pos": "", "nfl_team": "CAR"},
    ])
    out, _ = ids.resolve(df, name_col="player")
    assert out.iloc[0]["gsis_id"] == "00-0034796"
    assert out.iloc[0]["pos"] == "QB"
    assert out.iloc[1]["gsis_id"] == "00-0036152"
    assert out.iloc[1]["pos"] == "CB"


def test_an_unambiguous_name_still_resolves_normally():
    """The fix must not turn every name lookup unresolved — only genuine ties."""
    df = pd.DataFrame([{"player": "Christian McCaffrey", "pos": "", "nfl_team": ""}])
    out, _ = ids.resolve(df, name_col="player")
    row = out.iloc[0]
    assert row["gsis_id"] == "00-0033280"
    assert bool(row["resolved"]) is True
    assert row["match_method"] == "name"
