"""§20.1 The Odds API client: parsing, week matching and the credit guard.

Nothing here touches the network. The guard tests matter most — the free plan is 500
credits a month and one careless sweep is 112 of them.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from mega import odds as O


# ---------------------------------------------------------------- parsing
_EVENT = {
    "id": "evt1", "home_team": "Minnesota Vikings", "away_team": "Green Bay Packers",
    "bookmakers": [{
        "key": "draftkings",
        "markets": [
            {"key": "player_reception_yds", "outcomes": [
                {"name": "Over", "description": "Justin Jefferson", "price": -115, "point": 78.5},
                {"name": "Under", "description": "Justin Jefferson", "price": -105, "point": 78.5}]},
            {"key": "player_anytime_td", "outcomes": [
                {"name": "Yes", "description": "Justin Jefferson", "price": 130},
                {"name": "No", "description": "Justin Jefferson", "price": -160}]},
        ]}]}


def test_over_and_under_fold_into_one_row(monkeypatch):
    """A price is only de-viggable next to its other side, so the two outcomes have to
    land on the same row."""
    monkeypatch.setattr(O, "_get", lambda *a, **k: type("R", (), {"json": lambda s: _EVENT})())
    df = O.event_props("evt1")
    rec = df[df["market"] == "player_reception_yds"].iloc[0]
    assert rec["line"] == 78.5
    assert rec["price_over"] == -115 and rec["price_under"] == -105
    assert len(df) == 2                      # two markets, not four outcomes


def test_yes_no_markets_are_read_as_over_under(monkeypatch):
    """player_anytime_td prices Yes/No, not Over/Under. Reading only Over/Under would
    silently drop every touchdown market."""
    monkeypatch.setattr(O, "_get", lambda *a, **k: type("R", (), {"json": lambda s: _EVENT})())
    td = O.event_props("evt1")
    td = td[td["market"] == "player_anytime_td"].iloc[0]
    assert td["price_over"] == 130 and td["price_under"] == -160


def test_full_team_names_become_nflverse_codes(monkeypatch):
    monkeypatch.setattr(O, "_get", lambda *a, **k: type("R", (), {"json": lambda s: _EVENT})())
    df = O.event_props("evt1")
    assert set(df["home"]) == {"MIN"} and set(df["away"]) == {"GB"}


# ---------------------------------------------------------------- week matching
def test_events_are_matched_to_a_week_by_team_pair(monkeypatch):
    """Matched on the pair, not on kickoff time: a flexed Sunday-nighter and the Monday
    game closing the previous week both sit on the wrong side of any time cut."""
    ev = pd.DataFrame([{"event_id": "a", "home": "Minnesota Vikings", "away": "Green Bay Packers"},
                       {"event_id": "b", "home": "Dallas Cowboys", "away": "Chicago Bears"}])
    sched = pd.DataFrame([{"week": 5, "home_team": "MIN", "away_team": "GB"},
                          {"week": 6, "home_team": "DAL", "away_team": "CHI"}])
    import mega.season as S
    monkeypatch.setattr(S, "schedules", lambda s: sched)
    keep = O._week_events(ev, 2026, 5)
    assert list(keep["event_id"]) == ["a"]


def test_home_and_away_flipped_still_matches(monkeypatch):
    """The book's idea of home and nflverse's must not have to agree for a game to count."""
    ev = pd.DataFrame([{"event_id": "a", "home": "Green Bay Packers", "away": "Minnesota Vikings"}])
    sched = pd.DataFrame([{"week": 5, "home_team": "MIN", "away_team": "GB"}])
    import mega.season as S
    monkeypatch.setattr(S, "schedules", lambda s: sched)
    assert len(O._week_events(ev, 2026, 5)) == 1


# ---------------------------------------------------------------- the credit guard
def test_a_sweep_it_cannot_pay_for_is_refused(monkeypatch, tmp_path):
    """Half a swept week is worse than none: the priced games would outrank the unpriced
    ones in every table that compares players across games."""
    monkeypatch.setattr(O, "LEDGER", tmp_path / "b.json")
    O._save_ledger({"remaining": 50})
    monkeypatch.setattr(O, "key", lambda: "fake")
    ev = pd.DataFrame([{"event_id": f"e{i}", "home": "Minnesota Vikings",
                        "away": "Green Bay Packers"} for i in range(16)])
    monkeypatch.setattr(O, "events", lambda: ev)
    monkeypatch.setattr(O, "_week_events", lambda e, s, w: e)
    called = []
    monkeypatch.setattr(O, "event_props", lambda *a, **k: called.append(1))
    out, msg = O.refresh(2026, 5, force=True)
    assert not called, "must not spend a single credit once it knows it cannot finish"
    assert "skipped" in msg and "112 credits" in msg


def test_a_sweep_it_can_afford_proceeds(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "LEDGER", tmp_path / "b.json")
    monkeypatch.setattr(O, "DATA", tmp_path)
    monkeypatch.setattr(O, "props_csv", lambda s, w: tmp_path / f"p_{s}_{w}.csv")
    O._save_ledger({"remaining": 500})
    monkeypatch.setattr(O, "key", lambda: "fake")
    ev = pd.DataFrame([{"event_id": "e1", "home": "Minnesota Vikings", "away": "Green Bay Packers"}])
    monkeypatch.setattr(O, "events", lambda: ev)
    monkeypatch.setattr(O, "_week_events", lambda e, s, w: e)
    monkeypatch.setattr(O, "event_props", lambda *a, **k: pd.DataFrame(
        [{"player": "Justin Jefferson", "market": "player_reception_yds", "line": 78.5,
          "price_over": -115, "price_under": -105, "book": "dk", "home": "MIN",
          "away": "GB", "event_id": "e1"}]))
    monkeypatch.setattr(O, "_attach_ids", lambda d: d.assign(gsis_id="00-0035676", team="MIN"))
    out, msg = O.refresh(2026, 5, force=True)
    assert len(out) == 1 and "1 prop lines" in msg


def test_no_key_is_a_clean_off_switch(monkeypatch):
    monkeypatch.setattr(O, "key", lambda: "")
    out, msg = O.refresh(2026, 5, force=True)
    assert out.empty and "ODDS_API_KEY" in msg


def test_the_ledger_reads_the_accounts_own_numbers(monkeypatch, tmp_path):
    """Trusting a local tally of what we think we spent drifts; the header does not."""
    monkeypatch.setattr(O, "LEDGER", tmp_path / "b.json")
    class R:
        headers = {"x-requests-remaining": "388", "x-requests-used": "112",
                   "x-requests-last": "7"}
    O._note(R())
    assert O.remaining() == 388
    assert json.loads((tmp_path / "b.json").read_text())["used"] == 112


def test_a_missing_week_file_forces_a_sweep_regardless_of_the_clock(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "LEDGER", tmp_path / "b.json")
    monkeypatch.setattr(O, "props_csv", lambda s, w: tmp_path / "nope.csv")
    O._save_ledger({"swept": dt.datetime.now(dt.timezone.utc).isoformat()})
    assert O.due(150, 2026, 7) is True          # just swept, but not this week


def test_a_recent_sweep_of_this_week_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "LEDGER", tmp_path / "b.json")
    f = tmp_path / "p.csv"
    f.write_text("gsis_id\n")
    monkeypatch.setattr(O, "props_csv", lambda s, w: f)
    O._save_ledger({"swept": dt.datetime.now(dt.timezone.utc).isoformat()})
    assert O.due(150, 2026, 7) is False
