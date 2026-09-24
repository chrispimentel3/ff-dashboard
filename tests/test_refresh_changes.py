"""The change report in tools/refresh.py.

The defense cases are regressions. Yahoo renamed defenses between two scrapes on
2026-09-23 and the report called it four roster moves — twice, because my first fix keyed
on `pos` and `nfl_team`, which are exactly the columns Yahoo leaves EMPTY on the rows that
flip. Only the name identifies them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega.ask import team_from_name
from tools import refresh

COLS = ["seat", "team", "slot", "player", "yahoo_id", "nfl_team", "pos"]


def _roster(rows, tmp_path, name="yahoo_rosters.csv"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    return p


def test_team_from_name_resolves_city_and_nickname_alike():
    for city, nick, abbr in (("Baltimore", "Ravens", "BAL"), ("Houston", "Texans", "HOU"),
                             ("Green Bay", "Packers", "GB"), ("Denver", "Broncos", "DEN")):
        assert team_from_name(city) == abbr
        assert team_from_name(nick) == abbr


def test_a_real_player_name_resolves_to_nothing():
    """The whole safety of keying a blank-pos row by its name rests on this."""
    for n in ("Puka Nacua", "Bijan Robinson", "", "Josh Allen"):
        assert team_from_name(n) == ""


def test_los_angeles_is_ambiguous_and_stays_blank():
    """Two teams answer to it, so merging on it would be a guess."""
    assert team_from_name("Los Angeles") == ""
    assert team_from_name("Rams") == "LA"
    assert team_from_name("Chargers") == "LAC"


def test_a_renamed_defense_is_not_a_roster_move(tmp_path):
    """The exact 2026-09-23 false positive: nickname one scrape, city the next."""
    old = _roster([["1", "deez nuts", "DEF", "Ravens", "", "", ""]], tmp_path / "a")
    _roster([["1", "deez nuts", "DEF", "Baltimore", "", "", ""]], tmp_path / "b")
    refresh.DATA = tmp_path / "b"
    out = refresh.changes({"yahoo_rosters.csv": old})
    assert not [c for c in out if "Ravens" in c or "Baltimore" in c], out


def test_it_works_when_the_defense_sits_on_the_bench(tmp_path):
    """Denver was in a BN slot with an empty pos — `slot` alone would have missed it."""
    old = _roster([["1", "Crabcakes", "BN", "Broncos", "", "", ""]], tmp_path / "a")
    _roster([["1", "Crabcakes", "BN", "Denver", "", "", ""]], tmp_path / "b")
    refresh.DATA = tmp_path / "b"
    out = refresh.changes({"yahoo_rosters.csv": old})
    assert not [c for c in out if "Broncos" in c or "Denver" in c], out


def test_a_defense_that_really_changed_hands_still_reports(tmp_path):
    old = _roster([["1", "deez nuts", "DEF", "Ravens", "", "BAL", "DEF"]], tmp_path / "a")
    _roster([["2", "L'Omar", "DEF", "Baltimore", "", "BAL", "DEF"]], tmp_path / "b")
    refresh.DATA = tmp_path / "b"
    out = refresh.changes({"yahoo_rosters.csv": old})
    assert any("moved" in c and "BAL DEF" in c for c in out), out


def test_a_real_player_move_is_untouched(tmp_path):
    old = _roster([["1", "deez nuts", "WR", "Puka Nacua", "1", "LA", "WR"]], tmp_path / "a")
    _roster([["2", "L'Omar", "WR", "Puka Nacua", "1", "LA", "WR"]], tmp_path / "b")
    refresh.DATA = tmp_path / "b"
    out = refresh.changes({"yahoo_rosters.csv": old})
    assert any("Puka Nacua" in c and "moved" in c for c in out), out


def test_empty_roster_slots_are_not_players(tmp_path):
    """Yahoo writes an unfilled slot as a player called "(Empty)"."""
    old = _roster([["1", "deez nuts", "BN", "(Empty)", "", "", ""]], tmp_path / "a")
    _roster([["2", "L'Omar", "BN", "(Empty)", "", "", ""]], tmp_path / "b")
    refresh.DATA = tmp_path / "b"
    assert not [c for c in refresh.changes({"yahoo_rosters.csv": old}) if "Empty" in c]


def test_validation_catches_the_failures_that_have_actually_happened(tmp_path):
    refresh.DATA = tmp_path
    (tmp_path / "yahoo_rosters.csv").write_text("player,team,slot,seat\n")
    bad = refresh.validate()
    assert any("rosters" in b for b in bad)


# ---------------------------------------------------------------- transactions
def _tx(tmp_path, rows, name="a"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "yahoo_transactions.csv"
    pd.DataFrame({"text": rows}).to_csv(p, index=False)
    return p


def test_a_reworded_transaction_is_not_a_new_one(tmp_path):
    """Yahoo rewrites defense names between scrapes, so the same three transactions came
    back as "3 new" every time the spelling flipped."""
    old = _tx(tmp_path, [" Carolina Car - DEF $1 Waiver Tampa Bay TB - DEF To Waivers TaylorMade Sep 23, 5:24 am"], "a")
    _tx(tmp_path, [" Panthers Car - DEF $1 Waiver Buccaneers TB - DEF To Waivers TaylorMade Sep 23, 5:24 am"], "b")
    refresh.DATA = tmp_path / "b"
    out = refresh.changes({"yahoo_transactions.csv": old})
    assert any("transactions: 0 new" in c for c in out), out


def test_a_genuinely_new_transaction_still_reports(tmp_path):
    old = _tx(tmp_path, [" Panthers Car - DEF $1 Waiver TaylorMade Sep 23, 5:24 am"], "a")
    _tx(tmp_path, [" Panthers Car - DEF $1 Waiver TaylorMade Sep 23, 5:24 am",
                   " Bijan Robinson Atl - RB $14 Waiver deez nuts Sep 24, 4:01 am"], "b")
    refresh.DATA = tmp_path / "b"
    out = refresh.changes({"yahoo_transactions.csv": old})
    assert any("transactions: 1 new" in c for c in out), out
    assert any("Bijan" in c for c in out), out


def test_two_different_bids_on_the_same_player_are_different_transactions(tmp_path):
    """The key must not collapse genuinely distinct rows."""
    a = " Panthers Car - DEF $1 Waiver TaylorMade Sep 23, 5:24 am"
    b = " Panthers Car - DEF $9 Waiver L'Omar Sep 23, 5:24 am"
    assert refresh._tx_key([a]) != refresh._tx_key([b])
