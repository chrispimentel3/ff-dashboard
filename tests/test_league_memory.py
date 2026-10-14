import math

import pandas as pd
import pytest

from mega import league_memory as LM


def test_rows_parse_into_claims_moves_and_trades():
    claim = LM.parse_row(" Taysom Hill NO - TE NA $8 Waiver Mark Andrews Bal - TE To Waivers Undisputed Dec 24, 4:35 am")
    assert claim == {"kind": "claim", "team": "Undisputed", "date": "Dec 24, 4:35 am", "pos": "TE", "bid": 8,
                     "player": "Taysom Hill"}
    assert LM.parse_row(" Titans Ten - DEF $21 Waiver Kimani Vidal LAC - RB To Waivers Undisputed Dec 24, 4:35 am")["pos"] == "DEF"
    move = LM.parse_row(" Eddy Pineiro SF - K Free Agent Bed Bath & Bijan Dec 20, 2:12 pm")
    assert move["kind"] == "move" and move["team"] == "Bed Bath & Bijan"
    trade = LM.parse_row("J.K. Dobbins Den - RB Q Quentin Johnston LAC - WR Traded to L'Omar Nov 4, 2:17 pm")
    assert trade == {"kind": "trade", "team": "L'Omar", "date": "Nov 4, 2:17 pm"}


def test_commissioner_notes_apostrophes_and_spacing_do_not_split_a_team():
    assert LM._name("Added ( by Commissioner ) PHPLTE ( Christopher )") == LM._name("PHPLTE")
    assert LM._name("L’Omar") == LM._name("L'Omar")
    assert LM._name("Bills Mafia") == LM._name("BillsMafia")


def test_aggression_compares_bids_on_the_same_player():
    # A always bids double the others on the same claim; B and C bid alike
    rows = []
    for i in range(10):
        rows += [{"claim": i, "current": "A", "bid": 20}, {"claim": i, "current": "B", "bid": 10},
                 {"claim": i, "current": "C", "bid": 10}]
    agg = LM.aggression(pd.DataFrame(rows))
    assert agg["A"][0] > 1.2 > agg["B"][0] and agg["A"][1] == 10
    assert agg["B"][0] == pytest.approx(agg["C"][0])
    shrunk = LM.aggression(pd.DataFrame(rows[:3]))      # one claim says less than ten
    assert 1.0 < shrunk["A"][0] < agg["A"][0]


def test_trade_habit_is_his_rate_against_the_league_shrunk():
    trades = [{"year": 2024, "date": f"Nov {d}, 1:00 pm", "kind": "trade", "current": t}
              for d in range(1, 7) for t in ("A", "B")]                      # A and B trade with each other 6 times
    out = LM.trade_habits(pd.DataFrame(trades), {"A": 3, "B": 3, "C": 3})
    assert out["A"][1] == 6 and out["C"][1] == 0
    assert out["A"][0] > 0 > out["C"][0]
    league = 12 / 9
    assert out["C"][0] == pytest.approx(math.log((0 + LM.TRADE_PRIOR * league) / (3 + LM.TRADE_PRIOR) / league), abs=1e-3)
