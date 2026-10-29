import pandas as pd

from mega import action_board as AB


def test_shop_and_hold_come_from_our_projection_against_his_pace():
    agg = pd.DataFrame([
        {"gsis_id": "hot", "player": "Hot", "pos": "WR", "slot": "WR", "games": 6, "half_ppr_pg": 16.0},
        {"gsis_id": "cold", "player": "Cold", "pos": "RB", "slot": "BN", "games": 6, "half_ppr_pg": 5.0},
        {"gsis_id": "fair", "player": "Fair", "pos": "WR", "slot": "WR", "games": 6, "half_ppr_pg": 10.0},
        {"gsis_id": "one", "player": "One", "pos": "RB", "slot": "BN", "games": 1, "half_ppr_pg": 2.0},
        {"gsis_id": "none", "player": "None", "pos": "TE", "slot": "TE", "games": 6, "half_ppr_pg": 20.0},
    ])
    proj = {"hot": {"ros_pg": 12.5}, "cold": {"ros_pg": 7.5}, "fair": {"ros_pg": 11.5}, "one": {"ros_pg": 9.0}}
    shop, hold = AB.shop_hold(agg, proj)
    assert shop["player"].tolist() == ["Hot"]                    # 3.5 below his pace
    assert hold["player"].tolist() == ["Cold"]                   # 2.5 above; Fair's +1.5 is inside GAP
    assert shop["per_g"].iloc[0] == -3.5 and hold["proj_pg"].iloc[0] == 7.5
    assert "we project 12.5/g" in shop["why"].iloc[0]
    # one game is no pace, and a player the projection doesn't cover is left alone
    assert "One" not in hold["player"].tolist() and "None" not in shop["player"].tolist()


def test_no_projection_means_no_calls():
    agg = pd.DataFrame([{"gsis_id": "x", "player": "X", "pos": "WR", "slot": "WR", "games": 5, "half_ppr_pg": 20.0}])
    shop, hold = AB.shop_hold(agg, None)
    assert shop.empty and hold.empty


def test_shortlist_is_best_first_one_per_manager():
    cards = [{"partner": "A", "i": 1}, {"partner": "A", "i": 2}, {"partner": "B", "i": 3}]
    assert [c["i"] for c in AB.shortlist(cards)] == [1, 3]
