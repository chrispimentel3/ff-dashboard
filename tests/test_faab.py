"""FAAB parsing and the bid curve."""
from __future__ import annotations

from mega import faab

HOME = """League
1\t Don R.I.C.O\t2-0-0\t267.18\t155.36\tW-2\t$93\t12\t1
2\t Crabcakes and Football\t2-0-0\t222.86\t179.70\tW-2\t$100\t11\t1
6\t TaylorMade\t1-1-0\t218.80\t219.68\tL-1\t$90\t7\t3
9\t Barkley’s Balls Deep\t1-1-0\t171.38\t208.84\tW-1\t$85\t4\t4
"""

OFFERS = """FAB Offers
\tBuccaneers TB - DEF
$3 Winning Offer
A Place in the Hampton $2 (Lower Offer)
Barkley’s Balls Deep $0 (Lower Offer)
Awarded To:
TaylorMade
Sep 16,5:16 am
\tDevaughn Vele NO - WR
$7 Winning Offer
Burrow-Chase combo $5 (Lower Offer)
Awarded To:
Don R.I.C.O
Sep 16,5:16 am
"""


def test_budgets_parse():
    df = faab.parse_budgets(HOME)
    assert len(df) == 4
    row = df[df["team"] == "TaylorMade"].iloc[0]
    assert row["faab_left"] == 90
    assert row["faab_spent"] == 10
    assert row["seat"] == 8


def test_bids_parse_winner_and_losers():
    df = faab.parse_bids(OFFERS)
    assert len(df) == 2
    first = df.iloc[0]
    # Yahoo prefixes the name with a Private Use Area glyph from its icon font.
    assert first["player"] == "Buccaneers TB - DEF"
    assert first["winning_bid"] == 3
    assert first["winner"] == "TaylorMade"
    assert first["n_bidders"] == 3
    assert first["runner_up"] == 2


def test_bid_rises_with_value_and_is_capped_by_budget():
    cheap = faab.suggest(0.5, 100, 3)["bid"]
    dear = faab.suggest(5.0, 100, 3)["bid"]
    assert 0 < cheap < dear <= faab.BUDGET * faab.MAX_SHARE
    assert faab.suggest(5.0, 6, 3)["bid"] == 6          # never bids more than you hold


def test_worthless_player_gets_no_bid_but_a_useful_one_gets_at_least_a_dollar():
    assert faab.suggest(0.0, 90, 3)["bid"] == 0
    assert faab.suggest(0.2, 90, 3)["bid"] >= 1


def test_late_season_pickups_are_worth_less():
    """Same player, fewer weeks to score for you."""
    assert faab.suggest(2.0, 100, 14)["bid"] < faab.suggest(2.0, 100, 3)["bid"]


def test_market_summary_reconciles_against_budgets():
    bids, buds = faab.parse_bids(OFFERS), faab.parse_budgets(HOME)
    m = faab.market_summary(bids, buds)
    assert m["claims"] == 2
    assert m["listed_spend"] == 10.0
    assert m["league_spend"] == 32.0          # 7 + 0 + 10 + 15
    assert m["unlisted_spend"] == 22.0


def test_merge_bids_keeps_claims_that_fell_off_the_feed(tmp_path):
    import pandas as pd

    path = tmp_path / "bids.csv"
    cols = ["player", "winning_bid", "winner", "date", "n_bidders", "runner_up", "losers"]
    old = pd.DataFrame([
        ["Ollie Gordon II Mia - RB", 25, "Heartbreak Drake", "Sep 30,4:20 am", 5, 22, "x"],
        ["Keenan Allen Ind - WR Q", 13, "Saquon Enjoyer", "Sep 30,4:20 am", 3, 3, "x"],
        ["Vikings Min - DEF", 26, "Saquon Enjoyer", "Sep 30,4:20 am", 4, 15, "x"],
    ], columns=cols).assign(season=2026)
    pd.concat([old, old.head(1).assign(season=2025)]).to_csv(path, index=False)
    # the same two claims re-scraped with a new tag, a renamed team and a city for a nickname
    new = pd.DataFrame([
        ["Keenan Allen Ind - WR O", 13, "Saquon Hater", "Sep 30,4:20 am", 3, 3, "x"],
        ["Minnesota Min - DEF", 26, "Saquon Hater", "Sep 30,4:20 am", 4, 15, "x"],
    ], columns=cols)
    out = faab.merge_bids(new, 2026, path)
    assert list(out["player"]) == ["Keenan Allen Ind - WR O", "Minnesota Min - DEF", "Ollie Gordon II Mia - RB"]
    assert set(out["season"]) == {2026}
    assert len(faab.merge_bids(new.iloc[0:0], 2026, path)) == 3     # an empty scrape loses nothing
