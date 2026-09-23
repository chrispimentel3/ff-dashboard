"""The nflverse column catalogue: name matching, and the folding that makes it work."""
from __future__ import annotations

import pytest

from mega import catalog


pytestmark = pytest.mark.skipif(not catalog.schema(),
                                reason="data/nflverse_schema.json not built")


def test_the_catalogue_covers_the_whole_database():
    sc = catalog.schema()
    assert len(sc) >= 20
    assert sum(r["n_cols"] for r in sc.values()) > 1000
    for must in ("player_stats", "snap_counts", "nextgen_stats_receiving", "pbp"):
        assert must in sc


def test_abbreviations_fold_both_ways():
    """'average' does not start with 'avg' — these are contractions, not prefixes, which
    is why prefix matching alone never found avg_separation."""
    assert catalog._norm("average separation") == catalog._norm("avg_separation")
    assert catalog._norm("expected points added") == catalog._norm("epa")
    assert catalog._norm("yards") == catalog._norm("yds")


@pytest.mark.parametrize("q,col", [
    ("average separation", "avg_separation"),
    ("average cushion", "avg_cushion"),
    ("time to throw", "avg_time_to_throw"),
    ("rush yards over expected", "rush_yards_over_expected"),
    ("completion percentage above expectation", "completion_percentage_above_expectation"),
    ("wopr", "wopr"),
    ("air yards share", "air_yards_share"),
])
def test_finds_real_columns_by_name(q, col):
    assert catalog.search(q, limit=3)[0].column == col


def test_nonsense_finds_nothing():
    assert catalog.search("xyz nonsense flibbertigibbet") == []


def test_position_steers_which_table_answers():
    """receiving_drop exists in the quarterback table too; a WR question must not land there."""
    assert catalog.search("receiving drops", positions=("WR",))[0].table == "pfr_advstats_rec"


def test_rate_columns_are_recognised():
    assert catalog.is_rate("avg_separation")
    assert catalog.is_rate("completion_pct")
    assert catalog.is_rate("target_share")
    assert not catalog.is_rate("targets")
    assert not catalog.is_rate("receiving_yards")


def test_identifiers_are_not_offered_as_stats():
    for h in catalog.search("player", limit=20, numeric_only=False):
        assert not h.column.endswith(("_id", "_name", "_url"))
