"""mega/history_web.py: seasons, records, champions and the all-time table from pulled files."""
from __future__ import annotations

import json

from mega import history_web as H


def _season(year, rows, champion):
    return {"year": year, "league_name": "Mega Bowl", "champion": champion,
            "standings": [{"year": year, "rank": i + 1, "made_playoffs": i < 2, "team": t, "wins": w, "losses": l,
                           "ties": 0, "points_for": pf, "points_against": pa, "streak": ""}
                          for i, (t, w, l, pf, pa) in enumerate(rows)],
            "rosters": {rows[0][0]: [{"slot": "QB", "player": "P", "pos": "QB", "nfl_team": "KC", "seat": 1,
                                      "team": rows[0][0], "yahoo_id": "1", "year": year}]}}


def _write(tmp_path, *seasons):
    for s in seasons:
        (tmp_path / f"{s['year']}.json").write_text(json.dumps(s))


def test_nothing_on_disk_is_unavailable_and_lists_whats_missing(tmp_path):
    out = H.build([2024, 2025], tmp_path)
    assert out["available"] is False and out["missing"] == [2025, 2024]


def test_seasons_sort_newest_first_and_report_what_is_still_missing(tmp_path):
    _write(tmp_path, _season(2024, [("A", 10, 4, 1500.0, 1300.0), ("B", 5, 9, 1200.0, 1400.0)], "A"),
           _season(2025, [("B", 11, 3, 1600.0, 1250.0), ("A", 4, 10, 1100.0, 1500.0)], "B"))
    out = H.build([2023, 2024, 2025], tmp_path)
    assert [s["year"] for s in out["seasons"]] == [2025, 2024]
    assert out["missing"] == [2023]
    assert out["seasons"][0]["standings"][0]["record"] == "11-3"
    assert out["seasons"][0]["rosters"]["B"][0]["player"] == "P"


def test_records_find_the_extremes_with_team_and_year(tmp_path):
    _write(tmp_path, _season(2024, [("A", 10, 4, 1500.0, 1300.0), ("B", 5, 9, 1200.0, 1400.0)], "A"),
           _season(2025, [("B", 11, 3, 1600.0, 1250.0), ("A", 4, 10, 1100.0, 1500.0)], "B"))
    recs = {r["label"]: r for r in H.build(None, tmp_path)["records"]}
    assert recs["Most points in a season"]["team"] == "B" and recs["Most points in a season"]["year"] == 2025
    assert recs["Fewest points in a season"]["value"] == "1,100.0"
    assert recs["Best record"]["value"] == "11-3" and recs["Worst record"]["value"] == "4-10"


def test_all_time_merges_known_names_by_seat_and_never_fuzzy_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "SEAT_BY_TEAM", {"A": 1, "B": 2})
    monkeypatch.setattr(H, "FORMER_NAMES", {"A2": 1})
    monkeypatch.setattr(H, "TEAM_BY_SEAT", {1: "A", 2: "B"})
    _write(tmp_path, _season(2023, [("A2", 9, 5, 1400.0, 1300.0), ("Zed", 5, 9, 1000.0, 1400.0)], "A2"),
           _season(2024, [("B", 10, 4, 1500.0, 1300.0), ("A", 6, 8, 1200.0, 1350.0)], "B"))
    at = H.build(None, tmp_path)["all_time"]
    a = next(r for r in at["rows"] if r["seat"] == 1)
    assert a["team"] == "A" and a["seasons"] == 2 and a["titles"] == 1 and a["years_won"] == [2023]
    assert a["record"] == "15-13" and a["pf"] == 2600.0
    assert next(r for r in at["rows"] if r["team"] == "Zed")["seat"] is None     # stays its own row
    assert at["named_only"] == 1


def test_apostrophes_match_whichever_way_yahoo_spelled_them():
    assert H._seat("Barkley's Balls Deep") == H._seat("Barkley’s Balls Deep") == 6
    assert H._seat("Nobody") is None
