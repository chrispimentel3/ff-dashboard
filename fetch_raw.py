"""
Pull raw nflverse-data CSVs to ./data/ without the dashboard.
Usage:  python fetch_raw.py 2025
"""
from __future__ import annotations

import sys
from pathlib import Path

import nflreadpy as nfl

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def dump(name: str, frame) -> None:
    try:
        df = frame.to_pandas()
    except AttributeError:
        df = frame
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  {name:22s} {df.shape[0]:>7} rows  ->  {path}")


def main(season: int) -> None:
    print(f"nflverse-data pull for {season}")
    dump("player_stats_week", nfl.load_player_stats(seasons=[season]))
    dump("ff_opportunity_week", nfl.load_ff_opportunity(seasons=[season], stat_type="weekly"))
    dump("snap_counts", nfl.load_snap_counts(seasons=[season]))
    dump("depth_charts", nfl.load_depth_charts(seasons=[season]))
    dump("injuries", nfl.load_injuries(seasons=[season]))
    dump("ngs_receiving", nfl.load_nextgen_stats(seasons=[season], stat_type="receiving"))
    dump("ngs_rushing", nfl.load_nextgen_stats(seasons=[season], stat_type="rushing"))
    dump("ngs_passing", nfl.load_nextgen_stats(seasons=[season], stat_type="passing"))
    dump("schedules", nfl.load_schedules(seasons=[season]))
    dump("ff_playerids", nfl.load_ff_playerids())
    dump("ff_rankings", nfl.load_ff_rankings())
    print("done ->", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2025)
