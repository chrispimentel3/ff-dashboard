"""Build config/faab_market.json (mega/faab_market.py) and backtest it season by season.

    PYTHONPATH=. .venv/bin/python tools/fit_faab_market.py

Leave one season out: build the price table from the other four, then bid on every claim in
the held-out season at each target win chance. A bid wins if it beats the best other bid
(a tie counts half — the worse record wins it). Reported against what the league's winners
actually paid for the same claims.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mega import faab_market as FM  # noqa: E402

TARGETS = (0.6, 0.7, 0.8, 0.9)


def main() -> None:
    df = FM.attach_last_game(FM.claims())
    report = {}
    for p in TARGETS:
        won, spent, paid, n = 0.0, 0.0, 0.0, 0
        for y in sorted(df["year"].unique()):
            table = FM.build_table(df[df["year"] != y])
            for r in df[df["year"] == y].itertuples():
                bid = FM.price(FM.lookup(r.pos, r.last_pts, table), p)
                won += 1.0 if bid > r.price else 0.5 if bid == r.price else 0.0
                spent += bid
                paid += r.bid
                n += 1
        report[str(p)] = {"claims": n, "win_rate": round(won / n, 3), "avg_bid": round(spent / n, 2),
                          "winners_avg_paid": round(paid / n, 2)}
        print(f"[faab] target {p:.0%}: won {won / n:.1%} of {n} claims, avg bid ${spent / n:.2f} "
              f"(the winners paid ${paid / n:.2f})")
    FM.TABLE.write_text(json.dumps({
        "_doc": "mega/faab_market.py price table: group -> sorted prices to win (the best other bid) on every "
                "2021-2025 Mega Bowl claim. Built by tools/fit_faab_market.py; backtest = leave one season out.",
        "target_p": FM.TARGET_P, "backtest": report, "groups": FM.build_table(df),
    }, indent=1) + "\n")
    print(f"[faab] wrote {FM.TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
