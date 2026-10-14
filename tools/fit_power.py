"""Pick mega/power.py's constants and backtest it. Writes config/power_params.json.

    PYTHONPATH=. .venv/bin/python tools/fit_power.py

For each 2021-2025 season and each "now" from week 4 to 12: fit on the lines posted by then
(through week now+1, as the market posts a week ahead), predict every later regular-season
game, and score the predicted implied team totals against the closing lines that were later
posted. The baseline is what the models used before: a team's own average implied total so
far (and a neutral spread). Constants are chosen on 2021-2023 and reported on 2024-2025.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mega import power  # noqa: E402

TRAIN, TEST = (2021, 2022, 2023), (2024, 2025)
NOWS = range(4, 13)
HALF_LIVES = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, float("inf"))
LAMBDAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def sides(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: implied total, from the closing lines."""
    h = pd.DataFrame({"team": g["home_team"], "opp": g["away_team"], "week": g["week"], "home": True,
                      "imp": (g["total_line"] + g["spread_line"]) / 2})
    a = pd.DataFrame({"team": g["away_team"], "opp": g["home_team"], "week": g["week"], "home": False,
                      "imp": (g["total_line"] - g["spread_line"]) / 2})
    return pd.concat([h, a], ignore_index=True)


def errors(sched: pd.DataFrame, hl: float, lam: float, weeks=None) -> tuple[list, list]:
    model_err, base_err = [], []
    for now in NOWS:
        train = sched[sched["week"] <= now + 1]
        test = sched[(sched["week"] > now + 1) & (sched["week"] <= 17)]
        if weeks:
            test = test[test["week"].isin(weeks)]
        m = power.fit(train, hl, lam)
        avg = sides(train).groupby("team")["imp"].mean()
        for g in test.itertuples():
            p = power.predict(m, g.home_team, g.away_team)
            if p is None:
                continue
            tot, sp = p
            for team, imp_true, imp_pred in ((g.home_team, (g.total_line + g.spread_line) / 2, (tot + sp) / 2),
                                             (g.away_team, (g.total_line - g.spread_line) / 2, (tot - sp) / 2)):
                model_err.append(abs(imp_pred - imp_true))
                base_err.append(abs(avg.get(team, np.nan) - imp_true))
    return model_err, base_err


def main() -> None:
    import nflreadpy as nfl
    all_s = nfl.load_schedules(seasons=list(TRAIN + TEST)).to_pandas()
    all_s = all_s[all_s["game_type"] == "REG"].dropna(subset=["total_line", "spread_line"])
    by = {s: g for s, g in all_s.groupby("season")}

    grid = {}
    for hl, lam in itertools.product(HALF_LIVES, LAMBDAS):
        e = [x for s in TRAIN for x in errors(by[s], hl, lam)[0]]
        grid[(hl, lam)] = float(np.mean(e))
    hl, lam = min(grid, key=grid.get)
    print(f"[power] chosen on {TRAIN}: half_life={hl} lambda={lam} (MAE {grid[(hl, lam)]:.3f})")

    out = {}
    for label, weeks in (("all_future", None), ("playoffs_15_17_from_now", (15, 16, 17))):
        m, b = [], []
        for s in TEST:
            mm, bb = errors(by[s], hl, lam, weeks)
            m += mm
            b += [x for x in bb if np.isfinite(x)]
        out[label] = {"n": len(m), "mae_model": round(float(np.mean(m)), 3), "mae_team_avg": round(float(np.mean(b)), 3)}
        print(f"[power] {TEST} {label}: model MAE {out[label]['mae_model']} vs team-average {out[label]['mae_team_avg']} "
              f"({len(m)} team-games)")
    power.PARAMS.write_text(json.dumps({
        "_doc": "mega/power.py constants, fitted by tools/fit_power.py: implied team totals for unposted games "
                "from ratings fit on posted lines. Chosen on 2021-23, backtest on 2024-25 vs the closing lines "
                "later posted; baseline = each team's average implied total so far (the old fallback).",
        "half_life": hl if np.isfinite(hl) else 1e9, "lambda": lam, "backtest": out,
        "grid_mae_train": {f"{h}|{l}": round(v, 4) for (h, l), v in grid.items()},
    }, indent=1) + "\n")
    print(f"[power] wrote {power.PARAMS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
