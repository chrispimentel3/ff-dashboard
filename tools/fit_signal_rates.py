"""Calibrate P(role expands | usage flag) for the waiver "Early signal" lane (HANDOFF v1.3 §3.3).

    python -m tools.fit_signal_rates [--seasons 2021 2022 2023 2024 2025]

Writes config/signal_rates.json, read by mega/waiver_value.py. Rerun each preseason.

**Walk-forward.** For each season and week w (3..14), roles and flags are rebuilt from only
the games played through week w — exactly what the live board would have shown that
Tuesday — using mega.roles.build, the same code the live site runs. A player "expands" if
his role reaches the next rung up his position's ladder (WR3 -> WR2, COMMITTEE -> LEAD,
TE1-BLK -> TE1-REC; mega.roles.LADDER) in any of weeks w+1..w+3.

**What is reported per flag:** the hit rate, its sample, and the base rate for players at
the same rungs with no flag at all, so a flag's lift is visible rather than assumed. A
flag is split by persistence (sustained vs spike, mega.roles.persistence) because a flag
that held for two games and one that fired once are different evidence.

Players already on the top rung (WR1, TE1-REC, LEAD, STARTER) can't expand and are
excluded. The depth-chart fallback for thin samples is skipped: historical depth snapshots
are not dated to the week, and using the end-of-season chart would leak the future.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

DEFAULT_SEASONS = [2021, 2022, 2023, 2024, 2025]
FIRST_WEEK, LAST_WEEK, AHEAD = 3, 14, 3
SIGNALS = ("TGT", "AIR", "SNAP", "LEAD", "GL", "ROLE+")
OUT = Path(__file__).resolve().parents[1] / "config" / "signal_rates.json"


def _roles_through(pw: pd.DataFrame, ffo: pd.DataFrame, week: int) -> pd.DataFrame:
    from mega import roles as rl
    sub = pw[pw["week"] <= week]
    f = ffo[pd.to_numeric(ffo["week"], errors="coerce") <= week] if ffo is not None and not ffo.empty else ffo
    tab = rl.build(sub, f, None, None, None).get("table")
    if tab is None or tab.empty:
        return pd.DataFrame(columns=["gsis_id", "pos", "role", "flags", "tags"])
    return tab[["gsis_id", "pos", "role", "flags", "tags"]]


def _rung(pos: str, role: str) -> int | None:
    from mega.roles import LADDER
    lad = LADDER.get(pos, ())
    return lad.index(role) if role in lad else None


def season_rows(season: int) -> list[dict]:
    from mega import season as S
    pw, ffo = S.player_week(season), S.ff_opportunity(season)
    by_week = {w: _roles_through(pw, ffo, w) for w in range(FIRST_WEEK, LAST_WEEK + AHEAD + 1)}
    rows = []
    for w in range(FIRST_WEEK, LAST_WEEK + 1):
        now = by_week[w]
        later = [dict(zip(by_week[w + k]["gsis_id"], by_week[w + k]["role"])) for k in range(1, AHEAD + 1)]
        for r in now.itertuples():
            rung = _rung(r.pos, r.role)
            if rung is None or rung == 0:
                continue
            expanded = any((lr := _rung(r.pos, fut.get(r.gsis_id, ""))) is not None and lr < rung
                           for fut in later)
            flags = [f for f in (r.flags if isinstance(r.flags, list) else []) if f in SIGNALS]
            tags = r.tags if isinstance(r.tags, dict) else {}
            rows.append({"season": season, "week": w, "pos": r.pos, "role": r.role,
                         "flags": flags, "tags": tags, "expanded": bool(expanded)})
    return rows


def summarise(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out: dict = {"base": {}, "flags": {}}
    none = df[df["flags"].map(len) == 0]
    for pos, g in none.groupby("pos"):
        out["base"][pos] = {"rate": round(float(g["expanded"].mean()), 4), "n": int(len(g))}
    ex = df.explode("flags").dropna(subset=["flags"])
    ex["persist"] = [("spike" if (t or {}).get(f) == "spike" else "sustained")
                     if f != "ROLE+" else "window" for f, t in zip(ex["flags"], ex["tags"])]
    for (flag, pos, persist), g in ex.groupby(["flags", "pos", "persist"]):
        out["flags"].setdefault(flag, {}).setdefault(pos, {})[persist] = {
            "rate": round(float(g["expanded"].mean()), 4), "n": int(len(g))}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    a = ap.parse_args(argv)
    rows = []
    for s in a.seasons:
        rows.extend(season_rows(s))
        print(f"  {s}: {len(rows)} player-weeks so far")
    out = {
        "_doc": "P(role reaches the next rung within 3 weeks | usage flag at week w), walk-forward "
                "2021-2025 weeks 3-14, with the no-flag base rate per position. Fitted by "
                "tools/fit_signal_rates.py; read by mega/waiver_value.py.",
        "fitted": dt.date.today().isoformat(), "seasons": a.seasons,
        "weeks": [FIRST_WEEK, LAST_WEEK], "ahead": AHEAD, **summarise(rows),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("base:", out["base"])
    for f, by_pos in out["flags"].items():
        for pos, d in by_pos.items():
            print(f"  {f:6s} {pos:3s} " + "  ".join(f"{k}={v['rate']:.2f} (n={v['n']})" for k, v in d.items()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
