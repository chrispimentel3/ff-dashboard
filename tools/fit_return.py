"""Do players come back from a missed game at full strength? Fit on nflverse 2021-2025.

    PYTHONPATH=. .venv/bin/python tools/fit_return.py

An absence is a run of his team's games he has no stat line for (byes excluded), after at
least BASE_N games played that season. For his k-th game back (k = 1, 2), the ratio of his
half-PPR points to his average over the BASE_N games before he left, pooled by position and by
how long he was out. The control is the same ratio for players who missed nothing — a
baseline of four games regresses on its own — so the return effect is ratio / control.
Writes config/return_curve.json, read by mega/proj_ros.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "config" / "return_curve.json"
SEASONS = (2021, 2022, 2023, 2024, 2025)
BASE_N = 4
MIN_BASE = 5.0           # fantasy-relevant before he left: 5+ half-PPR a game
POS = ("QB", "RB", "WR", "TE")
K = 50                   # (J) pseudo-events of the all-position ratio in every cell


def events(season: int) -> list[dict]:
    import nflreadpy as nfl
    from mega.intel import weekly
    w = weekly(season)
    w = w[w["season_type"].astype(str).str.upper().eq("REG") & w["pos"].isin(POS)]
    s = nfl.load_schedules(seasons=[season]).to_pandas()
    s = s[s["game_type"] == "REG"]
    team_weeks = {}
    for h, a, wk in zip(s["home_team"], s["away_team"], s["week"]):
        team_weeks.setdefault(h, set()).add(int(wk))
        team_weeks.setdefault(a, set()).add(int(wk))
    out = []
    for gid, g in w.sort_values("week").groupby("gsis_id"):
        team = g["team"].iloc[-1]
        played = dict(zip(g["week"].astype(int), g["half_ppr"]))
        sched = sorted(team_weeks.get(team, ()))
        seq = [(wk, played.get(wk)) for wk in sched]          # None = a team game he missed
        for i in range(BASE_N, len(seq)):
            before = [p for _, p in seq[i - BASE_N:i]]
            if any(p is None for p in before):
                continue
            base = float(np.mean(before))
            if base < MIN_BASE:
                continue
            # missed j games starting at i, then back
            j = 0
            while i + j < len(seq) and seq[i + j][1] is None:
                j += 1
            if j == 0:
                nxt = [seq[i + k][1] for k in range(2) if i + k < len(seq)]
                if len(nxt) == 2 and all(p is not None for p in nxt):
                    out.append({"pos": g["pos"].iloc[0], "missed": 0, "base": base, "g1": nxt[0], "g2": nxt[1]})
            else:
                back = [seq[i + j + k][1] for k in range(2) if i + j + k < len(seq)]
                if len(back) == 2 and all(p is not None for p in back):
                    out.append({"pos": g["pos"].iloc[0], "missed": j, "base": base, "g1": back[0], "g2": back[1]})
    return out


def main() -> None:
    df = pd.DataFrame([e for s in SEASONS for e in events(s)])
    ctl = df[df["missed"] == 0]
    ret = df[df["missed"] > 0].assign(span=lambda d: np.where(d["missed"] >= 2, "2+", "1"))
    curve, report = {}, {}
    # every position pooled, per span: what a thin cell is pulled toward
    pooled = {}
    for span, g in ret.groupby("span"):
        cs = ctl
        pooled[span] = [(g[k].sum() / g["base"].sum()) / (cs[k].sum() / cs["base"].sum()) for k in ("g1", "g2")]
    for pos in POS:
        c = ctl[ctl["pos"] == pos]
        c1, c2 = c["g1"].sum() / c["base"].sum(), c["g2"].sum() / c["base"].sum()
        for span, g in ret[ret["pos"] == pos].groupby("span"):
            r1, r2 = g["g1"].sum() / g["base"].sum(), g["g2"].sum() / g["base"].sum()
            # bootstrap the ratio for its spread: is it distinguishable from 1?
            rng = np.random.default_rng(7)
            boots = []
            for _ in range(400):
                b = g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31)))
                boots.append((b["g1"].sum() / b["base"].sum()) / c1)
            lo, hi = np.percentile(boots, [5, 95])
            # shrunk toward the pooled ratio by K pseudo-events, and never above 1: coming back
            # doesn't make anyone better, and a ratio over 1 here is the noise of a thin cell
            n = len(g)
            curve.setdefault(pos, {})[span] = [round(min(1.0, (n * x + K * pooled[span][i]) / (n + K)), 3)
                                               for i, x in enumerate((r1 / c1, r2 / c2))]
            report[f"{pos}|{span}"] = {"n": len(g), "game1": round(r1 / c1, 3), "game2": round(r2 / c2, 3),
                                       "game1_90ci": [round(lo, 3), round(hi, 3)]}
            print(f"[return] {pos} out {span}: n={len(g):4d}  game 1 back x{r1 / c1:.3f} "
                  f"(90% {lo:.2f}-{hi:.2f})  game 2 x{r2 / c2:.3f}")
    OUT.write_text(json.dumps({
        "_doc": "Points in a player's 1st/2nd game back from missing games, relative to his 4-game average "
                "before, divided by the same ratio for players who missed nothing (2021-2025 nflverse). "
                "Keyed pos -> games missed ('1' or '2+') -> [game1, game2]. Built by tools/fit_return.py.",
        "curve": curve, "pooled": {k: [round(x, 3) for x in v] for k, v in pooled.items()},
        "report": report}, indent=1) + "\n")
    print(f"[return] wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
