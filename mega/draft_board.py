"""Parse the pre-season draft board HTML into a tidy DataFrame.

The board stores the whole draft as a JS array literal:  const ROUNDS = [ [ "Name|POS|TEAM", ... ], ... ];
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

from .config import N_TEAMS, ROOT, TEAM_BY_SEAT

DEFAULT_HTML = ROOT / "megabowl2026_draft_board.html"


def _extract_rounds(html: str) -> list[list[str]]:
    m = re.search(r"const\s+ROUNDS\s*=\s*(\[.*?\]);", html, re.S)
    if not m:
        raise ValueError("Could not find `const ROUNDS = [...]` in the draft board HTML")
    # JS array of arrays of double-quoted strings — valid Python literal after minor cleanup.
    blob = m.group(1).replace("\n", " ")
    return ast.literal_eval(blob)


def load_draft(html_path: str | Path = DEFAULT_HTML) -> pd.DataFrame:
    html = Path(html_path).read_text(encoding="utf-8")
    rounds = _extract_rounds(html)

    rows = []
    for r, picks in enumerate(rounds):
        rnd = r + 1
        for i, raw in enumerate(picks):
            name, pos, team = (raw.split("|") + ["", "", ""])[:3]
            # snake order: odd rounds L->R, even rounds R->L
            seat = i + 1 if r % 2 == 0 else N_TEAMS - i
            overall = r * N_TEAMS + i + 1
            rows.append(
                dict(
                    overall=overall,
                    round=rnd,
                    pick_in_round=(overall - 1) % N_TEAMS + 1,
                    seat=seat,
                    drafted_by=TEAM_BY_SEAT.get(seat, f"seat {seat}"),
                    player=name.strip(),
                    pos=pos.strip().upper(),
                    nfl_team=team.strip().upper(),
                )
            )
    df = pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)
    # positional draft rank (e.g. RB7 = 7th RB off the board)
    df["pos_rank"] = df.groupby("pos").cumcount() + 1
    df["pos_slot"] = df["pos"] + df["pos_rank"].astype(str)
    return df


if __name__ == "__main__":
    d = load_draft()
    print(d.to_string())
    print("\npicks:", len(d), "| teams:", d["seat"].nunique(), "| rounds:", d["round"].max())
