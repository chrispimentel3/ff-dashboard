"""Fill Chris's xWins workbook from the scores the Yahoo refresh already collects.

    PYTHONPATH=. .venv/bin/python tools/update_xwins.py            # write what's new
    PYTHONPATH=. .venv/bin/python tools/update_xwins.py --dry-run  # show it, write nothing

`~/Desktop/Mega Bowl xWins.xlsx` is formula-driven: its only inputs are
`Weekly Scores!B2:O13` (each team's points, one column per week) and `B17:O28` (1 = won,
0 = lost). Every Week tab, the Summary, xWins, Power Score, CV and both charts recompute
from those. This used to be filled by pasting the Yahoo scoreboard into a text file and
running `~/Desktop/Mega Bowl updater/update_week.py`; the daily refresh already stores
every finished week in data/yahoo_scores.csv, so this writes from that instead.

Rows are fixed by draft seat, not by name — managers rename teams mid-season (two did on
2026-10-01) — and the names in column A are rewritten to what each team is called now.
Only a week with all twelve teams is written; a week already holding the same numbers is
left alone, so running this every day is safe. Before any write the workbook is copied
to `~/Desktop/Mega Bowl updater/backups/`.

The XML editing (no openpyxl, so the charts survive) is the updater's own xlsx_common.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

UPDATER = Path(os.environ.get("MEGA_XWINS_UPDATER", Path.home() / "Desktop" / "Mega Bowl updater"))
XLSX = Path(os.environ.get("MEGA_XWINS_XLSX", Path.home() / "Desktop" / "Mega Bowl xWins.xlsx"))

# The workbook's rows 2-13 (and 17-28), by draft seat — the 2025 order setup_2025.py
# wrote: TaylorMade, Heartbreak Drake, A Place in the Hampton, Crabcakes and Football,
# BillsMafia, Undisputed, Don R.I.C.O, Barkley's Balls Deep, deez nuts, TyRick Hill,
# Revenge of the Smith, L'Omar.
ROW_SEATS = [8, 4, 7, 12, 11, 3, 10, 6, 1, 2, 9, 5]
WEEKS = range(1, 15)          # columns B..O; the playoffs aren't in the workbook


def weeks_from_scores(sc) -> dict[int, dict[int, tuple[float, int]]]:
    """week -> {seat: (points, 1 if won else 0)}, for weeks with all twelve teams."""
    from mega import teams

    out: dict[int, dict[int, tuple[float, int]]] = {}
    for wk, g in sc.groupby("week"):
        if int(wk) not in WEEKS:
            continue
        row = {}
        for _, r in g.iterrows():
            seat = teams.seat_for(r["team"])
            if seat is None:
                raise SystemExit(f"week {wk}: no seat for team {r['team']!r} — "
                                 "add it to mega/config.py FORMER_NAMES")
            row[seat] = (round(float(r["points"]), 2), int(float(r["points"]) > float(r["opp_points"])))
        if set(row) == set(ROW_SEATS):
            out[int(wk)] = row
    return out


def _cell_values(sheet_xml: str) -> dict[str, str]:
    return {ref: v for ref, v in re.findall(r'<c r="([A-Z]+\d+)"[^>]*?>\s*<v>([^<]*)</v>', sheet_xml)}


def _team_names(ss_xml: str, first: int) -> list[str]:
    sis = re.findall(r"<si>(.*?)</si>", ss_xml, re.S)
    return [re.sub(r"<[^>]+>", "", s) for s in sis[first:first + len(ROW_SEATS)]]


def plan(xlsx: Path, weeks: dict, names_now: dict[int, str], X) -> tuple[dict, list[str] | None]:
    """What would change: ({week: {seat: (pts, win)}}, new column-A names or None)."""
    with zipfile.ZipFile(xlsx) as z:
        sheet = z.read(X.WEEKLY_SCORES_SHEET).decode("utf-8")
        ss = z.read("xl/sharedStrings.xml").decode("utf-8")
    have = _cell_values(sheet)
    todo = {}
    for wk, row in sorted(weeks.items()):
        col = X.col_letter(wk)
        for i, seat in enumerate(ROW_SEATS):
            pts, win = row[seat]
            cur_p, cur_w = have.get(f"{col}{2 + i}"), have.get(f"{col}{17 + i}")
            if cur_p is None or cur_w is None or abs(float(cur_p) - pts) > 0.005 or int(float(cur_w)) != win:
                todo[wk] = row
                break
    want = [names_now[s] for s in ROW_SEATS]
    rename = want if _team_names(ss, X.SS_FIRST_TEAM_IDX) != [X.xml_escape(n) for n in want] else None
    return todo, rename


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xlsx", type=Path, default=XLSX)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if not a.xlsx.is_file():
        print(f"xWins: no workbook at {a.xlsx} — skipped")
        return 0
    sys.path.insert(0, str(UPDATER))
    import xlsx_common as X

    from mega import teams
    from mega.yahoo import cached_scores

    weeks = weeks_from_scores(cached_scores())
    names_now = teams.current_names()
    todo, rename = plan(a.xlsx, weeks, names_now, X)
    if not todo and not rename:
        print(f"xWins: up to date (weeks {', '.join(map(str, sorted(weeks))) or 'none'})")
        return 0

    for wk, row in todo.items():
        line = ", ".join(f"{names_now[s]} {row[s][0]:.2f}{'W' if row[s][1] else 'L'}" for s in ROW_SEATS)
        print(f"xWins: week {wk} -> column {X.col_letter(wk)}: {line}")
    if rename:
        print("xWins: column A names ->", ", ".join(rename))
    if a.dry_run:
        print("xWins: --dry-run, nothing written")
        return 0

    backups = UPDATER / "backups"
    backups.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H%M%S")
    shutil.copy2(a.xlsx, backups / f"{a.xlsx.stem} {stamp}{a.xlsx.suffix}")

    def edit_sheet(xml: str) -> str:
        for wk, row in todo.items():
            col = X.col_letter(wk)
            for i, seat in enumerate(ROW_SEATS):
                pts, win = row[seat]
                xml = X.set_cell(xml, f"{col}{2 + i}", pts)
                xml = X.set_cell(xml, f"{col}{17 + i}", win)
        return xml

    def edit_names(xml: str) -> str:      # every sheet points at these twelve strings
        for k, name in enumerate(rename or []):
            sis = list(re.finditer(r"<si>.*?</si>", xml, re.S))
            tgt = sis[X.SS_FIRST_TEAM_IDX + k]
            xml = xml[:tgt.start()] + f"<si><t>{X.xml_escape(name)}</t></si>" + xml[tgt.end():]
        return xml

    edits = {X.WEEKLY_SCORES_SHEET: edit_sheet, "xl/workbook.xml": X.force_recalc}
    if rename:
        edits["xl/sharedStrings.xml"] = edit_names
    X.rewrite_xlsx(str(a.xlsx), str(a.xlsx), edits)
    print(f"xWins: wrote {len(todo)} week(s) to {a.xlsx} (backup in {backups})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
