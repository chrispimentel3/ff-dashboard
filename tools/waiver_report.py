"""What is worth claiming this week, and what to bid. Printed for the Tuesday report.

    PYTHONPATH=. .venv/bin/python tools/waiver_report.py [season]

Lives here rather than inline in the scheduled task because a multi-line `python -c` inside
a Markdown instruction is fragile — indentation and nested quoting both break it, and the
failure mode is a weekly job that silently skips the one section with a deadline attached.

`gain` is points per week added to the *optimal lineup*, after accounting for who the player
displaces and who gets cut to fit him. A player who cannot crack the lineup is worth 0 no
matter how well he has been scoring, which is why a third tight end never appears.

`role` and `role_flags` come from the role-context layer (HANDOFF §12). They are printed
because the scheduled task is told to report them: a column the task is asked to read and
the tool does not print produces a confident "no flags this week" over a board that had
them, which is exactly what happened on 2026-09-23.
"""
from __future__ import annotations

import sys

COLS = ["player", "pos", "role", "role_flags", "gain", "bid", "max_bid", "drop"]
SPEC_COLS = ["player", "pos", "role", "role_flags", "ppg", "upside"]


def main(season: int = 2026) -> int:
    import nflreadpy as nfl

    from mega import faab, needs, yahoo

    week = int(nfl.get_current_week())
    board = needs.board(season, week, yahoo_rosters=yahoo.cached_rosters(), top=25)

    r, m = faab.rivals(), faab.market_summary()
    print(f"WEEK {week} · waiver claims")
    if r.get("known"):
        print(f"FAAB ${r['mine']} of ${faab.BUDGET} · {r['richer']} of {r['teams'] - 1} "
              f"teams hold more (richest ${r['max_rival']})")
    if m.get("claims"):
        print(f"league has settled {m['claims']} claims, median ${m['median']:.0f}, "
              f"most ${m['max']:.0f}")
    print()

    if board.empty:
        print("NO BOARD — the valuation engine could not build. Report this and carry on.")
        return 1

    worth = board[board["bid"] >= 1]
    print("WORTH BIDDING ON")
    if worth.empty:
        print("  nothing — no free agent improves the starting lineup. Hold the budget.")
    else:
        print(worth[COLS].to_string(index=False))
    print()
    print("SPECULATIVE (a dollar at most)")
    spec = board[board["bid"] < 1]
    print(spec[[c for c in SPEC_COLS if c in spec.columns]].head(6).to_string(index=False)
          if not spec.empty else "  none")
    print()
    from mega.glossary import FLAGS

    print("WHAT THE TAGS MEAN")
    print("  role  — the job he actually has on his team now, from his last 3 games.")
    for code in ("ROLE+", "ROLE-", "TGT", "GL", "SNAP", "LEAD", "AIR", "1D/RR"):
        t = FLAGS[code]
        print(f"  {t.label:<13}— {t.plain}")
    print("  (1 game)     — it happened once, not across the window.")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2026))
