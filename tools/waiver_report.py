"""What is worth claiming this week, and what to bid. Printed for the Tuesday report.

    PYTHONPATH=. .venv/bin/python tools/waiver_report.py [season]

Lives here rather than inline in the scheduled task because a multi-line `python -c` inside
a Markdown instruction is fragile — indentation and nested quoting both break it, and the
failure mode is a weekly job that silently skips the one section with a deadline attached.

`gain` is points per week added to the *optimal lineup*, after accounting for who the player
displaces and who gets cut to fit him. A player who cannot crack the lineup is worth 0 no
matter how well he has been scoring, which is why a third tight end never appears.
"""
from __future__ import annotations

import sys

COLS = ["player", "pos", "gain", "bid", "max_bid", "drop"]


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
    print(spec[["player", "pos", "ppg", "upside"]].head(6).to_string(index=False)
          if not spec.empty else "  none")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2026))
