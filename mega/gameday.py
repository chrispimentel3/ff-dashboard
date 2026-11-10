"""Game-day check: who was ruled out after our numbers were set, and whether your Yahoo
lineup still matches Start/Sit for the games that haven't kicked off.

Run by hand on a Sunday after the inactives post (11:30 AM ET for the 1 PM games). Nothing
schedules it since 2026-11-10, when the jobs were cut to one daily refresh:

    python -m mega.gameday outs     Sleeper's Out/IR tags -> data/build/gameday_out.json,
                                    which mega/status.out_for_week reads, so the rebuild
                                    that follows zeroes them in Start/Sit
    python -m mega.gameday alert    rebuilt Start/Sit vs your Yahoo slots -> email

Why Sleeper: its injury tag flips to Out when the inactives post. ESPN's game feed only
carries the injury report (2026-11-09, BUF@MIN: Jauan Jennings and Michael Briscoe were
Out on Sleeper, missing from ESPN's list, and absent from the box score).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from zoneinfo import ZoneInfo

import pandas as pd

from .config import DATA, MY_TEAM, ROOT
from .ids import norm

ET = ZoneInfo("America/New_York")
GAMEDAY = DATA / "build" / "gameday_out.json"
OUT_TAGS = {"Out", "IR", "Sus", "PUP"}     # Doubtful stays a discount in the projection
SITE = "https://mega-bowl-web.vercel.app/start-sit"


def _now(now: dt.datetime | None) -> dt.datetime:
    return (now or dt.datetime.now(ET)).astimezone(ET)


def _web(name: str) -> dict:
    return json.loads((DATA / "web" / name).read_text())


def fetch_outs(season: int, week: int, now: dt.datetime | None = None) -> dict[str, str]:
    """norm -> label for every QB/RB/WR/TE/K Sleeper has as not playing. Keyed by Sleeper's
    name and, through yahoo_id, by the name Yahoo uses (Joshua vs Josh Palmer)."""
    from .sources import sleeper_players
    now = _now(now)
    p = sleeper_players()
    p = p[p["pos"].isin(["QB", "RB", "WR", "TE", "K"]) & p["injury_status"].isin(OUT_TAGS)]
    ro = pd.read_csv(DATA / "yahoo_rosters.csv", dtype=str)
    yahoo = dict(zip(ro["yahoo_id"], ro["player"]))
    at = now.strftime("%-I:%M %p")
    out = {}
    for name, yid, tag in zip(p["name"], p["yahoo_id"], p["injury_status"]):
        for n in {name, yahoo.get(str(yid).split(".")[0])} - {None, ""}:
            out[norm(n)] = f"{tag} (Sleeper, {at})"
    GAMEDAY.write_text(json.dumps({"season": season, "week": week, "at": now.isoformat(timespec="seconds"),
                                   "out": dict(sorted(out.items()))}, indent=1))
    return out


def kickoffs(season: int, week: int) -> dict[str, dt.datetime]:
    """NFL team -> its kickoff that week, ET."""
    from . import season as S
    s = S.schedules(season)
    s = s[pd.to_numeric(s["week"], errors="coerce") == int(week)]
    out = {}
    for day, t, h, a in zip(s["gameday"], s["gametime"], s["home_team"], s["away_team"]):
        k = dt.datetime.fromisoformat(f"{day}T{t}").replace(tzinfo=ET)
        out[h] = out[a] = k
    return out


def changes(ss: dict, slots: dict[str, str], kick: dict[str, dt.datetime], now: dt.datetime) -> list[str]:
    """What to change in Yahoo before kickoff: `ss` is start_sit.json, `slots` your Yahoo
    slot per player (BN = bench). Players whose game has started are left out."""
    def open_(r):
        k = kick.get(r.get("nfl_team"))
        return k is None or k > now

    def pts(r):
        return f"{float(r.get('proj_adj') or 0):.1f} pts"

    starters, bench = ss.get("starters") or [], ss.get("bench") or []
    lines = [f"Start {r['player']} ({r['pos']}, {pts(r)})"
             for r in starters if open_(r) and slots.get(r["player"]) in ("BN", "IR")]
    lines += [f"Bench {r['player']} ({r['pos']}, {r.get('report_status') or pts(r)})"
              for r in bench if open_(r) and slots.get(r["player"]) not in (None, "BN", "IR")]
    by = {r["player"]: r for r in starters}
    for r in bench:                                   # close calls the Sunday lines read the other way
        s = by.get(str(r.get("close_call") or "").removeprefix("≈ "))
        if s and open_(r) and open_(s) and r.get("vegas") is not None and s.get("vegas") is not None \
                and float(r["vegas"]) > float(s["vegas"]):
            lines.append(f"Close call: Sunday lines have {r['player']} {float(r['vegas']):.1f} vs "
                         f"{s['player']} {float(s['vegas']):.1f}; our model starts {s['player']} "
                         f"({pts(s)} vs {pts(r)}). Sunday lines aren't proven on close calls yet.")
    return lines


def alert(now: dt.datetime | None = None, send: bool = True) -> list[str]:
    now = _now(now)
    ss = _web("start_sit.json")
    season, week = int(_web("action_board.json")["season"]), int(ss.get("next_week") or 0)
    ro = pd.read_csv(DATA / "yahoo_rosters.csv", dtype=str)
    mine = ro[ro["team"] == MY_TEAM]
    lines = changes(ss, dict(zip(mine["player"], mine["slot"])), kickoffs(season, week), now)
    if not lines:
        print("[gameday] your Yahoo lineup matches Start/Sit for the games still to play")
        return []
    try:
        pulled = dt.datetime.fromisoformat(json.loads((DATA / "yahoo_pulled.json").read_text())["at"])
        as_of = pulled.astimezone(ET).strftime("%-I:%M %p %a")
    except (OSError, ValueError, KeyError):
        as_of = "the last pull"
    md = "\n".join(f"- {x}" for x in lines) + f"\n\nYour Yahoo lineup as of {as_of}. Start/Sit: {SITE}"
    print("[gameday]\n" + md)
    if send:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            from .digest import send_email
            send_email(md, subject=f"Mega Bowl week {week}: {lines[0]}")
        except Exception as e:
            print(f"[gameday] email not sent: {e}")
    return lines


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "outs":
        from . import data
        ab = _web("action_board.json")
        season = int(ab["season"])
        week = data.current_week(season, int(ab.get("next_week") or 1))
        print(f"[gameday] {len(fetch_outs(season, week))} names ruled out on Sleeper for week {week}")
    elif cmd == "alert":
        alert(send="--dry-run" not in sys.argv)
    else:
        sys.exit("usage: python -m mega.gameday outs | alert [--dry-run]")
