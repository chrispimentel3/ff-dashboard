"""Pull the league from Yahoo, check it is sane, and say what changed.

    PYTHONPATH=. .venv/bin/python tools/refresh.py [--season 2026] [--no-scores]

The data half of both scheduled tasks — the daily refresh and the Tuesday waiver run —
so the two cannot drift apart. It never commits; the caller does that, and only when this
exits 0.

Safety: every file is copied aside before the scrape and copied back if anything fails.
The failure that matters is a silent one — an expired sign-in returns a login page, which
parses to an empty frame and writes an empty CSV straight over good data. Validation is
what stands between that and a dashboard showing a league with no teams in it.

Exit codes
    0  refreshed and validated
    2  Yahoo sign-in expired — Chris has to run `python -m mega.yahoo login` himself
    3  validation failed, previous data restored
    4  something else failed, previous data restored
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Written by the scrape, and restored together — a half-updated set is worse than a stale one.
FILES = ("yahoo_rosters.csv", "yahoo_free_agents.csv", "yahoo_standings.csv",
         "yahoo_transactions.csv", "yahoo_faab.csv", "yahoo_faab_bids.csv", "yahoo_scores.csv")

OK, AUTH, INVALID, ERROR = 0, 2, 3, 4


def _backup(tmp: Path) -> dict[str, Path]:
    saved = {}
    for name in FILES:
        src = DATA / name
        if src.is_file():
            dst = tmp / name
            shutil.copy2(src, dst)
            saved[name] = dst
    return saved


def _restore(saved: dict[str, Path]) -> None:
    for name, src in saved.items():
        shutil.copy2(src, DATA / name)
    print(f"[restore] put back {len(saved)} file(s) — nothing was changed on disk.")


def validate() -> list[str]:
    """Every check that has ever caught a bad scrape. Returns the failures."""
    import pandas as pd

    bad: list[str] = []

    def read(name: str):
        p = DATA / name
        return pd.read_csv(p, dtype=str).fillna("") if p.is_file() else None

    r = read("yahoo_rosters.csv")
    if r is None or len(r) < 170:
        bad.append(f"rosters: {0 if r is None else len(r)} rows, expected 170+")
    elif r["team"].nunique() != 12:
        bad.append(f"rosters: {r['team'].nunique()} distinct teams, expected 12")

    s = read("yahoo_standings.csv")
    if s is None or len(s) != 12:
        bad.append(f"standings: {0 if s is None else len(s)} rows, expected 12")

    f = read("yahoo_free_agents.csv")
    if f is None or len(f) < 50:
        bad.append(f"free agents: {0 if f is None else len(f)} rows, expected 50+")

    t = read("yahoo_transactions.csv")
    if t is None or len(t) < 1:
        bad.append("transactions: empty")

    # The FAAB check earns its keep: a Yahoo layout change makes the parse return nothing,
    # and an empty file would otherwise be written straight over real balances.
    b = read("yahoo_faab.csv")
    if b is None or len(b) != 12:
        bad.append(f"faab: {0 if b is None else len(b)} rows, expected 12")
    elif "faab_left" in b.columns:
        v = pd.to_numeric(b["faab_left"], errors="coerce")
        if v.isna().any() or not v.between(0, 100).all():
            bad.append("faab: a balance is missing or outside $0-$100")

    return bad


def warnings_() -> list[str]:
    """Not worth restoring a backup over, but worth saying out loud."""
    import pandas as pd

    out = []
    p = DATA / "yahoo_rosters.csv"
    if p.is_file():
        r = pd.read_csv(p, dtype=str).fillna("")
        if "seat" in r.columns:
            blank = sorted(set(r.loc[r["seat"].str.strip() == "", "team"]))
            for team in blank:
                out.append(f"unknown seat for team {team!r} — a manager renamed their team. "
                           "mega/config.py TEAM_BY_SEAT needs updating; confirm the seat from "
                           "whose draft picks are on that roster, not from the name.")
    return out


def changes(saved: dict[str, Path]) -> list[str]:
    """What moved since the last run, in the order Chris cares about."""
    import pandas as pd

    out: list[str] = []

    def load(name, old=False):
        p = saved.get(name) if old else DATA / name
        if not p or not Path(p).is_file():
            return None
        return pd.read_csv(p, dtype=str).fillna("")

    # transactions: new rows at the top
    new, old = load("yahoo_transactions.csv"), load("yahoo_transactions.csv", old=True)
    if new is not None:
        if old is None:
            out.append(f"transactions: {len(new)} rows (no previous file to compare)")
        else:
            seen = set(map(tuple, old.astype(str).values.tolist()))
            fresh = [r for r in new.astype(str).values.tolist() if tuple(r) not in seen]
            out.append(f"transactions: {len(fresh)} new since the last run")
            for row in fresh[:12]:
                out.append("   · " + " | ".join(x for x in row if x)[:160])

    # settled FAAB claims carry the winning and losing bids; the transactions text does not
    nb, ob = load("yahoo_faab_bids.csv"), load("yahoo_faab_bids.csv", old=True)
    if nb is not None and ob is not None and len(nb) > len(ob):
        out.append(f"FAAB claims settled: {len(nb) - len(ob)} new")

    # roster moves, by player
    nr, orr = load("yahoo_rosters.csv"), load("yahoo_rosters.csv", old=True)
    if nr is not None and orr is not None and {"player", "team"} <= set(nr.columns):
        # Yahoo writes an unfilled roster slot as a player called "(Empty)". Left in, it
        # reads as a player changing hands every time two teams' bench counts differ.
        def _real(df):
            return df[~df["player"].str.strip().str.fullmatch(r"\(Empty\)", case=False, na=False)]

        nr, orr = _real(nr), _real(orr)
        now = dict(zip(nr["player"], nr["team"]))
        was = dict(zip(orr["player"], orr["team"]))
        added = [p for p in now if p not in was]
        dropped = [p for p in was if p not in now]
        moved = [(p, was[p], now[p]) for p in now if p in was and was[p] != now[p]]
        if added:
            out.append(f"added to a roster ({len(added)}): " + ", ".join(added[:10]))
        if dropped:
            out.append(f"off every roster ({len(dropped)}): " + ", ".join(dropped[:10]))
        for p, a, b in moved[:10]:
            out.append(f"moved: {p} {a} -> {b}")

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--no-scores", action="store_true", help="skip the weekly score fetch")
    args = ap.parse_args(argv)

    from mega.yahoo import STATE

    if not STATE.exists():
        print("NO SESSION — data/yahoo_state.json is missing.")
        return AUTH

    tmp = Path(tempfile.mkdtemp(prefix="megabowl-refresh-"))
    saved = _backup(tmp)
    print(f"[backup] {len(saved)} file(s) -> {tmp}")

    try:
        from mega.yahoo import AuthExpired, pull

        print("[scrape] pulling rosters, standings, free agents, transactions, FAAB …")
        try:
            pull()
        except AuthExpired as e:
            print(f"AUTH EXPIRED — {e}")
            _restore(saved)
            return AUTH

        bad = validate()
        if bad:
            print("VALIDATION FAILED:")
            for b in bad:
                print("   ✗ " + b)
            _restore(saved)
            return INVALID
        print("[validate] all checks passed")

        for w in warnings_():
            print("[warn] " + w)

        # id match: an unresolved player is invisible to every projection downstream
        try:
            import pandas as pd

            from mega import ids

            ros = pd.read_csv(DATA / "yahoo_rosters.csv", dtype=str).fillna("")
            _, rep = ids.resolve(ros, name_col="player")
            print("[ids] " + ids.report_line(rep))
        except Exception as e:
            print(f"[ids] skipped: {e}")

        # weekly scores — only the weeks not already on disk; completed weeks never change
        if not args.no_scores:
            try:
                import nflreadpy as nflp

                from mega.yahoo import refresh_scores

                wk = int(nflp.get_current_week()) - 1
                if wk >= 1:
                    sc = refresh_scores(wk)
                    n = len(sc) if sc is not None else 0
                    print(f"[scores] {n} rows on file through week {wk}")
            except Exception as e:
                print(f"[scores] skipped: {e}")

        print("\nCHANGES")
        ch = changes(saved)
        print("\n".join(ch) if ch else "   nothing moved since the last run")
        return OK

    except Exception as e:
        print(f"ERROR — {type(e).__name__}: {e}")
        _restore(saved)
        return ERROR
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
