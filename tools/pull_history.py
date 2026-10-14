"""Pull ONE completed season of Mega Bowl history from Yahoo and save it to
data/history/seasons/<year>.json. Resumable: a season already on disk is skipped, so run it
once per season (or `--next` for the newest one still missing) with a rest between runs.

    PYTHONPATH=. .venv/bin/python tools/pull_history.py 2025
    PYTHONPATH=. .venv/bin/python tools/pull_history.py --next
    PYTHONPATH=. .venv/bin/python tools/pull_history.py --tx --next     # transactions + FAAB bids

`--tx` pulls a season's transaction log (every add, drop, waiver claim and trade, as Yahoo's
row text) and its settled FAAB claims with the losing bids, into
data/history/transactions/<year>.json — the raw material for fitting each manager's bidding,
positional lean and trading habits. Rows are kept as text; parsing belongs to the fit.

Why one at a time: pulling all six seasons back-to-back made Yahoo answer "Request denied"
to every page, including the live league. A season is ~14 page loads at REQUEST_DELAY apart;
the rest between seasons is the caller's job. On a rate-limit or an expired session this
stops at once and writes nothing for that season — it never retries and never logs in.

Exit codes: 0 saved (or already had it) · 5 rate-limited · 6 session expired · 7 nothing found
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mega import yahoo_history as yh  # noqa: E402
from mega.yahoo import AuthExpired, LeagueSession  # noqa: E402

OUT = ROOT / "data" / "history" / "seasons"
TX_OUT = ROOT / "data" / "history" / "transactions"
PAGE = 25          # rows per transactions page; `count=` is the offset
MAX_PAGES = 80     # a season runs ~15-30 pages; this only stops a page that never ends
DELAY = 8.0     # seconds between page loads; the module's own 4s is what got us blocked
RETRIES = 3     # for a dropped connection only; "Request denied" and an expired session never retry
RETRY_WAIT = 30.0


def _patient(fn):
    """Retry a page load that failed on OUR network (timeout, network changed), waiting between
    tries. A rate-limit or auth error is not a network error and passes straight through."""
    import time

    def wrapped(*a, **k):
        for attempt in range(1, RETRIES + 1):
            try:
                return fn(*a, **k)
            except (yh.RateLimited, AuthExpired):
                raise
            except Exception as e:
                if "net::ERR_" not in str(e) or attempt == RETRIES:
                    raise
                print(f"[history] network hiccup ({str(e).splitlines()[0][-60:]}); retry {attempt}/{RETRIES - 1} in {RETRY_WAIT:.0f}s")
                time.sleep(RETRY_WAIT)
    return wrapped


def path(year: int, tx: bool = False) -> Path:
    return (TX_OUT if tx else OUT) / f"{year}.json"


def _paged(fetch, parse, url: str, sig) -> list[dict]:
    """Every page of a transactions view, until one comes back empty or repeats the last."""
    import time
    out, last = [], None
    for n in range(MAX_PAGES):
        time.sleep(DELAY)
        page = fetch(f"{url}&count={n * PAGE}")
        yh._check_denied(page)
        rows = _records(parse(page))
        key = [sig(r) for r in rows]
        if not rows or key == last:
            break
        out += rows
        last = key
        print(f"[history]   page {n + 1}: {len(rows)} rows")
    return out


def pull_tx(year: int) -> int:
    from mega.faab import parse_bids
    from mega.yahoo import parse_transactions
    base = f"https://football.fantasysports.yahoo.com/{year}/f1/{yh.SEASON_LEAGUE_IDS[year]}/transactions"
    print(f"[history] pulling {year} transactions, {DELAY:.0f}s between pages")
    try:
        with LeagueSession() as s:
            get, get_text = _patient(s.get), _patient(s.get_text)
            tx = _paged(get, parse_transactions, base + "?transactionsfilter=all", lambda r: r["text"])
            bids = _paged(get_text, parse_bids, base + "?transactionsfilter=faab",
                          lambda r: (r["player"], r["date"], r["winner"]))
    except yh.RateLimited as e:
        print(f"[history] RATE LIMITED on {year}: {e}\n[history] nothing saved; wait before the next run")
        return 5
    except AuthExpired as e:
        print(f"[history] session expired: {e}")
        return 6
    if not tx:
        print(f"[history] {year}: no transactions found — not saving")
        return 7
    TX_OUT.mkdir(parents=True, exist_ok=True)
    path(year, True).write_text(json.dumps({"year": year, "transactions": [r["text"] for r in tx],
                                            "faab_bids": bids}, indent=1, default=str))
    print(f"[history] saved {year}: {len(tx)} transactions, {len(bids)} FAAB claims -> "
          f"{path(year, True).relative_to(ROOT)}")
    return 0


def _records(df) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def main(argv: list[str]) -> int:
    tx = "--tx" in argv
    argv = [a for a in argv if a != "--tx"]
    missing = [y for y in sorted(yh.SEASON_LEAGUE_IDS, reverse=True) if not path(y, tx).exists()]
    if argv and argv[0] == "--next":
        if not missing:
            print("[history] every season is already saved")
            return 0
        year = missing[0]
    elif argv:
        year = int(argv[0])
        if year not in yh.SEASON_LEAGUE_IDS:
            print(f"[history] no league id for {year}")
            return 7
        if path(year, tx).exists():
            print(f"[history] {year} already saved ({path(year, tx).name})")
            return 0
    else:
        print(__doc__)
        return 7
    if tx:
        yh.REQUEST_DELAY = DELAY
        return pull_tx(year)

    yh.REQUEST_DELAY = DELAY
    yh.roster, yh.standings = _patient(yh.roster), _patient(yh.standings)
    print(f"[history] pulling {year} (league {yh.SEASON_LEAGUE_IDS[year]}), {DELAY:.0f}s between pages")
    try:
        with LeagueSession() as s:
            res = yh.pull_season(s, year)
    except yh.RateLimited as e:
        print(f"[history] RATE LIMITED on {year}: {e}\n[history] nothing saved; wait before the next run")
        return 5
    except AuthExpired as e:
        print(f"[history] session expired: {e}")
        return 6

    if res["standings"].empty or not res["rosters"]:
        print(f"[history] {year}: standings {len(res['standings'])} rows, {len(res['rosters'])} rosters — not saving a partial season")
        return 7
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "year": year, "league_id": res["league_id"], "league_name": res["league_name"],
        "champion": res["champion"],
        "standings": _records(res["standings"]),
        "rosters": {team: _records(df) for team, df in res["rosters"].items()},
    }
    path(year).write_text(json.dumps(payload, indent=1, default=str))
    print(f"[history] saved {year}: champion {res['champion']}, {len(res['standings'])} teams, "
          f"{len(res['rosters'])} rosters -> {path(year).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
