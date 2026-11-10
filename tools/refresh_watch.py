"""The league refresh: once a day by itself, and whenever the site's "Refresh league" button
asks for one. This is the only scheduled job.

Started every minute by a LaunchAgent on Chris's Mac (com.megabowl.refresh-watch). The first
check at or after 7 AM ET each day runs the refresh by itself (Tuesdays with the ffanalytics
projections too, ahead of that night's waivers); a Mac asleep at 7 runs it when it wakes.
Otherwise it reads the request the button left in the private repo (service/refresh_request.py)
through the `gh` sign-in, and when one is queued runs the same steps:

    tools/refresh.py --no-projections   scrape Yahoo, validate, restore on failure (odds when due)
    tools/publish_web.py                rebuild data/web/ and check it
    tools/update_xwins.py               the xWins workbook (never blocks)
    commit + push                       only the refreshed paths, by name

then waits until raw.githubusercontent.com serves the new files (the site reads them from
there), clears the site's cache (tools/clear_site_cache.py) and marks the request done.
Whatever the request file says, this only ever runs that fixed sequence.

    PYTHONPATH=. .venv/bin/python tools/refresh_watch.py      # one check, then exit
"""
from __future__ import annotations

import base64
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
REPO = "chrispimentel3/mega-bowl-private"
BRANCH = "offer-log"
PATH = "data/refresh.json"
SITE_REPO_RAW = "https://raw.githubusercontent.com/chrispimentel3/ff-dashboard/main/data/web/action_board.json"
LOCK = Path.home() / "Library" / "Caches" / "mega-bowl-refresh.lock"
YAHOO_FILES = ["data/yahoo_rosters.csv", "data/yahoo_free_agents.csv", "data/yahoo_standings.csv",
               "data/yahoo_transactions.csv", "data/yahoo_scores.csv", "data/yahoo_faab.csv",
               "data/yahoo_faab_bids.csv"]
MUST_IGNORE = [".env", ".streamlit/secrets.toml", "data/yahoo_state.json", "config/managers.json"]
RAW_WAIT = 420      # seconds to wait for raw.githubusercontent.com to serve the push (its cache is ~5 min)
YAHOO_SAYS = {
    2: "Your Yahoo sign-in expired, so rosters weren't updated. On your Mac, run: "
       "cd ~/Desktop/Test/ff-dashboard && .venv/bin/python -m mega.yahoo login",
    3: "Yahoo's data failed its checks, so the previous rosters were kept.",
    4: "The Yahoo scrape failed, so the previous rosters were kept.",
}


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"[{os.getpid()}]", msg, flush=True)


def gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True, timeout=60)


def read_state() -> tuple[dict, str | None]:
    r = gh(f"repos/{REPO}/contents/{PATH}?ref={BRANCH}")
    if r.returncode != 0:
        if "404" in r.stderr or "Not Found" in r.stdout:
            return {}, None
        raise RuntimeError(f"gh couldn't read the request: {r.stderr.strip() or r.stdout.strip()}")
    f = json.loads(r.stdout)
    return json.loads(base64.b64decode(f["content"]).decode() or "{}"), f["sha"]


def write_state(state: dict, sha: str | None) -> str | None:
    args = [f"repos/{REPO}/contents/{PATH}", "-X", "PUT", "-f", f"message=refresh: {state['status']}",
            "-f", f"branch={BRANCH}", "-f", "content=" + base64.b64encode(json.dumps(state).encode()).decode()]
    if sha:
        args += ["-f", f"sha={sha}"]
    r = gh(*args)
    if r.returncode != 0:
        raise RuntimeError(f"gh couldn't write the request: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout)["content"]["sha"]


def run(*cmd: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(list(cmd), cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    log(f"$ {' '.join(cmd)} -> {r.returncode}")
    for line in (r.stdout + r.stderr).splitlines()[-15:]:
        log("  " + line)
    return r


def git(*args: str) -> subprocess.CompletedProcess:
    return run("git", *args, timeout=300)


def publish(stage: list[str], what: str) -> str | None:
    """Commit exactly `stage` and push. Other changes in the checkout, staged or not, are
    left alone (`git commit -- <paths>`). Returns the new commit, or None if nothing changed."""
    if any(git("check-ignore", "-q", p).returncode != 0 for p in MUST_IGNORE):
        raise RuntimeError("a secrets file isn't gitignored; nothing was committed")
    git("add", "--", *stage)
    if git("diff", "--cached", "--quiet", "--", *stage).returncode == 0:
        return None
    msg = f"{what} {time.strftime('%Y-%m-%d %H:%M')}"
    if git("commit", "-q", "-m", msg, "--", *stage).returncode != 0:
        raise RuntimeError("git commit failed")
    if git("fetch", "-q", "origin").returncode != 0 or git("rebase", "-q", "--autostash", "origin/main").returncode != 0:
        git("rebase", "--abort")
        raise RuntimeError("couldn't rebase onto the latest main; nothing was pushed")
    if git("push", "-q").returncode != 0:
        raise RuntimeError("git push failed")
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def wait_for_raw() -> bool:
    """Until raw.githubusercontent.com serves the action board we just pushed — at the exact URL
    the site fetches: a cache-busting query would see the push before the site's URL does."""
    want = (ROOT / "data" / "web" / "action_board.json").read_bytes()
    deadline = time.time() + RAW_WAIT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(SITE_REPO_RAW, timeout=20) as r:
                if r.read() == want:
                    return True
        except OSError:
            pass
        time.sleep(20)
    return False


def pipeline(step, what: str, projections: bool = False) -> dict:
    """Yahoo pull, rebuild, xWins, commit + push, wait for GitHub, clear the site cache —
    the sequence both the button and the daily run use. `step(name)` reports progress."""
    step("Pulling your league from Yahoo")
    y = run(sys.executable, "tools/refresh.py", *([] if projections else ["--no-projections"]))
    yahoo_ok = y.returncode == 0

    step("Rebuilding the site")
    p = run(sys.executable, "tools/publish_web.py")
    stage = next((ln.split()[1:] for ln in p.stdout.splitlines() if ln.startswith("STAGE ")), []) if p.returncode == 0 else []
    run(sys.executable, "tools/update_xwins.py", timeout=300)

    paths = (YAHOO_FILES if yahoo_ok else []) + stage
    note = YAHOO_SAYS.get(y.returncode, f"The Yahoo scrape exited {y.returncode}; the previous rosters were kept.") if not yahoo_ok else None
    if p.returncode != 0:
        note = (note + " " if note else "") + f"The site rebuild failed its checks (exit {p.returncode}), so the site wasn't updated."
    if not paths:
        return {"failed": True, "commit": None, "note": note, "yahoo": yahoo_ok}

    step("Publishing")
    commit = publish(paths, what if yahoo_ok else "Site rebuild")
    if commit:
        step("Waiting for GitHub to serve the new data")
        if not wait_for_raw():
            note = (note + " " if note else "") + "GitHub was slow to serve the new data; the site may lag a few minutes."
        # The button's page clears the cache too, but only if its tab is still open when this
        # finishes; otherwise every page served its old copy (and stale banner) once (2026-11-04).
        run(sys.executable, "tools/clear_site_cache.py")
    return {"failed": False, "commit": commit, "note": note, "yahoo": yahoo_ok}


def refresh(state: dict, sha: str | None) -> None:
    def step(name: str) -> None:
        nonlocal sha
        state.update(status="running", step=name)
        sha = write_state(state, sha)

    r = pipeline(step, "League refresh from the site's Refresh button")
    if r["failed"]:
        state.update(status="failed", step=None, finished_at=int(time.time()), message=r["note"])
    else:
        state.update(status="done", step=None, finished_at=int(time.time()), commit=r["commit"], yahoo=r["yahoo"],
                     message=r["note"] if r["note"] else (None if r["commit"] else "Already up to date."))
    write_state(state, sha)


ET = ZoneInfo("America/New_York")
DAILY_AT = 7                    # ET hour; on Sundays this run also takes the week's odds sweep
DAILY_DONE = Path.home() / "Library" / "Caches" / "mega-bowl-daily.txt"


def daily_due(now) -> bool:
    """True on the first check at or after 7 AM ET each day."""
    try:
        last = DAILY_DONE.read_text().strip()
    except OSError:
        last = ""
    return now.hour >= DAILY_AT and last != now.date().isoformat()


def daily(now) -> None:
    DAILY_DONE.write_text(now.date().isoformat())       # marked first: a failing run mustn't retry every minute
    log("daily refresh")
    r = pipeline(log, "Daily refresh", projections=now.weekday() == 1)
    if r["note"]:
        log("  " + r["note"])


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0                       # the previous minute's run is still refreshing
        now = dt.datetime.now(ET)
        if daily_due(now):
            try:
                daily(now)
            except Exception as e:
                log(f"daily refresh failed: {type(e).__name__}: {e}")
            return 0
        state, sha = read_state()
        if state.get("status") != "queued":
            return 0
        log(f"refresh requested at {time.strftime('%H:%M:%S', time.localtime(state.get('requested_at', 0)))}")
        try:
            refresh(state, sha)
        except Exception as e:
            log(f"failed: {type(e).__name__}: {e}")
            state, sha = read_state()
            state.update(status="failed", step=None, finished_at=int(time.time()), message=str(e))
            write_state(state, sha)
            return 1
        log("done")
        return 0


if __name__ == "__main__":
    sys.exit(main())
