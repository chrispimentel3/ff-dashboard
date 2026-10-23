"""The site's "Refresh league" button: a request the Mac picks up.

The refresh has to run on Chris's Mac — the Yahoo scrape signs in with his saved session,
which never leaves it. So the button only records a request, as one small JSON file in a
private repo, through the GitHub contents API (Render's disk is wiped on every deploy);
tools/refresh_watch.py on the Mac checks it every minute, runs the refresh, and writes the
outcome back:

    {"status": "queued" | "running" | "done" | "failed", "requested_at": ..., "step": ...,
     "finished_at": ..., "message": ...}

Nothing in the file is ever executed: the Mac runs one fixed refresh whatever it says.

The repo (OFFER_LOG_REPO), the token (GITHUB_TOKEN, contents read/write on that repo only) and
the passcode (OFFER_PASSCODE) live in Render's environment. The names and the "offer-log"
branch are left over from the trade-offer log this used to share them with (removed
2026-10-22); they stay because Render and the Mac's watcher are configured with them.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import urllib.error
import urllib.request

API = "https://api.github.com"
BRANCH = "offer-log"
PATH = "data/refresh.json"
COOLDOWN = 600      # a press within this many seconds of the last finished refresh is ignored
STALE = 7200        # a request still "running" after this long died with the Mac (sleep, crash)


def configured() -> bool:
    return all(os.environ.get(k) for k in ("GITHUB_TOKEN", "OFFER_PASSCODE", "OFFER_LOG_REPO"))


def passcode_ok(given: str | None) -> bool:
    want = os.environ.get("OFFER_PASSCODE", "")
    return bool(want) and hmac.compare_digest((given or "").encode(), want.encode())


def _url(path: str) -> str:
    """The repo's API URL, `path` under it. No trailing slash on the repo itself: GitHub
    answers 404 to /repos/owner/name/."""
    base = f"{API}/repos/{os.environ['OFFER_LOG_REPO'].strip()}"
    return f"{base}/{path}" if path else base


def _call(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(
        _url(path), method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "mega-bowl-refresh"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read() or b"null")


_branch_ready = False


def _ensure_branch() -> None:
    global _branch_ready
    if _branch_ready:
        return
    try:
        _call("GET", f"git/ref/heads/{BRANCH}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        repo = _call("GET", "")
        base = _call("GET", f"git/ref/heads/{repo['default_branch']}")["object"]["sha"]
        _call("POST", "git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": base})
    _branch_ready = True


def read() -> tuple[dict, str | None]:
    _ensure_branch()
    try:
        f = _call("GET", f"contents/{PATH}?ref={BRANCH}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}, None
        raise
    return json.loads(base64.b64decode(f["content"]).decode() or "{}"), f["sha"]


def request(now: int) -> dict:
    """Queue a refresh unless one is already queued or running, or one finished within
    COOLDOWN. Returns the state as it now stands."""
    for _ in range(3):
        state, sha = read()
        if state.get("status") == "queued" or (
                state.get("status") == "running" and now - int(state.get("requested_at") or 0) < STALE):
            return state
        if state.get("status") == "done" and now - int(state.get("finished_at") or 0) < COOLDOWN:
            return {**state, "cooldown": True}
        new = {"status": "queued", "requested_at": now}
        body = {"message": "refresh: queued", "branch": BRANCH,
                "content": base64.b64encode(json.dumps(new).encode()).decode()}
        if sha:
            body["sha"] = sha
        try:
            _call("PUT", f"contents/{PATH}", body)
            return new
        except urllib.error.HTTPError as e:
            if e.code not in (409, 422):      # the Mac wrote at the same moment: read again
                raise
    raise RuntimeError("the refresh request kept conflicting")
