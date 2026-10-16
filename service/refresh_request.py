"""The site's "Refresh league" button: a request the Mac picks up.

The refresh has to run on Chris's Mac — the Yahoo scrape signs in with his saved session,
which never leaves it. So the button only records a request, as one small JSON file next to
the offer log (same private repo, same branch, same token); tools/refresh_watch.py on the
Mac checks it every minute, runs the refresh, and writes the outcome back:

    {"status": "queued" | "running" | "done" | "failed", "requested_at": ..., "step": ...,
     "finished_at": ..., "message": ...}

Nothing in the file is ever executed: the Mac runs one fixed refresh whatever it says.
"""
from __future__ import annotations

import base64
import json
import urllib.error

from service.offer_log import BRANCH, _call, _ensure_branch

PATH = "data/refresh.json"
COOLDOWN = 600      # a press within this many seconds of the last finished refresh is ignored
STALE = 7200        # a request still "running" after this long died with the Mac (sleep, crash)


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
