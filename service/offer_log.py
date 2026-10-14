"""The trade-offer log: offers Chris actually sent, and how each came back.

Yahoo only ever shows completed trades, never the offers turned down, so the acceptance
model's priors (config/acceptance_priors.json) have nothing to be fitted on. This is the
missing half: one line per event, "sent" with the model's estimate at the time, then
"accepted" / "declined" / "withdrawn" against the same id.

Stored as JSON lines in a GitHub repo through the contents API — Render's disk is wiped on
every deploy — on its own branch, so a log entry never triggers a redeploy or collides with
the daily data pushes. The repo is OFFER_LOG_REPO, meant to be a PRIVATE one: the log is
what TaylorMade offered whom, which a league-mate shouldn't be able to read. The token
(GITHUB_TOKEN, contents read/write on that repo only) and OFFER_PASSCODE live in Render's
environment; reads and writes both need the passcode.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import time
import urllib.error
import urllib.request

API = "https://api.github.com"
BRANCH = "offer-log"
PATH = "data/trade_offers.jsonl"
OUTCOMES = ("accepted", "declined", "withdrawn")


def configured() -> bool:
    return all(os.environ.get(k) for k in ("GITHUB_TOKEN", "OFFER_PASSCODE", "OFFER_LOG_REPO"))


def passcode_ok(given: str | None) -> bool:
    want = os.environ.get("OFFER_PASSCODE", "")
    return bool(want) and hmac.compare_digest((given or "").encode(), want.encode())


def _url(path: str) -> str:
    """The repo's API URL, `path` under it. No trailing slash on the repo itself: GitHub
    answers 404 to /repos/owner/name/ (which kept the log from ever creating its branch)."""
    base = f"{API}/repos/{os.environ['OFFER_LOG_REPO'].strip()}"
    return f"{base}/{path}" if path else base


def _call(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(
        _url(path), method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "mega-bowl-offer-log"})
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


def read() -> tuple[list[dict], str | None]:
    """Every logged event, oldest first, and the file's sha (None before the first entry)."""
    _ensure_branch()
    try:
        f = _call("GET", f"contents/{PATH}?ref={BRANCH}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return [], None
        raise
    text = base64.b64decode(f["content"]).decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()], f["sha"]


def append(entry: dict) -> dict:
    """Add one event. A write that lost a race (the file changed under it) re-reads and retries."""
    for attempt in range(3):
        rows, sha = read()
        if entry["action"] in OUTCOMES and not any(r.get("id") == entry["id"] and r["action"] == "sent" for r in rows):
            raise KeyError(entry["id"])
        text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows + [entry])
        body = {"message": f"offer log: {entry['action']} {entry.get('partner') or entry['id']}",
                "content": base64.b64encode(text.encode()).decode(), "branch": BRANCH}
        if sha:
            body["sha"] = sha
        try:
            _call("PUT", f"contents/{PATH}", body)
            return entry
        except urllib.error.HTTPError as e:
            if e.code in (409, 422) and attempt < 2:
                continue
            raise
    raise RuntimeError("offer log write kept conflicting")


def new_sent(partner: str, give: list, get: list, p_accept, d_me, d_them, source) -> dict:
    return {"ts": int(time.time()), "id": secrets.token_hex(4), "action": "sent", "partner": partner,
            "give": give, "get": get, "p_accept": p_accept, "d_me": d_me, "d_them": d_them, "source": source}


def offers(rows: list[dict]) -> list[dict]:
    """One row per sent offer, with its latest outcome (None while it's open), newest first."""
    by_id = {r["id"]: {**r, "outcome": None, "outcome_ts": None} for r in rows if r["action"] == "sent"}
    for r in rows:
        if r["action"] in OUTCOMES and r["id"] in by_id:
            by_id[r["id"]].update(outcome=r["action"], outcome_ts=r["ts"])
    return sorted(by_id.values(), key=lambda o: -o["ts"])
