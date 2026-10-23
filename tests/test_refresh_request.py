"""service/refresh_request.py — the site's Refresh button queues at most one refresh."""
from __future__ import annotations

import base64
import json
import urllib.error

from service import refresh_request as RR


class FakeRepo:
    def __init__(self, state=None):
        self.state, self.writes = state, 0

    def __call__(self, method, path, body=None):
        if method == "GET":
            if self.state is None:
                raise urllib.error.HTTPError(path, 404, "nf", {}, None)
            return {"content": base64.b64encode(json.dumps(self.state).encode()).decode(), "sha": "s"}
        self.writes += 1
        self.state = json.loads(base64.b64decode(body["content"]))
        return {}


def run(monkeypatch, state, now):
    repo = FakeRepo(state)
    monkeypatch.setattr(RR, "_call", repo)
    monkeypatch.setattr(RR, "_ensure_branch", lambda: None)
    return RR.request(now), repo


def test_queues_once_and_respects_a_running_or_recent_refresh(monkeypatch):
    out, repo = run(monkeypatch, None, 1000)
    assert out == {"status": "queued", "requested_at": 1000} and repo.writes == 1
    out, repo = run(monkeypatch, {"status": "queued", "requested_at": 900}, 1000)
    assert out["requested_at"] == 900 and repo.writes == 0
    out, repo = run(monkeypatch, {"status": "running", "requested_at": 900}, 1000)
    assert out["status"] == "running" and repo.writes == 0
    out, repo = run(monkeypatch, {"status": "done", "finished_at": 900}, 1000)          # within the cooldown
    assert out["cooldown"] and repo.writes == 0
    out, repo = run(monkeypatch, {"status": "done", "finished_at": 100}, 1000)
    assert out["status"] == "queued" and repo.writes == 1
    out, repo = run(monkeypatch, {"status": "running", "requested_at": 0}, RR.STALE + 1)   # died mid-run
    assert out["status"] == "queued" and repo.writes == 1


def test_the_passcode_gate(monkeypatch):
    monkeypatch.setenv("OFFER_PASSCODE", "hunter2")
    assert RR.passcode_ok("hunter2") and not RR.passcode_ok("hunter3") and not RR.passcode_ok(None)
    monkeypatch.delenv("OFFER_PASSCODE")
    assert not RR.passcode_ok("")                        # unset never opens the gate


def test_the_repo_url_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("OFFER_LOG_REPO", " chrispimentel3/mega-bowl-private ")
    assert RR._url("") == "https://api.github.com/repos/chrispimentel3/mega-bowl-private"
    assert RR._url("git/refs").endswith("/mega-bowl-private/git/refs")
