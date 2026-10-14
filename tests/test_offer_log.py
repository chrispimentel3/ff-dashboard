import base64
import io
import json
import urllib.error

import pytest

from service import offer_log as OL


class FakeGitHub:
    """Just enough of the contents API: one file on one branch, with sha checks."""

    def __init__(self, conflicts=0):
        self.text, self.sha, self.branch, self.conflicts = None, None, False, conflicts

    def __call__(self, method, path, body=None):
        err = lambda code: urllib.error.HTTPError(path, code, "x", {}, io.BytesIO(b""))
        if path.startswith("git/ref/heads/offer-log"):
            if not self.branch:
                raise err(404)
            return {"object": {"sha": "b"}}
        if path == "":
            return {"default_branch": "main"}
        if path.startswith("git/ref/heads/"):
            return {"object": {"sha": "m"}}
        if path == "git/refs":
            self.branch = True
            return {}
        if method == "GET":
            if self.text is None:
                raise err(404)
            return {"content": base64.b64encode(self.text.encode()).decode(), "sha": self.sha}
        if self.conflicts:                 # someone else's write landed first
            self.conflicts -= 1
            self.text = (self.text or "") + json.dumps({"ts": 0, "id": "other", "action": "sent"}) + "\n"
            self.sha = f"moved{self.conflicts}"
            raise err(409)
        assert body.get("sha") == self.sha and body["branch"] == "offer-log"
        self.text, self.sha = base64.b64decode(body["content"]).decode(), f"s{len(body['content'])}"
        return {}


@pytest.fixture
def gh(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(OL, "_call", fake)
    monkeypatch.setattr(OL, "_branch_ready", False)
    return fake


def test_sent_then_outcome_folds_into_one_offer(gh):
    s = OL.append(OL.new_sent("Allen Moore connection", ["Egbuka", "Boston"], ["Skattebo"], 0.82, 2.4, -1.5, "window"))
    assert gh.branch and json.loads(gh.text.splitlines()[0])["partner"] == "Allen Moore connection"
    OL.append({"ts": s["ts"] + 60, "id": s["id"], "action": "declined"})
    (o,) = OL.offers(OL.read()[0])
    assert o["id"] == s["id"] and o["outcome"] == "declined" and o["p_accept"] == 0.82


def test_an_outcome_needs_a_sent_offer(gh):
    with pytest.raises(KeyError):
        OL.append({"ts": 1, "id": "nope", "action": "accepted"})


def test_a_write_that_lost_a_race_retries(monkeypatch):
    fake = FakeGitHub(conflicts=1)
    monkeypatch.setattr(OL, "_call", fake)
    monkeypatch.setattr(OL, "_branch_ready", False)
    OL.append(OL.new_sent("X", ["a"], ["b"], 0.5, 1.0, -1.0, "card"))
    assert [json.loads(l)["id"] != "other" for l in fake.text.splitlines()] == [False, True]   # theirs kept, ours after


def test_the_passcode_gate(monkeypatch):
    monkeypatch.setenv("OFFER_PASSCODE", "hunter2")
    assert OL.passcode_ok("hunter2") and not OL.passcode_ok("hunter3") and not OL.passcode_ok(None)
    monkeypatch.delenv("OFFER_PASSCODE")
    assert not OL.passcode_ok("")                        # unset never opens the gate


def test_the_repo_url_has_no_trailing_slash(monkeypatch):
    monkeypatch.setenv("OFFER_LOG_REPO", " chrispimentel3/mega-bowl-private ")
    assert OL._url("") == "https://api.github.com/repos/chrispimentel3/mega-bowl-private"
    assert OL._url("git/refs").endswith("/mega-bowl-private/git/refs")
