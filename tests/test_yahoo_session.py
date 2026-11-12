"""mega/yahoo.py LeagueSession: one slow Yahoo page gets a second try instead of failing the pull."""
from __future__ import annotations

import pytest
from playwright.sync_api import TimeoutError as PWTimeout

from mega.yahoo import LeagueSession


class FakePage:
    def __init__(self, slow: bool):
        self.slow, self.closed, self.url = slow, False, "https://football.fantasysports.yahoo.com/f1/1"

    def goto(self, url, **kw):
        if self.slow:
            raise PWTimeout("Timeout 45000ms exceeded.")

    def wait_for_timeout(self, ms):
        pass

    def content(self):
        return "<html>league</html>"

    def close(self):
        self.closed = True


class FakeCtx:
    def __init__(self, slow_loads: int):
        self.slow_loads, self.pages = slow_loads, []

    def new_page(self):
        p = FakePage(slow=len(self.pages) < self.slow_loads)
        self.pages.append(p)
        return p


def _session(slow_loads: int) -> LeagueSession:
    s = object.__new__(LeagueSession)          # no browser, no saved login
    s._ctx = FakeCtx(slow_loads)
    return s


def test_a_slow_page_gets_one_more_try_in_a_fresh_tab():
    s = _session(slow_loads=1)
    assert s.get("https://football.fantasysports.yahoo.com/f1/1/transactions") == "<html>league</html>"
    assert len(s._ctx.pages) == 2 and s._ctx.pages[0].closed


def test_two_slow_loads_still_fail_the_pull():
    s = _session(slow_loads=2)
    with pytest.raises(PWTimeout):
        s.get("https://football.fantasysports.yahoo.com/f1/1/transactions")
    assert len(s._ctx.pages) == 2 and all(p.closed for p in s._ctx.pages)
