"""Run the whole app and assert it raised nothing.

The AST guards in test_app_order.py catch a name used before it is defined, but they cannot
see a name that is only bound inside an `if` or a `try` and read outside it — which is the
shape of the bug that survived them (MY_TEAM, imported inside `if not _o.empty`). Only
actually running the script finds that class.

Slow by the standards of this suite, and it touches the real on-disk caches, so it is
marked `slow` and skipped when the data it needs is not on this machine.
"""
from __future__ import annotations

import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"
DATA = APP.parent / "data"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("streamlit.testing.v1")
    if not (DATA / "yahoo_rosters.csv").exists():
        pytest.skip("no scraped league data on this machine")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


def test_the_app_runs_without_raising(app):
    assert app.exception == [], "\n".join(str(e) for e in app.exception)


def test_it_puts_something_on_screen(app):
    assert len(app.markdown) > 5, "the app ran but rendered almost nothing"
