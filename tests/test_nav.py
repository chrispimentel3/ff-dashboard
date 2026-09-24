"""The six pages.

app.py used to be 2,400 lines that ran top to bottom with every tab body inline, so a
helper defined below its call site did not exist yet when the body ran. That reached the
page four separate times. Each body is now a function, which means the whole module exists
before any of it executes — the hazard is designed out rather than guarded against.

These tests read the source. They cannot run the pages (Streamlit's AppTest can only switch
between FILE-based pages, not the function-based ones st.navigation takes here), so
tests/test_app_smoke.py runs the default page and this file checks the wiring of the rest.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"


def _tree() -> ast.Module:
    return ast.parse(APP.read_text())


def _funcs(prefix: str) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in _tree().body
            if isinstance(n, ast.FunctionDef) and n.name.startswith(prefix)}


def _pages() -> list[ast.Call]:
    return [n for n in ast.walk(_tree())
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "Page"]


def test_there_are_six_pages_and_exactly_one_opens_first():
    pages = _pages()
    assert len(pages) == 6
    defaults = [p for p in pages
                if any(k.arg == "default" and getattr(k.value, "value", False) for k in p.keywords)]
    assert len(defaults) == 1, "exactly one page may be the landing page"


# Names that describe where the data came from or what the code is called, rather than what
# the reader wants. Every one of these was a tab label before the rebuild.
FEATURE_NAMES = ("wopr", "archetype", "raw data", "waiver", "standings", "draft value",
                 "glossary", "news", "trade finder", "start/sit", "matchups", "digest")

# Six links have to sit in one row of the top bar. Nothing else stops a title growing until
# the bar wraps, and a wrapped bar is the thing this whole rebuild was about.
TITLE_CAP = 20


def test_every_page_is_a_question_or_an_instruction_not_a_feature_name():
    """The old tabs were named after features and data sources — WOPR, Archetypes, Raw
    Data — which only works if you already know what is in them."""
    titles = [k.value.value for p in _pages() for k in p.keywords if k.arg == "title"]
    assert len(titles) == 6
    for t in titles:
        assert t[0].isupper(), f"{t!r} should read as a sentence"
        bad = [f for f in FEATURE_NAMES if f in t.lower()]
        assert not bad, f"{t!r} is named after the feature inside it, not the question"
    assert sum(t.endswith("?") for t in titles) >= 3


def test_no_title_is_wide_enough_to_wrap_the_top_bar():
    """The nav moved from the sidebar, where a title could be any length, to a single row."""
    titles = [k.value.value for p in _pages() for k in p.keywords if k.arg == "title"]
    long = [t for t in titles if len(t) > TITLE_CAP]
    assert not long, f"too wide for one row of the top bar (cap {TITLE_CAP}): {long}"


def test_the_navigation_sits_at_the_top():
    """Not decoration: the sidebar is left to the four settings that rewrite every screen."""
    nav = [n for n in ast.walk(_tree())
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "navigation"]
    assert len(nav) == 1
    pos = [k.value.value for k in nav[0].keywords if k.arg == "position"]
    assert pos == ["top"], "st.navigation must be given position='top'"


def test_the_league_free_pages_come_last():
    """With the nav groups gone, order is the only thing on screen that shows the seam."""
    paths = [k.value.value for p in _pages() for k in p.keywords if k.arg == "url_path"]
    assert paths[-2:] == ["player", "ask"], f"league-free pages are no longer last: {paths}"


def test_every_page_has_a_url_of_its_own():
    """So that sending someone your Start/Sit is a link, not 'click the third tab'."""
    paths = [k.value.value for p in _pages() for k in p.keywords if k.arg == "url_path"]
    assert len(paths) == 6 and len(set(paths)) == 6
    assert all(p.islower() and p.isalpha() for p in paths)


def test_every_content_block_is_reachable_from_exactly_one_page():
    """Eighteen blocks came across from the old tab tree. A block nobody calls is a
    feature that silently disappeared in the move."""
    tabs = set(_funcs("_tab_"))
    pages = _funcs("_page_")
    assert tabs, "no content functions found"
    called: dict[str, list[str]] = {t: [] for t in tabs}
    for pname, node in pages.items():
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in tabs:
                called[n.id].append(pname)
    orphans = sorted(t for t, where in called.items() if not where)
    assert not orphans, f"content blocks no page renders: {orphans}"
    twice = {t: w for t, w in called.items() if len(w) > 1}
    assert not twice, f"blocks rendered by more than one page: {twice}"


def test_the_navigation_is_built_after_every_page_exists():
    """st.navigation must be the last thing the script does."""
    tree = _tree()
    nav = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "navigation"]
    assert len(nav) == 1
    last_def = max(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef))
    assert nav[0] > last_def, "navigation runs before some page is defined"


def test_no_tab_container_variables_survive_the_move():
    """The old `with tab_x:` containers are gone; a leftover reference would be a name
    bound nowhere."""
    src = APP.read_text()
    for stale in ("sec_now", "sec_team", "sec_get", "sec_more", "sec_players",
                  "with tab_", "st.tabs(["):
        if stale == "st.tabs([":
            continue          # sub-tabs are built by the _tabs helper, which is fine
        assert stale not in src, f"{stale!r} still referenced after the page split"


def test_sub_tabs_go_through_the_one_helper():
    """One place decides how a page's sub-tabs are built, so they cannot drift apart."""
    pages = _funcs("_page_")
    for name, node in pages.items():
        calls = {n.func.id for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_tabs" in calls, f"{name} builds its tabs by hand"


# ---------------------------------------------------------------- the league seam
LEAGUE_NAMES = {"MY_TEAM", "LEAGUE_ID", "LEAGUE_URL", "TEAM_BY_SEAT", "SEAT_BY_TEAM", "MY_SEAT"}


def _reachable_league_refs(page: str) -> dict[str, list[str]]:
    """Every league constant a page can reach, and the function it sits in."""
    tree = _tree()
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    found: dict[str, list[str]] = {}
    seen: set[str] = set()

    def walk(fn: ast.FunctionDef) -> None:
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id in LEAGUE_NAMES:
                found.setdefault(fn.name, []).append(node.id)
            elif isinstance(node, ast.Name) and node.id in funcs and node.id not in seen:
                seen.add(node.id)
                walk(funcs[node.id])
            elif isinstance(node, ast.alias) and node.name in LEAGUE_NAMES:
                found.setdefault(fn.name, []).append(node.name)

    walk(funcs[page])
    return found


def test_the_ask_page_never_touches_the_league():
    """It runs on ask, catalog and roles, none of which have heard of Mega Bowl."""
    assert _reachable_league_refs("_page_ask") == {}


def test_the_player_page_touches_the_league_in_exactly_one_place():
    """The card names who holds him, which is the one league-aware thing on an otherwise
    league-free page. Keeping it in a single named function is what makes pointing this at
    a second league — or at none — a contained change rather than a hunt."""
    refs = _reachable_league_refs("_page_player")
    assert set(refs) == {"_owner_badge"}, (
        f"league constants have spread beyond the ownership badge: {refs}"
    )


def test_the_ownership_badge_disappears_when_there_is_no_league():
    """With no scraped rosters the card must be a plain NFL player card, not a broken
    Mega Bowl one — so the very first thing it does is give up."""
    fn = {n.name: n for n in _tree().body
          if isinstance(n, ast.FunctionDef)}["_owner_badge"]
    first = next(s for s in fn.body if not isinstance(s, ast.Expr))   # skip the docstring
    assert isinstance(first, ast.If), "the no-league case must be handled before anything else"
    assert isinstance(first.body[0], ast.Return)
    assert first.body[0].value.value == "", "with no league the badge renders nothing"
