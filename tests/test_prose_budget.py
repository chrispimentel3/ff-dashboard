"""How much prose the page is allowed.

The dashboard had about a hundred and twenty-six separate pieces of explanation — sixty-six
captions, seventeen info boxes, twelve warnings — and most of the captions sat BELOW the
table they described, so a reader met the numbers first and the meaning second.

These are ratchets, not targets. Each count may fall and may not rise, so the cleanup
sticks without anyone having to remember it in review. Lower a ceiling when you beat it.

The rule they encode:

    ui.answer   the conclusion, one per page, above everything
    ui.h        a section title
    ui.lede     at most one per section, method only — never a conclusion, never a column
    ui.table    the evidence
    ui.note     at most one per section, below, a caveat only
    GLOSS       every per-column explanation, reaching every table that shows that column
"""
from __future__ import annotations

import ast
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"
SRC = APP.read_text()

# Ceilings, not goals. Beat one and lower it.
CEILING = {"st.caption": 58, "st.info": 15, "st.warning": 4, "ui.lede": 13}


@pytest.mark.parametrize("call,ceiling", sorted(CEILING.items()))
def test_prose_only_goes_down(call, ceiling):
    n = SRC.count(f"{call}(")
    assert n <= ceiling, (
        f"{call} is used {n} times, above the ceiling of {ceiling}. If the extra one really "
        "explains a column, it belongs in GLOSS, which reaches every table showing that "
        "column rather than only this one."
    )


def test_exception_handlers_speak_with_one_voice():
    """An outsider should not be reading raw Python exception names off the page."""
    assert 'unavailable: {' not in SRC, (
        "use ui.unavailable(what, exc) — it phrases the failure the same way everywhere"
    )


def test_every_page_leads_with_an_answer():
    """One sentence stating the conclusion, before any evidence."""
    tree = ast.parse(SRC)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    pages = {n: f for n, f in funcs.items() if n.startswith("_page_")}
    assert pages, "no page functions found"

    def answers(fn: ast.FunctionDef, seen: set[str]) -> int:
        """Count ui.answer calls reachable from a page.

        A page names its content functions rather than calling them — they are handed to
        the _tabs helper as values — so follow every Name that resolves to a function, not
        only the ones in call position.
        """
        total = 0
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "answer"):
                total += 1
            elif isinstance(node, ast.Name) and node.id in funcs and node.id not in seen:
                seen.add(node.id)
                total += answers(funcs[node.id], seen)
        return total

    # the two league-free pages answer per player and per question, not per page
    for name in ("_page_week", "_page_start", "_page_upgrade", "_page_review"):
        assert answers(pages[name], set()) >= 1, f"{name} opens with evidence, not an answer"


def test_the_answer_is_derived_rather_than_written():
    """A sentence typed into the source goes stale the week after, and a reader has no way
    to tell that it has."""
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "answer" and node.args):
            first = node.args[0]
            assert not (isinstance(first, ast.Constant) and isinstance(first.value, str)), (
                "ui.answer was given a fixed string; it should be built from this week's "
                "numbers"
            )


def test_column_meanings_live_in_one_place():
    """The Vegas median-to-mean explanation was written out near-verbatim on two tabs. It
    is one sentence in GLOSS now, and every table showing that column gets it."""
    assert SRC.count("corrected to a mean") == 0
    assert SRC.count("medians and corrected to means") == 0
    from mega import ui
    assert "MIDDLE outcome" in ui.GLOSS["VEGAS"]
