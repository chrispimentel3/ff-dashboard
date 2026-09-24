"""app.py is a Streamlit script: it runs top to bottom, once, every rerun.

So a helper defined BELOW the line that calls it simply does not exist yet, and the call
raises NameError. That bit twice in one sitting — the player card calling `_role_ctx` and
the League tab calling `_odds` — and both times a `try/except` turned it into a small grey
line nobody would read, so the feature was just missing from the page.

This walks the module's own statements in order and fails if a top-level call happens
before its definition. It is a cheap structural check for a mistake that is otherwise only
visible by opening the tab and noticing an absence.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"


def _tree() -> ast.Module:
    return ast.parse(APP.read_text())


def _module_level_names(tree: ast.Module) -> dict[str, int]:
    """Every function/class defined at module level, and the line it appears on."""
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node.lineno
    return out


def _calls_outside_functions(tree: ast.Module):
    """(name, lineno) for every call made in code that runs at import time.

    Bodies of `def`s are skipped: those run later, by which point the whole module exists.
    `with` and `if` bodies are NOT skipped — that is exactly where the tab code lives.
    """
    found = []

    def walk(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    found.append((sub.func.id, sub.lineno))
    walk(tree.body)
    return found


def test_no_helper_is_called_before_it_is_defined():
    tree = _tree()
    defined = _module_level_names(tree)
    bad = []
    for name, line in _calls_outside_functions(tree):
        at = defined.get(name)
        if at is not None and line < at:
            bad.append(f"{name}() called on line {line} but defined on line {at}")
    assert not bad, (
        "app.py runs top to bottom, so these calls hit a name that does not exist yet:\n  "
        + "\n  ".join(sorted(set(bad)))
    )


def test_the_check_would_actually_catch_the_bug_it_exists_for():
    """A guard that cannot fail is not a guard."""
    broken = ast.parse("with tab:\n    x = _later(1)\n\n\ndef _later(n):\n    return n\n")
    defined = _module_level_names(broken)
    hits = [(n, l) for n, l in _calls_outside_functions(broken)
            if n in defined and l < defined[n]]
    assert hits == [("_later", 2)]


def test_app_parses():
    assert _tree().body
