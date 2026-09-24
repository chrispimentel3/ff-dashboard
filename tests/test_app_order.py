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
import builtins
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


# ---------------------------------------------------------------- bare names, not calls
def _unbound_module_level_loads(tree: ast.Module):
    """(name, lineno) for every module-level name READ before anything binds it.

    The call check above misses a whole half of this mistake. `MY_TEAM` is imported inside
    four different functions and aliased at module level only near the bottom of the file;
    a tab two hundred lines above that referenced it bare, which is not a call, so nothing
    caught it — and because it sat behind a checkbox, the page looked fine until someone
    ticked the box.

    Function, class, lambda and comprehension bodies are skipped: they run later, or in
    their own scope, by which point the module is whole.
    """
    bound = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
    bad = []

    def bind(target) -> None:
        for n in ast.walk(target):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                bound.add(n.id)

    def visit(node) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add(a.asname or a.name.split(".")[0])
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                             ast.Lambda)):
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in bound:
                bad.append((node.id, node.lineno))
            else:
                bound.add(node.id)
            return
        # the value is evaluated before the target is bound, so order matters here
        if isinstance(node, ast.Assign):
            visit(node.value)
            for t in node.targets:
                bind(t)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            visit(node.iter)
            bind(node.target)
            for b in node.body + node.orelse:
                visit(b)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for it in node.items:
                visit(it.context_expr)
                if it.optional_vars is not None:
                    bind(it.optional_vars)
            for b in node.body:
                visit(b)
            return
        if isinstance(node, ast.Try):
            for b in node.body:
                visit(b)
            for h in node.handlers:
                if h.name:
                    bound.add(h.name)
                for b in h.body:
                    visit(b)
            for b in node.orelse + node.finalbody:
                visit(b)
            return
        for c in ast.iter_child_nodes(node):
            visit(c)

    for node in tree.body:
        visit(node)
    return bad


def test_no_module_level_name_is_read_before_it_exists():
    bad = _unbound_module_level_loads(_tree())
    assert not bad, (
        "app.py runs top to bottom, so these names do not exist yet where they are read:\n  "
        + "\n  ".join(f"{n} on line {l}" for n, l in sorted(set(bad), key=lambda x: x[1]))
    )


def test_the_name_check_would_catch_an_import_that_lands_too_late():
    """Exactly the shape of the real bug: used in a tab, imported further down."""
    broken = ast.parse(
        "tab = 1\nwith tab:\n    x = MY_TEAM\n\nfrom mega.config import MY_TEAM\n"
    )
    assert _unbound_module_level_loads(broken) == [("MY_TEAM", 3)]


def test_the_name_check_does_not_flag_ordinary_code():
    fine = ast.parse(
        "import os\nA = 1\nwith open(os.devnull) as f:\n    B = A + 1\n"
        "for i in range(B):\n    C = i\nD = [q for q in range(A)]\n"
    )
    assert _unbound_module_level_loads(fine) == []


def test_app_parses():
    assert _tree().body
