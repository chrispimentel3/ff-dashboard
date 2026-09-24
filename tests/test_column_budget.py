"""The column budget.

The three most-read tables render eighteen columns each. On a laptop the last six sit off
the right edge; on a phone it is a horizontal scroll with no way to know what is missing.
This narrows them to the columns the decision rests on and keeps the rest one click away.

It is additive on purpose: a `ui.table` call passing neither `view` nor `essential` must
behave exactly as it did before any of this existed, because thirty-seven of them do.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega import ui


@pytest.fixture(autouse=True)
def essentials(monkeypatch):
    """Default the whole suite to the narrowed view; the widening case asks for it."""
    monkeypatch.setattr(ui, "detail", lambda: "Essentials")


def _frame(codes):
    return pd.DataFrame([{c: 1 for c in codes}])


# ---------------------------------------------------------------- the additive guarantee
def test_a_table_that_asks_for_nothing_is_untouched():
    d = _frame(["PLAYER", "POS", "PPG", "TGT%", "xFP±", "G", "LAST"])
    out, hidden = ui._budget(d, None, None)
    assert out is d and hidden == []


def test_an_unknown_view_name_is_not_silently_a_blank_policy():
    """A typo in `view=` must not hide every column on the table."""
    d = _frame(["PLAYER", "POS", "PPG"])
    out, hidden = ui._budget(d, "no_such_view", None)
    assert list(out.columns) == list(d.columns) and hidden == []


# ---------------------------------------------------------------- what it keeps
def test_identity_columns_survive_every_policy():
    """A row without a name has no subject."""
    d = _frame(["LOGO", "PLAYER", "ROLE", "SLOT", "POS", "G", "LAST", "CAR", "TGT"])
    out, _ = ui._budget(d, None, ["POS"])
    for code in ("LOGO", "PLAYER", "ROLE", "SLOT"):
        assert code in out.columns, f"{code} is identity and must never be budgeted away"


def test_the_named_essentials_are_kept_and_the_rest_reported():
    d = _frame(["PLAYER", "POS", "ST", "PROJ*", "VEGAS", "NOTE", "G", "LAST", "CAR", "TGT"])
    out, hidden = ui._budget(d, "startsit_start", None)
    assert set(out.columns) == {"PLAYER", "POS", "ST", "PROJ*", "VEGAS", "NOTE"}
    assert set(hidden) == {"G", "LAST", "CAR", "TGT"}


def test_column_order_is_preserved_not_reordered_by_the_policy():
    """Readers scan left to right; a policy that reshuffles is a new table each click."""
    d = _frame(["PLAYER", "POS", "ST", "G", "PROJ*", "LAST", "VEGAS", "CAR", "NOTE"])
    out, _ = ui._budget(d, "startsit_start", None)
    assert list(out.columns) == ["PLAYER", "POS", "ST", "PROJ*", "VEGAS", "NOTE"]


# ---------------------------------------------------------------- when not to bother
def test_hiding_one_or_two_columns_is_not_worth_the_confusion():
    """The reader loses more in 'where did that go' than they gain in width."""
    d = _frame(["PLAYER", "POS", "ST", "PROJ*", "VEGAS", "NOTE", "G"])
    out, hidden = ui._budget(d, "startsit_start", None)
    assert list(out.columns) == list(d.columns) and hidden == []


def test_everything_mode_shows_everything():
    d = _frame(["PLAYER", "POS", "G", "LAST", "CAR", "TGT", "TM#"])
    ui.detail = lambda: "Everything"
    try:
        out, hidden = ui._budget(d, "startsit_start", None)
    finally:
        ui.detail = lambda: "Essentials"
    assert list(out.columns) == list(d.columns) and hidden == []


# ---------------------------------------------------------------- per-call override
def test_a_call_can_name_its_own_essentials():
    """The roster's form column is renamed per the rolling-window slider, so its code is
    not knowable at import time and cannot live in ESSENTIAL."""
    d = _frame(["PLAYER", "L3", "xFP±", "TGT%", "G", "LAST", "CAR", "TGT"])
    out, hidden = ui._budget(d, None, ["L3", "xFP±", "TGT%"])
    assert set(out.columns) == {"PLAYER", "L3", "xFP±", "TGT%"}
    assert len(hidden) == 4


def test_an_explicit_essential_list_beats_the_named_view():
    d = _frame(["PLAYER", "POS", "ST", "PROJ*", "VEGAS", "NOTE", "G", "LAST"])
    out, _ = ui._budget(d, "startsit_start", ["G"])
    assert "G" in out.columns and "PROJ*" not in out.columns


# ---------------------------------------------------------------- the policies themselves
def test_every_policy_names_real_codes():
    """A typo'd code would silently hide a column forever, and nothing would say so."""
    known = set(ui.COLS.values()) | ui._ALWAYS
    for view, codes in ui.ESSENTIAL.items():
        unknown = [c for c in codes if c not in known]
        assert not unknown, f"{view} names codes that no column maps to: {unknown}"


def test_every_essential_column_can_explain_itself():
    """These are the columns a reader sees by default. If one has no tooltip, the default
    view is the one place the glossary does not reach."""
    missing = {v: [c for c in codes if c not in ui.GLOSS and c not in ui._NO_KEY]
               for v, codes in ui.ESSENTIAL.items()}
    missing = {v: c for v, c in missing.items() if c}
    assert not missing, f"essential columns with no GLOSS entry: {missing}"


def test_no_policy_is_wider_than_it_is_worth():
    """Eight rendered columns is the ceiling that still reads without a sideways scroll;
    identity adds a logo and a role on top of whatever the policy names."""
    for view, codes in ui.ESSENTIAL.items():
        assert len(codes) <= 6, f"{view} names {len(codes)} columns before identity is added"
