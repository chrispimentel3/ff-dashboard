"""The glossary. Its job is that someone who knows football and has never seen this app
can read a tag and know what it claims. These tests hold it to that."""
from __future__ import annotations

import pytest

from mega import glossary as g
from mega import roles


# ---------------------------------------------------------------- coverage
def test_every_role_the_engine_can_assign_has_plain_english():
    missing = [r for r in roles.ROLES if r not in g.ROLES]
    assert not missing, f"roles with no glossary entry: {missing}"


def test_every_flag_the_engine_can_raise_has_plain_english():
    emitted = set(roles.FLAG_METRIC) | {"ROLE+", "ROLE-", "1D/RR", "CUFF"}
    missing = sorted(emitted - set(g.FLAGS))
    assert not missing, f"flags with no glossary entry: {missing}"


def test_nothing_is_defined_that_the_engine_never_emits():
    """A glossary entry for a tag nobody produces is a lie waiting to be believed."""
    assert set(g.ROLES) <= set(roles.ROLES)
    # CUFF is raised by mega/needs.py from the handcuff table, not by roles.py
    assert set(g.FLAGS) <= set(roles.FLAG_METRIC) | {"ROLE+", "ROLE-", "1D/RR", "CUFF"}


# ---------------------------------------------------------------- readable
# The terms being explained cannot do the explaining. Someone reading "target hog" should
# not be sent to look up "target share".
JARGON = ("target share", "air-yards share", "air yards share", "snap share", "snap %",
          "route participation", "routes run", "wopr", "tprr", "1d/rr", "role index",
          "z-score", "z score", "shrunk", "shrinkage", "percentile", "baseline",
          "expected points", "xfp", "denominator", "numerator", "handoff", "§")


@pytest.mark.parametrize("term", list(g.ROLES.values()) + list(g.FLAGS.values()))
def test_the_plain_english_contains_no_jargon(term):
    low = term.plain.lower()
    found = [j for j in JARGON if j in low]
    assert not found, f"{term.code}: {found} in {term.plain!r}"


@pytest.mark.parametrize("term", list(g.ROLES.values()) + list(g.FLAGS.values()))
def test_every_term_says_what_it_is_and_why_it_matters(term):
    assert len(term.plain) > 25 and term.plain.endswith(".")
    assert term.matters, f"{term.code} has no 'why it matters'"
    assert term.label and term.label != term.code.lower()


def test_labels_are_short_enough_for_a_table_cell():
    for t in list(g.ROLES.values()) + list(g.FLAGS.values()):
        assert len(t.label) <= 14, f"{t.code}: {t.label!r} is too long for a cell"


# ---------------------------------------------------------------- rendering
def test_a_code_becomes_words():
    assert g.role_label("COMMITTEE") == "Committee back"
    assert g.role_label("TE1-BLK") == "TE1 blocking"
    assert g.flag_label("ROLE+") == "playing up"
    assert g.flag_label("GL") == "goal line"


def test_a_one_game_flag_says_so():
    assert g.flag_label("GL", "spike") == "goal line (1 game)"
    assert g.flag_label("GL", "sustained") == "goal line"


def test_a_player_with_no_snaps_says_so_rather_than_guessing():
    assert g.cell("", [], {}) == "no role yet"
    assert g.role_label(None) == "no role yet"


def test_a_change_flag_wins_the_limited_space():
    """ROLE+ says something is changing, which beats a description of what already is."""
    out = g.cell("WR3", ["TGT", "AIR", "ROLE+"], {})
    assert out.startswith("WR3 · playing up")


def test_a_durable_flag_outranks_a_one_game_flag():
    """A single good afternoon must not push a pattern out of the only two slots."""
    out = g.cell("LEAD", ["SNAP", "GL"], {"SNAP": "spike", "GL": "sustained"})
    assert out == "Lead back · goal line · every-down (1 game)"


def test_extra_flags_are_counted_not_dropped_silently():
    out = g.cell("WR1", ["TGT", "AIR", "GL", "1D/RR"], {})
    assert out.endswith("+2 more")


def test_an_unknown_code_passes_through_rather_than_vanishing():
    assert g.role_label("WR9") == "WR9"
    assert g.flag_label("NEWFLAG") == "NEWFLAG"


def test_the_glossary_frame_is_complete_and_grouped():
    f = g.frame()
    assert set(f["group"]) == {"Role", "Flag", "How long", "Vegas"}
    assert len(f) == len(g.ROLES) + len(g.FLAGS) + len(g.VEGAS) + 2
    assert f["what it means"].str.len().min() > 25


def test_every_vegas_column_the_app_shows_has_an_entry():
    """A column on screen with no glossary row is exactly the fault the glossary tab
    promises not to have."""
    from mega import ui
    shown = {ui.COLS[c] for c in ("vegas", "vegas_edge", "vegas_parts", "vegas_complete")}
    assert shown <= set(g.VEGAS)


def test_the_vegas_kind_filters_on_its_own():
    f = g.frame("vegas")
    assert set(f["group"]) == {"Vegas"} and len(f) == len(g.VEGAS)


def test_the_vegas_headline_explains_why_projections_beat_the_posted_line():
    """The one thing a reader will otherwise think is a bug."""
    h = g.VEGAS_HEADLINE.lower()
    assert "middle" in h and "above" in h
