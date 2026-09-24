"""The six charts.

Altair builds a full Vega-Lite spec without Streamlit, so every one of these assertions is
on the real chart definition rather than on a screenshot nobody re-checks.

The one that matters most is `test_a_comparison_is_grouped_not_stacked`. `st.bar_chart`
stacks a multi-column frame by default — Streamlit 1.63 documents `stack=None` as "Vega's
default" — so the season "actual vs expected" chart rendered actual PLUS expected: a total
nobody asked for, under a heading promising a comparison.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega import ui


@pytest.fixture
def drawn(monkeypatch):
    """Capture the chart instead of rendering it."""
    seen = {}

    def _cap(ch, caption):
        seen["spec"] = ch.to_dict()
        seen["caption"] = caption
    monkeypatch.setattr(ui, "_render", _cap)
    return seen


def _weeks(players, weeks=4, value=0.8):
    return pd.DataFrame([{"week": w, "who": p, "val": value}
                         for p in players for w in range(1, weeks + 1)])


def _find(node, key):
    """Every value stored under `key` anywhere in the spec."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                out.append(v)
            out += _find(v, key)
    elif isinstance(node, list):
        for v in node:
            out += _find(v, key)
    return out


# ---------------------------------------------------------------- the stacking bug
def test_a_comparison_is_grouped_not_stacked(drawn):
    """yOffset is what makes two bars sit side by side instead of end to end."""
    df = pd.DataFrame([{"player": "A", "kind": "Expected", "pts": 10.0},
                       {"player": "A", "kind": "Actual", "pts": 14.0}])
    ui.bar_compare(df, "player", "kind", "pts")
    assert "yOffset" in drawn["spec"]["encoding"]


def test_the_comparison_reads_expected_then_actual(drawn):
    df = pd.DataFrame([{"player": "A", "kind": "Expected", "pts": 10.0},
                       {"player": "A", "kind": "Actual", "pts": 14.0}])
    ui.bar_compare(df, "player", "kind", "pts", domain=["Expected", "Actual"],
                   palette=(ui.INK3, ui.NAVY))
    scale = drawn["spec"]["encoding"]["color"]["scale"]
    assert scale["domain"] == ["Expected", "Actual"]
    assert scale["range"] == [ui.INK3, ui.NAVY]


def test_the_category_goes_on_the_vertical_axis(drawn):
    """Fifteen full player names do not fit across an x axis."""
    df = pd.DataFrame([{"player": f"Player {i}", "kind": "Actual", "pts": float(i)}
                       for i in range(15)])
    ui.bar_compare(df, "player", "kind", "pts")
    assert drawn["spec"]["encoding"]["y"]["field"] == "player"
    assert drawn["spec"]["encoding"]["x"]["field"] == "pts"


def test_bar_height_grows_with_the_number_of_rows(drawn):
    ui.bar_compare(pd.DataFrame([{"p": f"P{i}", "k": "A", "v": 1.0} for i in range(20)]),
                   "p", "k", "v")
    assert drawn["spec"]["height"] >= 400


# ---------------------------------------------------------------- percent formatting
def test_a_share_reads_as_a_percentage_on_both_axis_and_tooltip(drawn):
    """Snap share and target share are 0-1 fractions. Without this they render as 0.8
    while every other surface in the app says 80%."""
    ui.line_chart(_weeks(["A"]), x="week", y="val", color="who", percent=True)
    assert ".0%" in _find(drawn["spec"]["encoding"]["y"], "format")
    assert ".0%" in _find(drawn["spec"]["encoding"]["tooltip"], "format")


def test_a_points_chart_is_not_forced_into_percentages(drawn):
    ui.line_chart(_weeks(["A"], value=14.2), x="week", y="val", color="who")
    assert ".0%" not in _find(drawn["spec"]["encoding"]["y"], "format")
    assert ".1f" in _find(drawn["spec"]["encoding"]["tooltip"], "format")


def test_the_axis_does_not_carry_a_decimal_the_tooltip_needs(drawn):
    """A points axis formatted .1f ticks "180.0, 160.0, 140.0" — a decimal place on every
    label that never varies. The tooltip still wants it; the axis does not."""
    ui.line_chart(_weeks(["A"], value=151.7), x="week", y="val", color="who")
    assert not _find(drawn["spec"]["encoding"]["y"], "format")
    assert ".1f" in _find(drawn["spec"]["encoding"]["tooltip"], "format")


# ---------------------------------------------------------------- crowding
def test_a_handful_of_series_keeps_its_legend(drawn):
    ui.line_chart(_weeks(list("ABC")), x="week", y="val", color="who")
    assert drawn["spec"]["encoding"]["color"].get("legend") is not None


def test_twelve_series_drop_the_legend_and_label_the_lines_instead(drawn):
    """A twelve-entry legend at labelLimit=140 truncated the very names it existed to
    disambiguate, and nobody matches twelve hues to twelve labels anyway."""
    teams = [f"Team {c}" for c in "ABCDEFGHIJKL"]
    ui.line_chart(_weeks(teams), x="week", y="val", color="who", highlight=teams[:2])
    spec = drawn["spec"]
    assert "layer" in spec, "crowded charts layer a grey backdrop under the picked lines"
    assert any(m.get("type") == "text" for m in _find(spec, "mark") if isinstance(m, dict)), \
        "each highlighted line is named at its last point"
    assert all(e["color"].get("legend") is None
               for e in _find(spec, "encoding") if isinstance(e, dict) and "color" in e)


def test_crowding_without_a_pick_still_draws_every_line(drawn):
    """Nothing highlighted means nothing to dim — it must not render an empty chart."""
    ui.line_chart(_weeks([f"T{i}" for i in range(12)]), x="week", y="val", color="who")
    assert "layer" not in drawn["spec"]


# ---------------------------------------------------------------- shared theme
@pytest.mark.parametrize("draw", [
    lambda: ui.line_chart(_weeks(["A"]), x="week", y="val", color="who"),
    lambda: ui.bar_compare(pd.DataFrame([{"p": "A", "k": "x", "v": 1.0}]), "p", "k", "v"),
])
def test_every_chart_uses_the_one_grid_colour(drawn, draw):
    """The two helpers had drifted to a warm #F0EEE9 and a cool #F0F3F8."""
    draw()
    assert drawn["spec"]["config"]["axis"]["gridColor"] == ui.GRID


def test_series_colours_come_from_the_app_palette(drawn):
    """Every multi-series chart used to fall through to Altair's defaults while the app
    had a palette it only ever applied to table shading."""
    ui.line_chart(_weeks(list("ABC")), x="week", y="val", color="who")
    assert drawn["spec"]["encoding"]["color"]["scale"]["range"] == ui.SERIES


def test_a_chart_can_carry_its_own_title_and_caption(drawn):
    """No chart set a Vega title, so a downloaded PNG arrived unlabelled."""
    ui.line_chart(_weeks(["A"]), x="week", y="val", color="who",
                  title="Points by week", caption="what this shows")
    assert drawn["spec"]["title"]["text"] == "Points by week"
    assert drawn["caption"] == "what this shows"


def test_height_is_a_knob_not_a_constant(drawn):
    ui.line_chart(_weeks(["A"]), x="week", y="val", color="who", height=420)
    assert drawn["spec"]["height"] == 420


# ---------------------------------------------------------------- route plot
def _pool(n=30):
    return pd.DataFrame([{"player": f"P{i}", "pos": "WR", "team": "MIN", "short": f"P{i}",
                          "routes_pg": 10.0 + i, "fd_rr": 0.05 + i / 400} for i in range(n)])


def test_the_route_plot_explains_its_own_rules(drawn):
    """The yellow threshold and the dashed median were explained only in a caption."""
    ui.route_plot(_pool(), _pool(3), threshold=0.12)
    labels = [t for t in _find(drawn["spec"], "text") if isinstance(t, dict)]
    joined = str(drawn["spec"])
    assert "league-winner line" in joined
    assert "league median routes" in joined


def test_the_route_plot_has_a_legend_for_the_highlighted_points(drawn):
    ui.route_plot(_pool(), _pool(3), threshold=0.12)
    legends = [l for l in _find(drawn["spec"], "legend") if isinstance(l, dict)]
    assert any(l.get("title") == "Yours" for l in legends)


def test_an_empty_pool_draws_nothing_rather_than_raising(drawn):
    ui.route_plot(pd.DataFrame(), pd.DataFrame())
    assert "spec" not in drawn


def test_empty_frames_are_safe_everywhere(drawn):
    ui.line_chart(pd.DataFrame(), x="week", y="val", color="who")
    ui.bar_compare(pd.DataFrame(), "p", "k", "v")
    assert "spec" not in drawn


def test_highlighting_everything_is_not_highlighting(drawn):
    """A picker that selects the DATA leaves no subset to highlight. Labelling all of it
    stacked the names on top of each other where the lines converge."""
    teams = [f"Team {c}" for c in "ABCDEFGHIJKL"]
    ui.line_chart(_weeks(teams), x="week", y="val", color="who", highlight=teams)
    assert "layer" not in drawn["spec"]
    assert drawn["spec"]["encoding"]["color"].get("legend") is not None


def test_too_many_picks_fall_back_to_a_legend(drawn):
    teams = [f"Team {c}" for c in "ABCDEFGHIJKL"]
    ui.line_chart(_weeks(teams), x="week", y="val", color="who", highlight=teams[:8])
    assert "layer" not in drawn["spec"]


def test_a_small_pick_still_gets_named_lines(drawn):
    teams = [f"Team {c}" for c in "ABCDEFGHIJKL"]
    ui.line_chart(_weeks(teams), x="week", y="val", color="who", highlight=teams[:3])
    assert "layer" in drawn["spec"]


def test_rule_labels_do_not_widen_the_scale(drawn):
    """Pinning a label to the panel edge with alt.value() quietly extends the shared x
    scale to reach that pixel, which squashed every real point into a sliver at x=0."""
    ui.route_plot(_pool(), _pool(3), threshold=0.12)
    xs = [v for v in _find(drawn["spec"], "value") if isinstance(v, (int, float))]
    assert not [v for v in xs if v > 1000], f"a literal pixel value leaked into an encoding: {xs}"
