"""Shared UI theming for the Streamlit dashboard — matches the Mega Bowl draft board."""
from __future__ import annotations

import pandas as pd
import streamlit as st

BG = "#F2F1EE"
SURFACE = "#FFFFFF"
SURFACE2 = "#ECEAE6"
BORDER = "#D8D5CF"
INK = "#1C1C1A"
INK2 = "#5A574F"
INK3 = "#9A9690"
GOLD = "#D4960A"
MINE_BG = "#FFFBEF"

POS_COLORS = {
    "QB": "#C2452D", "RB": "#2E6DB4", "WR": "#B87A10",
    "TE": "#6B3FA0", "K": "#5A7080", "DEF": "#3A7048", "W/R": "#8A867E",
}

_NEG = (46, 109, 180)     # blue  — buy-low / underperforming
_POS = (176, 122, 16)     # amber — sell-high / overperforming
_HEAT = (212, 150, 10)    # gold  — sequential highlight


def _rgb(t: tuple[int, int, int]) -> str:
    return f"rgb({t[0]},{t[1]},{t[2]})"


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Archivo+Narrow:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root { --gold:#D4960A; --ink:#1C1C1A; --ink3:#9A9690; --border:#D8D5CF; }
.stApp { background:#F2F1EE; }
html, body, [class*="st-"], .stMarkdown, .stDataFrame { font-family:'Archivo Narrow', system-ui, sans-serif; }
h1,h2,h3,h4,.mb-title { font-family:'Archivo', system-ui, sans-serif; letter-spacing:-.01em; }
.block-container { padding-top:1.6rem; padding-bottom:2rem; max-width:1400px; }
section[data-testid="stSidebar"] { background:#ECEAE6; border-right:1px solid var(--border); }
.mb-mast { border-bottom:2px solid var(--border); padding-bottom:10px; margin-bottom:14px; }
.mb-title { font-weight:800; font-size:26px; color:var(--ink); }
.mb-title em { font-style:normal; color:var(--gold); }
.mb-sub { font-size:12.5px; color:var(--ink3); margin-top:3px; }
.mb-kpi { background:#fff; border:1px solid var(--border); border-radius:7px; padding:10px 14px; height:100%; }
.mb-kpi-l { font-size:10.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--ink3); }
.mb-kpi-v { font-family:'Archivo',sans-serif; font-weight:800; font-size:22px; color:var(--ink); line-height:1.15; margin-top:3px; }
.mb-kpi-s { font-size:11.5px; color:#5A574F; margin-top:2px; }
.stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-weight:700; font-size:13px; color:var(--ink3); padding:6px 12px; }
.stTabs [aria-selected="true"] { color:var(--ink) !important; border-bottom:2px solid var(--gold) !important; }
.mb-legend { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
.mb-chip { font-size:10px; font-weight:700; color:#fff; padding:2px 6px; border-radius:3px; }
[data-testid="stDataFrame"] { font-size:12.5px; }
.stAlert { border-radius:6px; }
</style>
"""


def inject_css() -> None:
    # st.html renders raw <style>/<link>; st.markdown escapes them in recent Streamlit.
    try:
        st.html(_CSS)
    except Exception:
        st.markdown(_CSS, unsafe_allow_html=True)


_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def short_name(full: str) -> str:
    """Surname without generational suffix — 'Marvin Mims Jr.' -> 'Mims'."""
    toks = [t for t in str(full).split() if t.lower().strip(".") not in _SUFFIXES]
    return toks[-1] if toks else str(full)


def masthead(bits: list[str]) -> None:
    sub = " &nbsp;·&nbsp; ".join(b for b in bits if b)
    st.markdown(
        f'<div class="mb-mast"><div class="mb-title">Mega&nbsp;Bowl <em>Command Center</em></div>'
        f'<div class="mb-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """items: (label, big value, small sub)."""
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        col.markdown(
            f'<div class="mb-kpi"><div class="mb-kpi-l">{label}</div>'
            f'<div class="mb-kpi-v">{value}</div><div class="mb-kpi-s">{sub}</div></div>',
            unsafe_allow_html=True,
        )


def pos_legend() -> None:
    chips = "".join(
        f'<span class="mb-chip" style="background:{c}">{p}</span>'
        for p, c in POS_COLORS.items() if p != "W/R"
    )
    st.markdown(f'<div class="mb-legend">{chips}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- Styler helpers
def _pos_text_color(v):
    return f"color:{POS_COLORS.get(str(v).upper().split()[0] if v else '', INK2)}; font-weight:700"


def _diverging(s: pd.Series):
    m = float(pd.to_numeric(s, errors="coerce").abs().max() or 1)
    out = []
    for v in pd.to_numeric(s, errors="coerce"):
        if pd.isna(v):
            out.append("")
            continue
        t = min(abs(v) / m, 1.0)
        col = _mix((255, 255, 255), _POS if v > 0 else _NEG, 0.12 + 0.55 * t)
        out.append(f"background-color:{_rgb(col)}")
    return out


def _sequential(s: pd.Series):
    v = pd.to_numeric(s, errors="coerce")
    lo, hi = float(v.min()), float(v.max())
    rng = (hi - lo) or 1
    out = []
    for x in v:
        if pd.isna(x):
            out.append("")
            continue
        t = (x - lo) / rng
        out.append(f"background-color:{_rgb(_mix((255, 255, 255), _HEAT, 0.10 + 0.6 * t))}")
    return out


def style_df(
    df: pd.DataFrame,
    diverging: list[str] = (),
    sequential: list[str] = (),
    pos_cols: list[str] = (),
    fmt: dict | None = None,
):
    sty = df.style
    for c in diverging:
        if c in df.columns:
            sty = sty.apply(_diverging, subset=[c])
    for c in sequential:
        if c in df.columns:
            sty = sty.apply(_sequential, subset=[c])
    for c in pos_cols:
        if c in df.columns:
            sty = sty.map(_pos_text_color, subset=[c])
    if fmt:
        sty = sty.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="—")
    return sty
