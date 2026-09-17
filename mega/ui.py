"""Shared UI theming for the Streamlit dashboard — matches the Mega Bowl draft board."""
from __future__ import annotations

import pandas as pd
import streamlit as st

BG = "#FBFAF8"
SURFACE = "#FFFFFF"
SURFACE2 = "#F4F2EE"
BORDER = "#E5E2DB"
INK = "#1C1C1A"
INK2 = "#5A574F"
INK3 = "#9A9690"
GOLD = "#D4960A"
MINE_BG = "#FFFDF5"

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
:root { --gold:#D4960A; --ink:#1C1C1A; --ink2:#5A574F; --ink3:#8A867E; --border:#E7E4DD; --surface2:#FAF9F7; }
.stApp { background:#FFFFFF; }
html, body, [class*="st-"], .stMarkdown, .stDataFrame { font-family:'Archivo Narrow', system-ui, sans-serif; }
/* the rule above also catches Streamlit's icon spans, which turns their ligatures into raw text */
[data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded' !important; }
h1,h2,h3,h4,.mb-title { font-family:'Archivo', system-ui, sans-serif; letter-spacing:-.01em; }
/* Streamlit's floating toolbar sits ~2.9rem tall; less top padding clips the masthead. */
.block-container { padding-top:3.6rem; padding-bottom:3rem; max-width:1400px; }
section[data-testid="stSidebar"] { background:var(--surface2); border-right:1px solid var(--border); }
.mb-mast { border-bottom:2px solid var(--border); padding-bottom:10px; margin-bottom:14px; }
.mb-title { font-weight:800; font-size:26px; color:var(--ink); line-height:1.25; }
.mb-title em { font-style:normal; color:var(--gold); }
.mb-sub { font-size:12.5px; color:var(--ink3); margin-top:3px; }
.mb-kpi { background:#fff; border:1px solid var(--border); border-radius:8px; padding:11px 14px; height:100%; }
.mb-kpi-l { font-size:10.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--ink3); }
.mb-kpi-v { font-family:'Archivo',sans-serif; font-weight:800; font-size:22px; color:var(--ink); line-height:1.15; margin-top:3px; }
.mb-kpi-s { font-size:11.5px; color:var(--ink2); margin-top:2px; }
.stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--border); overflow-x:auto; scrollbar-width:thin; }
.stTabs [data-baseweb="tab"] { font-weight:700; font-size:13px; color:var(--ink3); padding:6px 12px; white-space:nowrap; }
.stTabs [aria-selected="true"] { color:var(--ink) !important; border-bottom:2px solid var(--gold) !important; }
.mb-legend { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
.mb-chip { font-size:10px; font-weight:700; color:#fff; padding:2px 6px; border-radius:3px; }
.mb-key { display:flex; flex-wrap:wrap; gap:5px 14px; margin:7px 0 2px; font-size:11px; color:var(--ink2); }
.mb-key b { font-family:'Archivo',sans-serif; font-weight:700; color:var(--ink); letter-spacing:.03em; }
[data-testid="stDataFrame"] { font-size:12.5px; }
.stAlert { border-radius:7px; }
.mb-lede { font-size:13.5px; color:var(--ink2); margin:2px 0 10px; line-height:1.5; }
.mb-lede b { color:var(--ink); }

/* ---- phone ---------------------------------------------------------------
   Streamlit columns are flex children that never wrap on their own, so a KPI
   row squeezes to unreadable slivers on a narrow screen. Let them wrap. */
@media (max-width: 640px) {
  .block-container { padding-top:3.2rem; padding-left:.75rem; padding-right:.75rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:8px; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:calc(50% - 8px); flex:1 1 calc(50% - 8px); }
  .mb-title { font-size:20px; }
  .mb-sub { font-size:11.5px; }
  .mb-kpi-v { font-size:19px; }
  .stTabs [data-baseweb="tab"] { font-size:12px; padding:6px 9px; }
  [data-testid="stDataFrame"] { font-size:11.5px; }
  .mb-key { font-size:10.5px; gap:4px 10px; }
}
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


# Canonical short column labels. Keep every table speaking the same language:
# uppercase, no spaces, no slashes unless the unit demands one.
COLS = {
    "slot": "SLOT", "player": "PLAYER", "pos": "POS", "team": "TM", "nfl_team": "TM",
    "games": "G", "half_ppr_pg": "PPG", "last_wk": "LAST", "pg_recent": "PPG",
    "xfp_tot": "xFP", "xfp_diff": "xFP±",
    "tgt_pg": "TGT", "carry_pg": "CAR", "tgt_pct": "TGT%", "tm_rank": "TM#",
    "report_status": "ST", "opp": "OPP", "implied": "IMP",
    "value": "VAL", "add_rank": "ADD#", "trend_30d": "TR30", "add_score": "SCORE", "why": "WHY",
    "ease_rank": "MU#", "pa_pg": "PA/G", "proj_adj": "PROJ*", "proj_source": "SRC",
    "start_sit": "GRADE", "close_call": "NOTE", "lineup": "SLOT", "matchup": "MU",
    # archetypes
    "arch_fit": "FIT", "tags": "TAGS", "tgt_share": "TGT%", "age": "AGE", "carries_pg": "CAR",
    "exp_yrs": "EXP", "proj_ppg": "PROJ", "gms": "G", "vor": "VOR",
    # WOPR
    "owner": "OWNER", "team_2026_nfl": "TM", "wopr_anchored": "WOPR",
    "wopr_posrank": "WOPR#", "board_posrank": "DRAFT#", "rank_delta": "GAP",
    "ppg_minus_xppg": "xPPG±", "nfl_status": "ST",
    # league (Yahoo API)
    "rank": "RANK", "wins": "W", "losses": "L", "ties": "T",
    "points_for": "PF", "points_against": "PA", "streak": "STRK",
    "faab_balance": "FAAB", "moves": "MOV", "trades": "TRD",
    "week": "WK", "opponent": "OPP", "points": "PTS", "opp_points": "OPP PTS",
    "result": "RES", "when": "WHEN", "tx_type": "TYPE", "move": "MOVE",
    # trades
    "partner": "MANAGER", "give": "YOU GIVE", "give_val": "GIVE VAL",
    "get": "YOU GET", "get_val": "GET VAL", "fairness": "FAIR",
    "edge": "EDGE", "addresses": "FILLS", "they_need": "THEY NEED",
    # power rankings
    "power_rank": "PWR", "starters_pg": "LINEUP", "bench_pg": "BENCH",
    "matched": "MATCHED", "luck": "LUCK",
}

# What each abbreviation means, for the legend under a table.
GLOSS = {
    "G": "games played", "PPG": "half-PPR points per game", "LAST": "last week's points",
    "xFP": "expected fantasy points", "xFP±": "actual minus expected",
    "TGT": "targets per game", "CAR": "carries per game",
    "TGT%": "share of his NFL team's targets", "TM#": "target rank on his NFL team (1 = alpha)",
    "ST": "injury status", "OPP": "next opponent", "IMP": "Vegas implied team total",
    "VAL": "FantasyCalc trade value", "ADD#": "industry add rank", "TR30": "30-day value trend",
    "SCORE": "blended add score", "MU#": "matchup rank (1 = easiest of 32)",
    "PA/G": "points allowed per game", "PROJ*": "matchup-adjusted projection",
    "SRC": "projection source", "GRADE": "FantasyPros start/sit grade",
    "FIT": "archetype fit, 0–100", "TAGS": "blueprint traits he hits",
    "AGE": "age", "EXP": "seasons of NFL experience", "VOR": "draft-board value over replacement",
    "WOPR": "weighted opportunity rating (1.5·target share + 0.7·air-yards share)",
    "WOPR#": "his WOPR rank at the position", "DRAFT#": "where the board drafted him",
    "GAP": "draft rank minus opportunity rank — positive = drafted below his role",
    "xPPG±": "points above / below what his opportunity predicts",
    "OWNER": "fantasy manager who holds him",
    "PF": "points scored all season", "PA": "points scored against him",
    "STRK": "current win / loss streak", "FAAB": "free-agent budget left",
    "MOV": "roster moves made", "TRD": "trades made",
    "PTS": "what he scored that week", "OPP PTS": "what his opponent scored",
    "RES": "win / loss / tie",
    "MANAGER": "who you'd be trading with", "YOU GIVE": "the player you send",
    "YOU GET": "the player you receive",
    "GIVE VAL": "FantasyCalc value of the player you send",
    "GET VAL": "FantasyCalc value of the player you receive",
    "FAIR": "value parity — 1.0 is an even swap", "EDGE": "value you gain on the deal",
    "FILLS": "the hole on your roster this closes",
    "THEY NEED": "positions where their starters are below league average",
    "PWR": "rank by roster strength, ignoring record",
    "LINEUP": "points per game from his best legal starting lineup",
    "BENCH": "points per game from his three best bench players",
    "MATCHED": "roster players with stats to score — lower means a rougher estimate",
    "LUCK": "places the record sits above the roster — positive means they're overachieving",
}


def cols(df: pd.DataFrame, **extra: str) -> pd.DataFrame:
    """Rename to the canonical short labels; `extra` overrides for dynamic names."""
    return df.rename(columns={**COLS, **extra})


def col_config(df: pd.DataFrame, **custom: str) -> dict:
    """Hover tooltips on column headers, drawn from GLOSS.

    Phones have no hover, so `col_key` stays the readable fallback underneath.
    """
    gloss = {**GLOSS, **custom}
    return {c: st.column_config.Column(help=gloss[c]) for c in df.columns if c in gloss}


def lede(text: str) -> None:
    """One plain-English line saying what a section is for and how to act on it."""
    st.markdown(f'<div class="mb-lede">{text}</div>', unsafe_allow_html=True)


def line_chart(df: pd.DataFrame, x: str, y: str, color: str, y_title: str = "") -> None:
    """Long-form line chart with the dashboard's type and grid, not Streamlit's default."""
    import altair as alt

    ch = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=34, filled=True))
        .encode(
            x=alt.X(f"{x}:O", axis=alt.Axis(title=None, labelAngle=0, grid=False)),
            y=alt.Y(f"{y}:Q", axis=alt.Axis(title=y_title or None, grid=True)),
            color=alt.Color(f"{color}:N", legend=alt.Legend(title=None, orient="bottom",
                                                            columns=4, labelLimit=140)),
            tooltip=[color, x, alt.Tooltip(f"{y}:Q", format=".1f")],
        )
        .properties(height=290)
        .configure_view(strokeWidth=0)
        .configure_axis(labelFont="Archivo Narrow", titleFont="Archivo Narrow",
                        labelColor=INK2, titleColor=INK3, domainColor=BORDER,
                        tickColor=BORDER, gridColor="#F0EEE9", labelFontSize=11)
        .configure_legend(labelFont="Archivo Narrow", labelColor=INK2, labelFontSize=11,
                          symbolStrokeWidth=2.5)
    )
    st.altair_chart(ch, use_container_width=True)


def table(
    df: pd.DataFrame,
    rename: dict | None = None,
    diverging: list[str] = (),
    sequential: list[str] = (),
    pos_cols: list[str] = (),
    fmt: dict | None = None,
    help: dict | None = None,
    height: int | None = None,
) -> pd.DataFrame:
    """Rename to short labels, style, attach header tooltips, render. Returns the
    renamed frame so callers can pass its columns to `col_key`."""
    d = cols(df, **(rename or {}))
    st.dataframe(
        style_df(d, diverging=diverging, sequential=sequential, pos_cols=pos_cols, fmt=fmt),
        column_config=col_config(d, **(help or {})),
        width="stretch", hide_index=True,
        **({"height": height} if height else {}),
    )
    return d


def col_key(*names: str, **custom: str) -> None:
    """Compact 'ABBR meaning' legend under a table."""
    parts = [(n, GLOSS[n]) for n in names if n in GLOSS] + list(custom.items())
    st.markdown(
        '<div class="mb-key">'
        + "".join(f"<span><b>{k}</b> {v}</span>" for k, v in parts)
        + "</div>",
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


NA = "—"


def style_df(
    df: pd.DataFrame,
    diverging: list[str] = (),
    sequential: list[str] = (),
    pos_cols: list[str] = (),
    fmt: dict | None = None,
):
    # Streamlit 1.63 + pandas 3 drops Styler.na_rep and paints a literal "None", so
    # any column holding a blank is formatted to text here. Columns with no blanks stay
    # numeric so click-to-sort keeps working; the heat maps read the original numbers.
    fmt = {k: v for k, v in (fmt or {}).items() if k in df.columns}
    disp = df.copy()
    # A repeated label makes disp[c] a DataFrame, so .isna().any() below raises. Skip
    # those rather than crash the tab — the caller should give them distinct names.
    dupes = {c for c in disp.columns if list(disp.columns).count(c) > 1}
    for c in disp.columns:
        if c in dupes or not disp[c].isna().any():
            continue
        spec = fmt.pop(c, None)
        if spec:
            disp[c] = [NA if pd.isna(v) else spec.format(v)
                       for v in pd.to_numeric(disp[c], errors="coerce")]
        else:
            disp[c] = disp[c].astype(object).where(disp[c].notna(), NA)

    sty = disp.style
    for c in diverging:
        if c in df.columns:
            sty = sty.apply(lambda _s, c=c: _diverging(df[c]), subset=[c])
    for c in sequential:
        if c in df.columns:
            sty = sty.apply(lambda _s, c=c: _sequential(df[c]), subset=[c])
    for c in pos_cols:
        if c in df.columns:
            sty = sty.map(_pos_text_color, subset=[c])
    if fmt:
        sty = sty.format(fmt)
    return sty
