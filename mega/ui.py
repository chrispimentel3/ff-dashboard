"""Shared UI theming for the Streamlit dashboard — matches the Mega Bowl draft board."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

# NFL shield palette: navy #013369 and red #D50A0A on white.
NAVY = "#013369"
RED = "#D50A0A"

# Two families, one job each: Archivo for names, headings and the figures that lead;
# Archivo Narrow for labels and data. Named here so the CSS, the charts and the Streamlit
# theme in .streamlit/config.toml all say the same thing.
FONT_HEAD = "Archivo"
FONT_BODY = "Archivo Narrow"

BG = "#FFFFFF"
SURFACE = "#FFFFFF"
SURFACE2 = "#F5F7FA"
BORDER = "#DDE3EC"
INK = "#12161C"
INK2 = "#4E5866"
INK3 = "#8A93A0"
GOLD = NAVY          # the accent name the rest of the module already reads
# Broadcast yellow, used for one thing only: the first-down line and clearing it.
FIRST_DOWN = "#FFC40C"
MINE_BG = "#F2F6FC"

# Drawn from real team palettes rather than only navy/red — the position column has to
# stay readable at a glance, and a two-colour set can't do that across six positions.
POS_COLORS = {
    "QB": "#D50A0A", "RB": "#013369", "WR": "#B8860B",
    "TE": "#4B2E83", "K": "#5A7080", "DEF": "#1F5C34", "W/R": "#8A93A0",
}

_NEG = (1, 51, 105)       # navy — buy-low / underperforming
_POS = (213, 10, 10)      # red  — sell-high / overperforming
_HEAT = (1, 51, 105)      # navy — sequential highlight


def _rgb(t: tuple[int, int, int]) -> str:
    return f"rgb({t[0]},{t[1]},{t[2]})"


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


_CSS = """
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Archivo+Narrow:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root { --navy:#013369; --red:#D50A0A; --gold:#013369; --ink:#12161C; --ink2:#4E5866; --ink3:#8A93A0; --border:#DDE3EC; --surface2:#F5F7FA;
  /* One radius scale. Small for chips and pills, medium for every panel, card and alert. */
  --r-sm:4px; --r-md:8px; }
.stApp { background:#FFFFFF; }
html, body, [class*="st-"], .stMarkdown, .stDataFrame { font-family:'Archivo Narrow', system-ui, sans-serif; }
/* the rule above also catches Streamlit's icon spans, which turns their ligatures into raw text */
[data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded' !important; }
h1,h2,h3,h4,.mb-title { font-family:'Archivo', system-ui, sans-serif; letter-spacing:-.01em; }
/* Streamlit's floating toolbar sits ~2.9rem tall; less top padding clips the masthead. */
.block-container { padding-top:3.6rem; padding-bottom:3rem; max-width:1400px; }
section[data-testid="stSidebar"] { background:var(--surface2); border-right:1px solid var(--border); }
.mb-mast { border-bottom:3px solid var(--navy); padding-bottom:10px; margin-bottom:14px; }
.mb-title { font-weight:800; font-size:26px; color:var(--navy); line-height:1.25; }
.mb-title em { font-style:normal; color:var(--red); }
.mb-sub { font-size:12.5px; color:var(--ink3); margin-top:3px; }
.mb-kpi { background:#fff; border:1px solid var(--border); border-top:3px solid var(--navy); border-radius:var(--r-md); padding:12px 14px; height:118px; overflow:hidden; }
.mb-kpi-l { font-size:11.5px; font-weight:700; letter-spacing:.01em; color:var(--ink3); }
/* The cards are a fixed height rather than a stretched one. Streamlit nests each card six
   divs deep with no definite height anywhere in the chain, so `height:100%` resolves to auto
   and the tallest card overflows its row; forcing the row to grid instead collapsed the
   columns to a character wide. A fixed height is the honest fix: the label is one line, the
   note is two, and the row is uniform whatever the text says. */
.mb-kpi-l { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.mb-kpi-s { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
/* Section titles: one size, one weight, one rhythm — see ui.h() */
.mb-h { font-family:'Archivo', system-ui, sans-serif; font-weight:700; color:var(--ink);
  letter-spacing:-.005em; margin:22px 0 8px; }
.mb-h4 { font-size:17px; }
.mb-h3 { font-size:19px; }
.mb-kpi-v { font-family:'Archivo',sans-serif; font-weight:800; font-size:22px; color:var(--navy); line-height:1.15; margin-top:3px; }
.mb-kpi-s { font-size:11.5px; color:var(--ink2); margin-top:2px; }
.stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--border); overflow-x:auto; scrollbar-width:thin; }
.stTabs [data-baseweb="tab"] { font-weight:700; font-size:13px; color:var(--ink3); padding:6px 12px; white-space:nowrap; }
.stTabs [aria-selected="true"] { color:var(--navy) !important; border-bottom:3px solid var(--red) !important; }
.mb-legend { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
.mb-chip { font-size:10px; font-weight:700; color:#fff; padding:2px 6px; border-radius:var(--r-sm); }
/* column key: label | what it means, one row per column */
.mb-key { display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; margin:2px 0; font-size:13px; line-height:1.45; }
.mb-key dt { font-family:'Archivo',sans-serif; font-weight:700; color:var(--navy); white-space:nowrap; }
[data-testid="stExpander"] { border-radius:var(--r-md); }
.mb-key dd { margin:0; color:var(--ink2); }
[data-testid="stExpander"] details summary p { font-size:12.5px; color:var(--ink3); }
[data-testid="stExpander"] { border-color:var(--border); margin:-4px 0 10px; }
[data-testid="stDataFrame"] { font-size:12.5px; }
.stAlert { border-radius:var(--r-md); }
.mb-lede { font-size:13.5px; color:var(--ink2); margin:2px 0 10px; line-height:1.5; }
.mb-lede b { color:var(--ink); }
/* player lookup card */
.mb-card { padding-top:4px; }
.mb-card-name { font-family:'Archivo',sans-serif; font-weight:800; font-size:26px; color:var(--navy); line-height:1.15; }
.mb-card-team { display:flex; align-items:center; gap:6px; font-size:15px; color:var(--ink); margin:4px 0 2px; }
.mb-card-facts { font-size:13.5px; color:var(--ink2); margin:0 0 10px; }
.mb-card-logo { height:26px; width:auto; }
.mb-card-row { font-size:14px; color:var(--ink); margin:3px 0; }
.mb-card-row span:first-child { display:inline-block; min-width:104px; font-size:12px; font-weight:700;
  letter-spacing:.01em; color:var(--ink3); }
.mb-flag { background:var(--red); color:#fff; font-weight:700; font-size:11.5px; padding:2px 7px;
  border-radius:var(--r-sm); margin-right:6px; }

/* ---- phone ---------------------------------------------------------------
   Streamlit columns are flex children that never wrap on their own, so a KPI
   row squeezes to unreadable slivers on a narrow screen. Let them wrap. */
@media (max-width: 640px) {
  .block-container { padding-top:3.2rem; padding-left:.75rem; padding-right:.75rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:8px; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:calc(50% - 8px); flex:1 1 calc(50% - 8px); }
  .mb-kpi { height:126px; }
  .mb-title { font-size:20px; }
  .mb-sub { font-size:11.5px; }
  .mb-kpi-v { font-size:19px; }
  .mb-h4 { font-size:15.5px; } .mb-h3 { font-size:17px; }
  .stTabs [data-baseweb="tab"] { font-size:12px; padding:6px 9px; }
  [data-testid="stDataFrame"] { font-size:11.5px; }
  .mb-key { grid-template-columns:1fr; gap:1px; font-size:12.5px; }
  .mb-key dd { margin-bottom:7px; }
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
            f'<div class="mb-kpi"><div class="mb-kpi-l">{title_case(label)}</div>'
            f'<div class="mb-kpi-v">{value}</div><div class="mb-kpi-s">{sub}</div></div>',
            unsafe_allow_html=True,
        )


# Every table goes through `table()`, and every column has three layers:
#
#   COLS    raw column -> short internal code ("tgt_pct" -> "TGT%"). Call sites style and
#           format by code, so codes never change once a table uses them.
#   LABELS  code -> the header people read ("TGT%" -> "Target share").
#   GLOSS   code -> what the number says and what to do about it (header tooltip, and the
#           "What these columns mean" panel under each table — phones have no hover).
#
# The same idea gets the same name on every tab. Before this, points-vs-usage was
# xFP±, xFP±/G, diff_pg and xPPG± depending on where you looked.
COLS = {
    "slot": "SLOT", "player": "PLAYER", "pos": "POS", "team": "TM", "nfl_team": "TM",
    "games": "G", "gms": "G", "half_ppr_pg": "PPG", "last_wk": "LAST", "pg_recent": "PPG",
    "xfp_tot": "xFP", "expected": "xFP", "actual": "ACT", "xfp_diff": "xFP±", "diff": "xFP±",
    "per_g": "xFP±/G", "diff_pg": "xFP±/G", "signal": "SIGNAL",
    "tgt_pg": "TGT", "carry_pg": "CAR", "tgt_pct": "TGT%", "tm_rank": "TM#",
    # receiving routes (estimated — see mega/routes.py)
    "routes": "RTE", "routes_pg": "RTE/G", "tprr": "TPRR",
    "fd_rr": "1D/RR", "fd": "1D", "receiving_first_downs": "1D", "route_flag": "FLAG",
    "report_status": "ST", "opp": "OPP", "implied": "IMP",
    "value": "VAL", "add_rank": "ADD#", "trend_30d": "TR30", "add_score": "SCORE", "why": "WHY",
    # waivers priced against your own lineup (mega/needs.py) and FAAB (mega/faab.py)
    "gain": "GAIN", "fit": "FIT", "bid": "BID", "max_bid": "MAX", "drop": "CUT",
    "upside": "UPSIDE", "starts": "STARTS", "vacated": "VAC", "season_pts": "SEASON",
    "faab_left": "FAAB", "faab_spent": "SPENT", "headroom": "ROOM",
    "worst_starter": "WEAKEST", "replacement": "WIRE", "rostered": "HAVE", "starting": "START",
    "ease_rank": "MU#", "pa_pg": "PA/G", "proj_adj": "PROJ*", "proj": "PROJ", "proj_source": "SRC",
    "start_sit": "GRADE", "close_call": "NOTE", "lineup": "SLOT", "matchup": "MU",
    # draft value
    "drafted_by": "DRAFTED BY", "round": "RD", "value_delta": "VAL±",
    # archetypes
    "arch_fit": "FIT", "tags": "TAGS", "tgt_share": "TGT%", "age": "AGE", "carries_pg": "CAR",
    "exp_yrs": "EXP", "proj_ppg": "PROJ", "vor": "VOR",
    # WOPR
    "owner": "OWNER", "team_2026_nfl": "TM", "wopr_anchored": "WOPR", "name": "PLAYER",
    "wopr_posrank": "WOPR#", "board_posrank": "DRAFT#", "rank_delta": "GAP",
    "ppg_minus_xppg": "xPPG±", "nfl_status": "ST",
    # league
    "rank": "RANK", "wins": "W", "losses": "L", "ties": "T", "manager": "MGR",
    "points_for": "PF", "points_against": "PA", "streak": "STRK",
    "faab_balance": "FAAB", "moves": "MOV", "trades": "TRD",
    "week": "WK", "opponent": "OPP", "points": "PTS", "opp_points": "OPP PTS",
    "result": "RES", "when": "WHEN", "tx_type": "TYPE", "move": "MOVE",
    # trades
    "partner": "MANAGER", "give": "YOU GIVE", "give_val": "GIVE VAL",
    "get": "YOU GET", "get_val": "GET VAL", "fairness": "FAIR",
    "edge": "EDGE", "addresses": "FILLS", "they_need": "THEY NEED",
    "why_them": "THEY WANT", "costs_you": "COSTS YOU", "get_pos": "POS", "give_pos": "POS",
    "shape": "SHAPE", "d_me": "YOU ±", "d_them": "THEM ±", "mkt_ratio": "MARKET", "flag": "ODDS",
    # power rankings
    "power_rank": "PWR", "starters_pg": "LINEUP", "bench_pg": "BENCH",
    "xwins": "xW", "power": "xW%", "luck_w": "LUCK W", "ppg": "PPG", "cv": "SWING",
    "matched": "MATCHED", "luck": "LUCK",
    # game environment
    "players": "PLAYERS", "total": "TOT", "spread": "SPRD", "implied_pts": "IMP",
    "verdict": "VERDICT", "status": "OWN",
    # game log (player lookup) — raw per-game counts, distinct from the per-game rates above
    "game": "GAME", "snap_pct": "SNAP%", "half_ppr": "PTS", "xfp": "xFP", "vs_exp": "xFP±",
    "completions": "CMP", "attempts": "ATT", "passing_yards": "PASSYD", "passing_tds": "PASSTD",
    "passing_interceptions": "INT", "sacks_suffered": "SK", "carries": "CARRIES",
    "rushing_yards": "RUSHYD", "rushing_tds": "RUSHTD", "targets": "TARGETS", "receptions": "REC",
    "receiving_yards": "RECYD", "receiving_tds": "RECTD", "receiving_air_yards": "AIRYD",
    "receiving_yards_after_catch": "YAC",
    "stat": "STAT", "value_fmt": "VALUE", "pos_rank": "POSRANK", "means": "MEANS",
}

LABELS = {
    "SLOT": "Slot", "PLAYER": "Player", "POS": "Pos", "TM": "NFL", "LOGO": "NFL",
    "G": "Games", "PPG": "Pts/game", "LAST": "Last game",
    "xFP": "Expected pts", "ACT": "Actual pts", "xFP±": "Vs expected", "xFP±/G": "Vs expected/g",
    "SIGNAL": "Signal", "TGT": "Targets/g", "CAR": "Carries/g",
    "TGT%": "Target share", "TM#": "Team tgt rank",
    "RTE": "Routes (est.)", "RTE/G": "Routes/g (est.)",
    "TPRR": "Targets/route (est.)", "1D/RR": "1st downs/route (est.)", "1D": "1st downs",
    "FLAG": "Flag",
    "ST": "Status", "OPP": "Next opp", "IMP": "Vegas pts",
    "VAL": "Trade value", "ADD#": "Add rank", "TR30": "30-day trend", "SCORE": "Claim score",
    "WHY": "Why",
    "GAIN": "Adds pts/wk", "FIT": "Fit", "BID": "Bid", "MAX": "Walk-away",
    "CUT": "You'd drop", "UPSIDE": "Upside", "STARTS": "Starts now", "VAC": "Vacated tgt/g",
    "SEASON": "Season pts", "FAAB": "FAAB left", "SPENT": "FAAB spent", "ROOM": "Headroom",
    "WEAKEST": "Weakest starter", "WIRE": "Wire level", "HAVE": "Rostered", "START": "Starting",
    "MU#": "Matchup rank", "PA/G": "Pts allowed/g", "PROJ*": "Projection", "PROJ": "Raw proj",
    "SRC": "Source", "GRADE": "FP grade", "NOTE": "Close call", "MU": "Opponent",
    "DRAFTED BY": "Drafted by", "RD": "Round", "VAL±": "Value vs slot",
    "FIT": "Blueprint fit", "TAGS": "Traits", "AGE": "Age", "EXP": "NFL yrs", "VOR": "Value over repl.",
    "OWNER": "Owner", "WOPR": "WOPR", "WOPR#": "WOPR rank", "DRAFT#": "Draft rank",
    "GAP": "Role vs price", "xPPG±": "Vs role/g",
    "RANK": "Standing", "W": "W", "L": "L", "T": "T", "TEAM": "Team", "MGR": "Manager",
    "PF": "Pts for", "PA": "Pts against", "STRK": "Streak", "FAAB": "FAAB left",
    "MOV": "Moves", "TRD": "Trades",
    "WK": "Week", "PTS": "Pts", "OPP PTS": "Opp pts", "RES": "Result",
    "ROLE": "Role", "GIVE ROLE": "Their role", "GET ROLE": "Their role",
    "WHEN": "When", "TYPE": "Type", "MOVE": "Move",
    "MANAGER": "Manager", "YOU GIVE": "You give", "GIVE VAL": "Give value",
    "YOU GET": "You get", "GET VAL": "Get value", "FAIR": "Fairness", "EDGE": "Value gained",
    "FILLS": "Fixes", "THEY NEED": "They need",
    "THEY WANT": "Why they'd say yes", "COSTS YOU": "What it costs you",
    "SHAPE": "Shape", "YOU ±": "Your lineup ±", "THEM ±": "Their lineup ±",
    "MARKET": "Market ratio", "ODDS": "Likelihood", "GIVE LOGO": "", "GET LOGO": "",
    "PWR": "Power rank", "LINEUP": "Lineup pts/g", "BENCH": "Bench pts/g",
    "xW": "Expected wins", "xW%": "Expected win %", "LUCK W": "Luck (wins)", "SWING": "Week-to-week swing",
    "MATCHED": "Players scored", "LUCK": "Luck",
    "PLAYERS": "Your players", "TOT": "Game total", "SPRD": "Spread", "VERDICT": "Verdict",
    "OWN": "Status",
    "GAME": "Opponent", "SNAP%": "Snap %", "CMP": "Cmp", "ATT": "Att", "PASSYD": "Pass yds",
    "PASSTD": "Pass TD", "INT": "INT", "SK": "Sacked", "CARRIES": "Carries", "RUSHYD": "Rush yds",
    "RUSHTD": "Rush TD", "TARGETS": "Targets", "REC": "Rec", "RECYD": "Rec yds", "RECTD": "Rec TD",
    "AIRYD": "Air yds", "YAC": "YAC",
    "STAT": "Stat", "VALUE": "Value", "POSRANK": "Rank at position", "MEANS": "What it tells you",
}

GLOSS = {
    "SLOT": "Where he sits in your Yahoo lineup. BN = bench.",
    "TM": "His NFL team.",
    "G": "Games he's played this season.",
    "PPG": "Half-PPR fantasy points per game.",
    "LAST": "Fantasy points in his most recent game.",
    "xFP": "Points his usage should have produced — targets, carries, depth and red-zone looks, scored half-PPR.",
    "ACT": "Fantasy points he actually scored.",
    "xFP±": "Actual minus expected, season total. Plus = scoring above his usage (likely to cool off). "
            "Minus = below it (the points should come).",
    "xFP±/G": "Actual minus expected, per game. +2 or more: sell-high candidate. "
              "−1.5 or less: hold or buy — the work is there, the points will follow.",
    "SIGNAL": "BUY LOW = scoring well under his usage. SELL HIGH = well over it.",
    "TGT": "Targets per game.",
    "CAR": "Carries per game.",
    "TGT%": "His share of his NFL team's targets. 25%+ is a No. 1 receiver's role; under 15% is a part-timer.",
    "TM#": "Where he ranks in targets on his own NFL team. #1 = the go-to option.",
    "RTE": "Pass routes run. Estimated — his snap share × his team's dropbacks, because nflverse "
           "has no charted route count. It runs high for anyone who blocks or sits on passing downs.",
    "RTE/G": "Estimated pass routes run per game. Under ~25 is a part-time role.",
    "TPRR": "Targets per route run — how often he's thrown to when he's actually in a route. "
            "Around 20%+ is a featured receiver. It tells a real role apart from empty snaps, "
            "which raw target share can't.",
    "1D/RR": "First downs per route run — the single best read on a receiver. 12%+ is the "
             "league-winner line for a WR. Tight ends read lower: the route estimate counts "
             "their blocking snaps, so compare them to other TEs, not to WRs.",
    "1D": "Catches that moved the chains.",
    "FLAG": "\"League-winner 1D/RR\" when a WR clears 12% on enough routes to believe it. "
            "\"Under 50 routes\" means the sample is too small to read the rates at all.",
    "ST": "Out, doubtful, questionable or IR — from the NFL injury report, or from Chris's own list (data/player_status.csv), which wins when the report hasn't caught up.",
    "OPP": "Next opponent (@ = away game).",
    "IMP": "Vegas's expected points for his offense next game. Higher = more scoring to go around.",
    "VAL": "FantasyCalc trade value — what the trade market says he's worth.",
    "ADD#": "Rank among the most-added players across Sleeper leagues. Lower = hotter pickup.",
    "TR30": "Change in trade value over the last 30 days. Plus = rising.",
    "SCORE": "Blend of recent points, usage, trade value and add rate. Higher = better claim.",
    "WHY": "The main reasons behind this row, in plain words.",
    "MU#": "How easy next week's defense is for his position. #1 = easiest of 32, #32 = toughest.",
    "PA/G": "Fantasy points that defense allows per game to this position.",
    "PROJ*": "This week's projection after the matchup adjustment — the number the lineup is built on.",
    "PROJ": "Projection before the matchup adjustment.",
    "SRC": "Where the projection comes from. FantasyPros = expert projection. "
           "nflverse-est = modelled from his usage, so treat close calls as coin flips.",
    "GRADE": "FantasyPros start/sit grade.",
    "NOTE": "Flagged when a bench player projects within ~2 pts of a starter — worth a closer look.",
    "MU": "Next opponent (@ = away game).",
    "DRAFTED BY": "Manager who drafted him.",
    "RD": "Draft round.",
    "VAL±": "Trade value now minus what his draft slot should be worth. Plus = he's beaten his draft cost.",
    "FIT": "How closely he matches the league-winner blueprint, 0–100.",
    "TAGS": "Blueprint traits he hits.",
    "EXP": "Seasons in the NFL.",
    "VOR": "Draft-board value over a replacement-level player.",
    "OWNER": "Fantasy manager who has him.",
    "WOPR": "Weighted opportunity rating: 1.5 × target share + 0.7 × air-yards share. "
            "About 0.5+ is a starting-calibre WR role.",
    "WOPR#": "His WOPR rank at his position.",
    "DRAFT#": "Where the draft board ranked him at his position.",
    "GAP": "Draft rank minus opportunity rank. Plus = his role is bigger than his price.",
    "ROLE": "His job on his own offence over his last 3 games (WR1-WR4+, TE1-REC/BLK, RB "
            "LEAD/COMMITTEE/RECEIVING/BACKUP, QB STARTER). Flags after it: ROLE+ = producing "
            "like the rung above him; a + on a flag means it held across the window, not once.",
    "xPPG±": "Points per game above or below what his WOPR predicts. Plus = running hot; minus = due.",
    "RANK": "Place in the league standings.",
    "PF": "Points scored all season.", "PA": "Points scored against him all season.",
    "STRK": "Current win or loss streak.", "FAAB": "Free-agent budget left.",
    "MOV": "Roster moves made.", "TRD": "Trades made.",
    "PTS": "Points scored that week.", "OPP PTS": "Opponent's points that week.",
    "RES": "Win, loss or tie.",
    "MANAGER": "Who you'd be trading with.",
    "GIVE VAL": "FantasyCalc value of the player you send.",
    "GET VAL": "FantasyCalc value of the player you receive.",
    "FAIR": "How even the swap is by trade value. 1.00 = dead even; offers below 0.85 are filtered out.",
    "EDGE": "Trade value you gain. Plus = the deal favours you.",
    "FILLS": "The weak spot on your roster this trade fixes.",
    "THEY NEED": "Their weakest positions — lead with these when you pitch it.",
    "THEY WANT": "The reason this manager takes the call: a hole at that position, or simply "
                 "that the values match.",
    "COSTS YOU": "Flagged when the player you'd send is one you can't spare — giving him up "
                 "opens a hole where you were already below league average.",
    "SHAPE": "How many players each side sends.",
    "YOU ±": "Points per week your best legal lineup gains, after both rosters are rebuilt, "
             "cut back to size and re-optimised. Bench depth counts, at a discount.",
    "THEM ±": "The same number for them. Near zero or positive and they have no reason to say "
              "no; deeply negative and you're asking them to weaken their team.",
    "MARKET": "What they receive divided by what they give, on consensus rankings. Around 1.00 "
              "looks like a fair deal on paper — which is what gets an offer opened.",
    "ODDS": "LIKELY = helps their lineup and looks fair. EXPLOIT = looks fair on rankings but "
            "costs them lineup points; that's your edge, and the harder sell. NEEDS PITCH = "
            "fine for their lineup but looks lopsided, so lead with the fit.",
    "PWR": "Rank by roster strength alone, ignoring record.",
    "xW": "Wins his scores were worth against the whole league, not just the one opponent the "
          "schedule gave him each week. A close score is treated as a coin flip and a blowout "
          "as near-certain, judged against how spread out scoring was that week.",
    "xW%": "Expected wins per week played — the power ranking itself. 0.500 is an average team.",
    "LUCK W": "Real wins minus expected wins. Plus means the schedule has been kind and the "
              "record should come back; minus means a team is better than its record says.",
    "SWING": "How much his weekly score bounces around, relative to his own average. Low is a "
             "team you can predict; high is one that wins big and loses big.",
    "LINEUP": "Points per game his best legal starting lineup is worth.",
    "BENCH": "Points per game from his three best bench players — cover for byes and injuries.",
    "MATCHED": "Roster players with enough stats to score. Low = a rougher estimate.",
    "LUCK": "Power rank minus standing. Plus = the record flatters the roster (expect a slide). "
            "Minus = better than the record shows.",
    "PLAYERS": "Your players in this game.",
    "TOT": "Vegas over/under for the whole game.",
    "SPRD": "Point spread. Plus = this team is favoured by that many.",
    "VERDICT": "The matchup in one word.",
    "OWN": "Whether he's free, on another roster, or already yours.",
    "GAME": "Who he played (@ = away game).",
    "SNAP%": "Share of his offense's plays he was on the field for.",
    "SK": "Times sacked.",
    "AIRYD": "Air yards on his targets — how far downfield the ball was thrown, caught or not.",
    "YAC": "Yards after the catch.",
    "POSRANK": "Where he ranks at his position this season, among players with at least half "
               "the games of the most-used one.",
}

# Headers are Title Case across every table. Rather than hand-capitalising ninety strings in
# LABELS and watching them drift apart again, the rule is applied once at render.
# Short prepositions stay lowercase, the usual rule. "for" is deliberately not here: PF and
# PA are "Pts For" and "Pts Against" on every scoreboard ever printed.
_SMALL = {"a", "an", "and", "at", "by", "in", "of", "on", "or", "per", "the", "to", "vs"}


def title_case(label: str) -> str:
    """'Snap share' -> 'Snap Share'. Words that already carry capitals are left alone: xFP,
    WOPR, FAAB and TD are names, not prose, and title-casing them would damage them."""
    def cap(word: str, first: bool) -> str:
        m = re.search(r"[A-Za-z]", word)
        if not m:
            return word                       # "30", "±"
        i = m.start()
        if i and word[i - 1].isalnum():
            return word                       # "1st", "2nd" — the letters belong to the number
        core = word[i:]
        if core != core.lower():
            return word                       # xFP, WOPR, FAAB, TD
        if not first and core.strip(".,)").lower() in _SMALL:
            return word
        return word[:i] + core[0].upper() + core[1:]

    parts = re.split(r"([\s/\-]+)", str(label))
    out, first = [], True
    for part in parts:
        if not part or re.fullmatch(r"[\s/\-]+", part):
            out.append(part)
            continue
        out.append(cap(part, first))
        first = False
    return "".join(out)


# Ranks read as "#3", so they can't be mistaken for counts.
RANKS = {"TM#", "MU#", "ADD#", "WOPR#", "DRAFT#", "PWR", "RANK"}
# Obvious from the header; listing them in the legend is noise.
_NO_KEY = {"PLAYER", "POS", "LOGO", "GIVE LOGO", "GET LOGO", "AGE", "W", "L", "T", "TEAM",
           "MGR", "WK", "WHEN", "TYPE", "MOVE", "YOU GIVE", "YOU GET", "WHY"}
# Every number column is the same width, so tables line up with each other and a header like
# "Target Share" isn't clipped in one table and roomy in the next. Text columns are sized to
# what they hold; anything not named here is a number.
_NUM_W = 96
_FLEX = {"WHY", "MEANS", "PLAYERS", "MOVE", "NOTE", "TAGS"}
_ROLE_W = 130
_WIDTH = {
    "PLAYER": 170, "NAME": 170, "ROLE": _ROLE_W, "GIVE ROLE": _ROLE_W, "GET ROLE": _ROLE_W,
    "YOU GIVE": 150, "YOU GET": 150, "MANAGER": 140, "TEAM": 150, "DRAFTED BY": 140,
    "MGR": 120, "OWNER": 130, "STAT": 185, "SIGNAL": 115,
    "VERDICT": 115, "OWN": 130, "SRC": 125, "FILLS": 140, "THEY NEED": 140, "POSRANK": 115,
    "GAME": 105, "MU": 105, "OPP": 105, "SLOT": 72, "POS": 72, "ST": 105, "WHEN": 115,
    "TYPE": 115, "RES": 80, "STRK": 80, "W": 60, "L": 60, "T": 60, "G": 72, "WK": 68,
}
# Text reads from the left; every figure lines up on the right so columns can be compared
# down the page without the eye hunting for the decimal point.
_LEFT = {"PLAYER", "NAME", "WHY", "MEANS", "PLAYERS", "TAGS", "YOU GIVE", "YOU GET", "MANAGER",
         "TEAM", "DRAFTED BY", "MGR", "OWNER", "STAT", "NOTE", "MOVE", "SIGNAL", "VERDICT",
         "OWN", "SRC", "FILLS", "THEY NEED", "GAME", "MU", "OPP", "SLOT", "POS", "ST", "WHEN",
         "TYPE", "RES", "GRADE", "MATCHED", "ROLE", "GIVE ROLE", "GET ROLE"}
_LOGO_COLS = ("LOGO", "GIVE LOGO", "GET LOGO")


def cols(df: pd.DataFrame, **extra: str) -> pd.DataFrame:
    """Rename to the internal codes; `extra` overrides for dynamic names."""
    return df.rename(columns={**COLS, **extra})


# ---------------------------------------------------------------- team logos
_PLAYER_TEAM: dict[str, str] = {}


def set_player_teams(mapping: dict[str, str]) -> None:
    """normalized player name -> NFL team, so a table with a player but no team column
    still gets a logo. app.py fills this once per run."""
    _PLAYER_TEAM.clear()
    _PLAYER_TEAM.update({k: v for k, v in mapping.items() if k and v})


def _logo_for_name(name: object) -> str:
    from .ids import norm
    from .logos import logo_url

    # trade cells read "Name (WR)"
    n = norm(re.sub(r"\s*\([A-Z/]+\)\s*$", "", str(name)))
    return logo_url(_PLAYER_TEAM.get(n)) or ""


# ---------------------------------------------------------------- roles (HANDOFF §12)
_PLAYER_ROLE: dict[str, str] = {}


def set_player_roles(mapping: dict[str, str]) -> None:
    """normalized player name -> "WR3 · ROLE+", so every table that lists a player says
    what his job is without each tab having to join it in. app.py fills this once per run."""
    _PLAYER_ROLE.clear()
    _PLAYER_ROLE.update({k: v for k, v in mapping.items() if k and v})


def _role_for_cell(cell: object) -> str:
    """A cell is one name, or a trade package "A (WR, 9.1) + B (RB, 6.0)"."""
    from .ids import norm

    parts = [re.sub(r"\s*\([^)]*\)\s*$", "", p) for p in str(cell).split(" + ")]
    roles = [_PLAYER_ROLE.get(norm(p), "") for p in parts]
    return " + ".join(r or "—" for r in roles) if any(roles) else ""


def _with_roles(d: pd.DataFrame) -> pd.DataFrame:
    """Put a role column after each player column — after POS when that follows him."""
    d = d.copy()
    for src, dst in (("PLAYER", "ROLE"), ("YOU GIVE", "GIVE ROLE"), ("YOU GET", "GET ROLE")):
        if src not in d.columns or dst in d.columns:
            continue
        vals = d[src].map(_role_for_cell)
        if not vals.ne("").any():
            continue
        cols = list(d.columns)
        at = cols.index(src) + 1
        if at < len(cols) and cols[at] == "POS":
            at += 1
        d.insert(at, dst, vals.values)
    return d


def _with_logos(d: pd.DataFrame) -> pd.DataFrame:
    """Swap the NFL team text for its logo, or add one beside the player."""
    from .logos import logo_url

    d = d.copy()
    if "TM" in d.columns:
        logos = d["TM"].map(lambda t: logo_url(t) or "")
        at = list(d.columns).index("PLAYER") if "PLAYER" in d.columns else list(d.columns).index("TM")
        d = d.drop(columns=["TM"])
        d.insert(min(at, len(d.columns)), "LOGO", logos.values)
    elif "PLAYER" in d.columns and _PLAYER_TEAM:
        logos = d["PLAYER"].map(_logo_for_name)
        if logos.ne("").any():
            d.insert(list(d.columns).index("PLAYER"), "LOGO", logos.values)
    for side, col in (("GIVE LOGO", "YOU GIVE"), ("GET LOGO", "YOU GET")):
        if col in d.columns and _PLAYER_TEAM:
            d.insert(list(d.columns).index(col), side, d[col].map(_logo_for_name).values)
    return d


# ---------------------------------------------------------------- render
def col_config(df: pd.DataFrame, labels: dict | None = None, help: dict | None = None) -> dict:
    """Readable header, tooltip and width for every column; image cells for logos."""
    labels, gloss = {**LABELS, **(labels or {})}, {**GLOSS, **(help or {})}
    out = {}
    for c in df.columns:
        if c in _LOGO_COLS:
            # "small" is ~75px; a logo needs about half that, and trade rows carry two.
            out[c] = st.column_config.ImageColumn(labels.get(c, ""), help="NFL team", width=44)
            continue
        kw = {"label": title_case(labels.get(c, c)),
              "alignment": "left" if c in _LEFT else "right"}
        if c not in _FLEX:
            kw["width"] = _WIDTH.get(c, _NUM_W)
        if c in gloss:
            kw["help"] = gloss[c]
        out[c] = st.column_config.Column(**kw)
    return out


def table(
    df: pd.DataFrame,
    rename: dict | None = None,
    diverging: list[str] = (),
    sequential: list[str] = (),
    pos_cols: list[str] = (),
    fmt: dict | None = None,
    help: dict | None = None,
    labels: dict | None = None,
    height: int | None = None,
    legend: bool = True,
    logos: bool = True,
    container=None,
    roles: bool = True,
) -> pd.DataFrame:
    """The one way to put a table on screen: internal codes -> readable headers,
    tooltips, team logos, heat shading, and a plain-English column key underneath.
    Returns the frame as rendered (codes as column names)."""
    box = container or st
    d = cols(df, **(rename or {}))
    if roles and _PLAYER_ROLE:
        d = _with_roles(d)
    if logos:
        d = _with_logos(d)
    fmt = {**(fmt or {}), **{r: "#{:.0f}" for r in RANKS if r in d.columns}}
    box.dataframe(
        style_df(d, diverging=diverging, sequential=sequential, pos_cols=pos_cols, fmt=fmt),
        column_config=col_config(d, labels=labels, help=help),
        width="stretch", hide_index=True,
        **({"height": height} if height else {}),
    )
    if legend:
        col_key(*d.columns, _labels=labels, _help=help, _container=box)
    return d


def col_key(*names: str, _labels: dict | None = None, _help: dict | None = None,
            _container=None, **custom: str) -> None:
    """'What these columns mean' panel. Collapsed, so it costs one line until it's needed."""
    labels, gloss = {**LABELS, **(_labels or {})}, {**GLOSS, **(_help or {}), **custom}
    seen, rows = set(), []
    for n in list(names) + list(custom):
        lab = title_case(labels.get(n, n))
        if n in _NO_KEY or n not in gloss or lab in seen:
            continue
        seen.add(lab)
        rows.append(f"<dt>{lab}</dt><dd>{gloss[n]}</dd>")
    if not rows:
        return
    with (_container or st).expander("What these columns mean"):
        st.markdown(f'<dl class="mb-key">{"".join(rows)}</dl>', unsafe_allow_html=True)


def h(text: str, level: int = 4) -> None:
    """Every section title, in one place: Title Case, one size, one rhythm above and below.
    Before this the app had st.subheader in some tabs and #### markdown in others."""
    st.markdown(f'<div class="mb-h mb-h{level}">{title_case(text)}</div>', unsafe_allow_html=True)


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
        .configure_axis(labelFont=FONT_BODY, titleFont=FONT_BODY,
                        labelColor=INK2, titleColor=INK3, domainColor=BORDER,
                        tickColor=BORDER, gridColor="#F0EEE9", labelFontSize=11)
        .configure_legend(labelFont=FONT_BODY, labelColor=INK2, labelFontSize=11,
                          symbolStrokeWidth=2.5)
    )
    st.altair_chart(ch, width="stretch")


def route_plot(pool: pd.DataFrame, mine: pd.DataFrame, threshold: float | None = None,
               x: str = "routes_pg", y: str = "fd_rr", y_title: str = "1st downs per route",
               x_title: str = "Routes per game (est.)") -> None:
    """Receivers on two axes: how much he is on the field, and what it is worth when he is.

    The league is the grey backdrop and `mine` is what the eye lands on, because the question
    is never "who leads the NFL" — it's "where do my guys sit". `threshold` draws the
    first-down line: 12% of routes, the mark a WR has to clear.
    """
    import altair as alt

    if pool.empty:
        return
    base = alt.Chart(pool)
    enc = dict(
        x=alt.X(f"{x}:Q", axis=alt.Axis(title=x_title, grid=False),
                scale=alt.Scale(nice=True, zero=False)),
        y=alt.Y(f"{y}:Q", axis=alt.Axis(title=y_title, format="%", grid=True),
                scale=alt.Scale(nice=True)),
    )
    tip = [alt.Tooltip("player:N", title="Player"), alt.Tooltip("pos:N", title="Pos"),
           alt.Tooltip("team:N", title="NFL"),
           alt.Tooltip(f"{x}:Q", title=x_title, format=".1f"),
           alt.Tooltip(f"{y}:Q", title=y_title, format=".1%")]

    layers = [base.mark_point(size=42, filled=False, strokeWidth=1.3, opacity=.45,
                              color=INK3).encode(**enc, tooltip=tip)]
    if threshold is not None:
        layers.append(alt.Chart(pd.DataFrame({"t": [threshold]}))
                      .mark_rule(color=FIRST_DOWN, strokeWidth=2.5)
                      .encode(y=alt.Y("t:Q")))
    med = pd.to_numeric(pool[x], errors="coerce").median()
    if pd.notna(med):
        layers.append(alt.Chart(pd.DataFrame({"m": [med]}))
                      .mark_rule(color=BORDER, strokeWidth=1, strokeDash=[3, 3])
                      .encode(x=alt.X("m:Q")))
    if mine is not None and not mine.empty:
        m = alt.Chart(mine)
        # Not POS_COLORS here: its WR gold sits in the same family as the first-down line, and
        # the line has to be the only thing on the chart wearing that colour.
        layers.append(m.mark_point(size=110, filled=True, opacity=1)
                      .encode(**enc, color=alt.Color("pos:N", legend=None,
                              scale=alt.Scale(domain=["WR", "TE"], range=["#1D5FA8", "#6B3FA0"])),
                              tooltip=tip))
        layers.append(m.mark_text(align="left", dx=9, dy=1, font=FONT_HEAD, fontSize=11,
                                  fontWeight=600, color=INK)
                      .encode(**enc, text=alt.Text("short:N")))

    ch = (alt.layer(*layers).properties(height=340).configure_view(strokeWidth=0)
          .configure_axis(labelFont=FONT_BODY, titleFont=FONT_BODY,
                          labelColor=INK2, titleColor=INK3, domainColor=BORDER,
                          tickColor=BORDER, gridColor="#F0F3F8", labelFontSize=11,
                          titleFontSize=11, titleFontWeight=700, titlePadding=8))
    st.altair_chart(ch, width="stretch")


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
