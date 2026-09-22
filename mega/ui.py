"""Shared UI theming for the Streamlit dashboard — matches the Mega Bowl draft board."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

# NFL shield palette: navy #013369 and red #D50A0A on white.
NAVY = "#013369"
RED = "#D50A0A"

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
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Archivo+Narrow:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root { --navy:#013369; --red:#D50A0A; --gold:#013369; --ink:#12161C; --ink2:#4E5866; --ink3:#8A93A0; --border:#DDE3EC; --surface2:#F5F7FA; }
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
.mb-kpi { background:#fff; border:1px solid var(--border); border-top:3px solid var(--navy); border-radius:6px; padding:11px 14px; height:100%; }
.mb-kpi-l { font-size:10.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--ink3); }
.mb-kpi-v { font-family:'Archivo',sans-serif; font-weight:800; font-size:22px; color:var(--navy); line-height:1.15; margin-top:3px; }
.mb-kpi-s { font-size:11.5px; color:var(--ink2); margin-top:2px; }
.stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--border); overflow-x:auto; scrollbar-width:thin; }
.stTabs [data-baseweb="tab"] { font-weight:700; font-size:13px; color:var(--ink3); padding:6px 12px; white-space:nowrap; }
.stTabs [aria-selected="true"] { color:var(--navy) !important; border-bottom:3px solid var(--red) !important; }
.mb-legend { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
.mb-chip { font-size:10px; font-weight:700; color:#fff; padding:2px 6px; border-radius:3px; }
/* column key: label | what it means, one row per column */
.mb-key { display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; margin:2px 0; font-size:13px; line-height:1.45; }
.mb-key dt { font-family:'Archivo',sans-serif; font-weight:700; color:var(--navy); white-space:nowrap; }
.mb-key dd { margin:0; color:var(--ink2); }
[data-testid="stExpander"] details summary p { font-size:12.5px; color:var(--ink3); }
[data-testid="stExpander"] { border-color:var(--border); margin:-4px 0 10px; }
[data-testid="stDataFrame"] { font-size:12.5px; }
.stAlert { border-radius:7px; }
.mb-lede { font-size:13.5px; color:var(--ink2); margin:2px 0 10px; line-height:1.5; }
.mb-lede b { color:var(--ink); }
/* player lookup card */
.mb-card { padding-top:4px; }
.mb-card-name { font-family:'Archivo',sans-serif; font-weight:800; font-size:26px; color:var(--navy); line-height:1.15; }
.mb-card-team { display:flex; align-items:center; gap:6px; font-size:15px; color:var(--ink); margin:4px 0 2px; }
.mb-card-facts { font-size:13.5px; color:var(--ink2); margin:0 0 10px; }
.mb-card-logo { height:26px; width:auto; }
.mb-card-row { font-size:14px; color:var(--ink); margin:3px 0; }
.mb-card-row span:first-child { display:inline-block; min-width:92px; font-size:11px; font-weight:700;
  letter-spacing:.06em; text-transform:uppercase; color:var(--ink3); }
.mb-flag { background:var(--red); color:#fff; font-weight:700; font-size:11.5px; padding:2px 7px;
  border-radius:4px; margin-right:6px; }

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
            f'<div class="mb-kpi"><div class="mb-kpi-l">{label}</div>'
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
    # power rankings
    "power_rank": "PWR", "starters_pg": "LINEUP", "bench_pg": "BENCH",
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
    "WHEN": "When", "TYPE": "Type", "MOVE": "Move",
    "MANAGER": "Manager", "YOU GIVE": "You give", "GIVE VAL": "Give value",
    "YOU GET": "You get", "GET VAL": "Get value", "FAIR": "Fairness", "EDGE": "Value gained",
    "FILLS": "Fixes", "THEY NEED": "They need", "GIVE LOGO": "", "GET LOGO": "",
    "PWR": "Power rank", "LINEUP": "Lineup pts/g", "BENCH": "Bench pts/g",
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
    "PWR": "Rank by roster strength alone, ignoring record.",
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

# Ranks read as "#3", so they can't be mistaken for counts.
RANKS = {"TM#", "MU#", "ADD#", "WOPR#", "DRAFT#", "PWR", "RANK"}
# Obvious from the header; listing them in the legend is noise.
_NO_KEY = {"PLAYER", "POS", "LOGO", "GIVE LOGO", "GET LOGO", "AGE", "W", "L", "T", "TEAM",
           "MGR", "WK", "WHEN", "TYPE", "MOVE", "YOU GIVE", "YOU GET", "WHY"}
_WIDTH = {"PLAYER": "medium", "WHY": "large", "MEANS": "large", "TAGS": "medium", "PLAYERS": "large",
          "YOU GIVE": "medium", "YOU GET": "medium", "MANAGER": "medium", "TEAM": "medium",
}
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
        kw = {"label": labels.get(c, c)}
        if c in gloss:
            kw["help"] = gloss[c]
        if c in _WIDTH:
            kw["width"] = _WIDTH[c]
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
) -> pd.DataFrame:
    """The one way to put a table on screen: internal codes -> readable headers,
    tooltips, team logos, heat shading, and a plain-English column key underneath.
    Returns the frame as rendered (codes as column names)."""
    box = container or st
    d = cols(df, **(rename or {}))
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
        lab = labels.get(n, n)
        if n in _NO_KEY or n not in gloss or lab in seen:
            continue
        seen.add(lab)
        rows.append(f"<dt>{lab}</dt><dd>{gloss[n]}</dd>")
    if not rows:
        return
    with (_container or st).expander("What these columns mean"):
        st.markdown(f'<dl class="mb-key">{"".join(rows)}</dl>', unsafe_allow_html=True)


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
        layers.append(m.mark_text(align="left", dx=9, dy=1, font="Archivo", fontSize=11,
                                  fontWeight=600, color=INK)
                      .encode(**enc, text=alt.Text("short:N")))

    ch = (alt.layer(*layers).properties(height=340).configure_view(strokeWidth=0)
          .configure_axis(labelFont="Archivo Narrow", titleFont="Archivo Narrow",
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
