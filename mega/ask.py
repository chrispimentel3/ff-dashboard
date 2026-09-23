"""Ask: plain-English questions answered from the nflverse tables the app already loads.

    "list WRs by snap %"      "top 10 RB by targets last 3 weeks"
    "worst catch rate"        "who leads the Rams in target share"

Deterministic, not a language model. A registry of metrics with aliases plus a small
parser for position / team / week / ordering / volume. It costs nothing per query, needs
no API key, runs offline, is testable — and, the part that matters, it can always state
how it read the question, so a misread shows on screen instead of being laundered into a
confident wrong answer. `Result.restated` is that sentence; the UI prints it every time.

Every rate here is summed numerator over summed denominator, never the mean of weekly
ratios — the same rule as target share and WOPR everywhere else in this app. A player with
two targets and one catch is not a 50% receiver, and averaging his week into a season rate
is how he ends up looking like one.

Pure pandas over frames the caller supplies, so it can be tested without Streamlit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as _field

import numpy as np
import pandas as pd

POS_ALL = ("QB", "RB", "WR", "TE")
DEFAULT_TOP = 25


# ---------------------------------------------------------------- metric registry
@dataclass(frozen=True)
class Field:
    """One answerable metric.

    kind="rate"     value = sum(num) / sum(den)      e.g. snap %, catch rate
    kind="total"    value = sum(col)                 e.g. targets, receiving yards
    kind="per_game" value = sum(col) / games         e.g. PPG
    kind="max"      value = max(col)                 e.g. best single week
    """
    key: str
    label: str
    aliases: tuple[str, ...]
    kind: str = "total"
    col: str = ""                      # total / per_game / max
    num: str = ""                      # rate numerator
    den: str = ""                      # rate denominator
    pct: bool = False                  # render as a percentage
    fmt: str = "{:.1f}"
    min_den: float = 0.0               # below this the rate is noise, not a reading
    den_label: str = ""                # what min_den counts, for the restatement
    pos: tuple[str, ...] = POS_ALL     # positions the question defaults to
    show: tuple[str, ...] = ()         # supporting columns, so the number is auditable
    note: str = ""


def _f(**kw) -> Field:
    return Field(**kw)


FIELDS: tuple[Field, ...] = (
    # ------------------------------------------------------------ playing time
    _f(key="snap_pct", label="snap %", kind="rate", num="offense_snaps", den="team_snaps",
       pct=True, fmt="{:.1%}", show=("offense_snaps", "team_snaps"),
       aliases=("snap %", "snap pct", "snap percent", "snap percentage", "snap share",
                "snaps %", "snap rate", "playing time", "snap"),
       note="Offensive snaps played over the snaps his team ran in those games."),
    _f(key="offense_snaps", label="snaps", kind="total", col="offense_snaps",
       fmt="{:.0f}", aliases=("snaps", "snap count", "snap counts", "offensive snaps")),
    _f(key="snaps_pg", label="snaps/g", kind="per_game", col="offense_snaps", fmt="{:.1f}",
       aliases=("snaps per game", "snaps a game", "snaps/g")),

    # ------------------------------------------------------------ receiving volume
    _f(key="targets", label="targets", kind="total", col="targets", fmt="{:.0f}",
       pos=("WR", "TE", "RB"), aliases=("targets", "target", "looks")),
    _f(key="targets_pg", label="targets/g", kind="per_game", col="targets", fmt="{:.1f}",
       pos=("WR", "TE", "RB"), aliases=("targets per game", "targets a game", "targets/g")),
    _f(key="target_share", label="target share", kind="rate", num="targets", den="team_targets",
       pct=True, fmt="{:.1%}", pos=("WR", "TE", "RB"), show=("targets", "team_targets"),
       aliases=("target share", "target %", "target pct", "target percentage", "tgt share", "tgt%"),
       note="His targets over his own offence's targets, in the weeks he played."),
    _f(key="receptions", label="catches", kind="total", col="receptions", fmt="{:.0f}",
       pos=("WR", "TE", "RB"), aliases=("receptions", "catches", "rec")),
    _f(key="receiving_yards", label="rec yards", kind="total", col="receiving_yards", fmt="{:.0f}",
       pos=("WR", "TE", "RB"), aliases=("receiving yards", "rec yards", "receiving yds", "yards receiving")),
    _f(key="receiving_tds", label="rec TD", kind="total", col="receiving_tds", fmt="{:.0f}",
       pos=("WR", "TE", "RB"), aliases=("receiving touchdowns", "receiving tds", "rec tds", "receiving td")),
    _f(key="air_yards", label="air yards", kind="total", col="receiving_air_yards", fmt="{:.0f}",
       pos=("WR", "TE"), aliases=("air yards", "receiving air yards")),
    _f(key="adot", label="aDOT", kind="rate", num="receiving_air_yards", den="targets",
       fmt="{:.1f}", min_den=10, den_label="targets", pos=("WR", "TE"),
       show=("receiving_air_yards", "targets"),
       aliases=("adot", "average depth of target", "depth of target"),
       note="Average depth of target: air yards over targets."),
    _f(key="yac", label="YAC", kind="total", col="receiving_yards_after_catch", fmt="{:.0f}",
       pos=("WR", "TE", "RB"), aliases=("yac", "yards after catch", "yards after the catch")),

    # ------------------------------------------------------------ receiving efficiency
    _f(key="catch_rate", label="catch rate", kind="rate", num="receptions", den="targets",
       pct=True, fmt="{:.1%}", min_den=10, den_label="targets", pos=("WR", "TE", "RB"),
       show=("receptions", "targets"),
       aliases=("catch rate", "catch %", "catch pct", "catch percentage", "reception rate")),
    _f(key="ypt", label="yds/target", kind="rate", num="receiving_yards", den="targets",
       fmt="{:.2f}", min_den=10, den_label="targets", pos=("WR", "TE", "RB"),
       show=("receiving_yards", "targets"),
       aliases=("yards per target", "yards/target", "ypt", "yds per target")),
    _f(key="ypr", label="yds/catch", kind="rate", num="receiving_yards", den="receptions",
       fmt="{:.2f}", min_den=8, den_label="catches", pos=("WR", "TE", "RB"),
       show=("receiving_yards", "receptions"),
       aliases=("yards per catch", "yards per reception", "yards/catch", "ypr", "ypc receiving")),

    # ------------------------------------------------------------ routes (estimated)
    _f(key="routes", label="routes", kind="total", col="routes", fmt="{:.0f}",
       pos=("WR", "TE"), aliases=("routes", "routes run", "route"),
       note="Estimated — snap share x team dropbacks. See mega/routes.py."),
    _f(key="routes_pg", label="routes/g", kind="per_game", col="routes", fmt="{:.1f}",
       pos=("WR", "TE"), aliases=("routes per game", "routes a game", "routes/g")),
    _f(key="tprr", label="TPRR", kind="rate", num="targets", den="routes", fmt="{:.3f}",
       min_den=20, den_label="routes", pos=("WR", "TE"), show=("targets", "routes"),
       aliases=("tprr", "targets per route", "targets per route run", "target rate"),
       note="Targets per route run. Routes are estimated, so read it as a ranking, not a rate."),
    _f(key="yprr", label="YPRR", kind="rate", num="receiving_yards", den="routes", fmt="{:.2f}",
       min_den=20, den_label="routes", pos=("WR", "TE"), show=("receiving_yards", "routes"),
       aliases=("yprr", "yards per route", "yards per route run")),
    _f(key="fd_rr", label="1D/RR", kind="rate", num="receiving_first_downs", den="routes",
       fmt="{:.3f}", min_den=20, den_label="routes", pos=("WR", "TE"),
       show=("receiving_first_downs", "routes"),
       aliases=("1d/rr", "first downs per route", "first down rate")),

    # ------------------------------------------------------------ rushing
    _f(key="carries", label="carries", kind="total", col="carries", fmt="{:.0f}",
       pos=("RB", "QB"), aliases=("carries", "rushing attempts", "rush attempts", "totes")),
    _f(key="carries_pg", label="carries/g", kind="per_game", col="carries", fmt="{:.1f}",
       pos=("RB",), aliases=("carries per game", "carries a game", "carries/g")),
    _f(key="rush_share", label="rush share", kind="rate", num="carries", den="team_carries",
       pct=True, fmt="{:.1%}", pos=("RB",), show=("carries", "team_carries"),
       aliases=("rush share", "carry share", "rushing share", "backfield share")),
    _f(key="rushing_yards", label="rush yards", kind="total", col="rushing_yards", fmt="{:.0f}",
       pos=("RB", "QB"), aliases=("rushing yards", "rush yards", "rushing yds", "yards rushing")),
    _f(key="rushing_tds", label="rush TD", kind="total", col="rushing_tds", fmt="{:.0f}",
       pos=("RB", "QB"), aliases=("rushing touchdowns", "rushing tds", "rush tds", "rushing td")),
    _f(key="ypc", label="yds/carry", kind="rate", num="rushing_yards", den="carries",
       fmt="{:.2f}", min_den=20, den_label="carries", pos=("RB", "QB"),
       show=("rushing_yards", "carries"),
       aliases=("yards per carry", "yards/carry", "ypc", "rushing average")),
    _f(key="touches", label="touches", kind="total", col="touches", fmt="{:.0f}",
       pos=("RB", "WR"), aliases=("touches", "opportunities", "opps"),
       note="Carries plus targets."),

    # ------------------------------------------------------------ scoring position
    # Derived from play-by-play (mega/redzone.py). No nflverse column holds these: the
    # weekly `rushing_10` / `rushing_20` columns look like red zone stats and are not —
    # they count runs of 10+ and 20+ YARDS.
    _f(key="rz_carries", label="RZ carries", kind="total", col="rz_carries", fmt="{:.0f}",
       pos=("RB", "QB"), aliases=("red zone carries", "redzone carries", "rz carries",
                                  "carries inside the 20", "red zone rushing attempts",
                                  "red zone rush attempts"),
       note="Carries inside the opponent's 20. From play-by-play; two-point plays excluded."),
    _f(key="rz_targets", label="RZ targets", kind="total", col="rz_targets", fmt="{:.0f}",
       pos=("WR", "TE", "RB"), aliases=("red zone targets", "redzone targets", "rz targets",
                                        "targets inside the 20")),
    _f(key="rz_touches", label="RZ touches", kind="total", col="rz_touches", fmt="{:.0f}",
       pos=("RB", "WR", "TE"), aliases=("red zone touches", "redzone touches", "rz touches",
                                        "red zone opportunities", "red zone work", "red zone"),
       note="Red zone carries plus red zone targets."),
    _f(key="rz_carry_share", label="RZ carry share", kind="rate", num="rz_carries",
       den="team_rz_carries", pct=True, fmt="{:.1%}", pos=("RB",),
       show=("rz_carries", "team_rz_carries"), min_den=5, den_label="team RZ carries",
       aliases=("red zone carry share", "red zone rush share", "rz carry share")),
    _f(key="rz_target_share", label="RZ target share", kind="rate", num="rz_targets",
       den="team_rz_targets", pct=True, fmt="{:.1%}", pos=("WR", "TE", "RB"),
       show=("rz_targets", "team_rz_targets"), min_den=5, den_label="team RZ targets",
       aliases=("red zone target share", "rz target share")),
    _f(key="i10_carries", label="carries in-10", kind="total", col="i10_carries", fmt="{:.0f}",
       pos=("RB", "QB"), aliases=("carries inside the 10", "inside the 10 carries",
                                  "rushes inside the 10")),
    _f(key="i10_targets", label="targets in-10", kind="total", col="i10_targets", fmt="{:.0f}",
       pos=("WR", "TE", "RB"), aliases=("targets inside the 10", "inside the 10 targets")),
    _f(key="gl_carries", label="goal line carries", kind="total", col="gl_carries", fmt="{:.0f}",
       pos=("RB", "QB"), aliases=("goal line carries", "goalline carries", "goal-line carries",
                                  "carries inside the 5", "goal line work"),
       note="Carries inside the 5 — the touches that actually become touchdowns."),
    _f(key="gl_touches", label="goal line touches", kind="total", col="gl_touches", fmt="{:.0f}",
       pos=("RB", "WR", "TE"), aliases=("goal line touches", "goalline touches",
                                        "touches inside the 5")),

    # ------------------------------------------------------------ passing
    _f(key="passing_yards", label="pass yards", kind="total", col="passing_yards", fmt="{:.0f}",
       pos=("QB",), aliases=("passing yards", "pass yards", "passing yds", "yards passing")),
    _f(key="passing_tds", label="pass TD", kind="total", col="passing_tds", fmt="{:.0f}",
       pos=("QB",), aliases=("passing touchdowns", "passing tds", "pass tds", "passing td")),
    _f(key="interceptions", label="INT", kind="total", col="passing_interceptions", fmt="{:.0f}",
       pos=("QB",), aliases=("interceptions", "picks", "ints")),
    _f(key="attempts", label="attempts", kind="total", col="attempts", fmt="{:.0f}",
       pos=("QB",), aliases=("pass attempts", "passing attempts", "dropbacks", "attempts")),
    _f(key="comp_pct", label="comp %", kind="rate", num="completions", den="attempts",
       pct=True, fmt="{:.1%}", min_den=30, den_label="attempts", pos=("QB",),
       show=("completions", "attempts"),
       aliases=("completion percentage", "completion %", "completion rate", "comp %", "comp pct")),
    _f(key="ypa", label="yds/att", kind="rate", num="passing_yards", den="attempts",
       fmt="{:.2f}", min_den=30, den_label="attempts", pos=("QB",),
       show=("passing_yards", "attempts"),
       aliases=("yards per attempt", "yards/attempt", "ypa")),

    # ------------------------------------------------------------ fantasy
    _f(key="half_ppr", label="points", kind="total", col="half_ppr", fmt="{:.1f}",
       aliases=("fantasy points", "points", "half ppr", "half-ppr", "total points", "pts")),
    _f(key="ppg", label="PPG", kind="per_game", col="half_ppr", fmt="{:.1f}",
       aliases=("points per game", "ppg", "fantasy points per game", "scoring average")),
    _f(key="best_week", label="best week", kind="max", col="half_ppr", fmt="{:.1f}",
       aliases=("best week", "highest week", "ceiling", "best game")),
    _f(key="xfp", label="xFP", kind="total", col="half_ppr_exp", fmt="{:.1f}",
       aliases=("expected points", "expected fantasy points", "xfp", "expected"),
       note="nflverse ff_opportunity, scored with this league's half-PPR weights. See docs/expected-points.md."),
    _f(key="xfp_pg", label="xFP/g", kind="per_game", col="half_ppr_exp", fmt="{:.1f}",
       aliases=("expected points per game", "xfp per game", "xfp/g")),
    _f(key="xfp_diff", label="xFP±", kind="total", col="xfp_diff", fmt="{:+.1f}",
       aliases=("points over expected", "over expected", "xfp diff", "xfp difference",
                "luck", "efficiency vs expected"),
       note="Actual minus expected. Positive = outscoring his opportunity."),
    _f(key="first_downs", label="1D", kind="total", col="first_downs", fmt="{:.0f}",
       aliases=("first downs", "first down", "1d")),
    _f(key="games", label="games", kind="total", col="one", fmt="{:.0f}",
       aliases=("games", "games played", "appearances")),
)

FIELD_BY_KEY = {f.key: f for f in FIELDS}


def _w(pat: str) -> str:
    """Word boundary that also works next to '%' and '/'.

    `\b` is defined against word characters, so `\bsnap %\b` can never match — the
    trailing boundary wants a word character after the '%' and there is never one. That
    silently handed "snaps %" to the bare "snaps" counting stat instead of the rate.
    """
    return r"(?<![a-z0-9])" + pat + r"(?![a-z0-9])"


# ---------------------------------------------------------------- vocabulary
_POS_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("wide receivers", ("WR",)), ("wide receiver", ("WR",)), ("receivers", ("WR",)),
    ("receiver", ("WR",)), ("wrs", ("WR",)), ("wr", ("WR",)),
    ("running backs", ("RB",)), ("running back", ("RB",)), ("rbs", ("RB",)), ("rb", ("RB",)),
    ("backs", ("RB",)),
    ("tight ends", ("TE",)), ("tight end", ("TE",)), ("tes", ("TE",)), ("te", ("TE",)),
    ("quarterbacks", ("QB",)), ("quarterback", ("QB",)), ("qbs", ("QB",)), ("qb", ("QB",)),
    ("pass catchers", ("WR", "TE")), ("pass-catchers", ("WR", "TE")),
    ("flex", ("RB", "WR")), ("skill players", POS_ALL), ("everyone", POS_ALL),
]

# nflverse abbreviations. Bare "no" is deliberately absent — it is a word, and
# "no receiver over 20%" should not quietly become a New Orleans query.
_TEAM_WORDS: dict[str, str] = {}
for _abbr, _names in {
    "ARI": ("arizona", "cardinals", "cards"), "ATL": ("atlanta", "falcons"),
    "BAL": ("baltimore", "ravens"), "BUF": ("buffalo", "bills"),
    "CAR": ("carolina", "panthers"), "CHI": ("chicago", "bears"),
    "CIN": ("cincinnati", "bengals"), "CLE": ("cleveland", "browns"),
    "DAL": ("dallas", "cowboys"), "DEN": ("denver", "broncos"),
    "DET": ("detroit", "lions"), "GB": ("green bay", "packers"),
    "HOU": ("houston", "texans"), "IND": ("indianapolis", "colts"),
    "JAX": ("jacksonville", "jaguars", "jags"), "KC": ("kansas city", "chiefs"),
    "LA": ("los angeles rams", "rams",), "LAC": ("los angeles chargers", "chargers"),
    "LV": ("las vegas", "raiders"), "MIA": ("miami", "dolphins"),
    "MIN": ("minnesota", "vikings", "vikes"), "NE": ("new england", "patriots", "pats"),
    "NO": ("new orleans", "saints"), "NYG": ("new york giants", "giants"),
    "NYJ": ("new york jets", "jets"), "PHI": ("philadelphia", "eagles"),
    "PIT": ("pittsburgh", "steelers"), "SEA": ("seattle", "seahawks"),
    "SF": ("san francisco", "49ers", "niners"), "TB": ("tampa bay", "buccaneers", "bucs"),
    "TEN": ("tennessee", "titans"), "WAS": ("washington", "commanders"),
}.items():
    for _n in _names:
        _TEAM_WORDS[_n] = _abbr
    if _abbr != "NO":
        _TEAM_WORDS[_abbr.lower()] = _abbr

_ASC_WORDS = ("worst", "lowest", "fewest", "least", "bottom", "ascending", "smallest")
_WORDS_TO_N = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
               "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
               "twenty": 20, "twentyfive": 25, "thirty": 30}


def _clean(q: str) -> str:
    """Lowercase, split a glued '%', drop punctuation that carries no meaning here."""
    s = str(q).lower().replace("%", " % ").replace("+", " + ")
    s = re.sub(r"[^a-z0-9%+\-/ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Longest alias first, so "snap %" wins over "snap" and "targets per game" over "targets".
# Aliases go through _clean too, so the question and the alias are spelled the same way.
_ALIASES: list[tuple[str, Field]] = sorted(
    ((_clean(a), f) for f in FIELDS for a in f.aliases), key=lambda t: -len(t[0])
)


# ---------------------------------------------------------------- query
@dataclass
class Query:
    field: Field
    positions: tuple[str, ...] = POS_ALL
    teams: tuple[str, ...] = ()
    weeks: tuple[int, ...] = ()          # empty = every week in the data
    top: int = DEFAULT_TOP
    ascending: bool = False
    min_den: float = 0.0
    scope: str = ""                      # "", "mine", "fa", "rostered"
    window_label: str = "the season"
    unmatched: tuple[str, ...] = ()
    source: str = "pw"                   # "pw" = the curated player-week table,
    hit: object = None                   # otherwise an nflverse table name + its Hit


class AskError(ValueError):
    """The question named no metric we hold."""


def parse(text: str, weeks_available: tuple[int, ...] = ()) -> Query:
    """Read a question into a Query. Raises AskError if no metric matches.

    Each matcher removes what it consumed, so "top 10 RB by targets" does not let the
    position matcher find a stray letter inside a metric name.
    """
    s = _clean(text)

    # --- ordering (looked for before anything strips the words)
    ascending = any(re.search(rf"\b{w}\b", s) for w in _ASC_WORDS)

    # --- scope
    scope = ""
    if re.search(r"\b(my (team|roster|guys|players)|i own|i have|on my)\b", s):
        scope = "mine"
    elif re.search(r"\b(free agents?|waiver wire|waivers|available|unowned|unrostered)\b", s):
        scope = "fa"
    elif re.search(r"\b(rostered|owned|taken)\b", s):
        scope = "rostered"
    s = re.sub(r"\b(my (team|roster|guys|players)|i own|i have|on my|free agents?|waiver wire"
               r"|waivers|available|unowned|unrostered|rostered|owned|taken)\b", " ", s)

    # --- weeks. Ordered most specific first; each returns and strips.
    weeks: tuple[int, ...] = ()
    label = "the season"
    avail = tuple(sorted(int(w) for w in weeks_available)) if len(weeks_available) else ()
    if m := re.search(r"\bweeks? (\d+)\s*(?:-|to|thru|through)\s*(\d+)\b", s):
        a, b = int(m.group(1)), int(m.group(2))
        weeks = tuple(range(min(a, b), max(a, b) + 1))
        label = f"weeks {min(a, b)}-{max(a, b)}"
        s = s[: m.start()] + " " + s[m.end():]
    elif m := re.search(r"\b(?:last|past|recent) (\d+|" + "|".join(_WORDS_TO_N) + r") (?:weeks?|games?)\b", s):
        g = m.group(1)
        n = int(g) if g.isdigit() else _WORDS_TO_N[g]
        weeks = tuple(avail[-n:]) if avail else ()
        label = f"the last {n} weeks"
        s = s[: m.start()] + " " + s[m.end():]
    elif m := re.search(r"\bsince week (\d+)\b", s):
        a = int(m.group(1))
        weeks = tuple(w for w in avail if w >= a) if avail else ()
        label = f"week {a} onward"
        s = s[: m.start()] + " " + s[m.end():]
    elif m := re.search(r"\b(?:in |for )?(?:week|wk) (\d+)\b", s):
        a = int(m.group(1))
        weeks, label = (a,), f"week {a}"
        s = s[: m.start()] + " " + s[m.end():]
    elif m := re.search(r"\blast (?:week|game)\b", s):
        weeks = (avail[-1],) if avail else ()
        label = f"week {avail[-1]}" if avail else "last week"
        s = s[: m.start()] + " " + s[m.end():]

    # --- explicit volume floor ("min 20 targets", "at least 30 routes", "20+ targets").
    # Before teams, so the "min" in "minimum" is gone before MIN can match Minnesota.
    # The unit noun goes with it: "min 20 routes" left "routes" behind, and the metric
    # matcher then read the floor's unit as the question's stat (tprr became routes).
    _UNITS = r"(?:targets?|routes?|carries|catches|receptions?|attempts?|snaps?|games?|touches)"
    min_den = -1.0
    if m := re.search(r"(?:min|minimum|at least|over|more than) (\d+)\s*" + _UNITS + r"?", s):
        min_den = float(m.group(1))
        s = s[: m.start()] + " " + s[m.end():]
    elif m := re.search(r"(\d+)\s*\+\s*" + _UNITS + r"?", s):
        min_den = float(m.group(1))
        s = s[: m.start()] + " " + s[m.end():]

    # --- top N
    top = DEFAULT_TOP
    if m := re.search(r"\b(?:top|best|first|bottom|worst) (\d+|" + "|".join(_WORDS_TO_N) + r")\b", s):
        g = m.group(1)
        top = int(g) if g.isdigit() else _WORDS_TO_N[g]
        s = s[: m.start()] + " " + s[m.end():]
    elif re.search(r"\bwho leads\b|\bwho has the (most|best|fewest|worst|lowest)\b", s):
        top = 10

    # --- teams (longest name first: "los angeles rams" before "rams")
    teams: list[str] = []
    for name in sorted(_TEAM_WORDS, key=len, reverse=True):
        if re.search(_w(re.escape(name)), s):
            abbr = _TEAM_WORDS[name]
            if abbr not in teams:
                teams.append(abbr)
            s = re.sub(_w(re.escape(name)), " ", s)

    # --- positions. Before the metric, so that when no curated metric matches, what is
    # left to look up in the catalogue is the stat phrase alone: "wr by average separation"
    # found nothing, "average separation" finds it.
    positions: list[str] = []
    for word, ps in _POS_WORDS:
        if re.search(_w(re.escape(word)), s):
            for p in ps:
                if p not in positions:
                    positions.append(p)
            s = re.sub(_w(re.escape(word)), " ", s)

    # --- metric (longest alias first)
    stat_phrase = _phrase(s)
    fld, after = None, s
    for alias, f in _ALIASES:
        if re.search(_w(re.escape(alias)), s):
            fld = f
            after = re.sub(_w(re.escape(alias)), " ", s, count=1)
            break

    source, hit = "pw", None
    if fld is None:
        fld, source, hit, after = _generic(s, text, tuple(positions))
    elif _phrase(after):
        # A curated alias matched only part of the phrase. "rush yards over expected" hit
        # the xFP± alias "over expected" and answered with points over expected — a wrong
        # number, confidently formatted. If the whole phrase is the exact name of a real
        # nflverse column, that column is what was asked for.
        from . import catalog
        exact = catalog.search(stat_phrase, limit=1, positions=tuple(positions), min_score=90.0)
        if exact:
            fld, source, hit, after = _generic(s, text, tuple(positions), forced=exact[0])
    s = after
    pos = tuple(positions) if positions else fld.pos

    leftover = tuple(w for w in s.split() if len(w) > 2 and w not in _STOP)
    return Query(field=fld, positions=pos, teams=tuple(teams), weeks=weeks, top=top,
                 ascending=ascending, min_den=fld.min_den if min_den < 0 else min_den,
                 scope=scope, window_label=label, unmatched=leftover,
                 source=source, hit=hit)


_JOIN = {"by", "in", "on", "of", "at", "to", "vs", "is", "a", "an", "it", "or", "up"}


def _phrase(s: str) -> str:
    """What is left of a question once the scaffolding is gone — the stat, or nothing."""
    return " ".join(w for w in s.split() if w not in _STOP and w not in _JOIN).strip()


def _generic(remainder: str, original: str, positions: tuple[str, ...] = (),
             forced=None) -> tuple[Field, str, object, str]:
    """No curated metric matched: look the phrase up in the full nflverse catalogue.

    Returns a Field synthesised from the column. It is honest but blunt — a plain sum, or
    a plain mean when the column name says it is already a rate — because nobody has
    vetted what the right denominator for that column would be. Everything built this way
    is labelled on screen so the number is never mistaken for a curated one.
    """
    from . import catalog

    phrase = _phrase(remainder)
    if forced is not None:
        h = forced
    else:
        hits = (catalog.search(phrase, limit=1, positions=positions) if phrase else [])
        if not hits:
            hits = catalog.search(original, limit=1, positions=positions)
        if not hits:
            raise AskError(_suggest_for(original))
        h = hits[0]
    rec = catalog.schema().get(h.table, {})
    weekly = bool(rec.get("week_key"))
    return (
        Field(key=h.column, label=h.label, aliases=(),
              kind="mean" if h.is_rate else ("total" if weekly else "max"),
              col=h.column, fmt="{:.3f}" if h.is_rate else "{:.1f}", pos=POS_ALL,
              note=(f"`{h.column}` from nflverse `{h.table}`, found by name rather than "
                    + ("averaged across the weeks in range. " if h.is_rate else
                       ("summed across the weeks in range. " if weekly else "as recorded. "))
                    + "Nobody has vetted the right way to aggregate this one — check it "
                      "before you trade on it.")),
        h.table, h, "")


# "over", "above", "under" and "below" are deliberately NOT here: they carry meaning inside
# stat names ("rush yards over expected"), and the minimum-volume regex has already taken
# the "over 20" sense out of the string by the time this is used.
_STOP = {"list", "show", "give", "the", "and", "for", "with", "who", "what", "which", "has",
         "have", "had", "leads", "leading", "most", "best", "top", "all", "players", "player",
         "this", "that", "season", "year", "them", "their", "his", "from", "each", "per",
         "are", "was", "were", "get", "sorted", "sort", "rank", "ranked", "ranking", "order",
         "ordered", "worst", "lowest", "fewest", "least", "bottom", "ascending", "smallest",
         "highest", "many", "much", "there", "out", "than",
         "only", "just", "guys", "everyone", "anyone", "table", "chart", "data", "stats"}


def _suggest_for(text: str) -> str:
    """The error a question earns when it names no stat — with the nearest few, if any."""
    t = _clean(text)
    near = list(dict.fromkeys(
        f.label for f in FIELDS for a in f.aliases
        if any(len(w) > 3 and w in t for w in a.split())
    ))[:3]
    tip = f" Closest I hold: {', '.join(near)}." if near else ""
    return ("I could not find a stat in that question." + tip +
            " Name one like snap %, target share, routes, yards per carry or expected points "
            "\u2014 the full list is under 'What can I ask?'.")


# ---------------------------------------------------------------- the player-week table
_STAT_COLS = ("targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
              "receiving_first_downs", "receiving_yards_after_catch", "carries", "rushing_yards",
              "rushing_tds", "rushing_first_downs", "attempts", "completions", "passing_yards",
              "passing_tds", "passing_interceptions", "passing_first_downs", "half_ppr")


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else pd.Series(0.0, index=df.index)


def _normalize(stats: pd.DataFrame) -> pd.DataFrame:
    """nflverse column names -> the ones used here, one source per target.

    Renaming every candidate at once collapses `player_name` and `player_display_name`
    into two columns both called `player`, and every later lookup then hands back a
    2-D frame instead of a Series.
    """
    ren = {}
    for target, cands in (("gsis_id", ("gsis_id", "player_id")),
                          ("player", ("player", "player_display_name", "player_name", "full_name")),
                          ("pos", ("pos", "position", "position_group")),
                          ("team", ("team", "recent_team", "posteam"))):
        if target in stats.columns:
            continue
        for c in cands:
            if c in stats.columns:
                ren[c] = target
                break
    return stats.rename(columns=ren)


def player_week(stats: pd.DataFrame, snaps: pd.DataFrame | None = None,
                ffo: pd.DataFrame | None = None, routes: pd.DataFrame | None = None,
                crosswalk: pd.DataFrame | None = None, redzone: pd.DataFrame | None = None,
                season_type: str = "REG") -> pd.DataFrame:
    """One row per player per game, with everything a question can ask about.

    Accepts either the app's normalized weekly frame (gsis_id / player / pos / team) or
    nflverse's raw column names, so tools and tests can call it without going through
    app.py's loaders.
    """
    if stats is None or stats.empty:
        return pd.DataFrame()
    w = _normalize(stats)
    if "season_type" in w.columns and season_type:
        w = w[w["season_type"].astype(str).str.upper() == season_type.upper()]
    if w.empty:
        return pd.DataFrame()
    w = w[w["pos"].isin(POS_ALL)].copy()
    w["week"] = pd.to_numeric(w["week"], errors="coerce").astype("Int64")

    out = pd.DataFrame({"gsis_id": w["gsis_id"], "player": w["player"], "pos": w["pos"],
                        "team": w["team"].map(_canon_team), "week": w["week"]})
    for c in _STAT_COLS:
        out[c] = _num(w, c)
    if "half_ppr" not in w.columns:
        out["half_ppr"] = _score(w)
    out["touches"] = out["carries"] + out["targets"]
    out["first_downs"] = out["receiving_first_downs"] + out["rushing_first_downs"] + out["passing_first_downs"]
    out["one"] = 1.0

    # team denominators, from the same weeks (never a season-long team total against a
    # part-season player — that is the games-played denominator rule from the handoff)
    tt = out.groupby(["team", "week"], dropna=False)[["targets", "carries"]].sum()
    tt.columns = ["team_targets", "team_carries"]
    out = out.merge(tt.reset_index(), on=["team", "week"], how="left")

    out = _join_snaps(out, snaps, crosswalk)
    out = _join_routes(out, routes)
    out = _join_ffo(out, ffo)
    out = _join_redzone(out, redzone)
    return out


_RZ_COLS = ("rz_carries", "rz_targets", "rz_touches", "i10_carries", "i10_targets",
            "i10_touches", "gl_carries", "gl_targets", "gl_touches",
            "team_rz_carries", "team_rz_targets", "team_rz_touches",
            "team_i10_carries", "team_i10_targets", "team_gl_carries", "team_gl_targets")


def _join_redzone(out: pd.DataFrame, rz) -> pd.DataFrame:
    """Scoring-position work, from mega/redzone.py.

    A player-week absent from the red zone frame played no snap inside the 20, which is a
    real zero, not a missing value — so these fill with 0 rather than NaN. Filling with
    NaN would have dropped every such player out of an ascending 'fewest red zone carries'
    answer, which is precisely the player that question is looking for.
    """
    if rz is None or getattr(rz, "empty", True):
        for c in _RZ_COLS:
            out[c] = np.nan
        return out
    r = rz.copy()
    r["week"] = pd.to_numeric(r["week"], errors="coerce").astype("Int64")
    keep = ["gsis_id", "week"] + [c for c in _RZ_COLS if c in r.columns]
    out = out.merge(r[keep], on=["gsis_id", "week"], how="left")
    for c in _RZ_COLS:
        out[c] = out[c].fillna(0.0) if c in out.columns else 0.0
    return out


def _canon_team(t: object) -> str:
    try:
        from .ids import canon_team
        return canon_team(t)
    except Exception:
        return str(t).upper().strip()


def _score(w: pd.DataFrame) -> pd.Series:
    from .config import SCORING as s
    return (s["pass_yd"] * _num(w, "passing_yards") + s["pass_td"] * _num(w, "passing_tds")
            + s["pass_int"] * _num(w, "passing_interceptions")
            + s["rush_yd"] * _num(w, "rushing_yards") + s["rush_td"] * _num(w, "rushing_tds")
            + s["rec"] * _num(w, "receptions") + s["rec_yd"] * _num(w, "receiving_yards")
            + s["rec_td"] * _num(w, "receiving_tds")
            + s["fum_lost"] * (_num(w, "rushing_fumbles_lost") + _num(w, "receiving_fumbles_lost")
                               + _num(w, "sack_fumbles_lost"))
            + s["two_pt"] * (_num(w, "passing_2pt_conversions") + _num(w, "rushing_2pt_conversions")
                             + _num(w, "receiving_2pt_conversions"))
            + s["ret_td"] * _num(w, "special_teams_tds"))


def _join_snaps(out: pd.DataFrame, snaps, crosswalk) -> pd.DataFrame:
    """Snap counts carry no gsis_id, so they join through the pfr_id crosswalk — the same
    path mega/routes.py uses. Team snaps are the most any one player took in that game:
    a lineman goes wire to wire, so his count is the offence's."""
    out["offense_snaps"] = np.nan
    out["team_snaps"] = np.nan
    if snaps is None or getattr(snaps, "empty", True) or crosswalk is None or crosswalk.empty:
        return out
    sn = snaps.rename(columns={"pfr_player_id": "pfr_id"})
    if "pfr_id" not in sn.columns:
        return out
    if "game_type" in sn.columns:
        sn = sn[sn["game_type"].astype(str).str.upper() == "REG"]
    sn = sn.assign(week=pd.to_numeric(sn["week"], errors="coerce").astype("Int64"),
                   team=sn["team"].map(_canon_team),
                   offense_snaps=pd.to_numeric(sn["offense_snaps"], errors="coerce"))
    team_snaps = sn.groupby(["team", "week"])["offense_snaps"].max().rename("team_snaps").reset_index()
    per = sn.groupby(["pfr_id", "week"])["offense_snaps"].max().reset_index()
    xw = crosswalk.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("gsis_id")[["gsis_id", "pfr_id"]]
    out = out.drop(columns=["offense_snaps", "team_snaps"]).merge(xw, on="gsis_id", how="left")
    out = out.merge(per, on=["pfr_id", "week"], how="left")
    out = out.merge(team_snaps, on=["team", "week"], how="left")
    return out


def _join_routes(out: pd.DataFrame, routes) -> pd.DataFrame:
    out["routes"] = np.nan
    if routes is None or getattr(routes, "empty", True) or "routes" not in routes.columns:
        return out
    r = routes[["gsis_id", "week", "routes"]].copy()
    r["week"] = pd.to_numeric(r["week"], errors="coerce").astype("Int64")
    return out.drop(columns=["routes"]).merge(r, on=["gsis_id", "week"], how="left")


def _join_ffo(out: pd.DataFrame, ffo) -> pd.DataFrame:
    out["half_ppr_exp"] = np.nan
    if ffo is None or getattr(ffo, "empty", True):
        out["xfp_diff"] = np.nan
        return out
    f = ffo.rename(columns={"player_id": "gsis_id"})
    col = "half_ppr_exp" if "half_ppr_exp" in f.columns else "total_fantasy_points_exp"
    if "gsis_id" not in f.columns or col not in f.columns:
        out["xfp_diff"] = np.nan
        return out
    f = f.assign(week=pd.to_numeric(f["week"], errors="coerce").astype("Int64"))
    f = f.groupby(["gsis_id", "week"])[col].sum().rename("half_ppr_exp").reset_index()
    out = out.drop(columns=["half_ppr_exp"]).merge(f, on=["gsis_id", "week"], how="left")
    out["xfp_diff"] = out["half_ppr"] - out["half_ppr_exp"]
    return out


# ---------------------------------------------------------------- execution
@dataclass
class Result:
    query: Query
    df: pd.DataFrame
    restated: str
    fmt: dict = _field(default_factory=dict)
    warnings: list[str] = _field(default_factory=list)
    note: str = ""


def run(pw: pd.DataFrame, q: Query, mine: set | None = None,
        rostered: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Apply the query to the player-week table. Returns (rows, warnings)."""
    warns: list[str] = []
    if pw is None or pw.empty:
        return pd.DataFrame(), ["No weekly data loaded."]
    d = pw
    if q.weeks:
        d = d[d["week"].isin(list(q.weeks))]
    if q.positions:
        d = d[d["pos"].isin(list(q.positions))]
    if q.teams:
        d = d[d["team"].isin(list(q.teams))]
    if q.scope == "mine":
        d = d[d["gsis_id"].isin(mine or set())]
        if not (mine or set()):
            warns.append("Your roster was not available, so the roster filter did nothing.")
    elif q.scope in ("fa", "rostered"):
        owned = set(rostered or {})
        if not owned:
            warns.append("League rosters were not available, so the ownership filter did nothing.")
        elif q.scope == "fa":
            d = d[~d["gsis_id"].isin(owned)]
        else:
            d = d[d["gsis_id"].isin(owned)]
    if d.empty:
        return pd.DataFrame(), warns + ["Nothing matched those filters."]

    f = q.field
    need = [c for c in (f.col, f.num, f.den, *f.show) if c and c not in d.columns]
    if need:
        return pd.DataFrame(), warns + [f"`{f.label}` needs {', '.join(sorted(set(need)))}, "
                                        "which this season's data does not carry."]

    sum_cols = sorted({c for c in (f.col, f.num, f.den, *f.show) if c} | {"one"})
    g = d.sort_values("week").groupby("gsis_id")
    rows = g[sum_cols].sum()
    rows["games"] = g["one"].size()
    meta = g.agg(player=("player", "last"), pos=("pos", "last"), team=("team", "last"))
    rows = meta.join(rows).reset_index()

    if f.kind == "rate":
        den = rows[f.den].replace(0, np.nan)
        rows["value"] = rows[f.num] / den
        if q.min_den > 0:
            before = len(rows)
            rows = rows[rows[f.den] >= q.min_den]
            cut = before - len(rows)
            if cut:
                warns.append(f"{cut} player{'s' if cut != 1 else ''} below "
                             f"{q.min_den:g} {f.den_label or f.den} left out — a rate off a "
                             "handful of chances is noise, not a reading.")
    elif f.kind == "per_game":
        rows["value"] = rows[f.col] / rows["games"].replace(0, np.nan)
    elif f.kind == "mean":
        rows["value"] = d.groupby("gsis_id")[f.col].mean().reindex(rows["gsis_id"]).to_numpy()
    elif f.kind == "max":
        rows["value"] = d.groupby("gsis_id")[f.col].max().reindex(rows["gsis_id"]).to_numpy()
    else:
        rows["value"] = rows[f.col]

    rows = rows.dropna(subset=["value"])
    if rows.empty:
        return pd.DataFrame(), warns + [f"No player has a {f.label} to report in that window."]
    rows = rows.sort_values("value", ascending=q.ascending)
    rows["rank"] = range(1, len(rows) + 1)

    keep = ["rank", "player", "pos", "team", "games", "value", *[c for c in f.show if c in rows.columns]]
    out = rows[keep].head(max(1, q.top)).reset_index(drop=True)
    return out.rename(columns={"value": f.key}), warns


def run_table(df: pd.DataFrame, q: Query, xwalk: pd.DataFrame | None = None,
              mine: set | None = None, rostered: dict | None = None,
              pos_map: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Apply a query to a raw nflverse table the catalogue pointed at.

    Every table names its own keys differently — nextgen calls the player `player_gsis_id`,
    PFR and the snap counts use `pfr_player_id` — so the columns to group and filter on are
    read from the schema rather than assumed.
    """
    from . import catalog

    warns: list[str] = []
    rec = catalog.schema().get(q.source, {})
    if df is None or df.empty:
        return pd.DataFrame(), [f"`{q.source}` came back empty."]
    col = q.field.col
    if col not in df.columns:
        return pd.DataFrame(), [f"`{col}` is not in `{q.source}` for this season."]

    key = rec.get("player_key") or rec.get("team_key")
    if not key or key not in df.columns:
        return pd.DataFrame(), [f"`{q.source}` has no player or team key to group by."]
    name_col = rec.get("name_key") if rec.get("name_key") in df.columns else key
    team_col = rec.get("team_key") if rec.get("team_key") in df.columns else None
    week_col = rec.get("week_key") if rec.get("week_key") in df.columns else None
    pos_col = catalog.pos_key(q.source)
    pos_col = pos_col if pos_col in df.columns else None

    d = df.copy()

    # Most of these tables carry no position column — nextgen and the PFR advanced splits
    # identify a player and nothing else — so positions are resolved through the ids we
    # already hold rather than abandoned. Without this, "top 5 WR by average separation"
    # came back led by a tight end.
    gid = None
    if key in ("player_id", "gsis_id", "player_gsis_id"):
        gid = d[key]
    elif key in ("pfr_player_id", "pfr_id") and xwalk is not None and not xwalk.empty:
        xw = xwalk.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("pfr_id")
        gid = d[key].map(dict(zip(xw["pfr_id"], xw["gsis_id"])))

    if week_col:
        wk = pd.to_numeric(d[week_col], errors="coerce")
        # nflverse ships a week 0 row in the nextgen tables holding the SEASON TOTAL.
        # Left in, every season-long sum counts each player twice over — Kenneth Walker's
        # rush yards over expected read 251 when the real figure for the weeks played
        # was 125.6, which is week 0 plus the two weeks that make it up.
        pre = len(d)
        d = d[wk != 0]
        if len(d) < pre and not q.weeks:
            warns.append(f"Dropped {pre - len(d)} season-total rows (week 0) that "
                         f"`{q.source}` carries alongside the weekly ones.")
        if q.weeks:
            d = d[pd.to_numeric(d[week_col], errors="coerce").isin(list(q.weeks))]
        gid = gid.loc[d.index] if gid is not None else None
    elif q.weeks:
        warns.append(f"`{q.source}` is not weekly, so the week filter was ignored.")
    if q.positions and set(q.positions) != set(POS_ALL):
        if pos_col:
            keep_pos = d[pos_col].astype(str).str.upper().isin(list(q.positions))
            d, gid = d[keep_pos], (gid[keep_pos] if gid is not None else None)
        elif gid is not None and pos_map:
            keep_pos = gid.map(pos_map).isin(list(q.positions))
            d, gid = d[keep_pos], gid[keep_pos]
        else:
            warns.append(f"`{q.source}` carries no position column and its players could not "
                         "be matched to one, so every position is shown.")
    if q.teams and team_col:
        d = d[d[team_col].map(_canon_team).isin(list(q.teams))]

    # Ownership filters need gsis ids too — same resolution as the position filter above.
    if q.scope:
        if gid is None:
            warns.append(f"`{q.source}` is not keyed on a player id we can match to your "
                         "league, so the roster filter was ignored.")
        elif q.scope == "mine":
            d = d[gid.isin(mine or set())]
        elif q.scope == "fa":
            d = d[~gid.isin(set(rostered or {}))]
        elif q.scope == "rostered":
            d = d[gid.isin(set(rostered or {}))]

    if d.empty:
        return pd.DataFrame(), warns + ["Nothing matched those filters."]

    d = d.assign(_v=pd.to_numeric(d[col], errors="coerce"))
    g = d.groupby(key, dropna=True)
    agg = {"total": "sum", "mean": "mean", "max": "max", "per_game": "sum"}.get(q.field.kind, "sum")
    rows = g["_v"].agg(agg).rename("value").reset_index()
    rows["games"] = g["_v"].size().to_numpy()
    if q.field.kind == "per_game":
        rows["value"] = rows["value"] / rows["games"].replace(0, np.nan)
    meta = g.agg(player=(name_col, "last"), **({"team": (team_col, "last")} if team_col else {}),
                 **({"pos": (pos_col, "last")} if pos_col else {}))
    rows = rows.merge(meta.reset_index(), on=key, how="left")

    rows = rows.dropna(subset=["value"]).sort_values("value", ascending=q.ascending)
    if rows.empty:
        return pd.DataFrame(), warns + [f"No values for `{col}` in that window."]
    rows["rank"] = range(1, len(rows) + 1)
    keep = ["rank", "player"] + [c for c in ("pos", "team") if c in rows.columns] + ["games", "value"]
    out = rows[keep].head(max(1, q.top)).reset_index(drop=True)
    return out.rename(columns={"value": q.field.key}), warns


def answer(pw: pd.DataFrame, text: str, weeks_available: tuple[int, ...] = (),
           mine: set | None = None, rostered: dict | None = None,
           loader=None, xwalk: pd.DataFrame | None = None) -> Result:
    """Parse, run, and hand back a frame plus the sentence that says how it was read.

    `loader(table) -> DataFrame` fetches an nflverse table when the question names a stat
    outside the curated set. Without it, such a question is answered with a pointer to
    where the stat lives rather than a number.
    """
    q = parse(text, weeks_available)
    if q.source == "pw":
        df, warns = run(pw, q, mine=mine, rostered=rostered)
    elif loader is None:
        df, warns = pd.DataFrame(), [
            f"`{q.field.col}` lives in the nflverse `{q.source}` table, which is not loaded here."]
    else:
        pos_map = (dict(zip(pw["gsis_id"], pw["pos"]))
                   if pw is not None and not pw.empty and "pos" in pw.columns else {})
        try:
            df, warns = run_table(loader(q.source), q, xwalk=xwalk, mine=mine,
                                  rostered=rostered, pos_map=pos_map)
        except Exception as e:
            df, warns = pd.DataFrame(), [f"Could not load `{q.source}`: {type(e).__name__}: {e}"]
    if q.unmatched:
        warns.append("Ignored: " + ", ".join(q.unmatched) + ".")
    return Result(query=q, df=df, restated=restate(q), fmt={q.field.key: q.field.fmt},
                  warnings=warns, note=q.field.note)


def restate(q: Query) -> str:
    """Plain English, so a misparse is obvious before the numbers are believed."""
    f = q.field
    who = "/".join(q.positions) if len(q.positions) < 4 else "all skill players"
    bits = [f"{'Bottom' if q.ascending else 'Top'} {q.top} {who} by {f.label}"]
    if q.teams:
        bits.append("on " + "/".join(q.teams))
    if q.scope == "mine":
        bits.append("from your roster")
    elif q.scope == "fa":
        bits.append("among free agents")
    elif q.scope == "rostered":
        bits.append("among rostered players")
    bits.append("over " + q.window_label)
    s = ", ".join(bits)
    if q.min_den > 0 and f.kind == "rate":
        s += f", minimum {q.min_den:g} {f.den_label or f.den}"
    if q.source != "pw":
        s += f" (from nflverse `{q.source}`)"
    return s + "."


# ---------------------------------------------------------------- UI helpers
# Supporting columns get short headers, matching mega/ui.py's house style. Anything not
# named here falls through to the column code, which is ugly but never wrong.
HEADERS = {
    "rank": "#", "player": "PLAYER", "pos": "POS", "team": "TM", "games": "G",
    "offense_snaps": "SNAPS", "team_snaps": "TM SNAPS", "targets": "TGT",
    "team_targets": "TM TGT", "receptions": "REC", "receiving_yards": "REC YDS",
    "receiving_first_downs": "1D", "receiving_air_yards": "AIR", "routes": "RTE",
    "carries": "CAR", "team_carries": "TM CAR", "rushing_yards": "RUSH YDS",
    "completions": "COMP", "attempts": "ATT", "passing_yards": "PASS YDS",
}


def headers(q: Query) -> dict[str, str]:
    """Column -> table header for one query's result, metric included."""
    return {**HEADERS, q.field.key: q.field.label.upper()}


EXAMPLES = (
    "list WRs by snap %",
    "red zone carries by RB",
    "top 10 RB by goal line carries",
    "top 10 RB by targets last 3 weeks",
    "who leads the Rams in target share",
    "top 15 WR by yards per route run",
    "best free agent WR by points per game",
    "worst catch rate, min 20 targets",
    "top 10 WR by average separation",
    "top 10 RB by rush yards over expected",
    "top 12 QB by completion percentage above expectation",
    "my team by points over expected",
)


def catalogue() -> pd.DataFrame:
    """Every metric the box understands, for the 'what can I ask' panel."""
    return pd.DataFrame([
        {"stat": f.label, "say": f.aliases[0],
         "also": ", ".join(f.aliases[1:5]), "what it is": f.note}
        for f in FIELDS
    ])
