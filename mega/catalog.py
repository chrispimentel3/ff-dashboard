"""Every nflverse table and column, searchable by name.

The question box has a curated registry (mega/ask.py FIELDS) that knows the *right* way to
compute the couple of dozen stats worth asking about most — which denominator a rate takes,
what a sensible minimum volume is, which positions it applies to. This module is the other
half: the remaining ~1,300 columns across 27 tables, findable by name so a question about
something nobody curated still gets an answer.

The catalogue itself is built offline by tools/build_schema.py and committed as
data/nflverse_schema.json, so resolving a column costs a dict lookup rather than loading a
100 MB play-by-play file to discover that the column exists.

What comes back from here is honest but blunt: a generic sum or mean over a column whose
meaning nobody has vetted. Anything computed this way is labelled as such on screen.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass

import pandas as pd

from .config import DATA

SCHEMA_PATH = DATA / "nflverse_schema.json"

# Columns whose weekly values are already rates — summing them is meaningless, so a
# generic query averages instead and says so.
_RATE_HINT = re.compile(r"(?:^|_)(pct|percentage|share|rate|avg|average|mean|ratio|"
                        r"efficiency|epa_per|per_)(?:_|$)|^(?:avg|pct)_|_pct$|_pc$")
# Identifiers and labels: real columns, but not things to rank players by.
_NOT_A_STAT = re.compile(r"(_id|_key|_url|_name|_abbr|_code|_type|_desc|_text|_date|"
                         r"_time|_flag|^season$|^week$|^game_id$)$|^(id|name|team|player)$")

POS_KEYS = ("position", "pos", "position_group", "depth_chart_position")
# Loading these costs real time and memory; the UI warns before it does.
HEAVY = {"pbp", "depth_charts", "contracts", "participation"}


_POS_TABLE_ALL = {w for ws in {"WR": ("rec", "receiving"), "RB": ("rush", "rushing"),
                               "QB": ("pass", "passing"), "DEF": ("def",)}.values()
                  for w in ws}


@dataclass(frozen=True)
class Hit:
    table: str
    column: str
    score: float
    dtype: str
    numeric: bool
    is_rate: bool
    samples: tuple[str, ...] = ()
    rows: int = 0

    @property
    def label(self) -> str:
        return self.column.replace("_", " ")


@functools.lru_cache(maxsize=1)
def schema() -> dict:
    if not SCHEMA_PATH.is_file():
        return {}
    try:
        return json.loads(SCHEMA_PATH.read_text())
    except Exception:
        return {}


def pos_key(table: str) -> str | None:
    """Derived rather than stored, so adding a position alias needs no schema rebuild."""
    cols = schema().get(table, {}).get("columns", [])
    return next((c for c in POS_KEYS if c in cols), None)


# nflverse shorthand, expanded so a question and a column name meet in the middle. These
# are contractions, not prefixes — "average" does not start with "avg" — so no amount of
# prefix matching finds avg_separation from "average separation".
_ABBR = {
    "avg": "average", "pct": "percent", "percentage": "percent", "perc": "percent",
    "yds": "yards", "yd": "yards", "td": "touchdown", "tds": "touchdown",
    "att": "attempt", "atts": "attempt", "cmp": "completion", "comp": "completion",
    "tgt": "target", "tgts": "target", "int": "interception", "ints": "interception",
    "rec": "receiving", "recv": "receiving", "def": "defense", "defensive": "defense",
    "off": "offense", "offensive": "offense", "sep": "separation", "cush": "cushion",
    "temp": "temperature", "opp": "opponent", "exp": "expected", "expectation": "expected",
    "ret": "return", "fd": "first down", "1d": "first down", "rz": "red zone",
    "ay": "air yards", "pass": "passing", "rush": "rushing", "run": "rushing",
    "gm": "game", "st": "special teams", "ngs": "nextgen", "adot": "average depth target",
    "ypc": "yards per carry", "ypa": "yards per attempt", "ypt": "yards per target",
    "xfp": "expected fantasy points", "pts": "points", "fpts": "fantasy points",
    # acronyms that expand to several words — _norm re-splits, so these work like the rest
    "epa": "expected points added", "wpa": "win probability added",
    "cpoe": "completion percentage over expected", "ryoe": "rushing yards over expected",
    "wopr": "weighted opportunity rating", "yac": "yards after catch",
    "aypa": "air yards per attempt", "dakota": "adjusted epa",
}


def _depl(w: str) -> str:
    """De-pluralise. Applied to the question and the column name alike, so folding
    "yards" to "yard" costs nothing as long as both sides go through it."""
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _norm(s: str) -> str:
    raw = re.sub(r"\s+", " ", str(s).lower().replace("_", " ").replace("-", " ")).strip()
    toks: list[str] = []
    for w in raw.split():
        toks.extend(_ABBR.get(w, w).split())      # an abbreviation may expand to several
    return " ".join(_depl(t) for t in toks)


def is_rate(col: str) -> bool:
    return bool(_RATE_HINT.search(col.lower()))


def _score(query: str, col: str) -> float:
    """How well a column name answers the question. 0 means no."""
    q, c = _norm(query), _norm(col)
    if not q:
        return 0.0
    if q == c:
        return 100.0
    qw, cw = q.split(), c.split()
    if not qw:
        return 0.0
    if c.startswith(q + " ") or c.endswith(" " + q):
        return 80.0 - 0.1 * len(c)
    if q in c:
        return 65.0 - 0.1 * len(c)
    # whole words first, then word-prefixes ("temp" for temperature, "comp" for completions)
    hit = sum(1 for w in qw if w in cw)
    pre = sum(1.0 for w in qw if w not in cw
              and any(len(w) >= 4 and (x.startswith(w) or w.startswith(x)) for x in cw))
    got = hit + pre
    if got >= len(qw):                       # every word of the question is accounted for
        return 55.0 - 0.1 * len(c) - 2.0 * (len(cw) - len(qw)) - 8.0 * pre
    if got:
        return 20.0 * got / len(qw)
    return 0.0


# Where a stat should be looked for first when several tables carry the same column name.
_TABLE_RANK = {"player_stats": 0, "snap_counts": 1, "ff_opportunity_weekly": 2,
               "nextgen_stats_receiving": 3, "nextgen_stats_rushing": 3,
               "nextgen_stats_passing": 3, "pfr_advstats_rec": 4, "pfr_advstats_rush": 4,
               "pfr_advstats_pass": 4, "pfr_advstats_def": 4, "injuries": 5,
               "rosters_weekly": 6, "rosters": 7, "players": 8, "team_stats": 9,
               "schedules": 10, "draft_picks": 11, "combine": 12, "ftn_charting": 13,
               "pbp": 14, "depth_charts": 15, "contracts": 16}


# Which tables belong to which position family. "receiving drops" should come back from
# pfr_advstats_rec, not from the quarterback table that also happens to carry the column.
_POS_TABLE = {
    "WR": ("rec", "receiving"), "TE": ("rec", "receiving"),
    "RB": ("rush", "rushing"), "QB": ("pass", "passing"), "DEF": ("def",),
}


def search(text: str, limit: int = 8, numeric_only: bool = True,
           min_score: float = 30.0, positions: tuple[str, ...] = ()) -> list[Hit]:
    """Columns whose name matches `text`, best first.

    `positions` nudges the answer toward the table for those positions when several carry
    a column of the same name.
    """
    want = {w for p in positions for w in _POS_TABLE.get(p, ())}
    out: list[Hit] = []
    for table, rec in schema().items():
        num = set(rec.get("numeric", []))
        for col in rec.get("columns", []):
            if _NOT_A_STAT.search(col):
                continue
            if numeric_only and col not in num:
                continue
            s = _score(text, col)
            if s < min_score:
                continue
            s -= 0.4 * _TABLE_RANK.get(table, 20)      # prefer the usual home of a stat
            if want:
                fam = {w for w in _POS_TABLE_ALL if w in table}
                if fam & want:
                    s += 6.0
                elif fam:
                    s -= 6.0                            # a different position's table
            out.append(Hit(table=table, column=col, score=s,
                           dtype=rec.get("dtypes", {}).get(col, ""), numeric=col in num,
                           is_rate=is_rate(col),
                           samples=tuple(rec.get("samples", {}).get(col, [])),
                           rows=int(rec.get("rows", 0))))
    out.sort(key=lambda h: (-h.score, h.table))
    # one hit per column name: the best-ranked table wins
    seen, keep = set(), []
    for h in out:
        if h.column in seen:
            continue
        seen.add(h.column)
        keep.append(h)
    return keep[:limit]


def load(table: str, season: int) -> pd.DataFrame:
    """Fetch one catalogued table, using the loader and variant the schema recorded."""
    import inspect

    import nflreadpy as nfl

    rec = schema().get(table)
    if not rec:
        raise KeyError(f"unknown table {table!r}")
    if table == "pbp":                              # goes through the disk-cached path
        from . import pbp as _pbp
        return _pbp.load(season).to_pandas()
    fn = getattr(nfl, rec["loader"])
    kw = {}
    if "seasons" in inspect.signature(fn).parameters:
        kw["seasons"] = [season]
    if rec.get("variant"):
        kw["stat_type"] = rec["variant"]
    df = fn(**kw)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def tables() -> pd.DataFrame:
    """Everything catalogued, for the 'what's in here' panel."""
    rows = [{"table": t, "rows": r.get("rows", 0), "columns": r.get("n_cols", 0),
             "joins on": r.get("player_key") or r.get("team_key") or "—",
             "weekly": "yes" if r.get("week_key") else "no",
             "loader": f"nfl.{r.get('loader')}()" + (f" [{r['variant']}]" if r.get("variant") else "")}
            for t, r in sorted(schema().items())]
    return pd.DataFrame(rows)


def columns(table: str | None = None, query: str = "") -> pd.DataFrame:
    """Browsable column list, optionally filtered — the 'what can I ask' long tail."""
    rows = []
    for t, rec in sorted(schema().items()):
        if table and t != table:
            continue
        num = set(rec.get("numeric", []))
        for c in rec.get("columns", []):
            if query and _score(query, c) < 30:
                continue
            rows.append({"column": c, "table": t, "kind": "number" if c in num else "text",
                         "rate?": "yes" if c in num and is_rate(c) else "",
                         "example": ", ".join(rec.get("samples", {}).get(c, [])[:2])})
    return pd.DataFrame(rows)
