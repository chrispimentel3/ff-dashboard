"""Live API behind mega-bowl-web's "Ask anything" page.

Wraps mega/ask.py's answer() — a deterministic natural-language query engine over the
curated player-week index, not a language model — with the loaders in mega/data.py.
This is the one piece of mega-bowl-web that can't be statically generated: the question
space is open-ended, so unlike every other page (precomputed JSON in ff-dashboard's
data/web/), this runs the same computation the Streamlit "Ask anything" tab does, on
request, kept warm across requests with a small in-memory TTL cache instead of Streamlit's
st.cache_data (there is no Streamlit run context here).

Run locally:  .venv/bin/uvicorn service.main:app --reload --port 8008
"""
from __future__ import annotations

import os
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PydField

from mega import ask as ASK
from mega import data

try:
    import nflreadpy as nfl
except ImportError:  # pragma: no cover
    nfl = None

app = FastAPI(title="Mega Bowl — Ask API")

_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

PW_TTL = 6 * 3600       # matches app.py's CACHE_TTL for the player-week index
ROSTER_TTL = 30 * 60    # matches app.py's ownership/roster cache TTL

_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: int, build):
    now = time.time()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    val = build()
    _cache[key] = (now, val)
    return val


def _current_season() -> int:
    try:
        return int(nfl.get_current_season())
    except Exception:
        return data.SEASON_DEFAULT


def _player_week(season: int) -> pd.DataFrame:
    return _cached(f"pw:{season}", PW_TTL, lambda: data.ask_player_week(season))


def _my_gsis_ids() -> list[str]:
    def build():
        roster_raw, _src = data.my_roster()
        gsis_ids, _dropped = data.my_gsis_ids(roster_raw)
        return gsis_ids
    return _cached("gsis_ids", ROSTER_TTL, build)


def _rostered() -> dict:
    def build():
        own, _fa = data.ownership()
        return {g: t for g, (t, _sl) in own.items()}
    return _cached("rostered", ROSTER_TTL, build)


class AskRequest(BaseModel):
    text: str = PydField(min_length=1, max_length=300)
    seasons: list[int] = PydField(default_factory=list)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/meta")
def meta():
    season = _current_season()
    pw = _player_week(season)
    weeks = sorted(int(w) for w in pw["week"].dropna().unique()) if not pw.empty else []
    return {
        "season": season,
        "weeks_available": weeks,
        "seasons_selectable": list(range(season, 2014, -1)),
        "examples": list(ASK.EXAMPLES),
    }


@app.get("/catalogue")
def catalogue():
    return ASK.catalogue().to_dict(orient="records")


@app.post("/ask")
def ask(req: AskRequest):
    season = _current_season()
    pw = _player_week(season)
    weeks = tuple(sorted(int(w) for w in pw["week"].dropna().unique())) if not pw.empty else ()

    try:
        res = ASK.answer(
            pw, req.text, weeks_available=weeks,
            mine=set(_my_gsis_ids()), rostered=_rostered(),
            loader=data.nflverse_table, xwalk=data.player_ids.crosswalk(),
            pw_loader=_player_week, default_season=season,
            default_seasons=tuple(sorted(set(req.seasons))),
        )
    except ASK.AskError as e:
        raise HTTPException(status_code=400, detail=str(e))

    multi = len(res.query.seasons) > 1
    hdr = ASK.headers(res.query)
    if multi:
        hdr = {**hdr, "change": "CHANGE", **{str(y): str(y) for y in res.query.seasons}}

    df = res.df
    columns = [{"key": c, "label": hdr.get(c, c.upper())} for c in df.columns if c != "_id"]

    def fmt_value(col: str, v):
        f = res.fmt.get(col)
        if f is None or v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return f.format(v)
        except (ValueError, TypeError):
            return None

    rows = []
    for _, r in df.iterrows():
        row = {c: (None if pd.isna(r[c]) else r[c]) for c in df.columns if c != "_id"}
        for c in res.fmt:
            if c in row:
                row[f"{c}_display"] = fmt_value(c, r[c])
        rows.append(row)

    return {
        "restated": res.restated,
        "columns": columns,
        "rows": rows,
        "warnings": res.warnings,
        "note": res.note,
        "empty": df.empty,
    }
