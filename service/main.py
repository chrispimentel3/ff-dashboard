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

import ctypes
import gc
import os
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PydField

from mega import ask as ASK
from mega import data
from mega import sim as SIM
from mega import trade_league as TL
from mega.config import MY_TEAM

try:
    import nflreadpy as nfl
    # nflreadpy keeps every table it loads in memory for a day by default. On a 512MB host
    # that is the wrong place for it: disk costs nothing here, and a re-read is milliseconds.
    from nflreadpy import config as _nfl_cfg
    _nfl_cfg.update_config(cache_mode="filesystem")
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

# Give freed memory back to the OS after each request. glibc keeps what a request freed
# (the league build and the Ask loaders each spike 100MB+ and drop it) and Render counts it
# against the 512MB limit as if still in use. Linux-only; a no-op anywhere else. Pairs with
# MALLOC_ARENA_MAX=2 and ARROW_DEFAULT_MEMORY_POOL=system in render.yaml, which route
# pyarrow's buffers through the same malloc so this reaches them too.
try:
    _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
except (OSError, AttributeError):
    _malloc_trim = None


@app.middleware("http")
async def _release_memory(request, call_next):
    response = await call_next(request)
    if _malloc_trim is not None and request.url.path != "/health":
        gc.collect()
        _malloc_trim(0)
    return response


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


PW_KEEP = 4             # player-week tables held at once: this season plus a few past ones


def _player_week(season: int) -> pd.DataFrame:
    held = [k for k in _cache if k.startswith("pw:")]
    if f"pw:{season}" not in held and len(held) >= PW_KEEP:
        # drop the oldest-fetched past season; the current one is asked about most
        cur = f"pw:{_current_season()}"
        old = sorted((k for k in held if k != cur), key=lambda k: _cache[k][0])
        if old:
            _cache.pop(old[0], None)
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


ENGINE_TTL = 30 * 60      # matches app.py's _trade_engine/_trade_pool cache TTL
SEASON_MODEL_TTL = 6 * 3600  # matches app.py's _season_model cache TTL
SIM_TOP = 12              # matches app.py's SIM_TOP — only the best candidates get simulated


def _league(season: int) -> dict:
    """The priced league both the engine and the title-odds model are built from — built
    once, since building it is the heaviest step a trade search takes."""
    def build():
        from mega.trade_league import build_league
        from mega.yahoo import cached_rosters
        return build_league(season, cached_rosters())
    return _cached(f"league:{season}", ENGINE_TTL, build)


def _trade_engine(season: int):
    def build():
        from mega.trade_engine import create_engine
        from mega.trade_league import engine_config

        league = _league(season)
        if not league["teams"]:
            return None, league["report"]
        return create_engine(league, engine_config()), league["report"]
    return _cached(f"trade_engine:{season}", ENGINE_TTL, build)


def _trade_pool(season: int) -> pd.DataFrame:
    def build():
        from mega.intel import _valued_rosters
        from mega.trade_league import _pid
        from mega.yahoo import cached_rosters

        r = _valued_rosters(cached_rosters())
        if r.empty:
            return r
        r = r.sort_values("value", ascending=False)
        who = r["team"].map(lambda t: "yours" if str(t) == MY_TEAM else str(t))
        r["label"] = r["player"] + " · " + r["pos"].astype(str) + " · " + who
        r["pid"] = [_pid(row) for _, row in r.iterrows()]
        r["mine"] = r["team"].astype(str) == MY_TEAM
        return r[["label", "player", "pos", "team", "value", "pid", "mine"]]
    return _cached(f"trade_pool:{season}", ENGINE_TTL, build)


def _season_model(season: int, next_week: int):
    def build():
        from mega.config import DATA
        from mega.yahoo import cached_fixtures, cached_scores

        sc = cached_scores()
        st_path = DATA / "yahoo_standings.csv"
        if sc is None or sc.empty or not st_path.is_file():
            return None
        stand = pd.read_csv(st_path)
        ppw = sc.groupby("team")["points"].mean().to_dict()
        left = range(int(next_week), 15)
        try:
            fx = cached_fixtures()
        except Exception:
            fx = None
        return SIM.from_league(sc, stand, ppw, left, fixtures=fx)
    return _cached(f"season_model:{season}:{next_week}", SEASON_MODEL_TTL, build)


def _trade_odds(season: int, next_week: int, rows: list[tuple]) -> dict:
    """(key, partner, dMe, dThem) -> what the deal does to both teams' playoff chances —
    same 1,500-season simulation app.py's own trade search runs, on the same dice before
    and after so an offer that changes nothing reads as exactly zero."""
    s = _season_model(season, next_week)
    if s is None or MY_TEAM not in s.teams:
        return {}
    out = {}
    for key, partner, d_me, d_them in rows:
        if partner not in s.teams:
            continue
        out[key] = SIM.trade_delta(s, MY_TEAM, partner, float(d_me), float(d_them), n=1500)
    return out


LIVE_SEASONS = 3000        # half the offline 6,000: a live request has to answer in seconds
# On unless TITLE_ODDS_LIVE=0, which falls back to the team-level playoff odds. It was off
# at first because the model cost ~125MB (it stored its dice); it now regenerates them from
# the seed and costs ~10MB. Measured totals are in docs/service_memory.md.
TITLE_ODDS_LIVE = os.environ.get("TITLE_ODDS_LIVE", "1") != "0"


def _title_model(season: int, next_week: int):
    """HANDOFF v1.3 title-odds model (mega/title_odds.py), cached like the engine."""
    def build():
        from mega import title_odds as T
        return T.build(season, next_week, league=_league(season), n=LIVE_SEASONS)
    return _cached(f"title_model:{season}:{next_week}", ENGINE_TTL, build)


def _title_odds_rows(season: int, next_week: int, eng, my_id, results: list,
                     swap_ids: list | None = None) -> dict:
    """index -> both teams' change in playoff AND title odds for each offer, from player-
    level draws on common random numbers — the same model the offer cards use, so the
    search and the cards can't disagree. A netted 2-for-1 is measured against the free-
    swap roster (`swap_ids`), as the cards measure it."""
    from mega import trade_engine as te
    tm = _title_model(season, next_week)
    if tm is None:
        return {}
    ctx = eng.ctx
    my_name = ctx.teams[my_id]["name"]
    out = {}
    for i, r in enumerate(results):
        core = te.evaluate_core(ctx, my_id, r["partner"]["id"], r["giveIds"], r["getIds"])
        partner = r["partner"]["name"]
        ir = lambda team, tid: [p for p in tm.rosters.get(team, []) if p not in ctx.teams[tid]["roster"]]
        after = {my_name: list(core["_after"]["me"].ids) + ir(my_name, my_id),
                 partner: list(core["_after"]["them"].ids) + ir(partner, r["partner"]["id"])}
        if r.get("netted") and swap_ids:
            d = tm.delta_vs({my_name: list(swap_ids) + ir(my_name, my_id)}, after, (my_name, partner))
        else:
            d = tm.delta(after, (my_name, partner))
        us, them = d.get(my_name, {}), d.get(partner, {})
        out[i] = {"d_playoffs": us.get("d_playoffs"), "their_d_playoffs": them.get("d_playoffs"),
                  "d_title": us.get("d_title"), "their_d_title": them.get("d_title"),
                  "title_noise": bool(us.get("noise", True)),
                  "their_tag": SIM.partner_tag(them.get("p_playoffs", 0.0)),
                  "arms_rival": SIM.arms_rival(us.get("d_playoffs", 0.0), them.get("d_playoffs", 0.0),
                                               us.get("p_playoffs", 0.0), them.get("p_playoffs", 0.0))}
    return out


class AskRequest(BaseModel):
    text: str = PydField(min_length=1, max_length=300)
    seasons: list[int] = PydField(default_factory=list)


class TradeSearchRequest(BaseModel):
    pid: str
    mine: bool
    flags: list[str] = PydField(default_factory=lambda: ["LIKELY", "EXPLOIT", "NEEDS_PITCH"])
    two_player: bool = True
    order: str = "accept"  # "accept" (most likely accepted) | "gain" (best for me)


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


@app.get("/trade-pool")
def trade_pool():
    """Everyone rostered with a trade value, for the "pick a player" search box."""
    season = _current_season()
    pool = _trade_pool(season)
    if pool.empty:
        return []
    return pool.to_dict(orient="records")


@app.post("/trade-search")
def trade_search(req: TradeSearchRequest):
    """"Trade around one player" — the live search app.py's own Trades tab runs, wrapped
    as an endpoint so the statically-generated mega-bowl-web Trades page can offer it too
    (see mega/trades.py's docstring for why this couldn't just be precomputed: the question
    space is one pick out of the whole league, not a fixed list)."""
    season = _current_season()
    next_week = data.current_week(season, 1)
    eng, rep = _trade_engine(season)
    if eng is None:
        raise HTTPException(status_code=503, detail="League rosters unavailable this run.")
    me = rep.get("my_team_id")
    if me is None:
        raise HTTPException(status_code=503, detail="Could not resolve your team's roster.")

    shapes = ("1-for-1", "2-for-1") if req.two_player else ("1-for-1",)
    flags = tuple(f.upper().replace(" ", "_") for f in req.flags) or ("LIKELY", "EXPLOIT", "NEEDS_PITCH")

    try:
        if req.mine:
            out = TL.find_from_my_player(eng, me, [req.pid], includeFlags=list(flags),
                                         shapes=list(shapes), topN=40)
        else:
            out = TL.find_for_their_player(eng, me, req.pid, include_flags=flags,
                                           shapes=shapes, top_n=40)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    results, swap_ids = TL.net_free_swap(eng, me, out["results"])
    if req.order == "accept":
        rank = {"LIKELY": 0, "NEEDS_PITCH": 1, "EXPLOIT": 2, "LONGSHOT": 3}
        results = sorted(results, key=lambda r: (rank.get(r["flag"], 9), -r["dMe"]))
    else:
        results = sorted(results, key=lambda r: -r["dMe"])

    odds = {}
    if TITLE_ODDS_LIVE:
        try:
            odds = _title_odds_rows(season, next_week, eng, me, results[:SIM_TOP], swap_ids)
        except Exception:
            odds = {}
    if not odds:
        keyrows = [(i, r["partner"]["name"], r["dMe"], r["dThem"]) for i, r in enumerate(results[:SIM_TOP])]
        try:
            odds = _trade_odds(season, next_week, keyrows)
        except Exception:
            odds = {}

    rows = []
    for i, r in enumerate(results):
        o = odds.get(i, {})
        rows.append({
            "partner": r["partner"]["name"],
            "shape": r["shape"],
            "give": [p["name"] for p in r["give"]],
            "get": [p["name"] for p in r["get"]],
            "d_me": round(r["dMe"], 2),
            "d_them": round(r["dThem"], 2),
            "mkt_ratio": round(r["market"]["ratio"], 2),
            "flag": r["flag"],
            "fa_add": r.get("fa_add") or [],
            "netted": bool(r.get("netted")),
            "odds": o.get("d_playoffs"),
            "their_odds": o.get("their_d_playoffs"),
            "title": o.get("d_title"),
            "their_title": o.get("their_d_title"),
            "title_noise": o.get("title_noise"),
            "watch": "arms a rival" if o.get("arms_rival") else (o.get("their_tag") or None),
            "i_would_start": [s["name"] for s in r["me"]["startersIn"]],
            "i_would_bench": [s["name"] for s in r["me"]["startersOut"]],
            "they_would_start": [s["name"] for s in r["them"]["startersIn"]],
            "they_would_bench": [s["name"] for s in r["them"]["startersOut"]],
        })

    return {
        "evaluated": out["evaluated"], "padded": out["padded"], "matched": out["matched"],
        "sim_available": bool(odds),
        "rows": rows,
    }


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
