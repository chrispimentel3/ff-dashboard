"""Pure-data Player lookup: a searchable index plus a full detail card for every player.

`_tab_lookup`'s render body only computes anything for whichever player the search box
currently has selected — nothing, in a headless export run. So instead of extracting that
tab function the way action_board.py/start_sit.py do, this module precomputes every
player's full card once, up front, from the same season-level tables app.py already builds
(each pure and cheap to slice per player). The frontend does the "searching" client-side
over this one export — same pattern as every other search/filter control in mega-bowl-web.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from mega import glossary as GL
from mega import lookup as LK
from mega import roles as RL


def _clean(v):
    """NaN/NaT/pd.NA -> None so json.dumps never sees a NaN."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _height(v) -> str | None:
    v = _clean(v)
    if v is None:
        return None
    try:
        v = int(float(v))
        return f"{v // 12}'{v % 12}\""
    except (TypeError, ValueError):
        return None


def build_index(idx: pd.DataFrame, gsis_list: set) -> list[dict]:
    if idx is None or idx.empty:
        return []
    return [
        {
            "gsis_id": r["gsis_id"],
            "name": r["name"],
            "pos": r["pos"],
            "team": r["team"] or None,
            "last_team": _clean(r.get("last_team")) or None,
            "label": r["label"],
            "mine": r["gsis_id"] in gsis_list,
        }
        for _, r in idx.iterrows()
    ]


def _ownership_info(raw: dict | None) -> dict:
    """Normalizes app.py's `_owner_badge(...)[1]` (which only sets the keys relevant to its
    `kind`) into a fixed shape every player's export carries, whether or not a league is on
    file. `_owner_badge` is the only place that knows about MY_TEAM — this module never
    imports `mega.config` at all."""
    raw = raw or {}
    kind = raw.get("kind", "unknown")
    return {
        "kind": kind,
        "team": raw.get("team"),
        "slot": raw.get("slot"),
        "waiver_until": raw.get("waiver_until"),
    }


def _bio_dict(b: dict) -> dict:
    jersey = _clean(b.get("jersey_number"))
    weight = _clean(b.get("weight"))
    college = _clean(b.get("college"))
    draft_no = _clean(b.get("draft_number"))
    entry_yr = _clean(b.get("entry_year"))
    headshot = b.get("headshot_url")
    return {
        "jersey_number": int(jersey) if jersey is not None else None,
        "age": _clean(b.get("age")),
        "height": _height(b.get("height")),
        "weight": int(weight) if weight is not None else None,
        "college": str(college).split(";")[0] if college is not None else None,
        "draft_number": int(draft_no) if draft_no is not None else None,
        "entry_year": int(entry_yr) if entry_yr is not None else None,
        "headshot_url": headshot if isinstance(headshot, str) and headshot.startswith("http") else None,
    }


def _card_rows(table: pd.DataFrame, gid: str, ppos: str, rk: dict[str, str]) -> list[dict]:
    if table.empty or gid not in set(table["gsis_id"]):
        return []
    r = table[table["gsis_id"] == gid].iloc[0]
    rows = []
    for col, label, fmt, means in LK.CARD.get(ppos, []):
        v = r.get(col)
        v = _clean(v)
        rows.append({
            "stat": label,
            "value_fmt": fmt.format(v) if v is not None else "—",
            "pos_rank": rk.get(col, "—"),
            "means": means,
        })
    return rows


def _role_section(role_ctx: dict | None, gid: str) -> dict | None:
    if not role_ctx or role_ctx["table"].empty:
        return None
    table, baselines = role_ctx["table"], role_ctx["baselines"]
    rows = table[table["gsis_id"] == gid]
    if rows.empty:
        return None
    r0 = rows.iloc[0]
    flags = r0.get("flags") if isinstance(r0.get("flags"), list) else []
    tags = r0.get("tags") if isinstance(r0.get("tags"), dict) else {}
    headline = GL.cell(r0.get("role"), flags, tags)
    src = r0.get("role_src")
    card = RL.card(table, baselines, gid)
    card_rows = []
    if not card.empty:
        role_avg_col = [c for c in card.columns if c.endswith(" avg") and c != f"NFL {r0.get('pos')} avg"]
        for _, cr in card.iterrows():
            role_avg_key = next((c for c in cr.index if c.endswith(" avg") and not c.startswith("NFL")), None)
            nfl_avg_key = next((c for c in cr.index if c.startswith("NFL ")), None)
            card_rows.append({
                "metric": cr["metric"],
                "him": _clean(cr.get("him")),
                "role_avg": _clean(cr.get(role_avg_key)) if role_avg_key else None,
                "vs_role": _clean(cr.get("vs role")),
                "nfl_avg": _clean(cr.get(nfl_avg_key)) if nfl_avg_key else None,
                "vs_nfl": _clean(cr.get("vs NFL")),
                "fmt": cr.get("_fmt"),
            })
    return {
        "headline": headline,
        "from_usage": src == "usage",
        "games": int(_clean(r0.get("games")) or 0),
        "rows": card_rows,
    }


def _this_week_section(
    pteam: str, ppos: str, sched: pd.DataFrame, next_week: int, dvp: pd.DataFrame,
    out_reason: str | None, blended_proj: pd.DataFrame, gid: str,
) -> dict | None:
    if not pteam:
        return None
    wk = sched[(sched["week"] == next_week) & ((sched["home_team"] == pteam) | (sched["away_team"] == pteam))]
    if wk.empty:
        return {"bye": True}
    g0 = wk.iloc[0]
    is_home = g0["home_team"] == pteam
    opp = g0["away_team"] if is_home else g0["home_team"]
    out = {"bye": False, "opponent": opp, "home": bool(is_home), "ease_rank": None,
           "out_reason": out_reason, "projection": None, "proj_source": None}
    if dvp is not None and not dvp.empty:
        m = dvp[(dvp["defense"] == opp) & (dvp["pos"] == ppos)]
        if not m.empty:
            out["ease_rank"] = int(_clean(m["ease_rank"].iloc[0]))
    if not out_reason and blended_proj is not None and not blended_proj.empty and "gsis_id" in blended_proj.columns:
        pr = blended_proj[blended_proj["gsis_id"] == gid]
        if not pr.empty and _clean(pr["proj"].iloc[0]) is not None:
            out["projection"] = round(float(pr["proj"].iloc[0]), 1)
            out["proj_source"] = pr["proj_source"].iloc[0]
    return out


def _game_log_rows(w, gid, ffo, snaps, sched, pfr_id, routes, ppos) -> list[dict]:
    log = LK.game_log(w, gid, ffo, snaps, sched, pfr_id, routes)
    if log.empty:
        return []
    cols = [c for c in LK.LOG_COLS.get(ppos, LK.LOG_COLS["WR"]) if c in log.columns]
    out = []
    for _, row in log[cols].iterrows():
        out.append({c: _clean(row[c]) for c in cols})
    return out


def build_player(
    gid: str, idx_row: pd.Series, cur: int,
    seasons: dict[int, dict],
    rosters_bio: pd.DataFrame,
    owner_raw: dict | None,
    role_ctx: dict | None,
    dvp: pd.DataFrame | None,
    blended_proj: pd.DataFrame | None,
    out_reason: str | None,
    sched_current: pd.DataFrame,
    next_week: int,
) -> dict:
    ppos, pteam = idx_row["pos"], idx_row["team"]
    b = LK.bio(gid, rosters_bio)
    nfl_status = LK.STATUS_WORDS.get(str(b.get("status") or ""), str(b.get("status") or "")) or None

    payload = {
        "gsis_id": gid,
        "name": idx_row["name"],
        "pos": ppos,
        "team": pteam or None,
        "last_team": _clean(idx_row.get("last_team")) or None,
        "bio": _bio_dict(b),
        "ownership": _ownership_info(owner_raw),
        "nfl_status": nfl_status,
        "out_reason": out_reason,
        "seasons": {},
    }

    for season, s in seasons.items():
        table = s["table"]
        if table.empty or gid not in set(table["gsis_id"]):
            continue
        r = table[table["gsis_id"] == gid].iloc[0]
        rk = LK.ranks(table, gid, ppos)
        usage_key, usage_label, usage_fmt = {
            "QB": ("att_pg", "Pass att/g", "{:.1f}"),
            "RB": ("rush_share", "Carry share", "{:.0%}"),
        }.get(ppos, ("tgt_share", "Target share", "{:.1%}"))
        usage_val = _clean(r.get(usage_key))

        season_payload = {
            "games": int(_clean(r.get("games")) or 0),
            "pts_pg": _clean(r.get("pts_pg")),
            "xfp_pg": _clean(r.get("xfp_pg")),
            "vs_exp_pg": _clean(r.get("vs_exp_pg")),
            "usage_label": usage_label,
            "usage_value": usage_val,
            "usage_fmt": usage_fmt.format(usage_val) if usage_val is not None else None,
            "ranks": {
                "pts_pg": rk.get("pts_pg"),
                "xfp_pg": rk.get("xfp_pg"),
                usage_key: rk.get(usage_key),
            },
            "card": _card_rows(table, gid, ppos, rk),
            "game_log": _game_log_rows(
                s["w"], gid, s.get("ffo"), s.get("snaps"), s.get("sched"),
                s.get("pfr_ids", {}).get(gid), s.get("routes"), ppos,
            ),
            "role": _role_section(role_ctx, gid) if season == cur else None,
            "this_week": (
                _this_week_section(pteam, ppos, sched_current, next_week, dvp, out_reason, blended_proj, gid)
                if season == cur else None
            ),
        }
        payload["seasons"][str(season)] = season_payload

    return payload


def build(
    idx: pd.DataFrame,
    gsis_list: set,
    owner_info: dict[str, dict],
    seasons: dict[int, dict],
    rosters_bio: pd.DataFrame,
    role_ctx: dict | None,
    dvp: pd.DataFrame | None,
    blended_proj: pd.DataFrame | None,
    out_by_norm: dict,
    sched_current: pd.DataFrame,
    next_week: int,
    cur: int,
    norm_fn,
) -> dict:
    """Builds the whole export: the searchable index plus every player's full detail.

    `seasons` is `{season: {"table": season_table_df, "w": weekly_stats_df, "ffo":...,
    "snaps":..., "sched":..., "routes":..., "pfr_ids": {gsis_id: pfr_id}}}` — the same
    per-season frames app.py already loads once and caches. `owner_info` is
    `{gsis_id: app.py's _owner_badge(...)[1]}` — this module never touches MY_TEAM itself,
    only app.py's `_owner_badge` does (see test_nav.py's
    test_the_player_page_touches_the_league_in_exactly_one_place).
    """
    if idx is None or idx.empty:
        return {"available": False, "index": [], "players": {}}

    players: dict[str, dict] = {}
    for _, row in idx.iterrows():
        gid = row["gsis_id"]
        out_reason = out_by_norm.get(norm_fn(row["name"]))
        try:
            players[gid] = build_player(
                gid, row, cur, seasons, rosters_bio, owner_info.get(gid),
                role_ctx, dvp, blended_proj, out_reason, sched_current, next_week,
            )
        except Exception as e:  # one bad player shouldn't blank the whole index
            players[gid] = {"gsis_id": gid, "name": row["name"], "pos": row["pos"],
                            "team": row["team"] or None, "error": str(e), "seasons": {}}

    return {
        "available": True,
        "index": build_index(idx, gsis_list),
        "players": players,
    }
