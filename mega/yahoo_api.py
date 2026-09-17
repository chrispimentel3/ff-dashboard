"""Adapter: turn `pull_league.py` output (data/yahoo_api/*.json, via the official
Yahoo API / YFPY) into the DataFrames the dashboard + tuesday.py consume.

Pure JSON parsing — no yfpy import needed here, so it stays robust to yfpy version
quirks. Every extractor is defensive: missing fields degrade, they don't crash.
"""
from __future__ import annotations

import difflib
import glob
import json
import re
from pathlib import Path

import pandas as pd

from .config import DATA, DRAFT_BOARD_NAME_BY_SEAT, SEAT_BY_TEAM, TEAM_BY_SEAT

API_DIR = DATA / "yahoo_api"


def _norm(s: object) -> str:
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"[^a-z ]", " ", s).strip()


def available() -> bool:
    return API_DIR.is_dir() and bool(list(API_DIR.glob("rosters_week_*.json")))


def _load(name_glob: str):
    hits = sorted(glob.glob(str(API_DIR / name_glob)))
    if not hits:
        return None
    return json.loads(Path(hits[-1]).read_text(encoding="utf-8"))


def week() -> int | None:
    w = _load("_week.json")
    if isinstance(w, dict):
        return w.get("week")
    m = re.search(r"rosters_week_(\d+)", " ".join(glob.glob(str(API_DIR / "rosters_week_*.json"))))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- helpers
def _walk(obj):
    """Yield every dict found anywhere in a nested json structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _first(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, "", []):
            return d[k]
    return default


def _player_name(p: dict) -> str:
    n = p.get("name")
    if isinstance(n, dict):
        return _first(n, "full", "ascii_full", default="") or f"{n.get('first','')} {n.get('last','')}".strip()
    return str(n or "")


def _seat_for(team_name: str) -> int | None:
    if team_name in SEAT_BY_TEAM:
        return SEAT_BY_TEAM[team_name]
    pool = {**{v: k for k, v in TEAM_BY_SEAT.items()},
            **{v: k for k, v in DRAFT_BOARD_NAME_BY_SEAT.items()}}
    hit = difflib.get_close_matches(team_name, list(pool), n=1, cutoff=0.6)
    return pool[hit[0]] if hit else None


# ---------------------------------------------------------------- frames
def rosters_df() -> pd.DataFrame:
    data = _load("rosters_week_*.json")
    if not data:
        return pd.DataFrame()
    rows = []
    # data is {team_name: Roster}
    items = data.items() if isinstance(data, dict) else []
    for team_name, roster in items:
        seat = _seat_for(team_name)
        players = []
        for d in _walk(roster):
            if "name" in d and ("player_id" in d or "player_key" in d or "display_position" in d):
                players.append(d)
        seen = set()
        for p in players:
            pid = p.get("player_id") or p.get("player_key")
            if pid in seen:
                continue
            seen.add(pid)
            sel = p.get("selected_position")
            slot = sel.get("position") if isinstance(sel, dict) else (sel or "")
            rows.append(dict(
                seat=seat, team=team_name, slot=slot,
                player=_player_name(p),
                pos=_first(p, "display_position", "primary_position", default=""),
                nfl_team=_first(p, "editorial_team_abbr", "editorial_team_full_name", default=""),
                yahoo_id=str(p.get("player_id") or ""),
                status=p.get("status") or "",
                pct_owned=_first((p.get("percent_owned") or {}) if isinstance(p.get("percent_owned"), dict) else {},
                                 "value", default=p.get("percent_owned")),
            ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["norm"] = df["player"].map(_norm)
        df["pos"] = df["pos"].str.upper().str.replace("W/R/T", "W/R", regex=False)
    return df


def standings_df() -> pd.DataFrame:
    data = _load("league_standings.json") or _load("league_teams.json")
    if not data:
        return pd.DataFrame()
    rows = []
    for d in _walk(data):
        if "team_standings" in d or ("name" in d and "waiver_priority" in d):
            ts = d.get("team_standings") or {}
            ot = ts.get("outcome_totals") or {}
            name = _player_name(d) if isinstance(d.get("name"), dict) else str(d.get("name", ""))
            rows.append(dict(
                team=name, seat=_seat_for(name),
                rank=ts.get("rank"),
                wins=ot.get("wins"), losses=ot.get("losses"), ties=ot.get("ties"),
                points_for=_first(ts, "points_for") or d.get("points_for"),
                points_against=_first(ts, "points_against") or d.get("points_against"),
                streak=(ts.get("streak") or {}).get("value") if isinstance(ts.get("streak"), dict) else ts.get("streak"),
                faab_balance=d.get("faab_balance"),
                moves=d.get("number_of_moves"), trades=d.get("number_of_trades"),
            ))
    df = pd.DataFrame(rows).drop_duplicates("team")
    return df.sort_values("rank", na_position="last").reset_index(drop=True) if not df.empty else df


def _matchup_teams(m: dict) -> list[dict]:
    """The two team dicts inside one matchup, de-duplicated."""
    out, seen = [], set()
    for d in _walk(m.get("teams")):
        if "name" not in d or "team_points" not in d:
            continue
        key = d.get("team_id") or d.get("team_key") or _player_name(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _points(d: dict, *keys: str) -> float | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            v = v.get("total")
        if v not in (None, ""):
            try:
                return round(float(v), 2)
            except (TypeError, ValueError):
                continue
    return None


def matchups_df() -> pd.DataFrame:
    """One row per team per week: what they scored and who beat them."""
    rows = []
    for path in sorted(API_DIR.glob("scoreboard_week_*.json")):
        m_wk = re.search(r"scoreboard_week_(\d+)", path.name)
        if not m_wk:
            continue
        wk = int(m_wk.group(1))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for m in _walk(data):
            if "teams" not in m:
                continue
            pair = _matchup_teams(m)
            if len(pair) != 2:
                continue
            final = str(m.get("status", "")).lower() == "postevent"
            for side, other in ((pair[0], pair[1]), (pair[1], pair[0])):
                name = _player_name(side) if isinstance(side.get("name"), dict) else str(side.get("name", ""))
                opp = _player_name(other) if isinstance(other.get("name"), dict) else str(other.get("name", ""))
                pf = _points(side, "team_points")
                pa = _points(other, "team_points")
                rows.append(dict(
                    week=wk, team=name, seat=_seat_for(name), opponent=opp,
                    points=pf, opp_points=pa,
                    proj=_points(side, "team_projected_points"),
                    result=("W" if pf > pa else "L" if pf < pa else "T")
                           if final and pf is not None and pa is not None else "",
                    final=final,
                    playoffs=bool(int(m.get("is_playoffs") or 0)),
                ))
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["week", "team"]).sort_values(["week", "team"]).reset_index(drop=True) if not df.empty else df


def transactions_df() -> pd.DataFrame:
    data = _load("transactions.json")
    if not data:
        return pd.DataFrame()
    rows = []
    for tx in _walk(data):
        if tx.get("type") not in ("add", "drop", "add/drop", "trade", "commish"):
            continue
        ts = tx.get("timestamp")
        when = pd.to_datetime(int(ts), unit="s", errors="coerce") if ts else pd.NaT
        for p in _walk(tx):
            if "name" not in p or "transaction_data" not in p:
                continue
            td = p["transaction_data"]
            td = td[0] if isinstance(td, list) and td else td
            td = td if isinstance(td, dict) else {}
            rows.append(dict(
                when=when, tx_type=tx.get("type"),
                move=td.get("type"),  # 'add' or 'drop'
                player=_player_name(p),
                team=_first(td, "destination_team_name", "source_team_name", default=""),
            ))
    df = pd.DataFrame(rows)
    return df.sort_values("when", ascending=False).reset_index(drop=True) if not df.empty else df


def settings_summary() -> dict:
    data = _load("league_settings.json")
    out: dict = {}
    if not data:
        return out
    for d in _walk(data):
        for k in ("scoring_type", "waiver_type", "waiver_rule", "uses_faab", "trade_end_date",
                  "playoff_start_week", "num_playoff_teams"):
            if k in d and k not in out:
                out[k] = d[k]
        if "roster_positions" in d and "roster_positions" not in out:
            rp = d["roster_positions"]
            slots = {}
            for r in _walk(rp):
                if "position" in r and "count" in r:
                    slots[r["position"]] = r["count"]
            out["roster_positions"] = slots
    return out


def rostered_norms() -> set[str]:
    r = rosters_df()
    return set(r["norm"]) if not r.empty else set()


def dump_csv() -> None:
    for name, fn in [("rosters", rosters_df), ("standings", standings_df),
                     ("transactions", transactions_df)]:
        df = fn()
        df.to_csv(DATA / f"yahoo_api_{name}.csv", index=False)
        print(f"  yahoo_api_{name:12s} {len(df):>4} rows")
    print("  settings:", settings_summary())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pull":
        import pull_league  # noqa

        pull_league.main()
    if not available():
        print(f"No API data yet. Run:  uv run python pull_league.py   (writes to {API_DIR})")
    else:
        print(f"week: {week()}")
        dump_csv()
        print("\nrosters sample:")
        print(rosters_df().head(20).to_string())
