"""Yahoo player -> nflverse gsis_id resolution, with a match report.

Two-thirds of the rows the Yahoo roster scrape returns carry no position and no NFL
team (163 skill players on the last pull, 107 of them blank). Everything downstream —
lineup slotting, the matchup adjustment, positional strength — keys off those two
fields, so a player who fails to resolve doesn't error, he silently disappears from
every position-filtered view. That is how J.K. Dobbins ended up invisible rather
than losing a start/sit call.

Resolution order, highest priority first:

  1. data/id_overrides.csv   manual; a blank gsis_id means "intentionally unmapped"
  2. yahoo_id                exact, but ff_playerids is missing it for ~21% of
                             rostered skill players (almost all recent rookies)
  3. normalized name         + position, with NFL team as a tiebreaker only
  4. close name match        difflib, deliberately last and deliberately tight

Callers get back the frame plus a report; nothing here raises or exits. The app
surfaces unresolved players in a banner and drops them from position-filtered
views, which is the Streamlit equivalent of the handoff spec's non-zero exit.
"""
from __future__ import annotations

import difflib
import functools
import re
from pathlib import Path

import pandas as pd

from .config import DATA

OVERRIDES_CSV = DATA / "id_overrides.csv"

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def _s(v: object) -> str:
    """Blank-safe string. A real NaN is truthy, so `rec.get("pos") or ""` leaves the
    string "nan" behind and every downstream `if not pos` check silently passes."""
    t = str(v).strip()
    return "" if t.lower() in ("nan", "none", "<na>") else t


# One team vocabulary for the whole app: nflverse's, which is what the schedule, the
# defense-vs-position table and the logo file are keyed on. Yahoo says LAR/WSH/JAC
# and ff_playerids says GBP/KCC/SFO — before this, every Ram and Jaguar on a roster
# matched no opponent, so the matchup adjustment silently skipped them.
_TEAM_CANON = {
    "LAR": "LA", "RAM": "LA", "STL": "LA", "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI",
    "GBP": "GB", "KCC": "KC", "LVR": "LV", "OAK": "LV", "NEP": "NE", "NOS": "NO",
    "SFO": "SF", "TBB": "TB", "SDC": "LAC", "SD": "LAC", "BLT": "BAL", "CLV": "CLE",
    "HST": "HOU",
}


def canon_team(t: object) -> str:
    """Any team abbreviation -> nflverse's (LA, WAS, JAX, GB, ...). Blank stays blank."""
    v = _s(t).upper()
    return _TEAM_CANON.get(v, v)


# ff_playerids spells kickers PK and punters PN; Yahoo says K. The skill-player
# filters test `pos not in ("K", "DEF")`, so an un-normalized PK walks straight into
# the flex pool.
_POS_ALIAS = {"PK": "K", "PN": "P", "DST": "DEF", "D/ST": "DEF", "DEF/ST": "DEF"}

METHODS = ("override", "id", "name", "fuzzy")


def canon_pos(p: object) -> str:
    """Canonical position label: Yahoo's vocabulary, not nflverse's."""
    v = _s(p).upper()
    return _POS_ALIAS.get(v, v)


def norm(s: object) -> str:
    """Lowercase, strip punctuation and generational suffixes. Matches nflverse merge_name."""
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"[^a-z ]", " ", s).strip()


def _id_str(v: object) -> str:
    """Ids read out of a CSV arrive as floats — "40896.0" joins to nothing."""
    t = _s(v)
    return t[:-2] if t.endswith(".0") else t


@functools.lru_cache(maxsize=1)
def overrides() -> pd.DataFrame:
    """yahoo_id,name,gsis_id,note — a blank gsis_id marks a player we never expect to map.

    Keyed by yahoo_id where there is one; `name` covers the rows Yahoo ships without
    an id at all.
    """
    if not OVERRIDES_CSV.exists():
        return pd.DataFrame(columns=["yahoo_id", "name", "gsis_id", "note"])
    df = pd.read_csv(OVERRIDES_CSV, dtype=str).fillna("")
    for c in ("yahoo_id", "name", "gsis_id", "note"):
        if c not in df.columns:
            df[c] = ""
    for c in ("yahoo_id", "name", "gsis_id"):
        df[c] = df[c].astype(str).str.strip()
    return df[(df["yahoo_id"] != "") | (df["name"] != "")]


@functools.lru_cache(maxsize=1)
def team_defenses() -> dict[str, str]:
    """Nickname -> team abbreviation, e.g. {"bengals": "CIN"}.

    A team defense has no gsis_id, and on a bench row Yahoo gives it no position
    either — so "Bengals" reads as an unresolved skill player and lands in the
    lineup pool. Recognising the nickname keeps DSTs labelled and out of the
    skill views without a per-team override.
    """
    import nflreadpy as nfl

    try:
        t = nfl.load_teams().to_pandas()
    except Exception:
        return {}
    return {norm(n): str(a).upper() for n, a in zip(t["team_nick"], t["team_abbr"])}


@functools.lru_cache(maxsize=1)
def crosswalk() -> pd.DataFrame:
    """ff_playerids reduced to the join columns, with both name keys normalized."""
    import nflreadpy as nfl

    df = nfl.load_ff_playerids().to_pandas()
    keep = [c for c in ("name", "merge_name", "gsis_id", "pfr_id", "yahoo_id",
                        "sleeper_id", "position", "team") if c in df.columns]
    x = df[keep].copy()
    x["yahoo_key"] = pd.to_numeric(x.get("yahoo_id"), errors="coerce").astype("Int64").astype(str)
    x["norm"] = x["name"].map(norm)
    if "merge_name" in x.columns:
        # merge_name is already normalized upstream, but not identically to ours.
        x["norm_merge"] = x["merge_name"].map(norm)
    else:
        x["norm_merge"] = x["norm"]
    x["team_key"] = x.get("team", "").map(canon_team)
    x["pos_key"] = x.get("position", "").astype(str).str.upper().str.strip()
    return x


def _name_agrees(a: object, b: object, cutoff: float = 0.6) -> bool:
    """Loose check that two names refer to the same person.

    Deliberately loose — it exists to catch an id pointing at an unrelated player,
    not to adjudicate spellings. Nicknames ("Cam"/"Cameron") stay above the line.
    """
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return True
    if na == nb or na.split()[-1:] == nb.split()[-1:]:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= cutoff


def _pick(hits: pd.DataFrame, pos: str, team: str) -> tuple[pd.DataFrame, bool]:
    """Narrow multiple candidates by position, then team. Returns (hits, ambiguous)."""
    if len(hits) <= 1:
        return hits, False
    if pos:
        by_pos = hits[hits["pos_key"] == pos.upper()]
        if not by_pos.empty:
            hits = by_pos
    if len(hits) > 1 and team:
        by_team = hits[hits["team_key"] == canon_team(team)]
        if not by_team.empty:
            hits = by_team
    # Players change teams, so an unresolved tie is genuinely ambiguous, not a
    # coin flip we should quietly win.
    return hits, len(hits) > 1


def resolve(players: pd.DataFrame, name_col: str = "player") -> tuple[pd.DataFrame, dict]:
    """Attach gsis_id/pfr_id and backfill pos + nfl_team. Never raises.

    Adds: gsis_id, pfr_id, matched_name, match_method, resolved, unmapped.
    `unmapped` marks players an override deliberately blanks (empty roster slots,
    team defenses) so they don't pollute the unresolved count.
    """
    ids = crosswalk()
    _ov = overrides()
    ov = _ov.set_index("yahoo_id")["gsis_id"].to_dict()
    ov_name = {norm(n): g for n, g in zip(_ov.get("name", []), _ov["gsis_id"]) if str(n).strip()}
    dsts = team_defenses()
    by_gsis = ids.drop_duplicates("gsis_id").set_index("gsis_id")
    norms = ids["norm"].dropna().unique().tolist()
    counts = dict.fromkeys(METHODS, 0)
    counts["unresolved"] = 0
    counts["unmapped"] = 0
    rows, misses = [], []

    for _, r in players.iterrows():
        rec = dict(r)
        raw_name = _s(rec.get(name_col)) or _s(rec.get("name"))
        pos = _s(rec.get("pos"))
        team = _s(rec.get("nfl_team"))
        yid = _id_str(rec.get("yahoo_id"))
        method, hit, unmapped = "", ids.iloc[0:0], False

        # 0 — things that are not players. An empty roster slot and a team defense
        # both resolve to nothing; counting them as failures would bury the real ones.
        if not raw_name or raw_name.lower() in ("(empty)", "empty", "-"):
            unmapped = True
        elif norm(raw_name) in dsts and not yid:
            rec["pos"] = pos or "DEF"
            rec["nfl_team"] = team or dsts[norm(raw_name)]
            unmapped = True

        # 1 — manual override
        if not unmapped and yid and yid in ov:
            gid = ov[yid]
            if not gid:
                unmapped = True
            elif gid in by_gsis.index:
                hit = by_gsis.loc[[gid]].reset_index()
                method = "override"
            else:  # override points at an id nflverse doesn't carry; honour it anyway
                rec["gsis_id"] = gid
                method = "override"

        if not method and not unmapped and raw_name and norm(raw_name) in ov_name:
            gid = ov_name[norm(raw_name)]
            if not gid:
                unmapped = True
            else:
                rec["gsis_id"] = gid
                hit = by_gsis.loc[[gid]].reset_index() if gid in by_gsis.index else hit
                method = "override"

        # 2 — yahoo_id, but only when the name agrees. A stale or mistyped id is
        # silently a *different player*: we would attach his gsis_id, his team and
        # his position to this roster row and never notice.
        if not method and not unmapped and yid:
            h = ids[ids["yahoo_key"] == yid]
            if not h.empty:
                cand, _ = _pick(h, pos, team)
                if not cand.empty and _name_agrees(raw_name, cand.iloc[0].get("name")):
                    hit, method = cand, "id"

        # 3 — normalized name, position then team as tiebreaks
        if not method and not unmapped and raw_name:
            n = norm(raw_name)
            h = ids[(ids["norm"] == n) | (ids["norm_merge"] == n)]
            if not h.empty:
                hit, ambiguous = _pick(h, pos, team)
                method = "" if ambiguous else "name"

        # 4 — close match, last resort and deliberately tight
        if not method and not unmapped and raw_name:
            close = difflib.get_close_matches(norm(raw_name), norms, n=1, cutoff=0.90)
            if close:
                h = ids[ids["norm"] == close[0]]
                hit, ambiguous = _pick(h, pos, team)
                method = "" if ambiguous else "fuzzy"

        if not hit.empty:
            top = hit.iloc[0]
            rec["gsis_id"] = _s(rec.get("gsis_id")) or top.get("gsis_id")
            rec["pfr_id"] = top.get("pfr_id")
            rec["matched_name"] = top.get("name")
            if not pos and _s(top.get("pos_key")):
                rec["pos"] = canon_pos(top["pos_key"])
            if not team and _s(top.get("team")):
                rec["nfl_team"] = canon_team(top["team"])
        else:
            rec.setdefault("pfr_id", None)
            rec.setdefault("matched_name", None)
            if not method:
                rec["gsis_id"] = _s(rec.get("gsis_id")) or None

        rec["nfl_team"] = canon_team(rec.get("nfl_team"))
        rec["unmapped"] = unmapped
        rec["match_method"] = method or ("unmapped" if unmapped else "unresolved")
        rec["resolved"] = bool(method)
        if unmapped:
            counts["unmapped"] += 1
        elif method:
            counts[method] += 1
        else:
            counts["unresolved"] += 1
            misses.append({
                "player": raw_name, "yahoo_id": yid, "pos": pos, "nfl_team": team,
                "team": str(rec.get("team") or ""),
            })
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["norm"] = out[name_col if name_col in out.columns else "name"].map(norm)
        # Mixed ""/numeric ids make Arrow guess a float column and fail on the blanks.
        for c in ("yahoo_id", "gsis_id", "pfr_id"):
            if c in out.columns:
                out[c] = out[c].map(_id_str).replace("", None).astype("string")
    report = {"total": len(out), **counts, "unresolved_rows": misses}
    return out, report


def report_line(rep: dict) -> str:
    """One-line human summary of a match report."""
    parts = [f"{rep.get(m, 0)} by {m}" for m in METHODS if rep.get(m)]
    if rep.get("unmapped"):
        parts.append(f"{rep['unmapped']} intentionally unmapped")
    return f"{rep.get('total', 0)} players · " + ", ".join(parts or ["none matched"])


if __name__ == "__main__":
    ros = pd.read_csv(DATA / "yahoo_rosters.csv", dtype=str).fillna("")
    skill = ros[~ros["pos"].isin(["K", "DEF"]) & ~ros["slot"].isin(["K", "DEF"])]
    out, rep = resolve(skill)
    print(report_line(rep))
    print(f"unresolved: {rep['unresolved']}")
    for m in rep["unresolved_rows"]:
        print("  ", m)
    print("\nbackfilled pos:", (skill["pos"].str.strip() == "").sum(), "->",
          (out["pos"].astype(str).str.strip() == "").sum())
    print("backfilled team:", (skill["nfl_team"].str.strip() == "").sum(), "->",
          (out["nfl_team"].astype(str).str.strip() == "").sum())
