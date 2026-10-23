"""Worth to them: the chance another Mega Bowl manager says yes to an offer, and why.

One function for every trade surface: the offer cards (mega/trade_theses.py), the live
search, the team search and the evaluator (mega/trade_league.py deal_odds). The rules are
docs/trade_logic_v2.md's, agreed with Chris 2026-10-22:

  * A manager judges what they get by NAME and by their LINEUP. The name side is the
    perceived rank they can see (consensus blended with box scores, trade_league `ecr`) on
    the market curve, adjusted for position (a starting RB outweighs a non-elite TE at equal
    points; only an elite QB/TE carries weight), track record (a rookie is weak currency, a
    recent top finish is a brand), and their NFL team (a fan holds their own team's players).
  * The best player wins: a package is worth less than the sum of its parts.
  * A throw-in that won't start for them, or is worth next to nothing, costs them a roster
    spot: it counts AGAINST the deal. A piece that starts where they're thin counts extra.
  * A team out of the race still plays for wins now: its lineup counts more, names less.
  * A deal too small to matter to either side gets ignored.

There is no data to fit any of this on (Yahoo shows no declined offers, and Chris trades by
text), so every constant marked (J) is judgment, tuned until the calibration set Chris
labelled passes (tests/test_trade_worth.py). The odds are shown as estimates.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FANDOM = ROOT / "config" / "fandom.json"

# who counts as established: a season at starter level in a 12-team league (8+ games) last
# year or in two of the last three, or a breakout at RB1/WR1/top-6 level this year (4+ games)
STARTER_CUT = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
STAR_CUT = {"QB": 3, "RB": 8, "WR": 8, "TE": 3}   # a finish this high in the last three: a brand
MIN_GAMES, MIN_GAMES_NOW = 8, 4

PERCEIVED_FLOOR = 0.6     # (J) an unproven player's name, as a share of a proven one's
STAR_MULT = 1.5           # (J) a recent top finish (Kelce TE2 2023, Hall RB7 2023) is still a brand
W_NEED = 1.2              # (J) logit when a piece they get starts at a spot they're thin at
POS_MULT = {"RB": 1.45, "WR": 1.0}            # (J) starting backs are the scarce asset
TIER1_MULT = 1.1          # (J) an RB1/WR1 (top 12 at the position league-wide) on top
ELITE_MULT, OTHER_QB, OTHER_TE = 1.1, 0.45, 0.35   # (J) only an elite QB/TE carries a name
ELITE_GAP = 1.5           # pts/game: the elite tier ends at the first drop this big in the top 4
FAN_MULT = 1.3            # (J) their own NFL team's players, on either side of the deal
K = 1.3                   # (J) package value = (sum v^K)^(1/K): the best player wins
SIT_CREDIT = 0.35         # (J) a piece they'd hold on the bench, swapped one for one
MIN_PIECE = 12.0          # (J) a throw-in worth less than this (market points) is roster clog
ROSTER_COST = 1.5         # (J) logit per throw-in they'd have to cut someone for
TOO_SMALL = 1.0           # pts/wk: neither lineup moves this much
SMALL_NAMES = 30.0        # ...and nobody in it is much of a name (market points)
OUT_OF_RACE = 0.10        # playoff odds below this: playing for this week, not a title

A0 = -0.3                 # (J) a neutral offer is more likely declined than taken
W_NAME = 3.0              # (J) per unit of ln(what they get / what they give), by name
W_LINEUP = 0.2            # (J) per pt/wk their lineup gains
W_URGENCY = 0.5           # (J) managers at 25-50% playoff odds deal most
LOPSIDED = (0.55, 1.8)    # names this far from even read as a fleecing to the league

FLAGS = {"LIKELY": 0.5, "EXPLOIT": 0.3, "NEEDS_PITCH": 0.15}   # (J) bands on the odds


def flag(p: float, d_title_them: float = 0.0) -> str:
    if p >= FLAGS["LIKELY"]:
        return "LIKELY"
    if p >= FLAGS["EXPLOIT"] and d_title_them < 0:
        return "EXPLOIT"
    if p >= FLAGS["NEEDS_PITCH"]:
        return "NEEDS_PITCH"
    return "LONGSHOT"


def urgency(p_playoffs: float) -> float:
    """Teams at 25–50% playoff odds are the most motivated to deal."""
    if 0.25 <= p_playoffs <= 0.50:
        return 1.0
    if 0.10 <= p_playoffs < 0.25 or 0.50 < p_playoffs <= 0.70:
        return 0.5
    return 0.0


def fandom() -> dict:
    """Fantasy team name -> NFL team abbreviation, from Chris (config/fandom.json)."""
    try:
        return {k: v for k, v in json.loads(FANDOM.read_text()).items() if not k.startswith("_")}
    except Exception:
        return {}


def _ranks(pw, min_games: int) -> dict:
    """gsis id -> rank at the position by half-PPR points per game (min_games+ games)."""
    if pw is None or pw.empty:
        return {}
    g = pw[pw["pos"].isin(list(STARTER_CUT))].groupby(["gsis_id", "pos"])["half_ppr"].agg(["size", "mean"]).reset_index()
    g = g[g["size"] >= min_games]
    g["rk"] = g.groupby("pos")["mean"].rank(ascending=False, method="first")
    return dict(zip(g["gsis_id"], zip(g["pos"], g["rk"].astype(int))))


@lru_cache(maxsize=4)
def track_record(season: int) -> tuple[frozenset, frozenset]:
    """(established, stars) — gsis ids. Established: a starter-level season last year or in two
    of the last three, or a breakout at half the starter cut this season. Star: a finish
    inside STAR_CUT in any of the last three seasons."""
    from .season import player_week
    hits: dict[str, list[int]] = {}
    stars = set()
    for yr in (season - 3, season - 2, season - 1):
        try:
            ranks = _ranks(player_week(yr), MIN_GAMES)
        except Exception:
            continue
        for gid, (pos, rk) in ranks.items():
            if rk <= STARTER_CUT[pos]:
                hits.setdefault(gid, []).append(yr)
            if rk <= STAR_CUT[pos]:
                stars.add(gid)
    est = {g for g, ys in hits.items() if season - 1 in ys or len(ys) >= 2}
    try:
        now = _ranks(player_week(season), MIN_GAMES_NOW)
    except Exception:
        now = {}
    est |= {g for g, (pos, rk) in now.items() if rk <= STARTER_CUT[pos] // 2}
    return frozenset(est), frozenset(stars)


class Board:
    """What every manager sees about every player, built once per engine context."""

    def __init__(self, ctx, season: int, bias_by_id: dict | None = None, profiles: dict | None = None):
        from . import trade_engine as te
        from . import trade_league as tl
        self.ctx, self.te = ctx, te
        self.est, self.stars = track_record(season)
        self.bias = bias_by_id or {}
        prof = profiles if profiles is not None else tl.team_profiles(ctx)
        self.thin = {tid: {n["pos"] for n in (pr.get("needs") or [])} for tid, pr in prof.items()}
        fan = fandom()
        self.fan = {tid: fan.get(t.get("name")) for tid, t in ctx.teams.items()}
        self.rank, self.elite = {}, set()
        for pos in STARTER_CUT:
            ids = sorted((i for i, p in ctx.players.items() if p.get("pos") == pos),
                         key=lambda i: -te.value_at(ctx.players[i], None))
            self.rank.update({i: n + 1 for n, i in enumerate(ids)})
            if pos in ("QB", "TE"):
                v = [te.value_at(ctx.players[i], None) for i in ids[:5]]
                cut = next((n + 1 for n in range(min(4, len(v) - 1)) if v[n] - v[n + 1] >= ELITE_GAP), 0)
                self.elite.update(ids[:cut])

    def name_value(self, pid: str, team_id) -> tuple[float, list[str]]:
        """What `team_id`'s manager thinks `pid` is worth, on the market curve, and which
        adjustments moved it ("elite", "star", "unproven", "fan") for the "why"."""
        p = self.ctx.players[pid]
        pos, notes = p.get("pos"), []
        v = self.te.market_value(p, self.ctx.cfg)
        if pos in ("QB", "TE"):
            if pid in self.elite:
                v *= ELITE_MULT
                notes.append("elite")
            else:
                v *= OTHER_QB if pos == "QB" else OTHER_TE
        else:
            v *= POS_MULT.get(pos, 1.0) * (TIER1_MULT if self.rank.get(pid, 99) <= 12 else 1.0)
        if pid in self.stars:
            v *= STAR_MULT
            notes.append("star")
        elif pid not in self.est:
            v *= PERCEIVED_FLOOR
            notes.append("unproven")
        if self.fan.get(team_id) and p.get("nfl") == self.fan[team_id]:
            v *= FAN_MULT
            notes.append("fan")
        v *= float((self.bias.get(team_id) or {}).get(pos, 1.0))
        return v, notes


# the "why" for each adjustment, by which side of the deal the player is on (they get / they give)
NOTE = {
    "elite": ("{n} is an elite {pos}", "they'd give up an elite {pos} in {n}"),
    "star": ("{n} is a big name", "they'd give up a big name in {n}"),
    "unproven": ("{n} is unproven, so weak currency", "{n} is unproven, easier for them to part with"),
    "fan": ("{n} plays for {nfl}, their team", "{n} plays for {nfl}, their team: hard to pry loose"),
}


def notes_for(ctx, pid: str, codes: list[str], gets: bool) -> list[str]:
    p = ctx.players[pid]
    codes = [c for c in codes if not (c == "star" and "elite" in codes)]   # one is enough
    return [NOTE[c][0 if gets else 1].format(n=p["name"], pos=p.get("pos"), nfl=p.get("nfl")) for c in codes]


def package(vals: list[float]) -> float:
    return sum(max(v, 0.0) ** K for v in vals) ** (1 / K)


def score(board: Board, their_id, give_ids: list, get_ids: list, d_me: float, d_them: float,
          p_playoffs: float, habit: float = 0.0) -> dict:
    """The chance `their_id`'s manager says yes when I send `give_ids` for `get_ids`, and why
    (the reasons are written about them, for a card or a text).

    `d_me`/`d_them` are each lineup's change in pts/wk (the engine's dMe/dThem); `habit`
    is ln(their trades a season / the league's) from mega/league_memory.py."""
    from . import trade_league as tl
    ctx = board.ctx
    why: list[str] = []
    sits = tl.sits_for_them(ctx, their_id, give_ids, get_ids)
    rec = sorted(((pid, *board.name_value(pid, their_id)) for pid in give_ids), key=lambda x: -x[1])
    sent = [(pid, *board.name_value(pid, their_id)) for pid in get_ids]
    n_swap = len(get_ids)
    swaps, clog, need = [], 0, None
    for n, (pid, v, notes) in enumerate(rec):
        name = ctx.players[pid]["name"]
        if n >= n_swap and (pid in sits or v < MIN_PIECE):
            clog += 1
            why.append(f"{name} is a throw-in they'd have to cut someone for")
            continue
        if pid in sits:
            v *= SIT_CREDIT
            why.extend(tl.sits_text(ctx, {pid: sits[pid]}))
        elif need is None and ctx.players[pid]["pos"] in board.thin.get(their_id, ()):
            need = f"{name} starts at {ctx.players[pid]['pos']}, where they're thin"
        swaps.append(v)
        why.extend(notes_for(ctx, pid, notes, True))
    v_in, v_out = package(swaps), package([v for _, v, _ in sent])
    for pid, _, notes in sent:
        why.extend(notes_for(ctx, pid, notes, False))
    ratio = v_in / v_out if v_out > 0 else 2.0
    log_ratio = math.log(max(ratio, 1e-3))

    out = p_playoffs < OUT_OF_RACE
    w_name, w_lineup = (W_NAME * 0.5, W_LINEUP * 2.0) if out else (W_NAME, W_LINEUP)
    z = (A0 + w_name * log_ratio + w_lineup * d_them - ROSTER_COST * clog
         + W_NEED * (need is not None) + W_URGENCY * urgency(p_playoffs) + habit)
    p = 1.0 / (1.0 + math.exp(-z))

    small = abs(d_me) < TOO_SMALL and abs(d_them) < TOO_SMALL and max(v_in, v_out) < SMALL_NAMES
    if small:
        p *= 0.3
        why.insert(0, "Too small a change for either side to bother")
    lead = (f"Their lineup {'gains' if d_them >= 0 else 'loses'} {abs(d_them):.1f} pts/wk"
            + (" — and out of the race, they're playing for wins now" if out else ""))
    name_line = (f"By name they get {ratio:.0%} of what they give" if abs(ratio - 1) >= 0.05
                 else "By name it's about even")
    why = [lead, name_line] + ([need] if need else []) + why
    if p >= 0.3 and not LOPSIDED[0] <= ratio <= LOPSIDED[1]:
        why.append("Lopsided enough that the league could object")   # only if it might happen
    seen, uniq = set(), []
    for w in why:
        if w not in seen:
            seen.add(w)
            uniq.append(w[:1].upper() + w[1:])
    return {"p": p, "why": uniq, "ratio": ratio, "clog": clog, "small": small, "need": need is not None,
            "sits": sits, "out_of_race": out}
