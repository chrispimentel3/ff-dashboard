"""HANDOFF v1.3 Pass 3 — title odds from player-level draws, on common random numbers.

mega/sim.py plays the season with one number per team and a league-wide weekly spread, and
approximates the bracket as coin flips weighted by seed. That is fine for a standings page
and too blunt to price a trade: two rosters with the same weekly mean but different
players don't have the same title odds — a lineup of boom-or-bust receivers wins more
brackets from behind and loses more from in front. So here every starter draws his own
score each week:

  * **Distribution.** A split normal through the player's week median, p10 and p90 from
    data/proj_ros_<season>.json (fitted in the Pass 2 backtest), zero if he doesn't play
    (P(plays) from the same file) — and then his backup at that slot plays instead.
  * **Two layers of uncertainty.** Each player also draws ONE season-long projection error
    (the backtest's out-of-sample MAE by position, as an sd) — the player-level version of
    mega/sim.py's SE-of-the-mean team uncertainty. Week-to-week spread is scaled down so the
    two together still reproduce the fitted p10/p90, not double count it.
  * **The real bracket.** Remaining fixtures (scraped where available, sim.from_league),
    wins then points for, then Mega Bowl's 6-team, 2-bye bracket over weeks 15–17 played
    week by week on the same draws.
  * **Common random numbers.** Every draw is keyed to (player, week) — or (team, week) for
    K/DEF — from a fixed seed, never to a position in an array. A trade moves players
    between teams and their dice go with them, so a trade that changes no lineup changes
    the odds by exactly zero, and small real deltas aren't buried in resampling noise.
  * **Noise band.** Δ is reported with its standard error (paired, per simulated season);
    a Δ inside 2 SE is flagged as noise and callers suppress it.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from . import trade_engine as te

N_SEASONS = 6000          # handoff §5
SEED = 2026
REG_LAST = 14
PLAYOFF_WEEKS = (15, 16, 17)
KDEF_MEAN = 15.0          # (J) a half-PPR kicker + defense per week; same for every team,
KDEF_SD = 7.0             #     so it sets game-to-game noise and never a trade's delta
DEFAULT_Q = (0.35, 0.90, 1.75)   # (J) p10/p50/p90 multipliers for a player the projection lacks
NOISE_Z = 2.0             # |Δ| under 2 SE is noise
TW_CACHE_MAX = 1500       # team-week score arrays kept before the cache is flushed
POSTURE = ((0.70, "protect"), (0.35, "balanced"), (0.0, "swing"))   # handoff §5, on P(playoffs)


def posture(p_playoffs: float) -> str:
    for cut, name in POSTURE:
        if p_playoffs > cut or cut == 0.0:
            return name
    return "balanced"


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng((zlib.crc32(key.encode()) + SEED) & 0xFFFFFFFF)


@dataclass
class Model:
    season: object                 # mega.sim.Season: teams, schedule (weeks now..14), wins, pf
    ctx: te.Ctx                    # engine context carrying each player's weekly expected value
    dist: dict                     # pid -> {week: (median, p10, p90, p_active)}
    proj_sd: dict                  # pos -> sd of the season-long projection error
    rosters: dict                  # team -> [pids]
    n: int = N_SEASONS
    _draw: dict = field(default_factory=dict)
    _tw: dict = field(default_factory=dict)

    # ------------------------------------------------------------ dice
    def _dice(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        hit = self._draw.get(key)
        if hit is None:
            r = _rng(key)
            # float32: the live API holds these for hours on a 512MB host
            hit = (r.standard_normal(self.n, dtype=np.float32), r.random(self.n, dtype=np.float32))
            self._draw[key] = hit
        return hit

    def _player(self, pid: str, w: int) -> tuple[np.ndarray, np.ndarray]:
        """(points if he plays, whether he plays) for every simulated season."""
        p = self.ctx.players[pid]
        d = self.dist.get(pid, {}).get(w)
        if d is None:
            v = te.value_at(p, w)
            d = (v * DEFAULT_Q[1], v * DEFAULT_Q[0], v * DEFAULT_Q[2], 1.0 if v > 0 else 0.0)
        med, p10, p90, pa = d
        sd_proj = self.proj_sd.get(p["pos"], 0.0)
        sd_week = max((p90 - p10) / 2.563, 1e-6)
        f = math.sqrt(max(0.2, 1.0 - (sd_proj / sd_week) ** 2))
        z, u = self._dice(f"{pid}|{w}")
        z0, _ = self._dice(f"{pid}|season")
        lo, hi = (med - p10) / 1.2816 * f, (p90 - med) / 1.2816 * f
        pts = med + np.where(z < 0, lo * z, hi * z) + sd_proj * z0
        return np.maximum(pts, -2.0), u < pa

    # ------------------------------------------------------------ one team-week
    def team_week(self, team: str, ids, w: int) -> np.ndarray:
        key = (team, frozenset(ids), w)
        hit = self._tw.get(key)
        if hit is not None:
            return hit
        lu = te.lineup(list(ids), self.ctx, w)
        flex_ok = set(self.ctx.cfg["flexEligible"])
        bench = [e for e in lu.bench]
        total = np.zeros(self.n)
        for pid, pos, _pts, slot in lu.starters:
            x, on = self._player(pid, w)
            bk = next((e[0] for e in bench if e[1] == pos or (slot == "FLEX" and e[1] in flex_ok)), None)
            if bk is not None:
                xb, onb = self._player(bk, w)
                total += np.where(on, x, np.where(onb, xb, 0.0))
            else:
                total += np.where(on, x, 0.0)
        z, _ = self._dice(f"{team}|kdef|{w}")
        total += KDEF_MEAN + KDEF_SD * z
        if len(self._tw) > TW_CACHE_MAX:
            self._tw.clear()            # every new roster adds 15 weeks of arrays; a live
                                        # service would otherwise grow without bound
        self._tw[key] = total
        return total

    # ------------------------------------------------------------ a season
    def run(self, rosters: dict | None = None) -> dict:
        """Per-simulated-season outcomes for every team: made playoffs, won title."""
        rosters = rosters or self.rosters
        s = self.season
        teams = list(s.teams)
        idx = {t: i for i, t in enumerate(teams)}
        wins = np.array([[float(s.wins.get(t, 0))] * self.n for t in teams])
        pf = np.array([[float(s.points_for.get(t, 0.0))] * self.n for t in teams])
        for g in s.schedule:
            h, a, w = g["home"], g["away"], int(g["week"])
            if h not in idx or a not in idx or w > REG_LAST:
                continue
            hs = self.team_week(h, rosters.get(h, []), w)
            as_ = self.team_week(a, rosters.get(a, []), w)
            pf[idx[h]] += hs
            pf[idx[a]] += as_
            wins[idx[h]] += hs > as_
            wins[idx[a]] += as_ > hs
        order = np.argsort(-(wins + pf / 1e6), axis=0)          # order[k] = team at seed k+1
        made = np.zeros((len(teams), self.n), bool)
        for k in range(min(s.playoff_teams, len(teams))):
            made[order[k], np.arange(self.n)] = True
        champ = self._bracket(order, teams, rosters)
        title = np.zeros((len(teams), self.n), bool)
        title[champ, np.arange(self.n)] = True
        return {"teams": teams, "made": made, "title": title}

    def _bracket(self, order: np.ndarray, teams: list, rosters: dict) -> np.ndarray:
        """Mega Bowl: seeds 1–2 bye; week 15 3v6 and 4v5; week 16 the top seed meets the
        lowest seed left; week 17 the final. Higher seed wins a tie."""
        n = self.n
        cols = np.arange(n)
        score = {w: np.stack([self.team_week(t, rosters.get(t, []), w) for t in teams]) for w in PLAYOFF_WEEKS}
        s = [order[k, cols] for k in range(6)]                   # team index at seeds 1..6
        seed_of = {k + 1: s[k] for k in range(6)}

        def game(t1, s1, t2, s2, w):
            sc = score[w]
            a, b = sc[t1, cols], sc[t2, cols]
            win1 = a >= b
            return np.where(win1, t1, t2), np.where(win1, s1, s2)

        w3, ws3 = game(seed_of[3], np.full(n, 3), seed_of[6], np.full(n, 6), 15)
        w4, ws4 = game(seed_of[4], np.full(n, 4), seed_of[5], np.full(n, 5), 15)
        low_is_3 = ws3 > ws4                                     # the lower seed left is the larger number
        low_t, low_s = np.where(low_is_3, w3, w4), np.where(low_is_3, ws3, ws4)
        high_t, high_s = np.where(low_is_3, w4, w3), np.where(low_is_3, ws4, ws3)
        f1, fs1 = game(seed_of[1], np.full(n, 1), low_t, low_s, 16)
        f2, fs2 = game(seed_of[2], np.full(n, 2), high_t, high_s, 16)
        champ, _ = game(f1, fs1, f2, fs2, 17)
        return champ

    # ------------------------------------------------------------ odds and deltas
    def odds(self, rosters: dict | None = None) -> dict:
        r = self.run(rosters)
        return {t: {"p_playoffs": float(r["made"][i].mean()), "p_title": float(r["title"][i].mean())}
                for i, t in enumerate(r["teams"])}

    def delta(self, rosters_after: dict, teams: tuple) -> dict:
        """Δ odds for `teams` when `rosters_after` replaces the current rosters, with a
        paired standard error and a noise flag."""
        base = self._base()
        after = self.run({**self.rosters, **rosters_after})
        out = {}
        for t in teams:
            i = base["teams"].index(t)
            dt_ = after["title"][i].astype(float) - base["title"][i].astype(float)
            dp = after["made"][i].astype(float) - base["made"][i].astype(float)
            se_t = float(dt_.std(ddof=1) / math.sqrt(self.n)) if self.n > 1 else 0.0
            se_p = float(dp.std(ddof=1) / math.sqrt(self.n)) if self.n > 1 else 0.0
            out[t] = {"d_title": float(dt_.mean()), "se_title": se_t,
                      "d_playoffs": float(dp.mean()), "se_playoffs": se_p,
                      "p_title": float(after["title"][i].mean()),
                      "p_playoffs": float(after["made"][i].mean()),
                      "noise": abs(float(dt_.mean())) < NOISE_Z * se_t or float(dt_.mean()) == 0.0}
        return out

    def delta_vs(self, rosters_base: dict, rosters_after: dict, teams: tuple) -> dict:
        """Like `delta`, but against an alternative baseline (a free waiver swap already
        made) instead of the rosters as they stand — paired on the same dice either way."""
        a = self.run({**self.rosters, **rosters_base})
        b = self.run({**self.rosters, **rosters_after})
        out = {}
        for t in teams:
            i = a["teams"].index(t)
            dt_ = b["title"][i].astype(float) - a["title"][i].astype(float)
            dp = b["made"][i].astype(float) - a["made"][i].astype(float)
            se_t = float(dt_.std(ddof=1) / math.sqrt(self.n))
            out[t] = {"d_title": float(dt_.mean()), "se_title": se_t,
                      "d_playoffs": float(dp.mean()), "se_playoffs": float(dp.std(ddof=1) / math.sqrt(self.n)),
                      "p_title": float(b["title"][i].mean()), "p_playoffs": float(b["made"][i].mean()),
                      "noise": abs(float(dt_.mean())) < NOISE_Z * se_t or float(dt_.mean()) == 0.0}
        return out

    def _base(self):
        if not hasattr(self, "_base_run"):
            self._base_run = self.run(self.rosters)
        return self._base_run


# ======================================================================== from live data
def build(season: int, now: int, yahoo_rosters=None, n: int = N_SEASONS, league: dict | None = None,
          ctx: te.Ctx | None = None) -> Model | None:
    """The live model: engine lineups over weeks now..17 from the blended projection,
    player distributions from data/proj_ros_<season>.json, standings and fixtures from the
    Yahoo scrape."""
    import pandas as pd

    from . import proj_ros, sim as SIM
    from . import trade_league as tl
    from .config import DATA
    from .yahoo import cached_fixtures, cached_scores

    if league is None:
        league = tl.build_league(season, yahoo_rosters)
    if not league.get("teams"):
        return None
    if ctx is None:
        weeks = list(range(int(now), PLAYOFF_WEEKS[-1] + 1))
        cfg = te.merge(tl.engine_config(), {"horizon": {"weeks": weeks, "weights": {}}})
        ctx = te.build_context(league, cfg)
    sc = cached_scores()
    st_path = DATA / "yahoo_standings.csv"
    if sc is None or sc.empty or not st_path.is_file():
        return None
    stand = pd.read_csv(st_path)
    try:
        fx = cached_fixtures()
    except Exception:
        fx = None
    ppw = sc.groupby("team")["points"].mean().to_dict()
    season_obj = SIM.from_league(sc, stand, ppw, range(int(now), REG_LAST + 1), fixtures=fx)

    proj = (proj_ros.cached(season) or {}).get("players") or {}
    dist = {}
    for pid, p in ctx.players.items():
        pr = proj.get(pid)
        if not pr:
            continue
        d = {}
        for w, v in (pr.get("weeks") or {}).items():
            w = int(w)
            med, p10, p90 = float(v["median"]), float(v["p10"]), float(v["p90"])
            base = float(v.get("mean_if_active") or 0.0)
            if w == int(now) and v.get("p_active", 0) > 0 and base > 0:
                # this week's value is the Vegas number (trade_league) — slide the whole
                # distribution so its centre matches it, keeping the fitted shape
                k = te.value_at(p, w) / v["p_active"] / base
                med, p10, p90 = med * k, p10 * k, p90 * k
            d[w] = (med, p10, p90, float(v["p_active"]))
        dist[pid] = d
    proj_sd = {pos: float(v) for pos, v in (proj_ros.load_params().get("proj_err_sd") or {}).items()}
    rosters = {t["name"]: list(t["roster"]) + list(t.get("ir") or []) for t in league["teams"]}
    return Model(season_obj, ctx, dist, proj_sd, rosters, n=n)
