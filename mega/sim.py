"""§18 win probability and playoff odds.

Points per week treats every week and every opponent alike. It shouldn't: a point added to
a team already winning comfortably is worth less than the same point added to a team on the
bubble, and the only currency that matters in December is the title. So a trade is scored
twice — first in points per week, which is fast and ranks everything, then in championship
probability for the handful of candidates worth simulating.

Two things make this trustworthy rather than decorative:

  * **Common random numbers.** The baseline and every candidate are simulated on the same
    draws. Without that, the difference between two rosters is mostly sampling noise, and a
    trade that changes nothing comes back with a non-zero edge. Identical rosters must give
    exactly 0, and there is a test that says so.
  * **It has to agree with the dashboard.** The app already shows a pairwise win
    probability from a logistic on the point spread. Two different numbers for the same
    question on two screens is worse than one imperfect number, so `check_against()` holds
    the normal model to the logistic within 2 points and says when it fails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

N_SIMS = 10_000
SEED = 2026
TOP_CANDIDATES = 40          # §18.3: only the best candidates by pts/wk get simulated
ARMS_RIVAL = 0.05            # partner gains >= 5 points of playoff odds...
BAND = (0.20, 0.80)          # ...and both teams are live
LOGISTIC_DIVISOR = 1.7       # the dashboard's spread divisor — do not invent a second one
AGREE_TOL = 0.02


def win_prob_logistic(gap: float, spread: float, divisor: float = LOGISTIC_DIVISOR) -> float:
    """The dashboard's pairwise win probability, ported verbatim from mega/xwins.py.

        scale = spread / 1.7
        P     = 1 / (1 + exp(-gap / scale))

    Note where the 1.7 goes: it divides the SPREAD, not the gap. Dividing the gap instead
    makes a five-point win a 95% certainty, which is what my first version did and what
    §18.1's cross-check caught. `spread` is the standard deviation of the DIFFERENCE
    between the two scores being compared.
    """
    if spread <= 0:
        return 1.0 if gap > 0 else (0.0 if gap < 0 else 0.5)
    scale = float(spread) / float(divisor)
    return 1.0 / (1.0 + math.exp(-float(gap) / scale))


def win_prob_normal(gap: float, spread: float) -> float:
    """P(win) from a normal on the same difference distribution: Φ(gap / spread).

    `spread` is the sd of the difference. For two independent teams each with weekly sd σ
    that is σ√2 — passing σ itself is the easy mistake, and it makes every game look
    closer than it is."""
    if spread <= 0:
        return 1.0 if gap > 0 else (0.0 if gap < 0 else 0.5)
    return 0.5 * (1.0 + math.erf(float(gap) / (float(spread) * math.sqrt(2.0))))


def diff_spread(sigma: float) -> float:
    """The sd of the gap between two independent teams that each vary by `sigma`."""
    return float(sigma) * math.sqrt(2.0)


def check_against(spread: float, gaps=(0, 5, 10, 20, 30), tol: float = AGREE_TOL,
                  divisor: float = LOGISTIC_DIVISOR) -> list[str]:
    """§18.1 — the normal model must match the dashboard's logistic within 2 points.

    Returns the disagreements. A non-empty result means the two screens would give two
    different answers to the same question, and the honest move is to use the dashboard's
    function rather than ship both.
    """
    bad = []
    for g in gaps:
        a = win_prob_normal(g, spread)
        b = win_prob_logistic(g, spread, divisor)
        if abs(a - b) > tol:
            bad.append(f"gap {g}: normal {a:.3f} vs dashboard {b:.3f}")
    return bad


def weekly_sigma(scores: pd.DataFrame) -> float:
    """The spread of a weekly team score in THIS league, from its own history."""
    if scores is None or scores.empty or "points" not in scores.columns:
        return 0.0
    v = pd.to_numeric(scores["points"], errors="coerce").dropna()
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


@dataclass
class Season:
    teams: list                                   # team ids, any hashable
    means: dict                                   # {(team, week): expected starter points}
    schedule: list                                # [{"week", "home", "away"}]
    wins: dict = field(default_factory=dict)      # current record
    points_for: dict = field(default_factory=dict)
    sigma: float = 25.0
    playoff_teams: int = 6
    byes: int = 0
    approx_weeks: tuple = ()      # weeks with no real fixture on file
    mean_se: dict = field(default_factory=dict)   # {team: standard error of its own means}


def _draws(season: Season, n: int, rng: np.random.Generator) -> dict:
    """One block of standard normals per team-week, plus one per team for its mean shock
    (see `simulate`), reused across every candidate.

    This is the common-random-numbers trick: the SAME shocks are applied to the baseline
    and to each alternative, so what differs between them is the roster, not the dice.
    """
    weeks = sorted({g["week"] for g in season.schedule})
    draws = {(t, w): rng.standard_normal(n) for t in season.teams for w in weeks}
    draws.update({(t, "mu"): rng.standard_normal(n) for t in season.teams})
    return draws


def simulate(season: Season, n: int = N_SIMS, seed: int = SEED,
             draws: dict | None = None) -> pd.DataFrame:
    """Play the rest of the season `n` times. Returns per-team playoff, bye and title odds.

    `team_ppw` (season.means) is itself an estimate — an average over however many games a
    team has actually played — not a known constant, and treating it as one is what made
    early-season odds read as overconfident (a team 3-0 on three good games looked like a
    99% playoff lock, because the only randomness simulated was week-to-week score noise,
    which averages out over a long remaining season regardless of how shaky the 3-game
    estimate underneath it was). `mean_se` (§ from_league) is the standard error of that
    estimate — large with few games played, shrinking as more come in — and gets drawn
    ONCE per simulated season, not once per week: a team's true talent is either better or
    worse than its early-season average, consistently, not independently redrawn every
    Sunday. That single draw then rides along under every week's own score noise.
    """
    if not season.teams or not season.schedule:
        return pd.DataFrame(columns=["team", "p_playoffs", "p_bye", "p_title"])
    rng = np.random.default_rng(seed)
    draws = draws if draws is not None else _draws(season, n, rng)

    wins = {t: np.full(n, float(season.wins.get(t, 0))) for t in season.teams}
    pf = {t: np.full(n, float(season.points_for.get(t, 0.0))) for t in season.teams}
    mean_shock = {t: season.mean_se.get(t, 0.0) * draws[(t, "mu")][:n] for t in season.teams}

    for g in season.schedule:
        h, a, w = g["home"], g["away"], g["week"]
        if h not in wins or a not in wins:
            continue
        hs = season.means.get((h, w), 0.0) + mean_shock[h] + season.sigma * draws[(h, w)][:n]
        as_ = season.means.get((a, w), 0.0) + mean_shock[a] + season.sigma * draws[(a, w)][:n]
        pf[h] += hs
        pf[a] += as_
        wins[h] += (hs > as_).astype(float)
        wins[a] += (as_ > hs).astype(float)

    # rank by wins, then points for — the league's stated tiebreak
    order = np.argsort(
        -np.stack([wins[t] + pf[t] / 1e6 for t in season.teams], axis=0), axis=0)
    seed_of = np.empty_like(order)
    for s in range(len(season.teams)):
        seed_of[order[s], np.arange(n)] = s + 1

    rows = []
    for i, t in enumerate(season.teams):
        s = seed_of[i]
        rows.append({
            "team": t,
            "p_playoffs": float((s <= season.playoff_teams).mean()),
            "p_bye": float((s <= season.byes).mean()) if season.byes else 0.0,
            "p_title": float(_title_odds(s, season)),
            "mean_seed": float(s.mean()),
        })
    return pd.DataFrame(rows).sort_values("p_playoffs", ascending=False).reset_index(drop=True)


def _title_odds(seed: np.ndarray, season: Season) -> float:
    """A seeded bracket, approximated: making the field is most of it, and a better seed
    is worth more. Deliberately simple and deliberately labelled — the bracket itself is
    three games of coin-flips between teams that are close by construction."""
    field = seed <= season.playoff_teams
    if not field.any():
        return 0.0
    rounds = max(1, math.ceil(math.log2(max(2, season.playoff_teams))))
    # a top seed skips a round where byes exist
    per_round = 0.5
    base = per_round ** rounds
    lift = np.where(seed <= max(1, season.byes), 1.0 / per_round, 1.0)
    return float((field * base * lift).mean())


def delta(season: Season, means_after: dict, n: int = N_SIMS, seed: int = SEED,
          team=None) -> dict:
    """The change in a team's odds from a roster change, on common random numbers.

    `means_after` is the same {(team, week): points} map with the candidate applied. The
    two runs share every draw, so an unchanged roster returns exactly zero rather than
    sampling noise dressed up as an edge.
    """
    rng = np.random.default_rng(seed)
    shared = _draws(season, n, rng)
    before = simulate(season, n, seed, draws=shared).set_index("team")
    after_season = Season(**{**season.__dict__, "means": means_after})
    after = simulate(after_season, n, seed, draws=shared).set_index("team")
    out = {}
    for t in season.teams:
        out[t] = {
            "d_playoffs": float(after.loc[t, "p_playoffs"] - before.loc[t, "p_playoffs"]),
            "d_title": float(after.loc[t, "p_title"] - before.loc[t, "p_title"]),
            "p_playoffs": float(after.loc[t, "p_playoffs"]),
        }
    if team is not None:
        return out.get(team, {})
    return out


def partner_tag(p_playoffs: float) -> str:
    """§18.3 — how much the partner still has to play for."""
    if p_playoffs > 0.60:
        return "CONTENDER"
    if p_playoffs >= 0.20:
        return "BUBBLE"
    return "OUT"


def arms_rival(my_delta: float, their_delta: float, my_p: float, their_p: float) -> bool:
    """True when a trade meaningfully helps a team you are actually racing.

    A deal can be good for you in points and still be a mistake if it lifts the team most
    likely to take the last playoff place off you."""
    return (their_delta >= ARMS_RIVAL
            and BAND[0] <= my_p <= BAND[1] and BAND[0] <= their_p <= BAND[1])


# ---------------------------------------------------------------- building it from league data
PLAYOFF_TEAMS = 6           # Mega Bowl: 6 of 12 make it, weeks 15-17
PLAYOFF_BYES = 2


def from_league(scores: pd.DataFrame, standings: pd.DataFrame, team_ppw: dict,
                weeks_left, fixtures: pd.DataFrame | None = None,
                playoff_teams: int = PLAYOFF_TEAMS, byes: int = PLAYOFF_BYES) -> Season:
    """A Season from what the scrape already holds.

    `team_ppw` is each team's expected STARTER points per week — bench points are excluded
    because they do not score. sigma is this league's own weekly spread, not a
    league-average guess: a 12-team half-PPR league with these rosters is the only
    relevant population.

    `fixtures` is the real remaining schedule where it has been scraped. Strength of
    schedule is most of a fantasy season, so a real fixture beats an invented one every
    time; weeks with no fixture on file fall back to a round robin and `approx_weeks`
    records exactly which, so the caption can say so rather than imply more than is known.

    `team_ppw` is a mean over each team's own games played so far — its standard error is
    `sigma / sqrt(games)`, textbook standard-error-of-the-mean, using this league's own
    pooled week-to-week spread rather than a 3-game team-level variance that would itself
    be too noisy to trust. A team with no games on file gets the full `sigma` (the widest
    this ever gets) rather than a divide-by-zero.
    """
    if standings is None or standings.empty:
        return Season(teams=[], means={}, schedule=[])
    teams = [str(t) for t in standings["team"]]
    wins = {str(r["team"]): float(r.get("wins") or 0) for _, r in standings.iterrows()}
    pf, games_played, sigma = {}, {}, weekly_sigma(scores)
    if scores is not None and not scores.empty:
        pf = scores.groupby("team")["points"].sum().to_dict()
        games_played = scores.groupby("team")["points"].count().to_dict()
    weeks = [int(w) for w in weeks_left]
    means = {(t, w): float(team_ppw.get(t, 0.0)) for t in teams for w in weeks}
    mean_se = {t: sigma / math.sqrt(max(games_played.get(t, 0), 1)) for t in teams}

    known = set(teams)
    sched, real_weeks = [], set()
    if fixtures is not None and not fixtures.empty:
        fx = fixtures.assign(week=pd.to_numeric(fixtures["week"], errors="coerce"))
        for w in weeks:
            rows = fx[fx["week"] == w]
            rows = rows[rows["home"].isin(known) & rows["away"].isin(known)]
            if len(rows) >= len(teams) // 2:
                sched += [{"week": int(w), "home": str(r["home"]), "away": str(r["away"])}
                          for _, r in rows.iterrows()]
                real_weeks.add(int(w))
    missing = [w for w in weeks if w not in real_weeks]
    sched += _round_robin(teams, missing)
    return Season(teams=teams, means=means, schedule=sched, wins=wins,
                  points_for={str(k): float(v) for k, v in pf.items()},
                  sigma=sigma, playoff_teams=playoff_teams, byes=byes,
                  approx_weeks=tuple(missing), mean_se=mean_se)


def _round_robin(teams: list, weeks: list) -> list:
    """A stand-in schedule when the real one has not been scraped.

    Stated plainly because it matters: this is NOT Chris's actual remaining fixtures. It
    keeps everyone playing the same number of games so the odds are not nonsense, but a
    team's real strength of schedule is not in here. Odds built on it are for comparing
    two versions of the SAME roster, which is what a trade delta needs, and not for
    reading as a forecast.
    """
    n = len(teams)
    if n < 2:
        return []
    out = []
    for i, w in enumerate(weeks):
        rot = teams[:1] + teams[1:][i % max(1, n - 1):] + teams[1:][: i % max(1, n - 1)]
        for a, b in zip(rot[: n // 2], rot[n // 2:][::-1]):
            out.append({"week": int(w), "home": a, "away": b})
    return out


def schedule_note(season: Season) -> str:
    """Say exactly how much of the schedule is real — the honest version of a caveat."""
    weeks = sorted({g["week"] for g in season.schedule})
    if not weeks:
        return ""
    n_approx = len(season.approx_weeks)
    if not n_approx:
        return "Run on your real remaining fixtures."
    if n_approx == len(weeks):
        return ("Fixtures are a stand-in — none of the remaining schedule has been scraped "
                "yet. Use these to compare two versions of your own roster, not as a "
                "forecast of where you finish.")
    return (f"Real fixtures through week {max(w for w in weeks if w not in season.approx_weeks)}; "
            f"weeks {min(season.approx_weeks)}-{max(season.approx_weeks)} are a stand-in "
            "until the scrape reaches them.")


def trade_delta(season: Season, my_team: str, their_team: str, d_me: float, d_them: float,
                n: int = 2000, seed: int = SEED) -> dict:
    """§18.3 — what a trade does to both teams' odds.

    The engine already answers "how many points per week does this add", for both sides.
    That IS the change to each team's weekly mean, so it drops straight into the season
    without re-deriving anything: shift the two means, replay on the same draws, subtract.

    Returns the mover's change and the partner's, because a deal can be good for you in
    points and still be a mistake if it lifts the team you are racing for the last spot.
    """
    after = dict(season.means)
    for (t, w) in list(after):
        if t == my_team:
            after[(t, w)] = after[(t, w)] + float(d_me)
        elif t == their_team:
            after[(t, w)] = after[(t, w)] + float(d_them)
    out = delta(season, after, n=n, seed=seed)
    mine, theirs = out.get(my_team, {}), out.get(their_team, {})
    return {
        "d_playoffs": mine.get("d_playoffs", 0.0), "d_title": mine.get("d_title", 0.0),
        "p_playoffs": mine.get("p_playoffs", 0.0),
        "their_d_playoffs": theirs.get("d_playoffs", 0.0),
        "their_p_playoffs": theirs.get("p_playoffs", 0.0),
        "their_tag": partner_tag(theirs.get("p_playoffs", 0.0)),
        "arms_rival": arms_rival(mine.get("d_playoffs", 0.0), theirs.get("d_playoffs", 0.0),
                                 mine.get("p_playoffs", 0.0), theirs.get("p_playoffs", 0.0)),
    }
