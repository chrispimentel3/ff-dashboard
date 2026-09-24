"""§15 injury expectation and handcuff value, §16 replacement level v1.2.

Three ideas, all of which change what a bench player is worth:

  * **Every player is a little bit unavailable in future.** v1.1 said "out this week,
    available every week after", which is both false and asymmetric — it prices the healthy
    starter as certain and gives the backup nothing. §15.1 ramps a position's steady-state
    injury rate in over the next few weeks, so a starter loses a sliver of future value and
    his backup gains one, without either being double counted.

  * **A handcuff is worth what the job is worth, times the chance the job opens.** Not what
    he is doing now. The backup behind a lead back is carrying an option on a role, and
    §15.2 prices it at the ROLE's baseline share rather than the starter's own — a promoted
    backup inherits the touches, not the talent.

  * **Replacement level is the mean of the top three free agents, not the best one.** The
    maximum of a few hundred noisy per-game estimates is the luckiest of them and is biased
    high; one lucky free agent otherwise makes every bench player on every roster look
    worthless. This is the same winner's-curse correction already applied in mega/needs.py.

Every constant marked (J) is a judgment default. §19.4 says when each gets fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# §15.1 (J): weekly chance a healthy player picks up an injury, and how long it keeps him
# out. P_ss = p * L is the steady-state chance he is unavailable in any given future week.
ONSET = {"RB": 0.06, "WR": 0.05, "TE": 0.05, "QB": 0.04}
WEEKS_MISSED = 3.0
CUFF_HAIRCUT = 0.85            # (J) a committee backup inherits less than a clear number 2
CLEAR_TWO_SHARE = 0.60         # ...he is "clear" at 60% of the non-lead carries
REPLACEMENT_TOP_N = 3          # §16
IR_WEEKS_DEFAULT = 4           # §3.4

# Yahoo / injury-report status -> availability THIS week (§3.4)
AVAIL_NOW = {"OUT": 0.0, "O": 0.0, "DOUBTFUL": 0.25, "D": 0.25,
             "QUESTIONABLE": 0.85, "Q": 0.85, "IR": 0.0, "PUP": 0.0, "NFI": 0.0}
BENCHED = {"IR", "PUP", "NFI"}


def steady_state(pos: str) -> float:
    """P_ss — the chance a player at this position is unavailable in an arbitrary week."""
    return min(1.0, ONSET.get(str(pos).upper(), 0.05) * WEEKS_MISSED)


def p_out(pos: str, weeks_ahead: int) -> float:
    """§15.1 — ramps from what we know today to the position's steady state.

        P_out(w) = P_ss * min(1, (w - now) / L)

    Next week is mostly known, so the ramp starts near zero; by L weeks out it is the
    base rate and stays there."""
    if weeks_ahead <= 0:
        return 0.0
    return steady_state(pos) * min(1.0, weeks_ahead / WEEKS_MISSED)


def availability(status: object, pos: str, week: int, now: int,
                 ir_return: int | None = None) -> float:
    """§3.4 + §15.1 — how much of a week a player is expected to be there for.

    A known status governs the current week outright. Later weeks are the expected-
    availability curve, except for IR/PUP/NFI, which are zero until the return week.
    """
    s = str(status or "").strip().upper()
    ahead = int(week) - int(now)
    if s in BENCHED:
        back = ir_return if ir_return is not None else int(now) + IR_WEEKS_DEFAULT
        return 0.0 if week < back else 1.0 - p_out(pos, ahead)
    if ahead <= 0:
        return AVAIL_NOW.get(s, 1.0)
    return 1.0 - p_out(pos, ahead)


# ---------------------------------------------------------------- §15.2 handcuffs
def next_man_up(pw: pd.DataFrame, roles_tab: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Per team, the backup who would inherit each lead back's work.

    Highest snap share among the backs who are not the lead — the spec's tiebreak of depth
    chart `pos_rank` applies only when snaps cannot separate them. Only ONE player per
    starter gets contingent value: spreading it across a room double counts a job that can
    only be done by one man at a time.
    """
    cols = ["gsis_id", "team", "pos", "role"]
    if roles_tab is None or roles_tab.empty or pw is None or pw.empty:
        return pd.DataFrame(columns=["gsis_id", "cuff_of", "team", "share"])
    snaps = (pw.sort_values("week").groupby("gsis_id")
             .agg(snaps=("offense_snaps", "sum"), carries=("carries", "sum")))
    d = roles_tab[cols].merge(snaps, on="gsis_id", how="left")

    out = []
    for (team, pos), grp in d[d["pos"].isin(("RB", "WR", "TE", "QB"))].groupby(["team", "pos"]):
        lead_roles = {"RB": "LEAD", "WR": "WR1", "TE": "TE1-REC", "QB": "STARTER"}
        starters = grp[grp["role"] == lead_roles.get(pos)]
        if starters.empty:
            continue
        starter = starters.sort_values("snaps", ascending=False).iloc[0]
        rest = grp[grp["gsis_id"] != starter["gsis_id"]].sort_values(
            "snaps", ascending=False, na_position="last")
        if rest.empty:
            continue
        backup = rest.iloc[0]
        non_lead = rest["carries"].fillna(0).sum()
        share = (backup["carries"] or 0) / non_lead if non_lead else 0.0
        out.append({"gsis_id": backup["gsis_id"], "cuff_of": starter["gsis_id"],
                    "team": team, "pos": pos, "share": float(share)})
    return pd.DataFrame(out)


def contingent(cuffs: pd.DataFrame, base: dict, team_xfp: pd.DataFrame,
               weekly: dict, now: int, weeks) -> pd.DataFrame:
    """§15.2 — what the option on a job is worth, week by week.

        v_promoted = μ_share(role of the starter) * teamXFP * h
        c[w]       = P_out(starter, w) * max(0, v_promoted - weekly_backup[w])
        weekly_backup[w] += c[w]

    The promoted value uses the ROLE's baseline share, not the starter's own: a backup
    who takes over inherits the touches, not the player. `h` is 1.00 for a clear number
    two and 0.85 for one of a committee (J).
    """
    if cuffs is None or cuffs.empty:
        return pd.DataFrame(columns=["gsis_id", "week", "contingent"])
    lead_share = base.get("role", {}).get("xfp_share", {})
    rows = []
    for _, c in cuffs.iterrows():
        h = 1.0 if c["share"] >= CLEAR_TWO_SHARE else CUFF_HAIRCUT
        role_of_starter = {"RB": "LEAD", "WR": "WR1", "TE": "TE1-REC", "QB": "STARTER"}[c["pos"]]
        mu = lead_share.get(role_of_starter)
        if not mu:
            continue
        for w in weeks:
            tx = _team_xfp(team_xfp, c["team"], w)
            if tx is None:
                continue
            promoted = mu * tx * h
            own = float(weekly.get((c["gsis_id"], int(w)), 0.0) or 0.0)
            val = p_out(c["pos"], int(w) - int(now)) * max(0.0, promoted - own)
            rows.append({"gsis_id": c["gsis_id"], "cuff_of": c["cuff_of"],
                         "week": int(w), "contingent": round(val, 4)})
    return pd.DataFrame(rows)


def _team_xfp(team_xfp: pd.DataFrame, team: str, week: int):
    if team_xfp is None or team_xfp.empty:
        return None
    hit = team_xfp[(team_xfp["team"] == team) & (team_xfp["week"] == int(week))]
    if hit.empty:
        return None
    cols = [c for c in hit.columns if c.startswith("xfp_")]
    return float(hit[cols].iloc[0].sum())


# ---------------------------------------------------------------- §16 replacement level
def replacement(values: pd.Series, top_n: int = REPLACEMENT_TOP_N) -> float:
    """The mean of the best `top_n` free agents at a position.

    The single best is the maximum of a few hundred noisy estimates and is biased high —
    the winner's curse. One two-game cameo otherwise sets the bar for a whole position and
    every bench player behind it reads as worthless."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().sort_values(ascending=False)
    if v.empty:
        return 0.0
    return float(v.head(max(1, top_n)).mean())


def replacement_curve(now_level: float, supply: float, delta: float, week: int,
                      now: int, teams: int = 12) -> float:
    """§16 future supply — positions that refill get a rising replacement level.

        repl[w] = repl_now + supply * Δ * (w - now) / teams

    Wide receivers turn over on the wire; lead backs mostly do not. Where the bar rises,
    giving one away in a trade costs less than his current points suggest. Off until four
    weeks of transactions exist, because `supply` cannot be estimated before then.
    """
    return float(now_level) + float(supply) * float(delta) * max(0, int(week) - int(now)) / max(1, teams)
