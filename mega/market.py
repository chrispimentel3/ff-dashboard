"""§6.2 perceived market value and §17 manager bias.

Two corrections to the idea that a trade is fair when the two sides are worth the same.

**§6.2 — they cannot see what you can see.** League-mates read box scores and a ranking
list. They do not read expected points, target share or routes. Valuing a trade partner's
side on *your* model and calling the result fair is how a deal that is obviously good for
you reads as obviously good for them too. `perceivedRank` blends the rest-of-season
consensus with how a player's actual points have read, weighted toward the box score as
the season gives it more to say. That is what the EXPLOIT flag means: fair on what they
see, better for you on what the usage says.

**§17 — they do not all value positions the same.** One manager pays up for running backs;
another will not roster a second tight end. A trade has to clear the market test THEY
apply. Seeds come from the draft-day scouting notes and are replaced as soon as there is
enough 2026 evidence to fit a multiplier from their own behaviour.

Every constant marked (J) is a judgment default until §19.4 calibrates it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import DATA, ROOT

BOX_WEIGHT_MAX = 0.40        # §6.2 wb ramp, same shape as the usage blend
BOX_RAMP_GAMES = 3
UNRANKED = 300               # a player nobody ranks is not free, he is unranked
MIN_GAMES_FOR_BOX = 2        # §6.2: box rank needs at least two games to mean anything

# §17 (J) — seeds from the draft scouting notes, by MANAGER name. They only take effect
# once config/managers.json maps a manager to a Yahoo team; without that mapping every
# team is priced neutrally, which is the honest default. Guessing who is who would skew
# every trade flag in a way nobody could see.
SEEDS = {
    "jamie": {"RB": 1.15},
    "colin": {"WR": 1.15, "RB": 0.90},
    "roman": {"TE": 1.15},
    "steve": {"QB": 1.10},
    "fletch": {"QB": 1.10},
    "mike": {"TE": 0.85},
}
MANAGERS_JSON = ROOT / "config" / "managers.json"
BIAS_CLAMP = (0.80, 1.25)
REACH_SCALE = 40.0           # §17: rounds of draft reach that move a multiplier
FAAB_SCALE = 0.25
FAAB_MIN_SPEND = 20


# ---------------------------------------------------------------- §6.2 perceived rank
def box_rank(pw: pd.DataFrame, min_games: int = MIN_GAMES_FOR_BOX) -> pd.DataFrame:
    """Overall rank by actual points per game above the positional replacement.

    This is the league-mate's view: what the box score has said, ranked across positions
    so it can be compared with a consensus list. Players with fewer than `min_games` are
    left unranked rather than placed on one afternoon.
    """
    cols = ["gsis_id", "box_ppg", "box_var", "box_rank", "games"]
    if pw is None or pw.empty or "half_ppr" not in pw.columns:
        return pd.DataFrame(columns=cols)
    g = pw.groupby("gsis_id")
    d = pd.DataFrame({
        "pos": g["pos"].last(),
        "games": g["week"].nunique(),
        "box_ppg": g["half_ppr"].mean(),
    }).reset_index()
    d = d[d["games"] >= min_games]
    if d.empty:
        return pd.DataFrame(columns=cols)
    # replacement = the median starter-ish player at the position, in box-score terms
    repl = d.groupby("pos")["box_ppg"].transform(lambda s: s.nlargest(max(1, len(s) // 3)).min())
    d["box_var"] = d["box_ppg"] - repl
    d["box_rank"] = d["box_var"].rank(ascending=False, method="min")
    return d[cols]


def perceived(ecr_ros: pd.DataFrame, box: pd.DataFrame,
              max_weight: float = BOX_WEIGHT_MAX, ramp: int = BOX_RAMP_GAMES) -> pd.DataFrame:
    """gsis_id -> perceivedRank (§6.2).

        wb            = 0.40 * min(g, 3) / 3
        perceivedRank = (1 - wb) * ecrRos + wb * boxRank

    Early in a season the consensus is all anyone has; by the third game the box score has
    started to overwrite it in people's heads, which is exactly the thing being modelled.
    """
    e = (ecr_ros if ecr_ros is not None else pd.DataFrame()).copy()
    b = (box if box is not None else pd.DataFrame()).copy()
    if e.empty and b.empty:
        return pd.DataFrame(columns=["gsis_id", "perceived_rank", "ecr_ros", "box_rank"])
    if e.empty:
        e = pd.DataFrame({"gsis_id": b["gsis_id"], "ecr_ros": np.nan})
    if b.empty:
        b = pd.DataFrame({"gsis_id": e["gsis_id"], "box_rank": np.nan, "games": 0})
    out = e.merge(b[["gsis_id", "box_rank", "games"]], on="gsis_id", how="outer")
    out["games"] = pd.to_numeric(out["games"], errors="coerce").fillna(0)
    wb = max_weight * np.minimum(out["games"], ramp) / ramp
    ecr = pd.to_numeric(out["ecr_ros"], errors="coerce")
    bx = pd.to_numeric(out["box_rank"], errors="coerce")
    # whichever side is missing, the other carries the whole weight
    wb = np.where(ecr.isna(), 1.0, np.where(bx.isna(), 0.0, wb))
    out["perceived_rank"] = (1 - wb) * ecr.fillna(0) + wb * bx.fillna(0)
    out.loc[ecr.isna() & bx.isna(), "perceived_rank"] = np.nan
    out["perceived_rank"] = out["perceived_rank"].fillna(UNRANKED)
    return out[["gsis_id", "perceived_rank", "ecr_ros", "box_rank", "games"]]


# ---------------------------------------------------------------- §17 manager bias
def load_managers() -> dict:
    """manager name -> Yahoo team name, from config/managers.json.

    Absent, every team is priced neutrally. That is deliberate: guessing which manager
    owns "deez nuts" would put a silent thumb on every trade flag in the app, and a wrong
    bias is worse than no bias because nothing on screen would reveal it.
    """
    if not MANAGERS_JSON.is_file():
        return {}
    try:
        return json.loads(MANAGERS_JSON.read_text())
    except Exception:
        return {}


def seed_bias(managers: dict | None = None) -> dict[str, dict]:
    """team name -> {pos: multiplier}, from the draft scouting seeds."""
    managers = managers if managers is not None else load_managers()
    out = {}
    for manager, team in (managers or {}).items():
        seed = SEEDS.get(str(manager).strip().lower())
        if seed:
            out[str(team)] = dict(seed)
    return out


def draft_bias(picks: pd.DataFrame, adp: pd.DataFrame | None = None) -> dict[str, dict]:
    """§17 — how far each manager reached, by position, in rounds 1-10.

        reach = mean(ADP - pick) over their picks at that position
        m     = clamp(1 + reach / 40, 0.80, 1.25)

    Taking a back three rounds before the field did says more about what he will pay for
    one than anything he says in the group chat.
    """
    if picks is None or picks.empty:
        return {}
    d = picks.copy()
    if "round" in d.columns:
        d = d[pd.to_numeric(d["round"], errors="coerce") <= 10]
    if adp is not None and not adp.empty and "adp" not in d.columns:
        on = "norm" if "norm" in d.columns and "norm" in adp.columns else None
        if on:
            d = d.merge(adp[[on, "adp"]], on=on, how="left")
    if "adp" not in d.columns or "pick" not in d.columns:
        return {}
    d["reach"] = pd.to_numeric(d["adp"], errors="coerce") - pd.to_numeric(d["pick"], errors="coerce")
    out: dict[str, dict] = {}
    for (team, pos), grp in d.dropna(subset=["reach"]).groupby(["team", "pos"]):
        m = 1.0 + grp["reach"].mean() / REACH_SCALE
        out.setdefault(str(team), {})[str(pos)] = float(np.clip(m, *BIAS_CLAMP))
    return out


def faab_bias(bids: pd.DataFrame, min_spend: float = FAAB_MIN_SPEND) -> dict[str, dict]:
    """§17 — a team's share of its FAAB spent on a position, against the league's share.

    Only counts once a team has actually spent something; a multiplier fitted on $3 of
    evidence is noise wearing a number's clothes.
    """
    if bids is None or bids.empty:
        return {}
    need = {"team", "pos", "bid"}
    if not need <= set(bids.columns):
        return {}
    d = bids.copy()
    d["bid"] = pd.to_numeric(d["bid"], errors="coerce").fillna(0.0)
    league = d.groupby("pos")["bid"].sum()
    league_share = league / league.sum() if league.sum() else league
    out: dict[str, dict] = {}
    for team, grp in d.groupby("team"):
        spent = grp["bid"].sum()
        if spent < min_spend:
            continue
        share = grp.groupby("pos")["bid"].sum() / spent
        for pos, s in share.items():
            ls = float(league_share.get(pos, 0.0))
            if ls <= 0:
                continue
            m = 1.0 + FAAB_SCALE * (s / ls - 1.0)
            out.setdefault(str(team), {})[str(pos)] = float(np.clip(m, *BIAS_CLAMP))
    return out


def bias(picks: pd.DataFrame | None = None, bids: pd.DataFrame | None = None,
         managers: dict | None = None) -> dict[str, dict]:
    """The multipliers actually used: the mean of whichever evidence exists, falling back
    to the scouting seed, falling back to neutral.

    Fitted evidence replaces the seeds rather than averaging with them — a seed is a note
    from draft night, and once a manager's own behaviour has spoken it should not keep
    voting.
    """
    fitted = [b for b in (draft_bias(picks) if picks is not None else {},
                          faab_bias(bids) if bids is not None else {}) if b]
    seeds = seed_bias(managers)
    teams = {t for b in fitted for t in b} | set(seeds)
    out: dict[str, dict] = {}
    for team in teams:
        got = [b[team] for b in fitted if team in b]
        if got:
            keys = {p for g in got for p in g}
            out[team] = {p: float(np.mean([g[p] for g in got if p in g])) for p in keys}
        elif team in seeds:
            out[team] = seeds[team]
    return out
