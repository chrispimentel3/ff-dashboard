"""FAAB: what everyone has left, what the league has actually paid, and what to bid.

Mega Bowl runs "FAB w/ Reverse order of standings tiebreak" on a $100 season budget
(league settings, confirmed 2026-09-22). Three things live here:

  * **Budgets.** Yahoo puts each team's remaining balance on the league home page, in the
    standings block — not on /standings, which carries no money column at all. `budgets()`
    parses it; `refresh_budgets()` writes data/yahoo_faab.csv.
  * **History.** `?transactionsfilter=faab` lists every settled claim with the winning bid
    *and the losing ones*, which is the only honest read on what this league pays.
  * **A bid.** `suggest()` turns a pickup's marginal lineup value into dollars.

The bid model, in one line: a player who adds `gain` points per week for the rest of the
season is worth `gain x weeks_left` points, and that converts to a share of the *original*
budget through a saturating curve. The curve matters more than its constants — the point is
that the second point per week costs more than the first and the tenth costs much more,
because budget is finite and the season is not. It is deliberately not calibrated against
this league's own history: five settled claims is not a market, and pretending otherwise
would dress up a guess. What the history is good for is context, so `market_summary()`
reports it next to the suggestion and lets the reader judge.

Nothing here writes manager emails to disk. The /teams page carries them and this repo is
public, so budgets are read from the league home page, which does not.
"""
from __future__ import annotations

import math
import re

import pandas as pd

from .config import DATA, N_TEAMS, SEAT_BY_TEAM

BUDGET = 100           # season FAAB allowance, per team
# Chris confirmed the league accepts $0 claims (2026-09-23), so an uncontested add costs
# nothing. The $1 floor below still applies to anything a rival would also want: a $0 bid
# loses every tie, and the tiebreak is reverse standings, which does not favour him.
MIN_BID = 0
LAST_WEEK = 17         # Mega Bowl playoffs are weeks 15-17, so nothing is bought after 17
FAAB_CSV = DATA / "yahoo_faab.csv"
BIDS_CSV = DATA / "yahoo_faab_bids.csv"

# Half the budget on one player is already an enormous swing; past that you are buying a
# lottery ticket with the rest of your season. MAX_SHARE caps the suggestion, SATURATION is
# the season-points figure at which the curve reaches ~63% of that cap.
MAX_SHARE = 0.50
SATURATION = 90.0

# "1  Don R.I.C.O  2-0-0  267.18  155.36  W-2  $93  12  1"  — tab-separated on the league
# home page. Team names contain spaces and apostrophes, so anchor on the money column.
_HOME_ROW = re.compile(
    r"^\s*(?P<rank>\d{1,2})\s*\t\s*(?P<team>.+?)\t.*?\$(?P<faab>\d+)\b",
    re.MULTILINE,
)


# ─────────────────────────────────────────────────────────────────────── budgets
def parse_budgets(page_text: str) -> pd.DataFrame:
    """Remaining FAAB per team, from the league home page's rendered text."""
    rows = {}
    for m in _HOME_ROW.finditer(page_text):
        team = m.group("team").strip()
        if team and team not in rows:
            rows[team] = dict(
                team=team, seat=SEAT_BY_TEAM.get(team),
                rank=int(m.group("rank")), faab_left=int(m.group("faab")),
            )
    if not rows:
        return pd.DataFrame(columns=["team", "seat", "rank", "faab_left", "faab_spent"])
    df = pd.DataFrame(rows.values())
    df["faab_spent"] = BUDGET - df["faab_left"]
    return df.sort_values("rank").reset_index(drop=True)


def refresh_budgets() -> pd.DataFrame:
    """Scrape the league home page and cache the balances."""
    from .yahoo import BASE, LeagueSession

    with LeagueSession() as s:
        df = parse_budgets(s.get_text(BASE))
    if not df.empty:
        df.to_csv(FAAB_CSV, index=False)
    return df


def cached_budgets() -> pd.DataFrame:
    if not FAAB_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(FAAB_CSV)


# ─────────────────────────────────────────────────────────────────── bid history
# Yahoo prefixes the player line with a glyph from its own icon font, which lands in the
# Private Use Area and so survives .strip() — it is not whitespace.
_PUA = re.compile(r"[\ue000-\uf8ff]")


def _clean(s: str) -> str:
    return _PUA.sub("", s).strip()


_AWARD = re.compile(r"^\$(?P<amt>\d+)\s+Winning Offer\s*$")
_LOSER = re.compile(r"^(?P<team>.+?)\s+\$(?P<amt>\d+)\s+\((?P<why>[^)]+)\)\s*$")


def parse_bids(page_text: str) -> pd.DataFrame:
    """Settled FAAB claims, winners and losers alike.

    Yahoo renders each claim as a run of lines: the player, "$N Winning Offer", zero or
    more losing bids, "Awarded To:", the winner, then a date. Line-based rather than
    DOM-based for the same reason parse_standings is — the markup is flex divs.
    """
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    rows, i = [], 0
    while i < len(lines):
        m = _AWARD.match(lines[i])
        if not m:
            i += 1
            continue
        player = _clean(lines[i - 1]) if i else ""
        win_amt, losers, j = int(m.group("amt")), [], i + 1
        winner, date = "", ""
        while j < len(lines) and j < i + 30:
            if lines[j] == "Awarded To:":
                winner = lines[j + 1].strip() if j + 1 < len(lines) else ""
                date = lines[j + 2].strip() if j + 2 < len(lines) else ""
                break
            lm = _LOSER.match(lines[j])
            if lm:
                losers.append((lm.group("team").strip(), int(lm.group("amt")), lm.group("why")))
            j += 1
        rows.append(dict(player=player, winning_bid=win_amt, winner=winner, date=date,
                         n_bidders=len(losers) + 1,
                         runner_up=max((a for _, a, _ in losers), default=None),
                         losers="; ".join(f"{t} ${a}" for t, a, _ in losers)))
        i = j + 1
    return pd.DataFrame(rows)


def refresh_bids() -> pd.DataFrame:
    from .yahoo import BASE, LeagueSession

    with LeagueSession() as s:
        df = parse_bids(s.get_text(BASE + "?transactionsfilter=faab"))
    if not df.empty:
        df.to_csv(BIDS_CSV, index=False)
    return df


def cached_bids() -> pd.DataFrame:
    if not BIDS_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(BIDS_CSV)


def market_summary(bids: pd.DataFrame | None = None,
                   budgets: pd.DataFrame | None = None) -> dict:
    """What this league has actually paid. Context for a suggestion, not a calibration.

    The FAB Offers feed only lists claims that went to a waiver *run*, so an uncontested
    pickup and a plain free-agent add never appear on it. That makes it an undercount, and
    by a knowable amount: the budgets page says what each team has really spent. The gap
    comes back as `unlisted_spend` rather than being quietly absorbed, because a median
    computed off the contested claims alone reads far cheaper than this league truly is.
    """
    b = cached_bids() if bids is None else bids
    if b is None or b.empty:
        return {"claims": 0}
    won = pd.to_numeric(b["winning_bid"], errors="coerce").dropna()
    contested = b[pd.to_numeric(b["n_bidders"], errors="coerce") > 1]
    out = {
        "claims": int(len(won)),
        "median": float(won.median()),
        "max": float(won.max()),
        "contested": int(len(contested)),
        "listed_spend": float(won.sum()),
    }
    bud = cached_budgets() if budgets is None else budgets
    if bud is not None and not bud.empty and "faab_spent" in bud.columns:
        total = float(pd.to_numeric(bud["faab_spent"], errors="coerce").sum())
        out["league_spend"] = total
        out["unlisted_spend"] = round(total - out["listed_spend"], 2)
    return out


# ─────────────────────────────────────────────────────────────────────────── bid
def weeks_left(week: int) -> int:
    """Weeks the pickup can still score for you, counting the week about to be played."""
    return max(1, LAST_WEEK - int(week) + 1)


def suggest(gain: float, budget_left: int, week: int, aggression: float = 1.0) -> dict:
    """Dollars to bid on a player worth `gain` points per week from here.

    `gain` is marginal lineup value (mega.needs), not raw points: it is what the player adds
    to an *optimal* lineup once you account for who he displaces and who you cut for him. A
    player who cannot crack your lineup has gain 0 no matter how good he looks, which is the
    whole reason a third tight end never gets a bid here.
    """
    wl = weeks_left(week)
    pts = max(0.0, float(gain)) * wl
    share = MAX_SHARE * (1 - math.exp(-pts / SATURATION)) * max(0.0, aggression)
    raw = BUDGET * share
    bid = int(min(round(raw), budget_left))
    # A $0 claim loses every tie, so anything genuinely worth adding bids at least $1.
    if gain > 0.05 and budget_left >= 1:
        bid = max(1, bid)
    return {
        "bid": bid,
        "max_worth": int(min(round(raw * 1.5), budget_left)),
        "season_pts": round(pts, 1),
        "weeks_left": wl,
        "pct_budget": round(100 * bid / BUDGET, 1),
    }


def rivals(budgets: pd.DataFrame | None = None, my_team: str | None = None) -> dict:
    """Who can outbid you, and by how much."""
    from .config import MY_TEAM

    b = cached_budgets() if budgets is None else budgets
    me = my_team or MY_TEAM
    if b is None or b.empty or "faab_left" not in b.columns:
        return {"known": False}
    mine_row = b[b["team"] == me]
    mine = int(mine_row["faab_left"].iloc[0]) if not mine_row.empty else None
    others = b[b["team"] != me]
    return {
        "known": True,
        "mine": mine,
        "richer": int((others["faab_left"] > (mine or 0)).sum()),
        "max_rival": int(others["faab_left"].max()) if not others.empty else 0,
        "median_rival": float(others["faab_left"].median()) if not others.empty else 0.0,
        "teams": int(len(b)),
    }
