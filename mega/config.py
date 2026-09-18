"""League constants for Mega Bowl 2026 (Yahoo league 173489)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

LEAGUE_ID = "173489"
LEAGUE_KEY_GAME = "nfl"  # Yahoo game code; league_key = "<game_id>.l.173489" when using the API
LEAGUE_URL = f"https://football.fantasysports.yahoo.com/f1/{LEAGUE_ID}"

N_TEAMS = 12
N_ROUNDS = 15
MY_SEAT = 8
MY_TEAM = "TaylorMade"
SCORING_NAME = "half-PPR"

# Current Yahoo team names by draft seat (seat = R1 pick order from the draft results page).
TEAM_BY_SEAT = {
    1: "deez nuts",
    2: "Burrow-Chase combo",   # was "TyRick Hill"; confirmed by 13 of 15 draft picks still on that roster (2026-09-18)
    3: "Undisputed",
    4: "Heartbreak Drake",
    5: "L'Omar",
    6: "Barkley’s Balls Deep",
    7: "A Place in the Hampton",
    8: "TaylorMade",
    9: "Revenge of the Smith",
    10: "Don R.I.C.O",
    11: "BillsMafia",
    12: "Crabcakes and Football",
}
SEAT_BY_TEAM = {v: k for k, v in TEAM_BY_SEAT.items()}

# Names as they appear in the draft-board HTML (some were renamed since draft day).
DRAFT_BOARD_NAME_BY_SEAT = {
    1: "deez nuts", 2: "TyRick Hill", 3: "Undisputed", 4: "Any Given Ya…",
    5: "L'Omar", 6: "I Chase Brow…", 7: "Bed Bath & B…", 8: "TaylorMade",
    9: "JAH PLS", 10: "Play Action …", 11: "BillsMafia", 12: "Crabcakes an…",
}

# Yahoo default half-PPR scoring (edit if the league uses bonuses).
SCORING = dict(
    pass_yd=0.04, pass_td=4, pass_int=-1,
    rush_yd=0.10, rush_td=6,
    rec=0.5, rec_yd=0.10, rec_td=6,
    fum_lost=-2, two_pt=2, ret_td=6,
)

# Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 1 W/R (flex), 1 K, 1 DEF.
LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R": 1, "K": 1, "DEF": 1}
# The flex is W/R: RB or WR only. A TE can't start there in Mega Bowl (Chris, 2026-09-18) —
# this used to include TE, which let Start/Sit put a second tight end in an illegal slot.
FLEX_ELIGIBLE = {"RB", "WR"}

NEWS_FEEDS = [
    "https://www.espn.com/espn/rss/nfl/news",
    "https://www.rotowire.com/rss/news.php?sport=NFL",
    "https://profootballtalk.nbcsports.com/feed/",
]
