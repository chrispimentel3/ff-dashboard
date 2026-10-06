"""mega/live_web.py — what the site's live scoreboard scores each week."""
from __future__ import annotations

import pandas as pd

from mega import live_web as L


def test_the_match_key_strips_what_espn_and_yahoo_spell_differently():
    # must agree with liveKey() in mega-bowl-web's src/lib/liveScoring.ts
    assert L.key("Michael Penix Jr.") == L.key("Michael Penix") == "michael penix"
    assert L.key("Ka'imi Fairbairn") == "kaimi fairbairn"
    assert L.key("Jaxon Smith-Njigba") == "jaxon smith njigba"
    assert L.key("Deebo Samuel Sr.") == "deebo samuel"
    assert L.key("Luther Burden III") == "luther burden"


def test_starters_only_with_their_nfl_teams_and_defenses_by_city_or_nickname():
    ros = pd.DataFrame([
        dict(team="TaylorMade", slot="QB", player="Bryce Young", nfl_team=None),
        dict(team="TaylorMade", slot="BN", player="Travis Kelce", nfl_team="KC"),
        dict(team="TaylorMade", slot="K", player="Tyler Loop", nfl_team=None),
        dict(team="TaylorMade", slot="DEF", player="Carolina", nfl_team=None),
        dict(team="A Place in the Hampton", slot="DEF", player="Commanders", nfl_team=None),
        dict(team="A Place in the Hampton", slot="WR", player="(Empty)", nfl_team=None),
    ])
    fx = pd.DataFrame([{"week": 4, "home": "TaylorMade", "away": "Saquon Enjoyer"}])
    nfl = pd.DataFrame({"full_name": ["Tyler Loop"], "team": ["BAL"]})
    out = L.build(2026, 4, ros, fx, [{"player": "Bryce Young", "team": "CAR", "proj": 18.1}], nfl, "TaylorMade")
    me = {s["slot"]: s for s in out["teams"]["TaylorMade"]}
    assert set(me) == {"QB", "K", "DEF"}
    assert me["QB"]["nfl_team"] == "CAR" and me["QB"]["proj"] == 18.1
    assert me["K"]["nfl_team"] == "BAL" and me["K"]["proj_avg"]
    assert me["DEF"]["key"] == "DEF:CAR"
    # the old name is followed to the new one, and Washington is ESPN's WSH
    them = {s["slot"]: s for s in out["teams"]["Saquon Hater"]}
    assert them["DEF"]["key"] == "DEF:WSH" and them["WR"]["player"] is None
    assert out["matchups"] == [["TaylorMade", "Saquon Hater"]]
