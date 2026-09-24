"""Plain English for every role and flag the dashboard prints.

The role layer (mega/roles.py) thinks in codes — WR1, TE1-BLK, COMMITTEE, ROLE+, GL. They
are precise and they are useless to read. Someone who knows football but has never seen
this app should be able to look at a tag and know what it is claiming about the player,
without a spec open beside them.

So this module is the one place that turns a code into words, and every surface — the
tables, the player card, the waiver report, the digest — reads from here. One definition,
one wording, everywhere. If a tag ever means something different in two places, it is
because someone wrote a second definition instead of using this one.

Three levels of explanation, because three different amounts of room exist:

    label    two or three words, for a table cell
    plain    one sentence, assuming ordinary football knowledge and nothing else
    matters  why a fantasy manager would care

Deliberately no jargon in `plain`. Not "target share", not "route participation", not
"role index" — those are the things being explained, so they cannot do the explaining.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Term:
    code: str
    label: str
    plain: str
    matters: str = ""
    kind: str = "role"


# ---------------------------------------------------------------- roles (§12.1)
ROLES: dict[str, Term] = {
    "WR1": Term("WR1", "WR1",
                "His team's number one receiver — he gets thrown at more than any other "
                "wideout on the roster.",
                "The safest kind of receiver to start. Volume holds up even in bad games."),
    "WR2": Term("WR2", "WR2",
                "The second receiver on his team.",
                "Usually startable, but he needs the offence to move the ball to pay off."),
    "WR3": Term("WR3", "WR3",
                "The third receiver, and on the field for at least half his team's plays.",
                "A flex play in good matchups. One injury ahead of him changes everything."),
    "WR4+": Term("WR4+", "Depth WR",
                 "Fourth or lower in the receiver pecking order, or barely on the field.",
                 "Not startable as things stand. Worth watching only if his role grows."),
    "TE1-REC": Term("TE1-REC", "TE1 receiving",
                    "His team's starting tight end, and one they actually throw to.",
                    "The only kind of tight end worth a lineup spot most weeks."),
    "TE1-BLK": Term("TE1-BLK", "TE1 blocking",
                    "His team's starting tight end, but he is out there to block — the "
                    "passes rarely come his way.",
                    "Plays a lot and scores nothing. Snap counts will flatter him."),
    "TE2": Term("TE2", "Backup TE",
                "A second-string tight end, behind the starter in both playing time and "
                "targets.",
                "Only matters if the starter gets hurt."),
    "LEAD": Term("LEAD", "Lead back",
                 "He gets most of his team's carries — more than half of them.",
                 "The workhorse. Volume alone makes him startable."),
    "COMMITTEE": Term("COMMITTEE", "Committee back",
                      "He splits the carries with another back, taking roughly a third to "
                      "a half of them.",
                      "Touchdown-dependent. A good week needs the ball near the goal line."),
    "RECEIVING": Term("RECEIVING", "Receiving back",
                      "He does not get many carries, but he is the back they throw to — "
                      "often the one on the field on third down.",
                      "Worth more in this league than in most, because catches score half "
                      "a point each."),
    "BACKUP": Term("BACKUP", "Backup RB",
                   "A reserve back with no regular role.",
                   "A handcuff at best. He needs an injury ahead of him."),
    "STARTER": Term("STARTER", "Starting QB",
                    "His team's starting quarterback.",
                    "Only starters are worth rostering at quarterback."),
    "QB-BACKUP": Term("QB-BACKUP", "Backup QB",
                      "A backup quarterback — he is not starting unless something happens "
                      "to the man ahead of him.",
                      "Nothing to do here unless the starter goes down."),
}

# ---------------------------------------------------------------- flags (§12.5, §4.2)
FLAGS: dict[str, Term] = {
    "ROLE+": Term("ROLE+", "playing up",
                  "He is doing the job of the man above him — a third receiver producing "
                  "like a second, or a committee back running like the lead.",
                  "The strongest buy signal here. Offences tend to notice this before "
                  "fantasy managers do.", kind="flag"),
    "ROLE-": Term("ROLE-", "slipping",
                  "He is producing less than the man BELOW him on the depth chart.",
                  "His job is at risk. Sell or drop before the snap count catches up.",
                  kind="flag"),
    "TGT": Term("TGT", "target hog",
                "A big share of his team's throws go his way — far more than is normal "
                "for his spot in the pecking order.",
                "Targets are the most repeatable thing a receiver does. This is the flag "
                "that survives a quiet week.", kind="flag"),
    "AIR": Term("AIR", "downfield",
                "He is the one they throw deep to, not just the one they throw often to.",
                "Big-play upside, and more week-to-week swing. Boom or bust.", kind="flag"),
    "SNAP": Term("SNAP", "every-down",
                 "He rarely leaves the field.",
                 "Playing time is the floor under everything else. It usually comes "
                 "before the production does.", kind="flag"),
    "LEAD": Term("LEAD", "owns carries",
                 "He is taking the clear majority of his team's running plays.",
                 "Workload like this survives a bad game. Coaches do not bench it.",
                 kind="flag"),
    "GL": Term("GL", "goal line",
               "He is the one getting the ball inside the ten-yard line.",
               "This is where touchdowns come from, and touchdowns are six points.",
               kind="flag"),
    "1D/RR": Term("1D/RR", "moves chains",
                  "When he is thrown to, it tends to produce a first down.",
                  "Chain-movers keep drives alive, so offences keep going back to them.",
                  kind="flag"),
}

# ---------------------------------------------------------------- persistence (§4.2)
SUSTAINED = Term("sustained", "", "It has held for at least two of his last three games.",
                 "A pattern, not an afternoon.", kind="tag")
SPIKE = Term("spike", "1 game",
             "It happened in one game only, not across the three being looked at.",
             "Could be the start of something, could be a good matchup. Not proof of "
             "anything yet.", kind="tag")

UNKNOWN_ROLE = Term("", "no role yet",
                    "He has not played an offensive snap this season, so there is nothing "
                    "to read yet.",
                    "Blank rather than a guess — see for yourself once he plays.")


# ---------------------------------------------------------------- rendering
def role_label(code: object) -> str:
    """"COMMITTEE" -> "Committee back". Unknown codes come back as themselves."""
    c = str(code or "").strip()
    if not c:
        return UNKNOWN_ROLE.label
    t = ROLES.get(c)
    return t.label if t else c


def flag_label(code: object, tag: object = None) -> str:
    """"GL" with a one-game tag -> "goal line (1 game)"."""
    c = str(code or "").strip()
    t = FLAGS.get(c)
    label = t.label if t else c
    return f"{label} ({SPIKE.label})" if str(tag or "") == "spike" else label


def cell(role: object, flags=None, tags: dict | None = None, max_flags: int = 2) -> str:
    """The table-cell version: role, then the flags worth the width.

    Capped at two flags because a cell is a cell. Order of precedence:

      1. ROLE+ / ROLE-, which say something is CHANGING — worth more than a description
      2. flags that held across the window, ahead of ones that happened once, so a single
         good afternoon cannot push a durable signal out of the only two slots there are
      3. the registry order, which runs roughly most to least reliable
    """
    tags = tags or {}
    parts = [role_label(role)]
    fl = list(flags or [])
    fl.sort(key=lambda f: (f not in ("ROLE+", "ROLE-"),
                           tags.get(f) == "spike",
                           list(FLAGS).index(f) if f in FLAGS else 99))
    shown = [flag_label(f, tags.get(f)) for f in fl[:max_flags]]
    extra = len(fl) - len(shown)
    if extra > 0:
        shown.append(f"+{extra} more")
    return " · ".join(parts + shown)


def sentence(role: object, flags=None, tags: dict | None = None, name: str = "He") -> str:
    """A full plain-English line for the player card."""
    r = ROLES.get(str(role or "").strip())
    out = [f"**{role_label(role)}.** " + (r.plain if r else UNKNOWN_ROLE.plain)]
    tags = tags or {}
    for f in (flags or []):
        t = FLAGS.get(f)
        if not t:
            continue
        how = SUSTAINED.plain if tags.get(f) == "sustained" else SPIKE.plain
        out.append(f"**{t.label.capitalize()}** — {t.plain} {how}")
    return "  \n".join(out)


def frame(kind: str = "all") -> pd.DataFrame:
    """The glossary itself, for the panel on screen."""
    rows = []
    if kind in ("all", "role"):
        for t in ROLES.values():
            rows.append({"tag": t.label, "code": t.code, "what it means": t.plain,
                         "why it matters": t.matters, "group": "Role"})
    if kind in ("all", "flag"):
        for t in FLAGS.values():
            rows.append({"tag": t.label, "code": t.code, "what it means": t.plain,
                         "why it matters": t.matters, "group": "Flag"})
    if kind in ("all", "tag"):
        rows.append({"tag": f"({SPIKE.label})", "code": "spike", "what it means": SPIKE.plain,
                     "why it matters": SPIKE.matters, "group": "How long"})
        rows.append({"tag": "no marker", "code": "sustained", "what it means": SUSTAINED.plain,
                     "why it matters": SUSTAINED.matters, "group": "How long"})
    return pd.DataFrame(rows)


HEADLINE = (
    "**Role** is the job a player actually has on his own team right now, worked out from "
    "his last three games rather than from where he was drafted. **Flags** after it are "
    "things he is doing unusually well for that job. A flag marked *(1 game)* happened "
    "once; anything else held across the window."
)
