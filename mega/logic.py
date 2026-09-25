"""The Logic tab's content: why this dashboard is built the way it is.

Every other reference page here (Glossary) says what a tag on a player means. This page
says why the dashboard reaches for a particular number in the first place — Vegas lines
instead of a homemade model, what actually separates a WR1 from a WR2, why WOPR is weighted
the way it is. Each topic is a short plain-English answer first, then the actual mechanics
for anyone who wants them — same shape as docs/expected-points.md's "short version" lede,
just condensed for the dashboard instead of a standalone doc.

Static, hand-maintained content — there is nothing here to compute. Update this file
directly when a rule changes; nothing else derives from it.
"""
from __future__ import annotations

TOPICS = [
    {
        "key": "vegas",
        "title": "Why Vegas lines, not a homemade projection",
        "headline": (
            "A sportsbook's line already has the injury report, the weather and the game "
            "plan priced in — money is on the line, so it gets built by people with better "
            "information than a spreadsheet has. We take that price and re-score it to "
            "half-PPR instead of guessing at our own."
        ),
        "detail": (
            "Every starter has sportsbook lines on his catches, his yards and his chance of "
            "scoring. Scored in half-PPR, they become a projection someone else already paid "
            "to get right. Three corrections happen on the way in, each fitted against "
            "2021–2025 nflverse outcomes rather than assumed:\n\n"
            "- **A line is a median, not a mean.** Weekly yardage is lopsided — a receiver "
            "priced at 30.5 yards averages 39 — so projections sit deliberately above the "
            "posted line. Passing yards are the exception and are left alone.\n"
            "- **Anytime touchdown isn't expected touchdowns.** `6 × P(scores)` ignores "
            "multi-score games and undershoots the best scorers by about 12%; a measured "
            "multiplier is used instead.\n"
            "- **Receptions and passing touchdowns check out as Poisson-distributed** at the "
            "lines books actually post, so those two are inverted directly with no skew "
            "correction layered on top.\n\n"
            "The free odds API is metered (500 credits/month), so refreshes hold to roughly "
            "weekly and a sweep that can't finish a full slate is skipped outright — half a "
            "week of prices would rank priced games over unpriced ones, which is worse than "
            "no prices at all."
        ),
    },
    {
        "key": "roles",
        "title": "What makes a WR1 a WR1 (and every other role)",
        "headline": (
            "A raw number means little without the role attached to it. A 20% target share "
            "is a strong season for a WR3 and a warning sign for a WR1 — same number, "
            "opposite readings — so every player gets a role before his usage gets judged "
            "against it."
        ),
        "detail": (
            "Roles are read from each player's own last 3 games played, not the season to "
            "date — usage shifts (an injury ahead of him, a scheme change) should change his "
            "role immediately, not average in with games that no longer describe him. A "
            "player with fewer than 2 games played is placed from the depth chart instead, "
            "so a rookie who leads his team in targets in his one game so far isn't called a "
            "WR1 off it.\n\n"
            "- **WR1 / WR2 / WR3 / WR4+** — ranked by target share on his own team. Rank 3 "
            "only counts as **WR3** if he's also on the field at least half the snaps; "
            "otherwise he's **WR4+**.\n"
            "- **TE1-REC / TE1-BLK / TE2** — the highest-snap-share tight end on the team is "
            "**TE1-REC** if his target share clears 12%, else **TE1-BLK** (on the field, not "
            "the target). Every other tight end is **TE2**.\n"
            "- **RB: LEAD / COMMITTEE / RECEIVING / BACKUP** — by carry share: 50%+ is "
            "**LEAD**, 30%+ is **COMMITTEE**. Below that, an 8%+ target share makes him "
            "**RECEIVING**; otherwise **BACKUP**.\n"
            "- **QB: STARTER / QB-BACKUP** — by share of his own team's pass attempts, 70% "
            "the line.\n\n"
            "Once a player has a role, his usage is read three ways: his own rate (shrunk "
            "toward the role's typical rate, so two good games don't overstate a small "
            "sample), where that rate sits against players in the *same role* (100 = "
            "average for a WR1), and where it sits against the *position* as a whole (100 = "
            "NFL average WR). The first tells you if he's playing like his role; the other "
            "two tell you how good that role actually is."
        ),
    },
    {
        "key": "wopr",
        "title": "WOPR — Weighted Opportunity Rating",
        "headline": (
            "Targets alone miss how far downfield those targets travel; air yards alone miss "
            "how often he's targeted at all. WOPR blends both into one number for judging how "
            "much receiving opportunity a WR or TE is actually getting, independent of "
            "whether he's converted it into points yet."
        ),
        "detail": (
            "`WOPR = 1.5 × target share + 0.7 × air-yards share` — target share is his cut of "
            "his team's targets, air-yards share is his cut of his team's total intended "
            "depth of target. The weights match nflverse's own published `wopr` column to "
            "within rounding.\n\n"
            "Everything is a summed numerator over a summed denominator across the window, "
            "never the mean of weekly ratios — a single low-volume game once produced a "
            "window WOPR of 1.33 by averaging a week where team air yards happened to net "
            "out near zero. Early in a season, this year's WOPR is anchored to a 3-game prior "
            "from last year's data (`(g·wopr_now + 3·wopr_prior) / (g + 3)`) so four games of "
            "swingy small-sample usage don't get read at face value; the prior's weight fades "
            "out on its own as more of the current season accumulates. A player's own "
            "points-per-game against what his WOPR alone would predict (a per-position linear "
            "fit) is the efficiency signal — well ahead of it and he's due for regression "
            "(sell), well behind it and the role is better than the box score (buy)."
        ),
    },
    {
        "key": "xfp",
        "title": "Expected points (xFP) — and what it isn't",
        "headline": (
            "We do not model expected points ourselves. nflverse does — eight machine-learning "
            "models trained on years of real play-by-play — and we just re-score its output "
            "to this league's half-PPR rules."
        ),
        "detail": (
            "xFP answers: given the plays a player actually took part in, in the exact "
            "situations they happened in, how many points should they have produced? The "
            "key detail is that **air yards are the real throw, not a prediction** — the "
            "model takes each target as a given fact (it happened, and it went this far "
            "downfield) and asks only what it should have yielded. That makes xFP a measure "
            "of *efficiency given opportunity*, not a forecast that the opportunity "
            "continues. A receiver with a big target share and a large negative gap isn't "
            "being told \"expect this again\" — he's being told \"what he already got should "
            "have scored more than it did.\" That's the right input for a buy-low signal, and "
            "the wrong thing to use alone as a projection (which is why it's only one input "
            "among several elsewhere in the dashboard, not the whole projection)."
        ),
    },
    {
        "key": "archetypes",
        "title": "Blueprint fit — what \"archetype\" scoring means",
        "headline": (
            "Every league has a shape to who actually wins it. Blueprint fit scores each "
            "player 0–100 against the specific profile that's won Mega Bowl, not against a "
            "generic \"good fantasy player\" idea — a fit score is about whether this player "
            "matches what tends to work here."
        ),
        "detail": (
            "Four different profiles, one per position, thresholds prorated to a per-game "
            "rate so an in-season player is compared fairly to a full-season target:\n\n"
            "- **QB** — real rushing volume: 55+ rush attempts a season is the mark, 100+ is "
            "the top tier (the \"Konami code\" quarterback).\n"
            "- **RB** — early draft capital and youth: drafted in the first 3 rounds, in "
            "years 1–3 of his career. Winning backs average age 25; 27+ is a fade.\n"
            "- **WR** — a sticky first-down producer (12%+ of his routes end in a first down) "
            "in the breakout window (his 3rd–6th year), age 32 and under.\n"
            "- **TE** — the clear passing-game alpha on an offense without a top-60-ADP WR to "
            "compete with, averaging 20+ team points a game.\n\n"
            "Where the ideal stat isn't in free public data (routes run, specifically), a "
            "clearly labeled proxy stands in for it (first-down rate) rather than a silent "
            "guess. The score is a fit against this specific blueprint, not a projection — a "
            "low score doesn't mean a bad player, it means the profile that's worked in this "
            "league before doesn't describe him."
        ),
    },
    {
        "key": "rates",
        "title": "One house rule: rates are never averaged",
        "headline": (
            "Every rate anywhere in this dashboard — target share, snap share, catch rate, "
            "WOPR, xFP efficiency — is a summed numerator over a summed denominator across "
            "whatever window is in view. It is never the average of several weekly ratios."
        ),
        "detail": (
            "The difference matters more than it sounds like it should. Two games of 10-of-40 "
            "targets and 2-of-20 targets average to 17.5% if you average the two weekly "
            "rates — but the player actually caught 12 of 60 team targets, which is 20.0%. "
            "Averaging weekly ratios lets one low-volume week (a blowout, a short outing) "
            "swing a season number as much as a full one does. Summing first and dividing "
            "once treats every target, snap and carry the same regardless of which week it "
            "happened in — which is the only version of the question (\"what share of the "
            "offense did he get\") that's actually being asked."
        ),
    },
]


def frame() -> list[dict]:
    return TOPICS
