"""The Logic tab's content: why this dashboard is built the way it is.

Every other reference page here (Glossary) says what a tag on a player means. This page
says why the dashboard reaches for a particular number in the first place — Vegas lines
instead of a homemade model, what actually separates a WR1 from a WR2, why WOPR is weighted
the way it is. Each topic is a short plain-English answer first, then the actual mechanics
for anyone who wants them, as a bulleted "how it's calculated" — real constants pulled from
the modules that compute each number, not paraphrased, so this can't quietly drift from what
the code actually does.

Static, hand-maintained content — there is nothing here to compute. Update this file
directly when a rule (or a constant it quotes) changes; nothing else derives from it.
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
            "- **A line is a median, not a mean.** Weekly yardage is lopsided, so a receiving- "
            "or rushing-yards line is scaled up by a fitted curve, not a flat percentage — "
            "steep for a short line and fading toward zero for a long one (a 30-yard "
            "receiving line is worth about +29%, an 82-yard one only about +7%). Passing "
            "yards get no correction; a QB's yardage checks out symmetric.\n"
            "- **Anytime touchdown isn't `6 × P(scores)`.** That naive conversion overstates "
            "the top of the range by about 12%, because a player who scores twice in a game "
            "is rarer than treating each score as independent implies — the second score "
            "needs a second trip inside the ten. A curve fit on 21,458 player-games "
            "(`P(scores) → E[touchdowns | scored at least once]`) is used instead.\n"
            "- **Receptions and passing touchdowns are inverted straight from a Poisson "
            "distribution**, no skew correction — both check out as genuinely "
            "Poisson-distributed at the lines books actually post.\n\n"
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
            "role immediately. Fewer than 2 games played and he's placed from the depth "
            "chart instead, so a rookie who leads his team in targets in his one game so far "
            "isn't called a WR1 off it.\n\n"
            "- **WR1 / WR2** — simply the #1 / #2 target-share player on his own team.\n"
            "- **WR3 vs. WR4+** — the #3 target-share player only counts as **WR3** if he's "
            "also on the field at least half the snaps; otherwise he's **WR4+**.\n"
            "- **TE1-REC / TE1-BLK / TE2** — the highest-snap-share tight end on the team is "
            "**TE1-REC** if his target share clears 12%, else **TE1-BLK**. Every other tight "
            "end is **TE2**.\n"
            "- **RB: LEAD / COMMITTEE / RECEIVING / BACKUP** — by carry share: 50%+ is "
            "**LEAD**, 30%+ is **COMMITTEE**. Below that, an 8%+ target share makes him "
            "**RECEIVING**; otherwise **BACKUP**.\n"
            "- **QB: STARTER / QB-BACKUP** — 70%+ of his own team's pass attempts is the "
            "line.\n\n"
            "Once a player has a role, his own rate is shrunk toward his role's typical rate "
            "so two good games don't overstate a small sample: `x̂ = (his total + k × role "
            "average) ÷ (his opportunities + k)`. `k` is set per metric, not one universal "
            "number — heavier for noisier, more share-like metrics (target share: k=60; "
            "air-yards share: k=500; carry share: k=40; and so on).\n\n"
            "That shrunk rate `x̂` is then read two ways: `vs role = 100 × x̂ ÷ role average` "
            "(100 = average for his specific role — a WR1's 100 is a different real number "
            "than a WR3's 100) and `vs NFL = 100 × x̂ ÷ position average` (100 = NFL average "
            "at his position, full stop). The first tells you if he's playing like his role; "
            "the second tells you how good that role actually is."
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
            "- **Formula:** `WOPR = 1.5 × target share + 0.7 × air-yards share` — target "
            "share is his cut of his team's targets, air-yards share is his cut of his "
            "team's total intended depth of target. The weights match nflverse's own "
            "published `wopr` column to within rounding.\n"
            "- **Summed, never averaged.** Both shares are a summed numerator over a summed "
            "denominator across the whole window, never the mean of weekly ratios — a single "
            "low-volume game once produced a window WOPR of 1.33 by averaging in a week "
            "where team air yards happened to net out near zero.\n"
            "- **Early-season anchor.** This year's WOPR is blended with a 3-game prior from "
            "last year's data: `(g × wopr_now + 3 × wopr_prior) ÷ (g + 3)`, so a few games "
            "of swingy small-sample usage don't get read at face value. The prior's weight "
            "fades out on its own as more of the current season accumulates.\n"
            "- **Efficiency read.** A player's own points-per-game against what his WOPR "
            "alone would predict (a per-position linear fit) is the signal — well ahead of "
            "it and he's due for regression (sell), well behind it and the role is better "
            "than the box score (buy)."
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
            "- **What it answers.** Given the plays a player actually took part in, in the "
            "exact situations they happened in, how many points should they have produced?\n"
            "- **The model.** nflverse's `ff_opportunity` dataset — eight XGBoost models "
            "(three for run plays, five for pass plays), trained on public nflverse "
            "play-by-play. Nothing here is modeled from scratch; this dashboard only "
            "re-scores the output into half-PPR.\n"
            "- **Air yards are the real throw, not a prediction** — the model takes each "
            "target as a given fact (it happened, and it went this far downfield) and asks "
            "only what it should have yielded.\n"
            "- **What that makes xFP:** a measure of *efficiency given opportunity*, not a "
            "forecast that the opportunity continues. A receiver with a big target share and "
            "a large negative gap isn't being told \"expect this again\" — he's being told "
            "\"what he already got should have scored more than it did.\"\n"
            "- That's the right input for a buy-low signal, and the wrong thing to use alone "
            "as a projection — it's one input among several elsewhere in the dashboard, not "
            "the whole projection."
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
            "Four different profiles, one per position, each a weighted blend of components "
            "normalized 0–1 against a real anchor number, then combined into the 0–100 fit "
            "score:\n\n"
            "- **QB** — 80% rushing volume (his rush attempts per game, normalized against "
            "the elite bar of 100+ a season / 5.9 a game) + 20% his own points-per-game "
            "(normalized against a 24-point ceiling). 55+ rush attempts a season (3.2/g) is "
            "the **RUSH** mark, 100+ (5.9/g) is **RUSH+** — the \"Konami code\" quarterback.\n"
            "- **RB** — 40% volume (carries/game against an 18/game bellcow pace) + 35% age "
            "(full credit through 25, fading to near nothing by 28+) + 25% draft capital "
            "(full credit inside his draft's first 3 rounds, partial credit through pick "
            "72).\n"
            "- **WR** — 45% \"stickiness\" (half from his first-down-per-target rate against "
            "a 45% bar, half from first downs per game against a bar of 5) + 30% career "
            "window (peaks in years 3–6 of his career) + 25% an age gate (drops off sharply "
            "at 32+, with a couple of named exceptions for proven outliers).\n"
            "- **TE** — 40% target share (against a 25% bar) + 30% his offense's scoring "
            "rate (against a 26-points-a-game bar) + 30% whether his own team lacks a true "
            "#1 WR (a teammate inside the top-60 ADP cuts this component way down, since a "
            "thin WR room is exactly what routes a TE more volume).\n\n"
            "Where the ideal stat isn't in free public data (routes run, specifically), a "
            "clearly labeled proxy stands in for it (first-down rate) rather than a silent "
            "guess. The score is a fit against this specific blueprint, not a projection — a "
            "low score doesn't mean a bad player, it means the profile that's worked in this "
            "league before doesn't describe him."
        ),
    },
    {
        "key": "playoff_odds",
        "title": "Why playoff odds move so much early in the season",
        "headline": (
            "A team's projected points per week is itself a guess, especially a few weeks "
            "in — and the simulation now says so explicitly, instead of quietly treating "
            "an early-season average as gospel and simulating 6,000 seasons as if it were."
        ),
        "detail": (
            "- **The simulation.** 6,000 simulated seasons; every remaining game is a score "
            "drawn around each team's projected points per week, with this league's own "
            "week-to-week scoring spread layered on top.\n"
            "- **What was missing.** Early on, that scoring spread was the *only* "
            "randomness — the points-per-week estimate itself was treated as a known "
            "constant. Over a dozen-plus remaining weeks, even a modest, noisy edge from "
            "three games compounds into near-certainty through sheer repetition, which is "
            "exactly how a 3-0 start turned into a 99% playoff lock in week 3.\n"
            "- **The fix.** Each team's points-per-week estimate now carries its own "
            "uncertainty — textbook standard error of the mean: `this league's weekly "
            "scoring spread ÷ √(games played)` — wide with three games on file, narrowing "
            "every week as more come in.\n"
            "- **How it's applied.** Each simulated season draws that team's *true* level "
            "ONCE (not re-rolled every week — a team is consistently better or worse than "
            "its early-season average, not randomly different each Sunday), and every "
            "week's score is then drawn around that fixed level.\n"
            "- **Net effect.** Early-season odds spread out and pull back from the extremes, "
            "then tighten toward what the record actually shows as the season goes on and "
            "each team's true level becomes clearer."
        ),
    },
    {
        "key": "matchup_model",
        "title": "\"Matchup\" on a player's page — what it actually weighs",
        "headline": (
            "A defense that \"allows a lot of points\" isn't necessarily bad — it might just "
            "have faced a run of good offenses. This model asks the harder question: given "
            "who a defense actually faced, did it allow more or less than expected? That "
            "answer, blended with the Vegas line for this specific week, is what moves a "
            "player's number up or down from his normal week."
        ),
        "detail": (
            "- **Opponent-adjusted defense rating.** Points a defense allowed to a position, "
            "measured against those players' own normal week — not the raw total, which "
            "rewards a defense that got lucky with who it played. Formula: `rating = "
            "(actual points allowed + k × position average × last-year's prior) ÷ "
            "(those players' baselines + k × position average)`, then centered so the "
            "league average = 1.00.\n"
            "- **How fast the in-season sample takes over.** `k` (how many \"pseudo-games\" "
            "of the prior get blended in) and how much of last year carries over are both "
            "fit per position: run defense barely carries over year to year, so RB ratings "
            "lean on the in-season sample fast; pass defense — and especially wide receiver, "
            "where an in-season sample is close to pure noise for over a month — leans on "
            "last year's number much longer.\n"
            "- **Vegas implied team total**, relative to that team's own average line this "
            "season — forward-looking, and it prices things (a backup QB, bad weather, an "
            "injury) a defense rating can't see yet.\n"
            "- **Combining the two.** Not a straight multiply — each input is raised to a "
            "fitted exponent first: `mult = rating^b_def × implied_relative^b_impl`. A "
            "defense that's allowed 30% more than expected doesn't actually move a player "
            "30%; most of that 30% was noise, and the exponent (0.288 for QB, for example) "
            "is the fraction that's real signal.\n"
            "- **Applied to a matchup-*neutral* baseline** (this dashboard's usage-based "
            "estimate, not a projection that already has the matchup priced in elsewhere) "
            "to get an expected-points number for the week. A future week with no Vegas "
            "line posted yet falls back to the defense rating alone, labeled as such, "
            "rather than guessing at a line.\n\n"
            "**Treat anything inside about ±3% as within the noise floor** — the fitted "
            "model sorts reliably at the extremes (a bottom-5 matchup, a big Vegas total) "
            "and not reliably in the middle."
        ),
    },
    {
        "key": "trade_value",
        "title": "What the \"value\" number in a trade actually is",
        "headline": (
            "It's not this dashboard's opinion — it's FantasyCalc's crowd-sourced consensus "
            "trade value, the same number thousands of other managers are looking at when "
            "they weigh a deal. Higher is better, the scale runs roughly 0-10,000, and it's "
            "already tuned to this league's own settings (half-PPR, 12 teams, 1 QB)."
        ),
        "detail": (
            "- **The source.** FantasyCalc's live redraft value API, queried with "
            "`isDynasty=false, numQbs=1, numTeams=12, ppr=0.5` — this league's exact "
            "settings, not a generic default.\n"
            "- **Trades page** asks *is this specific deal fair, for these two teams, right "
            "now?* Both sides of a proposed trade get summed on this same value scale, and "
            "the fairness percentage is the smaller side over the larger — 95%+ is close to "
            "even. \"Addresses\" names the position the trade actually fixes for you, read "
            "off the other manager's own roster.\n\n"
            "**What it doesn't do yet:** neither view checks what a trade does to your "
            "actual starting lineup — whether the player you'd receive would even crack "
            "your starters, or fill a bye-week gap, versus just adding bench depth. That's "
            "a heavier per-trade computation (it existed as \"trade around one player\" "
            "search in the original dashboard) and isn't wired up here yet."
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
            "- **The rule.** `share = Σ(his count across the window) ÷ Σ(team count across "
            "the window)` — never `mean(his weekly ratio)`.\n"
            "- **Worked example.** Two games: 10-of-40 team targets, then 2-of-20. "
            "Averaging the two weekly rates gives 17.5%. Summing first gives the real "
            "answer — he caught 12 of 60 team targets, which is 20.0%.\n"
            "- **Why it matters.** Averaging weekly ratios lets one low-volume week (a "
            "blowout, a short outing) swing a season number as much as a full one does. "
            "Summing first treats every target, snap and carry the same regardless of which "
            "week it happened in — the only version of \"what share of the offense did he "
            "get\" that's actually being asked."
        ),
    },
]


def frame() -> list[dict]:
    return TOPICS
