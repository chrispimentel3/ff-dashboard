# Trade logic v2: agreed design

Agreed with Chris over several rounds on 2026-10-22, before any code. This is the reference for
the build; anything not written here is still open.

## Why

The trade cards felt like offers nobody in Mega Bowl would take. Checked against the data:

- **Fairness was league-blind.** Cards judged "fair" with FantasyCalc, and live search and the
  evaluator used a consensus-rank curve. Neither knows this league's waiver pool: Kelce (8.4/g,
  +1.4 over the waiver TE) was priced like Breece Hall, and Jayden Daniels scores less than
  Mahomes on waivers but still counted as an asset. All 11 cards were 2-for-1s, 6 padded with a
  QB or TE.
- **No schedule.** Cards ranked on a season-average number (`engine_config()` has no horizon),
  so byes, Vegas totals, injury returns and weeks 15–17 were ignored until the title sim, which
  only saw the top 25.
- **The acceptance model agreed with Chris on 7 of 11 labelled offers** (see Calibration). The
  four misses are the rules below.

What already works and stays: the engine's roster-rebuild valuation (`trade_engine.evaluate_core`)
measures points over replacement correctly, and the title sim prices the finalists.

## Currency: points over this league's waiver level, week by week

- **Waiver level** is the best healthy available player at each position. Waiver-wire (claim
  period) and free agents count the same. Recomputed every refresh.
- **Weekly horizon** from `proj_ros` weekly values: byes, Vegas implied totals, injury return
  weeks are already in them.
- **Week weights depend on record.** Each week counts by how much winning it moves that team's
  title odds, read off the season sim. A likely top-2 seed barely values week 15 (bye); the more
  a team wins, the more weight shifts to weeks 15–17. Exception: a team that is out of it
  (under ~10% playoff odds) still plays for wins now, so its next few weeks count and the
  playoffs don't.
- **Bench bonus** shrinks to injury insurance only; byes are now counted week by week.
- **Every candidate** is ranked on this weighted number; the top 25 still go through the full
  title sim. Making a rival stronger is already penalised there.

## Our side

Pure points in that currency (names don't matter to us). Cards add flags rather than changing
the value:

- **Role risk:** recent share of team opportunity falling. Chase Brown is 25%/29% in weeks 1–2,
  then 14%/20%/20%. Chris's read was "role could diminish after a 50/50 split week".
- **Rebound bet:** projection well above this season's points (Brown 14.8 projected vs 10.6).
- **Need view:** our weakest starting spot before and after (Chris on offer 6: "doesn't solve my
  third RB issue").
- **Back for the playoffs:** an injured player whose return week is before 15 is a buy, with the
  reason shown.

## Their side: one "worth to them" function

One function used by cards, live search, team search and the evaluator. FantasyCalc and the
consensus-rank curve leave the acceptance math.

1. **Their lineup gain** in the currency above, with their own week weights.
2. **Position tiers, league-wide.** RB1 = top-12 RB, RB2 = 13–24; same for WR. A starting RB
   outweighs a non-elite TE at equal points ("TE for an RB1 is too big a gap").
3. **Elite QB/TE.** The tier above the first big gap at the top of the position (today: TE Bowers
   14.1 and McBride 13.6, then a 2-pt gap to 11.6; QB only Josh Allen clears waiver Mahomes by
   more than a point). Other QBs/TEs are worth what they add to the partner's lineup, plus name
   value if established. Kittle has name value.
4. **Name value is recent.** Established = a top-24 RB/WR or top-12 QB/TE season last year, or in
   two of the last three (8+ games). Rookies and fading names (Rachaad White) count as unproven:
   weak currency to pay with, cheap to buy.
5. **Best player wins.** Packages don't add linearly: the side getting the best player carries
   extra weight, and a team's stud (overall top few) is effectively untouchable.
6. **Throw-ins that won't start for them count against the deal** (they have to cut someone),
   unless the throw-in fills their weakest spot (Kelce to a TE-less L'Omar worked; Daniels to
   Don R.I.C.O made it a no).
7. **Fandom.** A manager's favourite team's players weigh more for him: harder to buy from him,
   easier to sell to him.
8. **Too small to bother.** Deals that move neither side by about 1 pt/wk are dropped.
9. **Lopsided ceiling stays.** The league only vetoes a really lopsided trade.

**Output:** a percentage plus the "why": the two or three reasons that moved it, written for a
text message. The weights are judgment, labelled as estimates, and tuned until the calibration
set below passes.

### Fandom (from Chris)

| Team | NFL team |
|---|---|
| Crabcakes and Football | CIN |
| Saquon Hater | DET |
| Revenge of the Smith | PHI |
| Undisputed | CHI |
| L'Omar | BAL |
| deez nuts | BAL |
| Allen Moore connection | NE |
| BillsMafia | BUF |

Not given: Don R.I.C.O, Heartbreak Drake, Burrow-Chase combo.

## Calibration set

Chris's call on whether the partner says yes (Y / M / N), with the model before v2. Season-average
numbers from 2026-10-22 data. "→" means TaylorMade gives the left side and gets the right.

| # | Partner | Offer | Chris | Old model | Why (Chris) |
|---|---|---|---|---|---|
| 1 | deez nuts | Boston → Pickens | N | .01 | Pickens wouldn't be moved |
| 2 | Crabcakes | R. White → Monangai | N | .01 | White unknown, short term; too small to care |
| 3 | L'Omar | Kittle → B. Irving | N | .83 ✗ | TE for an RB1 is too big a gap |
| 4 | Undisputed | Kittle + B. Hall → Bowers | M | .77 ✗ | Likes Bowers; good intention |
| 5a | Don R.I.C.O | Watson → Hubbard | M | .26 | Fine trade |
| 5b | Don R.I.C.O | Watson + J. Daniels → Hubbard | N | .54 ✗ | Adding the QB makes it unlikely |
| 6 | L'Omar | B. Hall + Kelce → J. Warren | Y | .60 | Yes, but doesn't fix my RB3 |
| 8 | Allen Moore | R. White + Boston → Skattebo | N | .65 ✗ | Needs wins now; doesn't help him |
| 9 | deez nuts | Hall + Kelce + Daniels → Gibbs | N | .00 | Won't trade the overall RB1 |
| 10 | Crabcakes | Watson → Chase Brown | N | .32 | Bengals fan; question-mark RB |

**Pass bar:** v2 matches at least 9 of 10 (Y ≥ 50%, M 20–50%, N < 20%). This becomes a test.

Offer 7 (R. White + Egbuka → Davante Adams, out until week 12, BillsMafia) is an **our-side**
check: Chris wouldn't make it. With near-term weeks weighted, it should come out at or below
zero for us.

## Removed

The trade-offer log (`service/offer_log.py`, the `/offers` endpoints, "Your offers" on the Trades
page). Chris negotiates by text and never logged anything, so the acceptance model will never be
fitted from it; the calibration set replaces it.

## Build order (one commit each)

Status: steps 1–3 done 2026-10-22. The rules live in `mega/trade_worth.py`; its (J) constants
were picked by a grid search for the setting that matches all ten labels with the widest
margin and stays at 9+ when any one constant moves a step. `tests/test_trade_worth.py` holds the
bar on a league frozen that day (`tests/fixtures/trade_worth_calibration.json`).

Step 3 as built: `title_odds.Model.week_weights()` reads each week's weight off the title sim
(regular season: title odds after a win minus after a loss; playoffs: P(plays) × title odds when
it wins; under 10% playoff odds: the next 4 weeks evenly). The export writes them to
`data/week_weights_<season>.json` for the live service. **One deviation from the plan:** rather
than re-optimising a lineup every week (10× slower, ~20 s a search on Render), each player is
valued at their week-weighted expected points in a single lineup (`trade_engine` `weekWeights`),
mine with my weights and theirs with theirs. Byes, Vegas totals and injury return weeks are in
those weekly values; what it gives up is a bench player filling a bye, which the depth bonus
still stands in for — so step 6 should keep the bye part of the bench bonus. On 2026-10-22 it
put ~60% of a contender's weight on weeks 15–17; the calibration set still matched 10/10 with
weights on, and offer 7 came out −0.78 pts/wk for us once the free pickup (Mahomes) is netted.

1. Remove the offer log.
2. The shared "worth to them" function, its percentage and "why", and the calibration test.
3. Weekly horizon and record-based week weights; rank every candidate on them.
4. Our-side flags: role risk, rebound bet, need view, back for the playoffs.
5. Best / base / worst per remaining matchup (10th / 50th / 90th from the sim) on cards and in
   the evaluator.
6. Trim the bench bonus.
