# WOPR Targeting Module — Claude Code Handoff (v2: trades + waivers)

Repo: `chrispimentel3/MegaBowl2026` · League: Mega Bowl 2026 (Yahoo 173489, 12-team half-PPR, 1QB/2RB/2WR/1TE/1FLEX) · My team: TaylorMade (team id 1)
Spec status: data layer validated in a sandbox on 11 Sep 2026 against real 2025 data and the 2026 nflverse roster file. The Yahoo roster pull is specified but not yet run. Decisions in §1 carry recommended defaults and should be confirmed before building.

## 0. Purpose

Replace the legacy Excel receiving analysis (`Receiving_Analysis_-_Template_v1_1.xlsx`) with a reproducible script that computes Weighted Opportunity Rating (WOPR) for every WR/TE, then splits the results by who owns each player in the league. One Yahoo roster pull feeds three views: **My roster** (sell / hold), **Opponent rosters** (trade targets, grouped by manager), and **Free agents** (waiver adds). It fills the "air yards" gap noted in the draft doctrine and complements the board's existing `fdrr` field.

## 1. Decisions

Resolved: **D3 — both trades and waivers**, driven by a single ownership join (§5).

To confirm (recommended default in bold):

| # | Fork | Options | Default |
|---|------|---------|---------|
| D1 | Sample window | 2025 only · 2026 rolling only · both | **Both side by side.** 2025 is the baseline; 2026 becomes primary as weeks accrue (see D7) |
| D2 | Share denominator | Full-season team totals (legacy) · team totals in weeks the player appeared | **Games-played denominator** |
| D4 | Positions | WR+TE · WR+TE+RB | **WR+TE.** RB receiving work gets target share only, handled separately |
| D5 | Thresholds | Fixed · position percentiles recomputed each run | **Percentiles** (p50/p75) |
| D6 | Output destination | Standalone CSV + report · fields inside `index.html` | **Standalone first**; `index.html` integration is a separate, optional pass |
| D7 | Early-season signal (Weeks 1–4) | Raw 2026 WOPR · 2026 anchored to a 2025 prior | **Anchored:** `(g26·wopr26 + 3·wopr25) / (g26 + 3)`. The prior is worth 3 games and fades out on its own; rookies use the position median as prior. Raw 2026 is still shown alongside |
| D8 | Trade value axis | Board `vor` · Yahoo rest-of-season rank · own points projection | **Board `vor`** until a rest-of-season source is wired; label it "preseason value" in output |
| D9 | Trade partner fit | Targets only · targets + each manager's positional surplus/need | **Targets only in v1.** Partner fit (count of startable WR/TE per roster vs 3 WR + 1 TE slots) is a v2 add |
| D10 | Run cadence | Manual local run · scheduled GitHub Action | **Manual local run** first; schedule later using the same Yahoo OAuth secret as the power rankings workflow |

## 2. Data sources

nflverse weekly player stats (public, no auth; include `target_share`, `air_yards_share`, `wopr`, `receiving_air_yards` = intended air yards on all targets, `team`, `week`, `season_type`):

```
https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{SEASON}.csv
```

nflverse 2026 roster (public; gives current NFL team, status `ACT/RES/CUT/RET/DEV/INA`, `gsis_id`, `yahoo_id`). Use this as the authoritative current team and status, and as the primary gsis→Yahoo crosswalk:

```
https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{SEASON}.csv
```

Backup crosswalk (dynastyprocess; adds nothing for recent classes, see §4, but occasionally fills veteran gaps):

```
https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv
```

Yahoo league rosters (OAuth via `yfpy`, league 173489): pull all 12 team rosters for the current week. Each row needs `yahoo_player_id, name, editorial_team_abbr, display_position, fantasy_team_id, fantasy_team_name`. Reuse the existing `yahoo_pull.py` / OAuth token setup from the power rankings work if available (not currently in the repo).

Board data: the `PLAYERS` array embedded in `index.html` (regex `PLAYERS\s*=\s*(\[.*?\]);` → `json.loads`). Every board WR/TE that matched the 2026 roster file listed the same team in both sources.

## 3. Metric definitions

TS = player targets ÷ team targets. AYS = player intended air yards ÷ team intended air yards. WOPR = 1.5 × TS + 0.7 × AYS (matches nflverse's `wopr` column to within rounding).

Aggregate by summing numerators and denominators across the window, then dividing. Never average weekly ratios. Team totals are summed from the weekly file per team-week, only for the weeks the player appears (D2), and computed after any window filter.

xPPG: per-position linear fit of half-PPR PPG on WOPR, refit each season on players with ≥30 targets. 2025: WR `ppg = 18.83·WOPR − 0.52` (r = 0.88, n = 110); TE `ppg = 23.09·WOPR − 0.32` (r = 0.87, n = 49). `ppg_minus_xppg` is the efficiency/TD residual.

Half-PPR scoring uses league constants: rec/rush yd 0.1, rec/rush TD 6, reception 0.5, fumble lost −2.

## 4. Guardrails

**Availability must be proven, never assumed.** Both crosswalks have zero `yahoo_id` coverage for players with ≤1 year of experience. That means the entire 2025 and 2026 rookie classes, including McMillan, Egbuka, Warren, Loveland, Fannin, Golden and Burden. Coverage overall is 82% of active 2025 qualifiers. Match order: (1) `yahoo_id` from the crosswalk; (2) normalized name + position against the Yahoo roster pull, with team as tiebreaker; (3) a hand-maintained `overrides.csv` (`gsis_id, yahoo_id`). A player gets `view = FA` only if he is matched to a Yahoo ID and that ID is on no roster. Unmatched players get `view = UNKNOWN` and are printed for review, never listed as waiver adds. Otherwise rookies would show up as phantom free agents, the same failure mode as the board's `pts = 0` shells.

**Roster status:** only `ACT` players are eligible for waiver or trade target lists. `RES`, `CUT`, `RET` and `DEV` are shown with status only.

**Sample minimums:** season rows need ≥8 games to be tagged; window rows need ≥5 games in the window (2025) or ≥2 games (2026, where D7 anchoring absorbs the noise). A single low-volume game once produced a window WOPR of 1.33 because team air yards netted to 38.

**AYS** can exceed 1.0 or go negative in single weeks because behind-the-line targets carry negative air yards. Summing across the window handles it; do not clip.

**Names and teams:** normalize names by lowercasing, stripping `.`, `'`, `-` and suffixes (Jr, Sr, II–V). Map `LA→LAR`, `JAX→JAC` to match the board and Yahoo. Flag `MOVED_TEAM` when the last 2025 team differs from the 2026 roster team.

## 5. Ownership split and views

After the join, every player gets `view ∈ {MINE, OPP, FA, UNKNOWN}` and `owner` (fantasy team name).

| View | Purpose | Tags surfaced | Sort |
|------|---------|---------------|------|
| MINE | Sell / hold decisions | `SELL_HIGH`, `FADE` → sell candidates; `BUY_LOW` → hold, don't sell low | Tag, then `ppg_minus_xppg` |
| OPP | Trade targets, grouped by owner | `BUY_LOW`, `UNDERPRICED`, `RISER`, `ROLE_JUMP` | Anchored WOPR within tag |
| FA | Waiver adds | `ROLE_JUMP`, `RISER`, plus any FA with anchored WOPR ≥ position p50 | Anchored WOPR |
| UNKNOWN | Match review | none | name |

## 6. Tags

Applied to rows that pass §4.

| Tag | Rule (default) | Used in |
|-----|----------------|---------|
| `UNDERPRICED` | WOPR ≥ position p75 and board posrank worse than WOPR posrank by ≥6 | OPP |
| `BUY_LOW` | WOPR ≥ position p50 and `ppg_minus_xppg` ≤ −1.5 | OPP, MINE (hold) |
| `SELL_HIGH` | `ppg_minus_xppg` ≥ +2.0 and WOPR < position p75 (points ran ahead of role) | MINE |
| `RISER` | Late-window WOPR − season WOPR ≥ 0.08 and late-window WOPR ≥ 0.50 WR / 0.40 TE | OPP, FA |
| `ROLE_JUMP` | In-season: 2026 WOPR (≥2 games) ≥ position p75 and ≥ 2025 WOPR + 0.15, or no 2025 sample | FA, OPP |
| `FADE` | Board top-24 WR with WOPR < 0.52, or top-8 TE with WOPR < 0.37 | MINE |

2025 percentiles among board-matched players (≥6 games): WR p50 0.456 / p75 0.566 / p90 0.683; TE p50 0.343 / p75 0.388 / p90 0.455.

## 7. Output schema

`wopr_targets_{season}_wk{N}.csv`:

`view, owner, name, pos, team_2026_nfl, nfl_status, yahoo_id, gsis_id, match_method, age, board_posrank, board_vor, g_2025, wopr_2025, wopr_wk10plus, g_2026, wopr_2026, wopr_anchored, hppr_ppg, wopr_xppg, ppg_minus_xppg, fdrr, wopr_posrank, rank_delta, tags`

`match_method ∈ {crosswalk, name, override, none}` makes join quality auditable each run.

## 8. Repo layout and pass order

```
/wopr/
  config.yaml          # season, windows, thresholds, min-games, prior weight, team-code map
  fetch.py             # nflverse stats + roster, crosswalk (cached)
  yahoo_rosters.py     # yfpy pull of all 12 rosters -> cache/rosters_wk{N}.csv
  build.py             # aggregate, fit xPPG, join ownership, tag, write views
  overrides.csv        # manual gsis_id -> yahoo_id fixes
  outputs/
```

Pass 1 is the nflverse data build (no Yahoo). Pass 2 is the Yahoo roster pull plus the ownership join; stop and review the `UNKNOWN` list before moving on. Pass 3 produces the three views as CSV plus a simple report. Pass 4 (optional) patches `wopr`, `wopr_anchored`, `wtag`, `view` into `index.html` surgically, never touching existing enriched fields. Bump the localStorage key if board fields change, and deploy as `index.html` (versioned filenames 404 on Pages).

## 9. Known gaps

Route metrics (TPRR, first downs per route run) aren't in nflverse's free data; carry the board's `fdrr` through as-is. Snap counts (nflverse `snap_counts`) could replace "appeared in the weekly file" as the games-played definition. QB quality is the main confounder behind large `BUY_LOW` residuals and isn't modeled. Trade value uses preseason VOR (D8) and will drift as the season goes on.

## 10. Legacy template audit

The "2021 Receiving Analysis" tab computes WOPR as `=($S*1.5)+($R*0.7)` (S = air yards share, R = target share), which inverts the weights. On that tab's own data this swaps 3 of the top 24. That tab also derives air yards as `rec_yards − yac` (completed air yards only). The "FBRef Rec Data" tab uses the correct weights and aDOT × targets for air yards. Both tabs use full-season team denominators and break on "2TM" rows. The Pivot tab's team-level "Average of WOPR" has no meaning.

## 11. Reference implementation (validated core)

```python
import pandas as pd

URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{s}.csv"

def load_week(season):
    w = pd.read_csv(URL.format(s=season), low_memory=False)
    return w[w.season_type == "REG"].copy()

def team_totals(w):
    return (w.groupby(["team", "week"])[["targets", "receiving_air_yards"]]
             .sum().add_prefix("team_").reset_index())

def aggregate(w, min_week=None):
    if min_week:
        w = w[w.week >= min_week]
    w = w.merge(team_totals(w), on=["team", "week"]).sort_values("week")
    w["hppr"] = (0.1 * (w.receiving_yards + w.rushing_yards)
                 + 6 * (w.receiving_tds + w.rushing_tds)
                 + 0.5 * w.receptions
                 - 2 * (w.receiving_fumbles_lost + w.rushing_fumbles_lost))
    g = w.groupby(["player_id", "player_display_name", "position"])
    o = g.agg(games=("week", "nunique"), tgt=("targets", "sum"),
              ay=("receiving_air_yards", "sum"), ttgt=("team_targets", "sum"),
              tay=("team_receiving_air_yards", "sum"), pts=("hppr", "sum"),
              last_team=("team", "last")).reset_index()
    o["ts"] = o.tgt / o.ttgt
    o["ays"] = o.ay / o.tay
    o["wopr"] = 1.5 * o.ts + 0.7 * o.ays
    o["ppg"] = o.pts / o.games
    return o

def anchored(wopr26, g26, wopr25, prior_games=3):
    return (g26 * wopr26 + prior_games * wopr25) / (g26 + prior_games)
```

`player_id` in the weekly file is the `gsis_id`, which joins directly to the roster file.
