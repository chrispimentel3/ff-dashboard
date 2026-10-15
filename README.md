# FF Team Dashboard

A single-team fantasy dashboard built on [nflverse-data](https://github.com/nflverse/nflverse-data)
(pulled through `nflreadpy`). Configured for a **Yahoo half-PPR** league.

## Setup

`nflreadpy` needs **Python 3.10+**. Your system Python is 3.9, so use
[`uv`](https://docs.astral.sh/uv/) (it downloads a modern Python for you):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
cd ff-dashboard
uv venv --python 3.12
uv pip install -r requirements.txt
```

(If you'd rather not use `uv`, install Python 3.12 from python.org, then
`python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.)

## How it fits together

This repo is the data and analysis. The site people use is **mega-bowl-web** (Next.js on
Vercel), which reads the JSON this repo publishes:

```bash
PYTHONPATH=. .venv/bin/python tools/publish_web.py   # scrape, rebuild data/web/*.json, commit
```

`tools/export_web.py` builds every file the site reads from the `mega/` modules — one
`mega/*_web.py` per page. The interactive parts of the site (trade search, evaluating a
trade, Ask anything, the offer log) call the **Ask API** in `service/`, hosted on Render
(`render.yaml`).

---

# Mega Bowl command center

`mega/` adds league-wide intelligence for **Yahoo league 173489** (12-team, snake,
half-PPR, you = seat 8 "TaylorMade"). It works **without the Yahoo API** — either by
reusing a logged-in browser session, or fully offline from the draft board.

## The weekly run

```bash
uv run python tuesday.py                   # auto: Yahoo API if configured, else scrape, else offline
uv run python tuesday.py --email           # + email the digest (configure .env)
uv run python tuesday.py --no-yahoo        # offline: draft-board approximation
```

Writes `data/digest_YYYY-Www.md` — waiver targets, trade ideas, buy-low board,
your regression watch, draft-value movers, and news on your guys.

## Getting your league data

**A. Official Yahoo API (recommended) — `pull_league.py` + YFPY**

One-time setup:
1. Create an app at https://developer.yahoo.com/apps/create/
   - Application Type: **Installed Application**
   - Redirect URI: `https://localhost:8080`
   - API Permissions: **Fantasy Sports (Read)**
2. `cp .env.example .env` and paste in `YAHOO_CONSUMER_KEY` / `YAHOO_CONSUMER_SECRET`
3. `uv run python pull_league.py` — first run prints a Yahoo URL; approve, paste the
   code back. The token is cached into `.env`; later runs are silent.

That writes `data/yahoo_api/*.json`. `mega/yahoo_api.py` turns it into rosters /
standings / transactions, and `tuesday.py` and the export pick it up automatically.

**B. Browser scrape (no developer app)**
```bash
uv run playwright install chromium      # one time
uv run python -m mega.yahoo login       # sign in once; session saved ~weeks
uv run python -m mega.yahoo pull        # headless scrape -> data/yahoo_*.csv
```
Manual variant: *File ▸ Save Page As* the Players / roster / Standings / Transactions
pages into `data/manual/`, then `uv run python -m mega.yahoo pull --manual`.

**C. Nothing** — `tuesday.py --no-yahoo` still produces the full digest using the
draft board as a stand-in for current rosters.

## External data (all free, no auth)

- **nflverse** — snaps, targets, xFP, injuries, schedules
- **Sleeper API** — trending adds/drops across all leagues
- **FantasyCalc API** — redraft trade values + 30-day trend
- **RSS** — ESPN / Rotowire / PFT news

## Vegas projections (§20, needs a free key)

Every starter has sportsbook lines on his catches, his yards and his chance of scoring.
Scored in half-PPR they become a projection built by people with money at stake, and it
already contains the injury report, the weather and the game plan. Set `ODDS_API_KEY` in
`.env` (free tier, no card, from [the-odds-api.com](https://the-odds-api.com)) and the
column appears in Start/Sit and on a full-slate board under Matchups.

Three corrections happen on the way in, each fitted against 2021-2025 nflverse outcomes by
`python -m tools.fit_props` rather than assumed:

- **A line is a median, not a mean.** Weekly yardage is lopsided — a receiver priced at
  30.5 averages 39 — so projections deliberately sit *above* the posted line. Passing
  yards are the exception and are left alone.
- **Anytime touchdown is not expected touchdowns.** `6 × P(scores)` ignores two-score
  games; the Poisson fix overshoots the best scorers by ~12%. The measured multiplier is
  used instead.
- **Counts are Poisson.** Receptions and passing touchdowns check out against it at the
  lines books actually hang, so they are inverted directly and the skew correction is
  deliberately *not* applied on top.

**Budget.** The free plan is 500 credits a month. The event list is free; props cost
`markets × regions` per event, so one sweep of a 16-game slate over 7 markets is **112
credits** — about four sweeps a month. The refresh therefore holds itself to roughly
weekly, refuses a sweep it cannot finish (half a week would make priced games outrank
unpriced ones), and never spends the last 40. The $30 tier is 20,000 credits and makes it
daily: `ODDS_REFRESH_HOURS=24`.

Raw per-book prices stay out of git; the derived points are published to
`data/build/vegas_<season>_wk<NN>.csv`, which is what the site reads.

## Automating Tuesdays

- `crontab -e` → `0 9 * * 2 cd /path/to/ff-dashboard && ./.venv/bin/python tuesday.py --email`
- or the Claude Code `schedule` skill for a cloud agent
- or ask Claude to run `tuesday.py` and send the digest via its Gmail connector

## Your roster

Edit `roster.csv`. Columns: `name, slot, pos, nfl_team, gsis_id`.
Leave `gsis_id` blank — names are matched automatically against `ff_playerids`.
If a player can't be matched, the export says so — add a row to `data/id_overrides.csv`
(`yahoo_id,name,gsis_id,note`).

## Scoring

`SCORING` in `mega/config.py` holds Yahoo default half-PPR weights
(0.5/rec, 0.04/pass yd, 4/pass TD, -1/INT, 0.1/rush-rec yd, 6/rush-rec TD,
-2/fumble lost). Adjust if your league uses bonuses or different values.

## Data freshness

nflverse re-runs after each game and finalizes mid-week; every publish reads the latest.

## Raw pull without the UI

```bash
python fetch_raw.py 2025
```

Writes CSVs to `./data/`.
