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

## Run the dashboard

```bash
uv run streamlit run app.py
```

Opens at http://localhost:8501.

## Host it online (Streamlit Community Cloud)

GitHub alone can't run this app (Pages only serves static files — this is a Python
server). The free, purpose-built host is **Streamlit Community Cloud**, which runs
the app straight from your GitHub repo and redeploys on every push.

**1. Push this folder to a private GitHub repo**
```bash
cd ff-dashboard
git init && git add -A && git commit -m "FF dashboard"
gh repo create ff-dashboard --private --source=. --push
# (or create the repo in the GitHub UI and `git remote add origin … && git push -u origin main`)
```
`.env`, `data/`, and `.streamlit/secrets.toml` are gitignored, so no secrets or
cache ever leave your machine. Your `roster.csv` and draft-board HTML **are**
committed — keep the repo **private** so your team/league data stays private.

**2. Deploy**
1. Go to https://share.streamlit.io → **New app** → sign in with GitHub and
   authorize access to your private repo.
2. Pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings → Python version → 3.12**.
4. **Advanced settings → Secrets** → paste the contents of
   `.streamlit/secrets.toml.example` with your real values (at minimum
   `FANTASYPROS_API_KEY`). Save.
5. Deploy. First build takes a few minutes (nflverse data downloads on first load).

**What runs in the cloud vs. locally**
- ✅ Everything except the live Yahoo roster — projections, matchups, waivers,
  trades, draft value, archetypes, news all work from public data + your draft board.
- ⚠️ The **live Yahoo pull** (`pull_league.py` / browser scrape) is desktop-only
  (interactive OAuth / a real browser), so the cloud app uses the **offline
  draft-board approximation** for rosters. Run the live pull locally when you want
  the current roster, commit the refreshed `data/yahoo_api/` if you want the cloud
  app to see it (note: `data/` is gitignored by default — force-add it if so).
- ⚠️ Keep the deployed app **private**: in the app's Settings you can restrict
  viewers to specific emails, or keep the URL unlisted.

**Keeping it fresh / the Tuesday digest** — Community Cloud serves a web page; it
doesn't run scheduled jobs. For the automated Tuesday email, add a **GitHub Actions**
scheduled workflow that runs `tuesday.py --email` (ask me to wire it up).

## What it shows

| Tab | Content |
|-----|---------|
| Team overview | Per-player half-PPR/game, rolling avg, last week, expected points (xFP), target/carry share, injury status, next matchup implied team total |
| Actual vs expected | Season actual half-PPR vs `ff_opportunity` expected — sell-high / buy-low signal |
| Usage trends | Weekly snap share, target share, points, targets, carries |
| Matchups | Vegas totals/spreads and implied team points for your players' upcoming week |
| **Waiver wire** | Ranked pickup targets — recent points + volume + FantasyCalc value/trend + industry add rank |
| **Trades** | 1-for-1 ideas targeting your positional needs, fairness-scored |
| **Draft value** | Every pick vs its current trade value — risers/fallers, your regression watch |
| **News** | NFL headlines filtered to your roster / watchlist |
| Raw data | Mapped roster + CSV downloads |

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
standings / transactions, and `tuesday.py` + the dashboard pick it up automatically.

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

## Automating Tuesdays

- `crontab -e` → `0 9 * * 2 cd /path/to/ff-dashboard && ./.venv/bin/python tuesday.py --email`
- or the Claude Code `schedule` skill for a cloud agent
- or ask Claude to run `tuesday.py` and send the digest via its Gmail connector

## Your roster

Edit `roster.csv`. Columns: `name, slot, pos, nfl_team, gsis_id`.
Leave `gsis_id` blank — names are matched automatically against `ff_playerids`.
If a player shows as **unmatched** in the app, look up their `gsis_id` in the
Raw data tab's `ff_playerids` download and paste it into `roster.csv`.

## Scoring

`SCORING` at the top of `app.py` holds Yahoo default half-PPR weights
(0.5/rec, 0.04/pass yd, 4/pass TD, -1/INT, 0.1/rush-rec yd, 6/rush-rec TD,
-2/fumble lost). Adjust if your league uses bonuses or different values.

## Data freshness

nflverse re-runs after each game and finalizes mid-week. The app caches for 6h;
hit **Clear data cache** in the sidebar to force a refresh.

## Raw pull without the UI

```bash
python fetch_raw.py 2025
```

Writes CSVs to `./data/`.

## Not covered

Kicker and DST are not in the offensive datasets — track those separately
(`nfl.load_team_stats()` or manually).
