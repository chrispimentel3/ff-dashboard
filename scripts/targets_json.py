"""Build data/targets_2026.json for the Targets tab on the Mega Bowl Pages site.

Raw weekly counts only — targets, receiving first downs, receptions and estimated routes
per WR/TE per week. The page computes every ratio itself, so the rolling window can change
without rebuilding the file (targets handoff §5).

Routes are estimated. The handoff's proxy needs nflverse participation, which stops at
2025, so this uses the handoff's own fallback: snap share × team dropbacks. The estimate
and its caveats live in mega/routes.py; this script only reshapes and publishes it.

    python scripts/targets_json.py                         # write into the Pages checkout
    python scripts/targets_json.py --out path/to/file.json
    python scripts/targets_json.py --publish               # also commit and push the site
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mega import ids as player_ids          # noqa: E402
from mega import routes as rz               # noqa: E402

# The Pages repo, checked out beside ff-dashboard.
SITE = Path(__file__).resolve().parent.parent.parent / "MegaBowl2026"
RELEASES = "https://github.com/nflverse/nflverse-data/releases"


def _stats(season: int) -> pd.DataFrame:
    df = nfl.load_player_stats(seasons=[season]).to_pandas()
    return df.rename(columns={"player_display_name": "player", "position": "pos", "player_id": "gsis_id"})


def build(season: int) -> tuple[dict, pd.DataFrame]:
    """The JSON payload, and the season totals frame for the eyeball table."""
    stats = _stats(season)
    snaps = nfl.load_snap_counts(seasons=[season]).to_pandas().rename(columns={"pfr_player_id": "pfr_id"})
    db = rz.team_dropbacks(season)
    wk = rz.weekly(stats, snaps, player_ids.crosswalk(), db)
    if wk.empty:
        raise SystemExit(f"no {season} WR/TE weeks — is the season under way?")

    problems = rz.validate(wk)
    for m in problems:
        print(f"  ! {m}")

    # Team totals tie out (handoff §6, Pass 1). Every position, not just WR/TE — a team's
    # targets should equal its pass attempts less the throwaways nobody is credited with,
    # which runs a handful a game. More than that, or any surplus, means a bad join.
    reg = stats[stats["season_type"].astype(str).str.upper() == "REG"]
    gap = (reg.groupby(["team", "week"])["targets"].sum()
           - reg.groupby(["team", "week"])["attempts"].sum()).dropna()
    far = gap[(gap > 0) | (gap < -10)]
    if not far.empty:
        print(f"  ! {len(far)} team-weeks where targets don't tie out to pass attempts: "
              + ", ".join(f"{t} wk{w} {v:+d}" for (t, w), v in far.head(5).items()))

    last_week = int(wk["week"].max())
    played = {(t, w) for t, w in zip(db["team"], db["week"])}      # every team-week with a game

    players = []
    for gid, g in wk.groupby("gsis_id"):
        g = g.sort_values("week")
        last = g.iloc[-1]
        weeks = {}
        for r in g.itertuples():
            weeks[str(int(r.week))] = {
                "tgt": int(r.targets), "fd": int(r.fd), "rec": int(r.rec),
                "routes": None if pd.isna(r.routes) else round(float(r.routes), 1),
            }
        # His team played but he didn't: write zeros, so "inactive" reads differently from
        # "bye" (which stays absent). The page skips zero-route weeks in the rolling window.
        for w in range(1, last_week + 1):
            if str(w) not in weeks and (last.team, w) in played:
                weeks[str(w)] = {"tgt": 0, "fd": 0, "rec": 0, "routes": 0.0}
        players.append({"id": str(gid), "name": str(last.player), "team": str(last.team),
                        "pos": str(last.pos), "weeks": dict(sorted(weeks.items(), key=lambda kv: int(kv[0])))})
    players.sort(key=lambda p: (p["pos"], p["name"]))

    drop = {}
    for t, w, n in zip(db["team"], db["week"], db["dropbacks"]):
        drop.setdefault(t, {})[str(int(w))] = int(n)

    payload = {
        "meta": {
            "season": season,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "last_week": last_week,
            "routes_method": "proxy_snap_share_x_team_dropbacks",
            "routes_caveat": (
                "Routes are estimated: his share of his offense's snaps × his team's dropbacks that week. "
                "nflverse publishes no charted route count, and the participation file that would give a "
                "better proxy stops at 2025. The estimate counts blocking snaps, so tight ends read low — "
                "compare a TE with other TEs."),
            "min_routes": rz.MIN_ROUTES,
            "per_game_bar": rz.PER_GAME_BAR,
            "wr_fd_flag": rz.WR_FD_FLAG,
            "sources": {
                "player_stats": f"{RELEASES}/tag/player_stats",
                "snap_counts": f"{RELEASES}/tag/snap_counts",
                "pbp": f"{RELEASES}/tag/pbp",
            },
            "validation": problems or ["all checks passed"],
        },
        "team_dropbacks": dict(sorted(drop.items())),
        "players": players,
    }
    return payload, rz.totals(wk)


def repo_of(path: Path) -> Path | None:
    """The git checkout `path` lives in, if any."""
    d = path.resolve().parent
    while d != d.parent:
        if (d / ".git").is_dir():
            return d
        d = d.parent
    return None


def publish(out: Path, week: int) -> str:
    """Commit and push the data file alone. Never leaves the repo mid-rebase — this runs
    unattended on Tuesdays, so every failure has to end somewhere recoverable."""
    repo = repo_of(out)
    if repo is None:
        return f"not a git checkout, nothing published: {out}"
    rel = out.resolve().relative_to(repo)
    git = ["git", "-C", str(repo)]

    def run(*args):
        return subprocess.run([*git, *args], capture_output=True, text=True)

    if not run("status", "--porcelain", str(rel)).stdout.strip():
        return "data unchanged since the last run, nothing to publish"
    run("add", str(rel))
    c = run("commit", "-q", "-m",
            f"Targets data through week {week}\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>")
    if c.returncode:
        return f"commit failed: {c.stderr.strip() or c.stdout.strip()}"
    run("fetch", "-q", "origin")
    r = run("rebase", "-q", "origin/main")
    if r.returncode:
        run("rebase", "--abort")
        return ("rebase onto origin/main conflicted — committed locally but NOT pushed. "
                "Someone else changed the same file; sort it out by hand.")
    pu = run("push", "-q", "origin", "main")
    if pu.returncode:
        return f"committed but push failed: {pu.stderr.strip()}"
    return f"pushed {run('rev-parse', '--short', 'HEAD').stdout.strip()}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--publish", action="store_true", help="commit and push the data file")
    a = ap.parse_args(argv)

    season = a.season or int(nfl.get_current_season())
    out = a.out or SITE / "data" / f"targets_{season}.json"
    if a.out is None and repo_of(out) is None:
        raise SystemExit(f"no Pages checkout at {SITE} — clone it beside ff-dashboard:\n"
                         f"  git clone https://github.com/chrispimentel3/MegaBowl2026.git {SITE}\n"
                         f"or pass --out to write the file somewhere else.")
    print(f"building {season} targets -> {out}")
    data, tot = build(season)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Rebuilt from scratch every run, so the timestamp alone always differs. Compare
    # everything else and leave the file alone when nothing moved — otherwise the weekly
    # job would commit an identical file every Tuesday and the history would say nothing.
    def body(d: dict) -> str:
        return json.dumps({**d, "meta": {k: v for k, v in d["meta"].items() if k != "generated_at"}},
                          separators=(",", ":"), sort_keys=True)

    try:
        old = json.loads(out.read_text()) if out.is_file() else None
    except ValueError:
        old = None
    if old and body(old) == body(data):
        print(f"  unchanged since {old['meta'].get('generated_at', '?')}")
        data = old                                   # keep the old stamp; leave the file alone
    else:
        out.write_text(json.dumps(data, separators=(",", ":")) + "\n")

    n_routes = sum(1 for p in data["players"] for w in p["weeks"].values() if w["routes"])
    print(f"  {len(data['players'])} players, {n_routes} player-weeks with routes, "
          f"through week {data['meta']['last_week']}, {out.stat().st_size / 1024:.0f} KB")

    # Eyeball table (handoff §6, Pass 1)
    for pos in rz.POS:
        top = tot[(tot["pos"] == pos) & tot["qualified"]].nlargest(10, "fd_rr")
        print(f"\n== top 10 {pos} by 1D/RR (>= {rz.MIN_ROUTES} routes) ==")
        print(top[["player", "team", "routes", "targets", "fd", "tprr", "fd_rr"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if a.publish:
        print("\n" + publish(out, int(data["meta"]["last_week"])))


if __name__ == "__main__":
    main()
