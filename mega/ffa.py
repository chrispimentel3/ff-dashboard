"""§3.1 / §6.2 — projections and rest-of-season ECR from ffanalytics (R).

ffanalytics scrapes six projection sources and reduces them to a robust average, scored in
whatever league rules you hand it. That is a better `proj` than any single site, and its
`scrape_ecr` is what league-mates actually see when they value a player — which is the
whole point of the perceived-market rule in §6.2.

It only runs in R, so this module shells out to R/projections.R and reads the CSVs it
writes. Two consequences worth stating plainly:

  * **Streamlit Cloud has no R.** The hosted dashboard therefore reads whatever the last
    committed build produced; it cannot refresh projections itself. The scheduled task on
    Chris's Mac is what keeps them current.
  * **Scrapers break.** A source that fails is skipped, not fatal. `fresh()` says how old
    the numbers are so a stale file is visible rather than silently believed.

ffanalytics identifies players by MFL id; everything here is keyed on gsis_id, so the join
goes through `load_ff_playerids()` exactly as §2.1 specifies.
"""
from __future__ import annotations

import datetime as dt
import functools
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .config import DATA, ROOT

BUILD = DATA / "build"
SCRIPT = ROOT / "R" / "projections.R"
# The CRAN installer puts R here on macOS; it is not on PATH for a GUI-launched process.
R_CANDIDATES = ("/Library/Frameworks/R.framework/Resources/bin/Rscript",
                "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript")
STALE_AFTER = dt.timedelta(days=8)      # a week-old projection is last week's question


def rscript() -> str | None:
    """Where Rscript lives, or None if R was never installed."""
    found = shutil.which("Rscript")
    if found:
        return found
    return next((p for p in R_CANDIDATES if Path(p).is_file()), None)


def available() -> bool:
    return rscript() is not None and SCRIPT.is_file()


def proj_path(season: int, week: int | None) -> Path:
    tag = "ros" if week is None else f"{int(week):02d}"
    return BUILD / f"proj_{season}_wk{tag}.csv"


def ecr_path(season: int) -> Path:
    return BUILD / f"ecr_ros_{season}.csv"


def run(season: int, week: int | None = None, timeout: int = 900) -> dict:
    """Invoke the R scraper. Returns what happened rather than raising: a failed scrape
    should leave yesterday's numbers in place, not take the caller down."""
    exe = rscript()
    if exe is None:
        return {"ok": False, "why": "R is not installed on this machine"}
    if not SCRIPT.is_file():
        return {"ok": False, "why": f"missing {SCRIPT}"}
    BUILD.mkdir(parents=True, exist_ok=True)
    cmd = [exe, str(SCRIPT), "--season", str(season), "--out", str(BUILD)]
    if week is not None:
        cmd += ["--week", str(int(week))]
    env = {**os.environ, "R_LIBS_USER": os.path.expanduser("~/Library/R/arm64/4.5/library")}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(ROOT), env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": f"R scrape exceeded {timeout}s"}
    tail = "\n".join(p.stdout.strip().splitlines()[-6:])
    return {"ok": p.returncode == 0, "why": tail or p.stderr[-400:],
            "proj": proj_path(season, week).is_file(), "ecr": ecr_path(season).is_file()}


@functools.lru_cache(maxsize=1)
def _mfl() -> pd.DataFrame:
    """§2.1 — ffanalytics' `id` is an MFL id; everything here is keyed on gsis_id."""
    import nflreadpy as nfl

    ids = nfl.load_ff_playerids().to_pandas()
    keep = [c for c in ("mfl_id", "gsis_id", "name", "position", "team") if c in ids.columns]
    out = ids[keep].copy()
    out["mfl_id"] = pd.to_numeric(out["mfl_id"], errors="coerce").astype("Int64")
    return out.dropna(subset=["mfl_id"]).drop_duplicates("mfl_id")


def _age(path: Path) -> dt.timedelta | None:
    if not path.is_file():
        return None
    return dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)


def projections(season: int, week: int | None = None) -> pd.DataFrame:
    """gsis_id, player, pos, proj (half-PPR), with the floor/ceiling spread kept.

    `sd_pts` is the disagreement between sources — a wide spread is genuine uncertainty
    about the player, not noise to be averaged away, so it survives into the frame.
    """
    path = proj_path(season, week)
    if not path.is_file():
        return pd.DataFrame(columns=["gsis_id", "player", "pos", "proj"])
    df = pd.read_csv(path)
    if df.empty or "id" not in df.columns:
        return pd.DataFrame(columns=["gsis_id", "player", "pos", "proj"])
    if "avg_type" in df.columns:
        df = df[df["avg_type"].astype(str) == "robust"]
    df["mfl_id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    out = df.merge(_mfl(), on="mfl_id", how="left")
    out = out.rename(columns={"points": "proj", "name": "player", "pos": "pos_src"})
    out["pos"] = out.get("position", out.get("pos_src"))
    keep = ["gsis_id", "player", "pos", "proj", "floor", "ceiling", "sd_pts",
            "pos_rank", "tier", "mfl_id"]
    out = out[[c for c in keep if c in out.columns]]
    return out.dropna(subset=["gsis_id"]).reset_index(drop=True)


def ecr_ros(season: int) -> pd.DataFrame:
    """gsis_id, ecr_ros — the consensus rest-of-season rank, half-PPR.

    This is the number to value a trade partner's *perception* with (§6.2). They see
    rankings and box scores, not expected points."""
    path = ecr_path(season)
    if not path.is_file():
        return pd.DataFrame(columns=["gsis_id", "ecr_ros"])
    df = pd.read_csv(path)
    if df.empty or "id" not in df.columns:
        return pd.DataFrame(columns=["gsis_id", "ecr_ros"])
    df["mfl_id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    rank = "ecr_rank" if "ecr_rank" in df.columns else "avg"
    out = df.merge(_mfl(), on="mfl_id", how="left")
    out["ecr_ros"] = pd.to_numeric(out[rank], errors="coerce")
    return out.dropna(subset=["gsis_id"])[["gsis_id", "ecr_ros"]].reset_index(drop=True)


def status(season: int, week: int | None = None) -> dict:
    """Is R here, are the files here, and how old are they — for the report line."""
    pa, ea = _age(proj_path(season, week)), _age(ecr_path(season))
    def days(a):
        return None if a is None else round(a.total_seconds() / 86400, 1)
    return {
        "r_installed": rscript() is not None,
        "script": SCRIPT.is_file(),
        "proj_rows": len(projections(season, week)),
        "ecr_rows": len(ecr_ros(season)),
        "proj_age_days": days(pa), "ecr_age_days": days(ea),
        "stale": any(a is not None and a > STALE_AFTER for a in (pa, ea)),
    }


def line(season: int, week: int | None = None) -> str:
    s = status(season, week)
    if not s["r_installed"]:
        return "ffanalytics: R not installed — projections fall back to FantasyPros."
    if not s["proj_rows"]:
        return "ffanalytics: no projection build on disk yet — run R/projections.R."
    age = s["proj_age_days"]
    warn = "  STALE" if s["stale"] else ""
    return (f"ffanalytics: {s['proj_rows']} projections, {s['ecr_rows']} ROS ECR, "
            f"{age:.1f} days old{warn}")
