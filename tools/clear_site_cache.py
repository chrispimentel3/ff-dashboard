"""Clear the live site's cached data after a push, the way the Refresh button's page does.

Without this the site keeps serving its cached copy for up to an hour after a scheduled
rebuild, and the first visitor after that still gets the old copy once. It waits until
raw.githubusercontent.com serves what was pushed (the site reads its data from there), wakes
the Render service (the site checks the passcode through it), then POSTs the site's
/api/refresh-site with OFFER_PASSCODE from .env. Never blocks a run: it always exits 0 and
prints one [site] line saying what happened.

    PYTHONPATH=. .venv/bin/python tools/clear_site_cache.py
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refresh_watch import ROOT, wait_for_raw  # noqa: E402

SITE = "https://mega-bowl-web.vercel.app/api/refresh-site"
SERVICE = "https://ff-dashboard-o8f9.onrender.com/health"


def _call(url: str, method: str = "GET", headers: dict | None = None, timeout: int = 90) -> int:
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except OSError:
        return 0


def main() -> int:
    load_dotenv(ROOT / ".env")
    passcode = os.environ.get("OFFER_PASSCODE")
    if not passcode:
        print("[site] OFFER_PASSCODE isn't in .env, so the cache wasn't cleared; the site catches up within the hour.")
        return 0
    if not wait_for_raw():
        print("[site] GitHub was slow to serve the push; clearing the cache anyway.")
    status = 0
    for attempt in range(3):           # Render's free tier can take a minute to wake
        _call(SERVICE)
        status = _call(SITE, "POST", {"X-Offer-Passcode": passcode})
        if status == 200:
            print("[site] cache cleared: the live site shows the new data now.")
            return 0
        time.sleep(30)
    print(f"[site] cache not cleared (HTTP {status}: wrong passcode, or the service didn't wake); "
          "the site catches up within the hour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
