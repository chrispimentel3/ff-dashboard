"""Weekly orchestrator — run every Tuesday.

  uv run python tuesday.py                 # auto: Yahoo API if configured, else scrape, else offline
  uv run python tuesday.py --api           # force the official Yahoo API (YFPY) path
  uv run python tuesday.py --scrape        # force the browser-scrape path
  uv run python tuesday.py --no-yahoo      # offline: draft-board approximation only
  uv run python tuesday.py --manual        # parse hand-saved Yahoo HTML in data/manual/
  uv run python tuesday.py --email         # also email the digest (needs .env SMTP_*)
  uv run python tuesday.py --no-targets    # skip the Targets page rebuild
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from mega import intel
from mega.digest import build_digest, send_email, write_digest


def resolve_rosters(args) -> tuple[object, str]:
    """Return (rosters_df_or_None, human-readable source)."""
    if args.no_yahoo:
        return None, "offline (draft-board approximation)"

    # 1) Official Yahoo API via YFPY
    if not args.scrape:
        try:
            from mega import yahoo_api

            has_keys = bool(os.environ.get("YAHOO_CONSUMER_KEY"))
            if args.api or yahoo_api.available() or has_keys:
                if not yahoo_api.available() and has_keys:
                    print("[yahoo-api] no cached pull — running pull_league.py")
                    import pull_league

                    pull_league.main()
                df = yahoo_api.rosters_df()
                if df is not None and not df.empty:
                    yahoo_api.dump_csv()
                    return df, f"Yahoo API (week {yahoo_api.week()})"
                if args.api:
                    print("[yahoo-api] pull produced no rosters — check pull_league.py output")
        except Exception as e:
            print(f"[yahoo-api] skipped: {e}")
            if args.api:
                sys.exit(2)

    # 2) Browser scrape
    if not args.api:
        try:
            from mega import yahoo

            res = yahoo.pull(manual=args.manual)
            df = res.get("rosters")
            if df is not None and not df.empty:
                return df, "browser scrape"
        except Exception as e:
            print(f"[scrape] skipped: {e}")
            if args.scrape:
                sys.exit(2)

    return None, "offline (draft-board approximation)"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--api", action="store_true", help="force the official Yahoo API path")
    ap.add_argument("--scrape", action="store_true", help="force the browser-scrape path")
    ap.add_argument("--no-yahoo", action="store_true", help="offline only")
    ap.add_argument("--manual", action="store_true", help="scrape from hand-saved HTML in data/manual/")
    ap.add_argument("--email", action="store_true")
    ap.add_argument("--no-targets", action="store_true",
                    help="skip rebuilding and publishing the Targets page data")
    args = ap.parse_args(argv)

    rosters, source = resolve_rosters(args)
    print(f"\nrosters: {source}")
    print(f"building digest (season {args.season}, form {intel.form_season(args.season)}) …")

    md = build_digest(args.season, yahoo_rosters=rosters)
    path = write_digest(md, args.season)
    print(f"digest -> {path}")

    if args.email:
        try:
            send_email(md)
        except Exception as e:
            print(f"[email] {e}", file=sys.stderr)

    # Targets page (chrispimentel3.github.io/MegaBowl2026/power.html). Its data is raw
    # nflverse counts, no Yahoo involved, so it runs whether or not the scrape worked.
    if not args.no_targets:
        try:
            from scripts.targets_json import main as targets_main

            print("\nrebuilding Targets page data …")
            targets_main(["--season", str(args.season), "--publish"])
        except SystemExit as e:
            print(f"[targets] {e}", file=sys.stderr)
        except Exception as e:
            print(f"[targets] skipped: {e}", file=sys.stderr)

    print("\n" + md[:1600] + "\n…")


if __name__ == "__main__":
    main()
