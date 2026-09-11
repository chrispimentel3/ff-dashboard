"""Assemble the weekly 'Tuesday Report' as Markdown, and (optionally) email it."""
from __future__ import annotations

import datetime as dt
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from .config import DATA, MY_SEAT, MY_TEAM, SCORING_NAME
from . import intel
from .draft_board import load_draft
from .sources import news_for_players


def _tbl(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    d = df.head(n)[[c for c in cols if c in df.columns]].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(1)
    head = "| " + " | ".join(d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = "\n".join("| " + " | ".join(str(x) for x in row) + " |" for row in d.values)
    return f"{head}\n{sep}\n{body}\n"


def build_digest(season: int, my_players: list[str] | None = None,
                 yahoo_rosters: pd.DataFrame | None = None) -> str:
    today = dt.date.today()
    draft = load_draft()
    have_ros = yahoo_rosters is not None and not yahoo_rosters.empty
    if have_ros:
        rostered = set(yahoo_rosters["norm"])
        mine = yahoo_rosters[yahoo_rosters["seat"] == MY_SEAT]["player"].tolist()
        my_players = my_players or mine or draft[draft["drafted_by"] == MY_TEAM]["player"].tolist()
    else:
        my_players = my_players or draft[draft["drafted_by"] == MY_TEAM]["player"].tolist()
        rostered = set(intel._norm(p) for p in draft["player"])

    wb = intel.waiver_board(season, rostered, top=12)
    tr = intel.trade_finder(season, yahoo_rosters=yahoo_rosters if have_ros else None)
    dv = intel.draft_value_delta(season)
    bl = intel.buy_low_sell_high(season)

    my_norm = {intel._norm(p) for p in my_players}
    my_bl = bl[bl["norm"].isin(my_norm)] if not bl.empty else bl
    league_buy = bl[(bl["signal"] == "BUY LOW") & (~bl["norm"].isin(my_norm))] if not bl.empty else bl

    news = news_for_players(my_players)

    L = []
    L.append(f"# Mega Bowl — Tuesday Report · {today:%b %d, %Y}")
    src = "live Yahoo rosters" if have_ros else "draft-board approximation"
    L.append(f"_{SCORING_NAME} · team **{MY_TEAM}** · form data: {intel.form_season(season)} season · rosters: {src}_\n")

    L.append("## 🔎 Waiver targets")
    L.append(_tbl(wb, ["player", "pos", "pg_recent", "tgt_pg", "carry_pg", "add_rank", "why"], 10))

    L.append("## 🤝 Trade ideas (approximate — verify rosters)")
    L.append(_tbl(tr, ["partner", "give", "give_pos", "get", "get_pos", "addresses", "fairness"], 8))

    L.append("## 📉 Buy-low around the league")
    L.append(_tbl(league_buy, ["player", "pos", "gms", "diff_pg"], 8))

    L.append("## ⚠️ Your regression watch (actual vs expected)")
    L.append(_tbl(my_bl.sort_values("diff_pg"), ["player", "pos", "diff_pg", "signal"], 12))

    L.append("## 📈 Draft value movers")
    L.append("**Risers**")
    L.append(_tbl(dv[dv["value"] > 0], ["player", "pos", "drafted_by", "round", "value_delta"], 6))
    L.append("**Fallers**")
    L.append(_tbl(dv[dv["value"] > 0].sort_values("value_delta"), ["player", "pos", "drafted_by", "round", "value_delta"], 6))

    L.append("## 📰 News on your guys")
    if news is not None and not news.empty:
        for _, r in news.head(12).iterrows():
            L.append(f"- **{r['title']}** ({r['source']})  \n  {r['link']}")
    else:
        L.append("_(nothing new)_")

    return "\n".join(L) + "\n"


def write_digest(md: str, season: int) -> str:
    iso = dt.date.today().isocalendar()
    path = DATA / f"digest_{iso.year}-W{iso.week:02d}.md"
    path.write_text(md, encoding="utf-8")
    return str(path)


def _md_to_html(md: str) -> str:
    try:
        import markdown  # optional

        return markdown.markdown(md, extensions=["tables"])
    except Exception:
        return "<pre style='font-family:ui-monospace,monospace;white-space:pre-wrap'>" + (
            md.replace("&", "&amp;").replace("<", "&lt;")
        ) + "</pre>"


def send_email(md: str, to: str | None = None, subject: str | None = None) -> None:
    """SMTP send. Configure via .env:  SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASS DIGEST_TO
    For Gmail: host smtp.gmail.com, port 587, user = your address, pass = 16-char App Password."""
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    to = to or os.environ.get("DIGEST_TO") or user
    if not (user and pw and to):
        raise RuntimeError("Email not configured — set SMTP_USER, SMTP_PASS, DIGEST_TO in .env")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject or f"Mega Bowl — Tuesday Report {dt.date.today():%b %d}"
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(md, "plain", "utf-8"))
    msg.attach(MIMEText(_md_to_html(md), "html", "utf-8"))

    with smtplib.SMTP(host, port) as srv:
        srv.starttls(context=ssl.create_default_context())
        srv.login(user, pw)
        srv.sendmail(user, [to], msg.as_string())
    print(f"emailed digest -> {to}")


if __name__ == "__main__":
    import sys

    s = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    md = build_digest(s)
    p = write_digest(md, s)
    print(md)
    print(f"\n--- written to {p}")
