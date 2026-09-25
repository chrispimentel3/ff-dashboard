"""Pure-data Action board: shop/hold candidates, waiver claims, trades.

Extracted out of app.py's `_tab_action` so the numbers have one source of truth. The
Streamlit tab and `tools/export_web.py` (which feeds the new mega-bowl-web frontend)
both call `build()` and render the same dict differently — neither recomputes the
underlying logic, so the two surfaces can never drift apart on what counts as a
shop/hold candidate or a worthwhile waiver claim.
"""
from __future__ import annotations

import json

import pandas as pd

ACOLS = ["player", "pos", "slot", "half_ppr_pg", "per_g", "tgt_pct", "tm_rank", "why"]


def _why_sell(r: pd.Series) -> str:
    bits = [f"scoring {r['per_g']:+.1f}/g more than his opportunity"]
    if pd.notna(r.get("tgt_pct")) and r["tgt_pct"] < 0.20:
        bits.append(f"only {r['tgt_pct']:.0%} of targets")
    if pd.notna(r.get("tm_rank")) and r["tm_rank"] >= 3:
        bits.append(f"#{int(r['tm_rank'])} option on his own offense")
    return "; ".join(bits)


def _why_buy(r: pd.Series) -> str:
    bits = [f"scoring {abs(r['per_g']):.1f}/g less than his opportunity"]
    if pd.notna(r.get("tgt_pct")) and r["tgt_pct"] >= 0.20:
        bits.append(f"{r['tgt_pct']:.0%} target share")
    if pd.notna(r.get("tm_rank")) and r["tm_rank"] <= 2:
        bits.append(f"his team's #{int(r['tm_rank'])} option")
    return "; ".join(bits)


def worth_claiming(wv: pd.DataFrame) -> pd.DataFrame:
    """Free agents who would actually change your lineup.

    One definition, because two surfaces had their own and contradicted each other on the
    same week: the action board counted any positive gain and said "6 free agents would
    start for you", while the waiver page counted a bid of a dollar or more and said "0
    would change your lineup". A gain too small to be worth a dollar is not a claim.
    """
    if wv is None or wv.empty:
        return wv
    if "bid" in wv.columns:
        return wv[wv["bid"] >= 1]
    return wv[pd.to_numeric(wv.get("gain"), errors="coerce").fillna(0) > 0]


def shop_hold(agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sell-high / buy-low candidates on the user's own roster, ranked by points vs opportunity."""
    mine = agg.copy() if agg is not None and not agg.empty else pd.DataFrame()
    if mine.empty or "xfp_diff" not in mine.columns:
        return pd.DataFrame(columns=ACOLS), pd.DataFrame(columns=ACOLS)

    mine["per_g"] = mine["xfp_diff"] / mine["games"].clip(lower=1)

    sell = mine[mine["per_g"] >= 2.0].sort_values("per_g", ascending=False).head(5)
    if not sell.empty:
        sell = sell.assign(why=sell.apply(_why_sell, axis=1))

    buy = mine[mine["per_g"] <= -1.5].sort_values("per_g").head(5)
    if not buy.empty:
        buy = buy.assign(why=buy.apply(_why_buy, axis=1))

    return sell.reindex(columns=ACOLS), buy.reindex(columns=ACOLS)


def headline(IB: dict | None) -> tuple[str, str]:
    if not IB:
        return "", ""
    w = IB.get("waivers")
    tr = IB.get("trades")
    nw = 0 if w is None or w.empty else len(worth_claiming(w))
    nt = 0 if tr is None or tr.empty else len(tr)
    bits = []
    if nw:
        bits.append(f"{nw} free agent{'' if nw == 1 else 's'} would start for you")
    if nt:
        bits.append(f"{nt} trade{'' if nt == 1 else 's'} clear the fairness filter")
    said = " and ".join(bits)
    main = (f"{said[0].upper()}{said[1:]}." if said else
            "Nothing on the wire or the trade board beats what you already have.")
    sub = "Everything below is ranked by what it adds to your starting nine, not by name."
    return main, sub


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def build(agg: pd.DataFrame, IB: dict | None, BASIS: dict, season: int) -> dict:
    """Full Action-board payload as JSON-safe plain Python — no Streamlit calls."""
    main, sub = headline(IB)
    sell, buy = shop_hold(agg)
    xfp_available = agg is not None and not agg.empty and "xfp_diff" in agg.columns

    payload: dict = {
        "headline": main,
        "subhead": sub,
        # Structured, not a pre-formatted sentence: each renderer (Streamlit tab, web
        # frontend) writes its own copy/markup from these fields rather than parsing markdown
        # out of a shared string.
        "basis": None if BASIS.get("current") else {
            "prior_season": BASIS["season"],
            "season": int(season),
            "weeks": BASIS["weeks"],
        },
        "xfp_available": xfp_available,
        "shop": _records(sell),
        "hold": _records(buy),
        "waivers": [],
        "trades": [],
        "roster_src": None,
    }

    if IB is not None:
        wv = IB.get("waivers")
        if wv is not None and not wv.empty:
            if "bid" in wv.columns:
                worth = wv[wv["bid"] >= 1]
                cols = ["player", "pos", "gain", "bid", "max_bid", "drop", "why"]
            else:
                worth = wv
                cols = ["player", "pos", "pg_recent", "tgt_pct", "tm_rank", "add_score", "why"]
            payload["waivers"] = _records(worth.head(5).reindex(columns=cols))

        tr = IB.get("trades")
        if tr is not None and not tr.empty:
            tr5 = tr.head(5).copy()
            tr5["give"] = tr5["give"] + " (" + tr5["give_pos"] + ")"
            tr5["get"] = tr5["get"] + " (" + tr5["get_pos"] + ")"
            tr5["addresses"] = tr5["addresses"].str.replace(r"^my (\S+) need$", r"\1", regex=True)
            cols = ["partner", "give", "give_val", "get", "get_val", "addresses", "fairness"]
            payload["trades"] = _records(tr5.reindex(columns=cols))
        payload["roster_src"] = IB.get("roster_src")

    return payload
