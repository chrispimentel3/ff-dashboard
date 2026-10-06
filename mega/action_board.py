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
    # HANDOFF v1.3: the "Bid now" lane — lineup gain over the next three weeks above
    # waiver_value.TAU_BID — is the claim worth making. Stash and early-signal players are
    # real, but they are not what "would change your lineup" means.
    if "lane" in wv.columns:
        return wv[wv["lane"] == "bid_now"]
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
    th = IB.get("theses") or {}
    if th.get("available"):
        # a consolidation card can cost title odds while it adds lineup points; count only the gains
        nt = sum(1 for c in th.get("cards") or [] if (c.get("us") or {}).get("d_title", 0) > 0
                 and not (c.get("us") or {}).get("title_noise", True))
        trade_bit = "raise{} your title odds"
    else:
        nt, trade_bit = (0 if tr is None or tr.empty else len(tr)), "clear{} the fairness filter"
    bits = []
    if nw:
        bits.append(f"{nw} free agent{'' if nw == 1 else 's'} would start for you")
    if nt:
        bits.append(f"{nt} trade{'' if nt == 1 else 's'} " + trade_bit.format("s" if nt == 1 else ""))
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
                worth = worth_claiming(wv)
                cols = ["player", "pos", "gain", "bid", "max_bid", "drop", "why"]
            else:
                worth = wv
                cols = ["player", "pos", "pg_recent", "tgt_pct", "tm_rank", "add_score", "why"]
            payload["waivers"] = _records(worth.head(5).reindex(columns=cols))

        tr = IB.get("trades")
        cards = (IB.get("theses") or {}).get("cards") or []
        if cards:
            # HANDOFF v1.3: the same engine-built offers as the Trades tab, best first, one
            # per manager — the home board is a shortlist, not three asks of one roster
            seen, top = set(), []
            for c in cards:
                if c["partner"] not in seen and len(top) < 5:
                    seen.add(c["partner"])
                    top.append(c)
            payload["trades"] = [{
                "partner": c["partner"],
                "give": " + ".join(f"{g['name']} ({g['pos']})" for g in c["give"]),
                "give_val": round(sum(g["ros_pg"] for g in c["give"]), 1),
                "get": " + ".join(f"{g['name']} ({g['pos']})" for g in c["get"]),
                "get_val": round(sum(g["ros_pg"] for g in c["get"]), 1),
                "addresses": c["tags"][0].replace("_", " ").lower(),
                "fairness": c["fairness"],
                "d_title": c["us"]["d_title"], "thesis": c["thesis"],
                "d_me": c["us"]["d_ros"], "p_accept": c["p_accept"], "flag": c["flag"],
            } for c in top]
        elif tr is not None and not tr.empty:
            tr5 = tr.head(5).copy()
            tr5["give"] = tr5["give"] + " (" + tr5["give_pos"] + ")"
            tr5["get"] = tr5["get"] + " (" + tr5["get_pos"] + ")"
            tr5["addresses"] = tr5["addresses"].str.replace(r"^my (\S+) need$", r"\1", regex=True)
            cols = ["partner", "give", "give_val", "get", "get_val", "addresses", "fairness"]
            payload["trades"] = _records(tr5.reindex(columns=cols))
        payload["roster_src"] = IB.get("roster_src")

    return payload
