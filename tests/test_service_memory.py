"""The Render service's memory fixes (2026-09-27) and the live search's free-swap netting.

The service is a 512MB host. Asking about a past season used to rebuild that season's
Ask table from a full season of play-by-play (~300MB each), and the title-odds model kept
every die it rolled. These pin the replacements so they can't quietly regress.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mega import catalog, data
from mega import trade_engine as te
from mega import trade_league as TL
from test_trade_theses import _ctx  # tests/ is on sys.path (rootdir, no __init__)


def test_the_prebuilt_ask_history_matches_the_code_that_builds_it():
    """If this fails, the code behind the Ask table changed: the pre-built seasons are now
    being ignored (safe, but heavy on the service). Rebuild them:
    .venv/bin/python tools/build_ask_history.py"""
    man = json.loads((data.ASK_HISTORY / "manifest.json").read_text())
    assert man["stamp"] == data.ask_pw_stamp()
    assert set(range(2015, 2026)) <= set(man["seasons"])


def test_a_finished_season_comes_from_disk_not_nflverse(monkeypatch):
    from mega import ask as ASK

    def boom(*a, **k):
        raise AssertionError("rebuilt from nflverse")
    stamp = data.ask_pw_stamp()                       # patching player_week changes the stamp
    monkeypatch.setattr(data, "ask_pw_stamp", lambda: stamp)
    monkeypatch.setattr(ASK, "player_week", boom)
    pw = data.ask_player_week(2024)
    assert len(pw) > 5000 and {"gsis_id", "targets", "routes", "rz_touches"} <= set(pw.columns)


def test_a_stale_stamp_falls_back_to_the_live_build(monkeypatch):
    monkeypatch.setattr(data, "ask_pw_stamp", lambda: "not-the-stamp")
    assert data._ask_history(2024) is None


def test_a_catalogue_question_asks_only_for_the_columns_it_uses():
    cols = catalog.needed("pbp", "air_yards")
    assert "air_yards" in cols
    assert len(cols) < 12                  # not the 372 a full play-by-play read returns


def test_the_title_model_keeps_no_dice():
    """Dice are regenerated from the seed: same key, same roll, nothing stored."""
    from mega import title_odds as T
    m = T.Model(season=None, ctx=None, dist={}, proj_sd={}, rosters={}, n=50)
    a, b = m._dice("x|5"), m._dice("x|5")
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert not hasattr(m, "_draw")


def _results(ctx, trades):
    out = []
    for give, get in trades:
        core = te.evaluate_core(ctx, 1, 2, give, get)
        r = te.with_detail(ctx, 1, core)
        r["shape"] = f"{len(give)}-for-{len(get)}"
        out.append(r)
    return out


def test_the_live_search_nets_a_two_for_one_like_the_cards_do():
    ctx = _ctx()
    rows = _results(ctx, [(["r2", "w3"], ["or1"]), (["w2"], ["or1"])])
    raw_two, raw_one = rows[0]["dMe"], rows[1]["dMe"]
    net, swap_ids = TL.net_free_swap(SimpleNamespace(ctx=ctx), 1, rows)
    assert "fa_wr" in swap_ids
    two = next((r for r in net if r["shape"] == "2-for-1"), None)
    one = next(r for r in net if r["shape"] == "1-for-1")
    assert one["dMe"] == pytest.approx(raw_one) and not one.get("netted")   # 1-for-1s untouched
    fa_alone = te.team_value(swap_ids, ctx) - ctx.base[1].value
    if two is None:
        assert raw_two - fa_alone < 0.01                                  # nothing left: dropped
    else:
        assert two["netted"] and two["fa_add"] == ["fa_wr"]
        assert two["dMe"] == pytest.approx(raw_two - fa_alone, abs=1e-9)


def test_offers_come_back_first_and_the_odds_call_reuses_the_search(monkeypatch):
    """The page asks for the offers without odds (a few seconds on Render), shows them,
    then asks again with odds — which must not run the search a second time."""
    from service import main as M

    ctx = SimpleNamespace(players={"a": {"name": "A"}, "b": {"name": "B"}}, ir_owner={})
    eng = SimpleNamespace(ctx=ctx)
    row = {"partner": {"id": 2, "name": "Them"}, "shape": "1-for-1", "give": [{"name": "A"}],
           "get": [{"name": "B"}], "giveIds": ["a"], "getIds": ["b"], "dMe": 1.0, "dThem": 0.5,
           "market": {"ratio": 1.0}, "flag": "LIKELY",
           "me": {"startersIn": [], "startersOut": []}, "them": {"startersIn": [], "startersOut": []}}
    searched = []

    def fake_search(eng, me, req, flags, shapes, lap):
        searched.append(req.pid)
        return {"results": [row], "evaluated": 1, "padded": 0, "matched": 1}, [row], None

    monkeypatch.setattr(M, "_current_season", lambda: 2026)
    monkeypatch.setattr(M.data, "current_week", lambda s, d: 4)
    monkeypatch.setattr(M, "_trade_engine", lambda s: (eng, {"my_team_id": 1}))
    monkeypatch.setattr(M, "_search", fake_search)
    monkeypatch.setattr(M, "TITLE_ODDS_LIVE", True)
    monkeypatch.setattr(M, "_title_odds_rows", lambda *a, **k: {0: {"d_playoffs": 0.04, "d_title": 0.01}})
    monkeypatch.setattr(M, "_searches", M.OrderedDict())

    first = M.trade_search(M.TradeSearchRequest(pid="a", mine=True, odds=False))
    assert first["odds_pending"] and first["rows"][0]["odds"] is None and first["odds_model"] is None
    full = M.trade_search(M.TradeSearchRequest(pid="a", mine=True, odds=True))
    assert not full["odds_pending"] and full["rows"][0]["odds"] == 0.04
    assert searched == ["a"] and "search_cached" in full["timings"]
