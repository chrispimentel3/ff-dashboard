"""mega.history.snapshot_new: the scheduled refresh runs it every morning, so a week's
trend snapshot must be taken once and then left alone."""
from __future__ import annotations

import pandas as pd

from mega import history


def test_a_week_is_snapshotted_once_and_then_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HIST_DIR", tmp_path)
    calls = []

    def fake(season, week):
        calls.append(week)
        history._append(tmp_path / "wopr_weekly.csv", season, week,
                        pd.DataFrame({"norm": ["a"], "player": ["A"], "pos": ["WR"], "v": [float(len(calls))]}))

    monkeypatch.setattr(history, "SNAPSHOTS", ((fake, "wopr_weekly.csv"),))
    assert history.snapshot_new(2026, 4) == ["wopr_weekly.csv"]
    assert history.snapshot_new(2026, 4) == []           # the next morning: already taken
    assert calls == [4]
    assert history.snapshot_new(2026, 5) == ["wopr_weekly.csv"]
    d = pd.read_csv(tmp_path / "wopr_weekly.csv")
    assert list(d["week"]) == [4, 5] and list(d["v"]) == [1.0, 2.0]


def test_one_failing_metric_does_not_block_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HIST_DIR", tmp_path)

    def boom(season, week):
        raise RuntimeError("source down")

    def ok(season, week):
        history._append(tmp_path / "b.csv", season, week, pd.DataFrame({"x": [1]}))

    monkeypatch.setattr(history, "SNAPSHOTS", ((boom, "a.csv"), (ok, "b.csv")))
    assert history.snapshot_new(2026, 4) == ["b.csv"]
