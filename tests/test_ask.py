"""The question box: parsing, and the arithmetic behind each kind of metric.

Frames are built by hand so the maths is checkable by eye — every expected number in
here can be worked out on paper from the rows above it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mega import ask


# ---------------------------------------------------------------- fixtures
def _stats() -> pd.DataFrame:
    """Two teams, two weeks. LA throws a lot; SF runs."""
    rows = [
        # gsis, player, pos, team, wk, tgt, rec, recyd, rec1d, carries, rushyd, half_ppr
        ("A", "Adams",   "WR", "LA", 1, 10, 6,  80, 4, 0,  0, 14.0),
        ("A", "Adams",   "WR", "LA", 2,  8, 4,  40, 2, 0,  0,  6.0),
        ("B", "Nacua",   "WR", "LA", 1,  6, 5,  70, 3, 0,  0, 12.5),
        ("B", "Nacua",   "WR", "LA", 2, 12, 9, 120, 6, 0,  0, 21.0),
        ("C", "Parkins", "TE", "LA", 1,  4, 2,  20, 1, 0,  0,  3.0),
        ("C", "Parkins", "TE", "LA", 2,  2, 1,   8, 0, 0,  0,  1.8),
        ("D", "McCaff",  "RB", "SF", 1,  5, 4,  30, 2, 20, 90, 17.0),
        ("D", "McCaff",  "RB", "SF", 2,  3, 3,  25, 1, 18, 70, 13.0),
        ("E", "Guerend", "RB", "SF", 1,  1, 1,   5, 0,  4, 12,  2.7),
        ("E", "Guerend", "RB", "SF", 2,  1, 0,   0, 0,  6, 18,  1.8),
    ]
    df = pd.DataFrame(rows, columns=["gsis_id", "player", "pos", "team", "week", "targets",
                                     "receptions", "receiving_yards", "receiving_first_downs",
                                     "carries", "rushing_yards", "half_ppr"])
    df["season_type"] = "REG"
    return df


def _snaps() -> pd.DataFrame:
    """Team snaps come from the biggest count in the game — a lineman plays every one."""
    rows = []
    for wk in (1, 2):
        for team, players in {"LA": [("A", 40), ("B", 55), ("C", 50)],
                              "SF": [("D", 45), ("E", 15)]}.items():
            rows.append(dict(pfr_id=f"LINE{team}", week=wk, team=team, offense_snaps=60.0,
                             game_type="REG"))
            for pid, n in players:
                rows.append(dict(pfr_id=f"pfr{pid}", week=wk, team=team,
                                 offense_snaps=float(n), game_type="REG"))
    return pd.DataFrame(rows)


def _crosswalk() -> pd.DataFrame:
    return pd.DataFrame({"gsis_id": list("ABCDE"), "pfr_id": [f"pfr{c}" for c in "ABCDE"]})


def _routes() -> pd.DataFrame:
    rows = [("A", 1, 30.0), ("A", 2, 28.0), ("B", 1, 38.0), ("B", 2, 40.0),
            ("C", 1, 25.0), ("C", 2, 24.0)]
    return pd.DataFrame(rows, columns=["gsis_id", "week", "routes"])


@pytest.fixture(scope="module")
def pw() -> pd.DataFrame:
    return ask.player_week(_stats(), _snaps(), None, _routes(), _crosswalk())


# ---------------------------------------------------------------- parsing
def test_the_question_that_started_it():
    q = ask.parse("list of WR by snap %")
    assert q.field.key == "snap_pct"
    assert q.positions == ("WR",)
    assert not q.ascending


@pytest.mark.parametrize("text,key", [
    ("top 10 RB by targets", "targets"),
    ("WR by targets per game", "targets_pg"),          # longer alias beats "targets"
    ("who leads in target share", "target_share"),
    ("snap counts for the Bears", "offense_snaps"),    # "snaps" is a count, "snap %" a rate
    ("best yards per route run", "yprr"),
    ("yards per carry", "ypc"),
    ("expected fantasy points", "xfp"),
    ("points over expected", "xfp_diff"),
    ("TE by routes this season", "routes"),
    ("RB by points per game", "ppg"),
])
def test_metric_aliases(text, key):
    assert ask.parse(text).field.key == key


def test_longest_alias_wins_over_substring():
    # "snap %" must not be swallowed by the bare "snap" alias
    assert ask.parse("WR by snap %").field.key == "snap_pct"
    assert ask.parse("WR by snap share").field.key == "snap_pct"
    assert ask.parse("WR snaps").field.key == "offense_snaps"


def test_position_defaults_come_from_the_metric():
    assert ask.parse("who leads in yards per carry").positions == ("RB", "QB")
    assert ask.parse("leaders in routes").positions == ("WR", "TE")
    assert ask.parse("leaders in passing yards").positions == ("QB",)


def test_explicit_position_beats_the_default():
    assert ask.parse("TE by yards per route run").positions == ("TE",)
    assert ask.parse("pass catchers by targets").positions == ("WR", "TE")
    assert ask.parse("flex by points per game").positions == ("RB", "WR")


def test_teams():
    assert ask.parse("who leads the Rams in target share").teams == ("LA",)
    assert ask.parse("49ers RB by carries").teams == ("SF",)
    assert ask.parse("targets for KC").teams == ("KC",)


def test_los_angeles_rams_is_not_the_chargers():
    assert ask.parse("los angeles rams WR by targets").teams == ("LA",)
    assert ask.parse("los angeles chargers WR by targets").teams == ("LAC",)


def test_bare_no_is_not_new_orleans():
    """'no' is a word. Only the spelled-out team names reach NO."""
    assert ask.parse("WR with no targets").teams == ()
    assert ask.parse("new orleans WR by targets").teams == ("NO",)
    assert ask.parse("saints WR by targets").teams == ("NO",)


def test_minimum_is_not_minnesota():
    """'min 20 targets' must be a volume floor, not a Vikings filter."""
    q = ask.parse("worst catch rate min 20 targets")
    assert q.min_den == 20
    assert q.teams == ()
    assert ask.parse("minnesota WR by targets").teams == ("MIN",)


def test_weeks():
    assert ask.parse("WR by targets in week 3").weeks == (3,)
    assert ask.parse("WR by targets weeks 2-4").weeks == (2, 3, 4)
    assert ask.parse("WR by targets last 3 weeks", (1, 2, 3, 4, 5)).weeks == (3, 4, 5)
    assert ask.parse("WR by targets since week 4", (1, 2, 3, 4, 5)).weeks == (4, 5)
    assert ask.parse("WR by targets last week", (1, 2, 3)).weeks == (3,)
    assert ask.parse("WR by targets this season").weeks == ()


def test_top_n_and_direction():
    assert ask.parse("top 10 WR by targets").top == 10
    assert ask.parse("top five WR by targets").top == 5
    assert ask.parse("WR by targets").top == ask.DEFAULT_TOP
    assert ask.parse("worst WR by catch rate").ascending
    assert not ask.parse("best WR by catch rate").ascending


def test_scope():
    assert ask.parse("my team by points per game").scope == "mine"
    assert ask.parse("best free agent WR by targets").scope == "fa"
    assert ask.parse("rostered RB by carries").scope == "rostered"


def test_a_question_with_no_stat_says_so():
    with pytest.raises(ask.AskError) as e:
        ask.parse("who should I start this week")
    assert "could not find a stat" in str(e.value)


# ---------------------------------------------------------------- the table
def test_player_week_shape(pw):
    assert len(pw) == 10
    assert set(pw["gsis_id"]) == set("ABCDE")
    assert pw["one"].sum() == 10


def test_team_targets_are_that_weeks_team_total(pw):
    la1 = pw[(pw["team"] == "LA") & (pw["week"] == 1)]
    assert la1["team_targets"].unique().tolist() == [20]   # 10 + 6 + 4
    sf2 = pw[(pw["team"] == "SF") & (pw["week"] == 2)]
    assert sf2["team_targets"].unique().tolist() == [4]    # 3 + 1


def test_team_snaps_is_the_fullest_workload_in_the_game(pw):
    assert pw[pw["team"] == "LA"]["team_snaps"].unique().tolist() == [60.0]


def test_touches_is_carries_plus_targets(pw):
    mc = pw[(pw["gsis_id"] == "D") & (pw["week"] == 1)].iloc[0]
    assert mc["touches"] == 25          # 20 carries + 5 targets


# ---------------------------------------------------------------- arithmetic
def test_snap_pct_is_summed_over_summed(pw):
    """Adams: (40 + 40) / (60 + 60) = 0.667. Not the mean of two weekly ratios."""
    r = ask.answer(pw, "list WR by snap %")
    row = r.df[r.df["player"] == "Adams"].iloc[0]
    assert row["snap_pct"] == pytest.approx(80 / 120)
    assert row["offense_snaps"] == 80
    assert row["team_snaps"] == 120


def test_a_rate_is_not_the_mean_of_weekly_rates(pw):
    """Nacua caught 5/6 then 9/12. Summed = 14/18 = .778; mean of the two = .792.
    They differ, and the summed one is the one we report."""
    r = ask.answer(pw, "WR by catch rate, min 5 targets")
    row = r.df[r.df["player"] == "Nacua"].iloc[0]
    assert row["catch_rate"] == pytest.approx(14 / 18)
    assert row["catch_rate"] != pytest.approx((5 / 6 + 9 / 12) / 2)


def test_target_share(pw):
    """Adams 18 targets of LA's 20 + 22 = 42."""
    r = ask.answer(pw, "WR by target share")
    row = r.df[r.df["player"] == "Adams"].iloc[0]
    assert row["target_share"] == pytest.approx(18 / 42)


def test_totals_and_per_game(pw):
    tot = ask.answer(pw, "WR by targets").df.set_index("player")["targets"]
    assert tot["Nacua"] == 18
    pg = ask.answer(pw, "WR by targets per game").df.set_index("player")["targets_pg"]
    assert pg["Nacua"] == pytest.approx(9.0)


def test_best_week_is_a_max_not_a_sum(pw):
    r = ask.answer(pw, "WR by best week")
    assert r.df.set_index("player")["best_week"]["Nacua"] == pytest.approx(21.0)


def test_tprr_uses_routes(pw):
    """Adams 18 targets over 58 routes."""
    r = ask.answer(pw, "WR by tprr, min 20 routes")
    row = r.df[r.df["player"] == "Adams"].iloc[0]
    assert row["tprr"] == pytest.approx(18 / 58)


def test_ordering_and_top_n(pw):
    # receiving yards, not targets: Adams and Nacua both have 18 targets, and a tie
    # would test pandas' sort stability rather than the ordering this parses.
    r = ask.answer(pw, "top 2 WR by receiving yards")
    assert r.df["player"].tolist() == ["Nacua", "Adams"]      # 190 then 120
    assert r.df["rank"].tolist() == [1, 2]
    worst = ask.answer(pw, "worst 2 WR by receiving yards")
    assert worst.df["player"].tolist() == ["Adams", "Nacua"]   # Parkins is a TE, excluded


def test_week_filter_changes_the_answer(pw):
    wk1 = ask.answer(pw, "WR by targets in week 1").df.set_index("player")["targets"]
    assert wk1["Adams"] == 10 and wk1["Nacua"] == 6
    wk2 = ask.answer(pw, "WR by targets in week 2").df.set_index("player")["targets"]
    assert wk2["Adams"] == 8 and wk2["Nacua"] == 12


def test_team_filter(pw):
    r = ask.answer(pw, "Rams pass catchers by targets")
    assert set(r.df["player"]) == {"Adams", "Nacua", "Parkins"}


def test_min_den_drops_the_noise_and_says_it_did(pw):
    r = ask.answer(pw, "WR by catch rate min 30 targets")
    assert r.df.empty or "Nacua" not in set(r.df["player"])
    assert any("below 30" in w for w in r.warnings)


def test_positions_are_filtered(pw):
    r = ask.answer(pw, "RB by carries")
    assert set(r.df["player"]) == {"McCaff", "Guerend"}


# ---------------------------------------------------------------- scope + reporting
def test_my_roster_scope(pw):
    r = ask.answer(pw, "my team by targets", mine={"A", "D"})
    assert set(r.df["player"]) == {"Adams", "McCaff"}


def test_free_agent_scope_excludes_the_owned(pw):
    r = ask.answer(pw, "free agent WR by targets", rostered={"A": "TaylorMade"})
    assert "Adams" not in set(r.df["player"])
    assert "Nacua" in set(r.df["player"])


def test_scope_without_league_data_warns_rather_than_lying(pw):
    r = ask.answer(pw, "my team by targets", mine=set())
    assert any("roster was not available" in w for w in r.warnings)


def test_restated_says_how_it_was_read():
    q = ask.parse("top 10 WR by snap % on the Rams last 3 weeks", (1, 2, 3, 4, 5))
    s = ask.restate(q)
    assert "Top 10" in s and "WR" in s and "snap %" in s and "LA" in s and "last 3 weeks" in s


def test_unmatched_words_are_reported_not_swallowed(pw):
    r = ask.answer(pw, "WR by targets in the rain")
    assert any("Ignored" in w and "rain" in w for w in r.warnings)


def test_missing_column_is_explained_not_crashed():
    bare = ask.player_week(_stats())          # no snaps, no crosswalk
    r = ask.answer(bare, "WR by snap %")
    assert r.df.empty
    assert any("snap" in w.lower() or "does not carry" in w for w in r.warnings)


def test_every_example_parses_and_runs(pw):
    for q in ask.EXAMPLES:
        r = ask.answer(pw, q, weeks_available=(1, 2), mine={"A"}, rostered={"A": "X"})
        assert isinstance(r.df, pd.DataFrame), q
        assert r.restated.endswith("."), q


def test_every_alias_resolves_to_its_own_field():
    """No alias may be shadowed by a longer one belonging to a different metric."""
    for f in ask.FIELDS:
        for a in f.aliases:
            got = ask.parse(f"players by {a}").field
            assert got.key == f.key, f"{a!r} -> {got.key}, expected {f.key}"


# ---------------------------------------------------------------- red zone
def _redzone() -> pd.DataFrame:
    """Adams and Nacua see red zone targets; McCaffrey does the goal-line work."""
    rows = [
        # gsis, wk, rz_car, rz_tgt, i10_car, i10_tgt, gl_car, gl_tgt
        ("A", 1, 0, 3, 0, 2, 0, 1), ("A", 2, 0, 1, 0, 1, 0, 0),
        ("B", 1, 0, 1, 0, 0, 0, 0), ("B", 2, 0, 4, 0, 2, 0, 1),
        ("D", 1, 6, 1, 4, 1, 3, 0), ("D", 2, 4, 0, 3, 0, 2, 0),
        ("E", 1, 1, 0, 0, 0, 0, 0), ("E", 2, 2, 0, 1, 0, 1, 0),
    ]
    df = pd.DataFrame(rows, columns=["gsis_id", "week", "rz_carries", "rz_targets",
                                     "i10_carries", "i10_targets", "gl_carries", "gl_targets"])
    for tag in ("rz", "i10", "gl"):
        df[f"{tag}_touches"] = df[f"{tag}_carries"] + df[f"{tag}_targets"]
    team = {"A": "LA", "B": "LA", "D": "SF", "E": "SF"}
    df["_t"] = df["gsis_id"].map(team)
    cols = [c for c in df.columns if c.startswith(("rz_", "i10_", "gl_"))]
    tt = df.groupby(["_t", "week"])[cols].sum()
    tt.columns = [f"team_{c}" for c in cols]
    return df.merge(tt.reset_index(), on=["_t", "week"], how="left").drop(columns=["_t"])


@pytest.fixture(scope="module")
def pwrz() -> pd.DataFrame:
    return ask.player_week(_stats(), _snaps(), None, _routes(), _crosswalk(), _redzone())


def test_red_zone_carries(pwrz):
    r = ask.answer(pwrz, "red zone carries by RB")
    assert r.df.set_index("player")["rz_carries"]["McCaff"] == 10   # 6 + 4


def test_goal_line_carries_are_inside_the_5_not_the_20(pwrz):
    r = ask.answer(pwrz, "top 5 RB by goal line carries")
    v = r.df.set_index("player")["gl_carries"]
    assert v["McCaff"] == 5 and v["Guerend"] == 1
    assert ask.parse("goal line carries").field.key == "gl_carries"
    assert ask.parse("red zone carries").field.key == "rz_carries"


def test_red_zone_target_share(pwrz):
    """Adams 4 RZ targets of LA's 9."""
    r = ask.answer(pwrz, "red zone target share for WR min 5")
    assert r.df.set_index("player")["rz_target_share"]["Adams"] == pytest.approx(4 / 9)


def test_absent_from_the_red_zone_is_a_zero_not_a_blank(pwrz):
    """A player with no red zone work must still appear when asked for the fewest —
    filling with NaN would drop exactly the player that question is looking for."""
    r = ask.answer(pwrz, "fewest red zone carries by WR")
    assert r.df.iloc[0]["rz_carries"] == 0


def test_red_zone_columns_are_zero_when_no_frame_is_given(pw):
    r = ask.answer(pw, "red zone carries by RB")
    assert r.df.empty or set(r.df["rz_carries"]) == {0}


# ---------------------------------------------------------------- generic catalogue
def test_unknown_stat_without_a_loader_says_where_it_lives(pw):
    r = ask.answer(pw, "top 5 WR by average separation")
    assert r.query.source == "nextgen_stats_receiving"
    assert any("not loaded here" in w for w in r.warnings)


def test_curated_alias_does_not_hijack_an_exact_column_name():
    """'over expected' is an xFP± alias. 'rush yards over expected' is a real nflverse
    column, and the longer exact match has to win or the answer is a different stat."""
    assert ask.parse("points over expected").field.key == "xfp_diff"
    q = ask.parse("top 5 RB by rush yards over expected")
    assert q.source == "nextgen_stats_rushing"
    assert q.field.col == "rush_yards_over_expected"


def test_generic_hit_reports_its_source_and_carries_a_caveat():
    q = ask.parse("WR by average cushion")
    assert q.source == "nextgen_stats_receiving"
    assert "nextgen_stats_receiving" in ask.restate(q)
    assert "vetted" in q.field.note


def test_a_rate_column_is_averaged_not_summed():
    assert ask.parse("WR by average separation").field.kind == "mean"
    assert ask.parse("RB by rush yards over expected").field.kind == "total"


# ---------------------------------------------------------------- raw-table execution
def _fake_nextgen() -> pd.DataFrame:
    """Shaped like nflverse nextgen: a week 0 row holding the SEASON TOTAL."""
    return pd.DataFrame({
        "player_gsis_id": ["A", "A", "A", "B", "B", "B"],
        "player_display_name": ["Adams"] * 3 + ["Nacua"] * 3,
        "team_abbr": ["LA"] * 6,
        "week": [0, 1, 2, 0, 1, 2],
        "avg_separation": [3.0, 2.0, 4.0, 5.0, 6.0, 4.0],
    })


def test_week_zero_season_totals_are_dropped(pw):
    """Summing week 0 with the weeks that make it up double-counts every player."""
    q = ask.parse("WR by average separation")
    out, warns = ask.run_table(_fake_nextgen(), q, pos_map={"A": "WR", "B": "WR"})
    assert out.set_index("player")["avg_separation"]["Adams"] == pytest.approx(3.0)  # (2+4)/2
    assert out.set_index("player")["games"]["Adams"] == 2
    assert any("week 0" in w for w in warns)


def test_positions_resolve_through_ids_when_the_table_has_no_position_column():
    q = ask.parse("top 5 WR by average separation")
    out, _ = ask.run_table(_fake_nextgen(), q, pos_map={"A": "WR", "B": "TE"})
    assert set(out["player"]) == {"Adams"}


def test_table_query_without_a_position_map_says_so():
    q = ask.parse("top 5 WR by average separation")
    out, warns = ask.run_table(_fake_nextgen(), q, pos_map={})
    assert any("every position is shown" in w for w in warns)
    assert set(out["player"]) == {"Adams", "Nacua"}


def test_missing_column_in_a_raw_table_is_explained():
    q = ask.parse("WR by average separation")
    out, warns = ask.run_table(_fake_nextgen().drop(columns=["avg_separation"]), q)
    assert out.empty and any("not in" in w for w in warns)


# ---------------------------------------------------------------- seasons
@pytest.mark.parametrize("text,years", [
    ("WR by targets in 2024", (2024,)),
    ("WR by targets 2023-2025", (2023, 2024, 2025)),
    ("compare 2024 and 2025 WR target share", (2024, 2025)),
    ("WR by targets 2022 vs 2025", (2022, 2025)),
    ("WR by targets 2023 to 2025", (2023, 2024, 2025)),
])
def test_season_parsing(text, years):
    assert ask.parse(text).seasons == years


def test_last_n_seasons_anchors_on_the_current_one():
    assert ask.parse("WR by targets last 3 seasons", default_season=2026).seasons == (2024, 2025, 2026)


def test_a_year_is_not_mistaken_for_a_week_or_a_row_limit():
    q = ask.parse("top 10 WR by targets in 2024")
    assert q.seasons == (2024,) and q.top == 10 and q.weeks == ()
    q2 = ask.parse("WR by targets weeks 2-4 in 2024")
    assert q2.seasons == (2024,) and q2.weeks == (2, 3, 4)


def test_a_year_in_the_question_beats_the_picker(pw):
    r = ask.answer(pw, "WR by targets in 2024", default_seasons=(2019, 2020))
    assert r.query.seasons == (2024,)


def test_the_picker_supplies_years_when_the_question_does_not(pw):
    r = ask.answer(pw, "WR by targets", default_seasons=(2024, 2025),
                   pw_loader=lambda s: pw.assign(season=s))
    assert r.query.seasons == (2024, 2025)


def test_most_improved_ranks_by_the_change_not_the_level():
    assert ask.parse("most improved WR by targets 2024 vs 2025").by_change
    assert not ask.parse("top WR by targets 2024 vs 2025").by_change


def test_xfp_difference_still_resolves_despite_the_change_words():
    """'difference' signals a change ranking AND is part of an alias; the alias wins."""
    assert ask.parse("WR by xfp difference").field.key == "xfp_diff"


def _season_pw(season: int) -> pd.DataFrame:
    """Same players, different volumes, so a change is checkable by eye."""
    bump = {2024: 1.0, 2025: 2.0}[season]
    st = _stats().copy()
    st["targets"] = (st["targets"] * bump).astype(int)
    st["season"] = season
    return ask.player_week(st, season=season)


def test_two_seasons_come_back_side_by_side_with_the_change():
    r = ask.answer(None, "compare 2024 and 2025 WR by targets", pw_loader=_season_pw)
    assert list(r.df.columns) == ["rank", "player", "pos", "team", "2024", "2025", "change"]
    row = r.df.set_index("player").loc["Nacua"]
    assert row["2024"] == 18 and row["2025"] == 36 and row["change"] == 18


def test_each_season_is_ranked_without_the_row_limit_before_merging():
    """Applying the limit per season and merging afterwards compares one year's top N
    against another's, and a player just outside it comes back blank instead of lower."""
    r = ask.answer(None, "top 2 WR by targets 2024 vs 2025", pw_loader=_season_pw)
    assert len(r.df) == 2
    assert r.df[["2024", "2025"]].notna().all().all()


def test_comparison_lines_players_up_on_id_not_name():
    """A name can change spelling between nflverse seasons; an id does not."""
    def loader(season):
        d = _season_pw(season)
        if season == 2025:                       # same player, different spelling
            d["player"] = d["player"].replace({"Nacua": "Puka Nacua"})
        return d
    r = ask.answer(None, "compare 2024 and 2025 WR by targets", pw_loader=loader)
    assert len(r.df[r.df["player"].str.contains("Nacua")]) == 1


def test_restate_names_the_years():
    assert "in 2024" in ask.restate(ask.parse("WR by targets in 2024"))
    s = ask.restate(ask.parse("most improved WR by targets 2024 vs 2025"))
    assert "2024, 2025" in s and "ranked by the change" in s


def test_a_season_with_no_rows_is_reported_not_silently_dropped():
    def loader(season):
        return _season_pw(2024) if season == 2024 else pd.DataFrame()
    r = ask.answer(None, "compare 2024 and 2025 WR by targets", pw_loader=loader)
    assert any("2025 returned no rows" in w for w in r.warnings)
