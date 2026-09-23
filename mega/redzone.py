"""Scoring-position opportunity: carries and targets inside the 20, the 10 and the 5.

Play-by-play is the only source for this. The weekly player stats carry `rushing_10`,
`rushing_12` and `rushing_20`, which look like red zone columns and are not — they count
runs of 10+, 12+ and 20+ YARDS. Reading them as "carries inside the 20" would have
overstated Derrick Henry's red zone work by more than double.

Verified against nflverse's own totals: summing rushes by `rusher_player_id` across
2026 reproduces every one of 1,306 players' `carries` in load_player_stats exactly.
"""
from __future__ import annotations

import pandas as pd

# name -> yards from the opposing goal line. "rz" is the red zone proper; the tighter
# two matter because goal-line work is what actually converts to touchdowns.
ZONES = {"rz": 20, "i10": 10, "gl": 5}

_COLS = ["season_type", "week", "posteam", "yardline_100", "rush_attempt", "pass_attempt",
         "rusher_player_id", "receiver_player_id", "two_point_attempt"]


def weekly(season: int) -> pd.DataFrame:
    """One row per player per week: carries and targets in each zone, plus his team's
    totals in the same week, so a share can be taken without a second pass."""
    import polars as pl

    from . import pbp as _pbp

    try:
        p = _pbp.load(season, _COLS)
    except Exception:
        return pd.DataFrame()
    if p is None or p.height == 0:
        return pd.DataFrame()

    # Two-point plays are excluded: they are snapped from the 2 and would otherwise
    # inflate goal-line work for players who never got a real scoring-position touch.
    p = p.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("two_point_attempt").fill_null(0) != 1)
        & pl.col("posteam").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )

    frames: list[pd.DataFrame] = []
    for tag, line in ZONES.items():
        z = p.filter(pl.col("yardline_100") <= line)
        carries = (
            z.filter((pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null())
            .group_by(["rusher_player_id", "week", "posteam"]).len()
            .rename({"rusher_player_id": "gsis_id", "len": f"{tag}_carries", "posteam": "team"})
        )
        # A target is a pass with a named receiver; throwaways carry a null receiver_player_id.
        targets = (
            z.filter((pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null())
            .group_by(["receiver_player_id", "week", "posteam"]).len()
            .rename({"receiver_player_id": "gsis_id", "len": f"{tag}_targets", "posteam": "team"})
        )
        frames.append(carries.to_pandas())
        frames.append(targets.to_pandas())

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["gsis_id", "week", "team"], how="outer")
    if out.empty:
        return out
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")

    count_cols = [c for c in out.columns if c.endswith(("_carries", "_targets"))]
    out[count_cols] = out[count_cols].fillna(0.0)

    # team totals for the same week, so shares use a matching denominator
    team = out.groupby(["team", "week"], dropna=False)[count_cols].sum()
    team.columns = [f"team_{c}" for c in count_cols]
    out = out.merge(team.reset_index(), on=["team", "week"], how="left")

    for tag in ZONES:
        out[f"{tag}_touches"] = out[f"{tag}_carries"] + out[f"{tag}_targets"]
        out[f"team_{tag}_touches"] = out[f"team_{tag}_carries"] + out[f"team_{tag}_targets"]
    return out.drop(columns=["team"])


def validate(w: pd.DataFrame) -> list[str]:
    """Cheap invariants, for the same reason mega/routes.py has them."""
    msgs = []
    if w is None or w.empty:
        return ["no red zone rows"]
    for tag, line in ZONES.items():
        if tag == "rz":
            continue
        wider = w["rz_carries"] + w["rz_targets"]
        inner = w[f"{tag}_carries"] + w[f"{tag}_targets"]
        bad = int((inner > wider).sum())
        if bad:
            msgs.append(f"{bad} player-weeks with more work inside the {line} than inside the 20")
    return msgs
