# Expected fantasy points (xFP), end to end

What `xFP` means everywhere in this dashboard, where the number comes from, and exactly how
it is calculated. Written 2026-09-22 against ffopportunity model **v1.0.0**.

The short version: **we do not model expected points. nflverse does, and we re-score its
output to Mega Bowl's half-PPR rules.** Everything below is the detail behind those two
clauses.

---

## 1. What xFP is — and the thing it is not

Expected points answers: *given the opportunities a player actually got, in the exact game
situations they arrived in, how many fantasy points should they have produced?*

The critical detail, and the one that decides how you're allowed to use the number:

> **Air yards are the real throw, not a prediction.** The model takes each target as given —
> that it happened, and how far downfield it went — and asks only what it should have
> yielded.

So xFP measures **efficiency given opportunity**. It does *not* forecast whether the
opportunity continues. A receiver with a 29% target share and a large negative gap is not
being told "you will keep getting these targets"; he is being told "the targets you already
got should have scored more than they did."

That is the right input to a buy-low signal and the wrong thing to use alone as a
projection, which is why `projections.nflverse_estimate` weights it at 25% rather than
using it directly.

---

## 2. Where it comes from

[`ffverse/ffopportunity`](https://github.com/ffverse/ffopportunity), an R package. Per its
README the models are xgboost + tidymodels, trained on public nflverse play-by-play from
2006–2020. We consume the published output through `nflreadpy`:

```python
nfl.load_ff_opportunity(seasons=[season], stat_type="weekly", model_version="latest")
```

`"latest"` is an alias for `v1.0.0`; they are the same artifact. `stat_type` also accepts
`"pbp_pass"` and `"pbp_rush"` for the play-level rows, which is what you want when auditing
a single game.

---

## 3. The eight models

Eight XGBoost boosters, three for runs and five for passes. Each is applied per play by
`.forge_and_predict()` in `R/ep_predict.R`, in this order — and the order matters, because
each model can see the ones before it:

| # | Model | Type | Predicts |
|---|---|---|---|
| **Rush** | | | |
| 1 | `rushing_yards` | regression | yards on a carry |
| 2 | `rushing_td` | classification | P(rushing touchdown) |
| 3 | `rushing_fd` | classification | P(first down) |
| **Pass** | | | |
| 1 | `pass_completion` | classification | P(catch) on a target |
| 2 | `yards_after_catch` | regression | YAC if caught |
| 3 | `pass_touchdown` | classification | P(touchdown) |
| 4 | `pass_first_down` | classification | P(first down) |
| 5 | `passing_int` | classification | P(interception) |

### The chain is real, and it is visible in the binaries

Between models 2 and 3 on the pass side, `ep_predict.R` computes an intermediate:

```r
yardline_exp = yardline_100 - air_yards - yards_after_catch_exp
```

— i.e. *where the catch is expected to end up* — and feeds it to the touchdown model.

You can confirm the chaining without running any R. The feature count in each model's
header steps up by exactly the number of upstream predictions it consumes:

| Rush | features | | Pass | features |
|---|---|---|---|---|
| `rushing_yards` | 48 | | `pass_completion` | 41 |
| `rushing_td` | 49 | | `yards_after_catch` | 42 |
| `rushing_fd` | 50 | | `pass_touchdown` | 44 |
| | | | `pass_first_down` | 45 |
| | | | `passing_int` | 46 |

(Read from the `binf` legacy-format header: 4-byte magic, float `base_score`, uint32
`num_feature`. `pass_touchdown` gains two because it takes both `yards_after_catch_exp` and
the derived `yardline_exp`.)

### The models never learn who the player is

That same feature count is the proof. 41–50 predictors is a situational model; carrying
player identity would require thousands of dummy columns. The only player attribute in the
matrix is **`position`**. `rusher_player_id`, `passer_player_id`, `receiver_player_id` and
`full_name` ride along in the frame purely as join keys for the later rollup.

This is the whole point of the metric. Two different backs, same down, distance, field
position, game script and blocking direction, get the *same* expected value — so the gap
between a player's actual and expected points is attributable to him rather than to his
situation.

---

## 4. The features

Built in `R/ep_preprocess.R`. Shared by both models:

| Group | Columns |
|---|---|
| Game state | `down`, `ydstogo`, `yardline_100`, `goal_to_go`, `qtr`, `score_differential`, `half_seconds_remaining`, `game_seconds_remaining`, `fixed_drive`, `ep` |
| Vegas | `vegas_wp`, `total_line`, `implied_total` |
| Environment | `roof`, `surface`, `temp`, `wind` |
| Tendency | `xpass`, `shotgun`, `no_huddle` |
| Other | `position`, `era`, `posteam_type` |

Rush adds `run_location`, `run_gap`, `run_gap_dir`, `qb_dropback`, `qb_scramble`.

Pass adds `air_yards`, `pass_location`, `qb_hit`, `passer_position`, `receiver_position`,
and two engineered terms:

```r
relative_to_sticks  = air_yards - ydstogo       # is the throw past the marker?
relative_to_endzone = air_yards - yardline_100  # is the throw to the end zone?
```

Three derived inputs worth knowing about:

- **`implied_total`** — the team's implied points, reconstructed from `total_line` and
  `spread_line` with a sign convention for home/away favourite.
- **`temp` / `wind`** — forced to 68°F and 0mph indoors; missing outdoor values default to
  60°F and 8mph.
- **`roof`** collapses to indoors/outdoors, **`surface`** to grass/turf, **`era`** splits at
  2018.

Two-point attempts are handled specially: `down` is set to 4, `air_yards` becomes
`yardline_100`, and `xpass` is pinned to 0.75.

---

## 5. Play → player-week

From `R/ep_summarize.R`. These are the formulas that produce the columns we read.

**A target is worth P(catch) × (air yards + expected YAC):**

```r
yards_gained_exp = complete_pass_exp * (yards_after_catch_exp + air_yards)
receptions_exp   = rec_complete_pass_exp     # i.e. the sum of catch probabilities
```

A carry's expected yards come straight from the model, with hard-coded overrides: a kneel is
−1, an aborted snap is 0, and neither can score.

Those play rows are then summed per `player_id × week`, and each play is credited to **both**
the passer and the receiver via a pivot on `player_type`, which is why a QB and his receiver
both accrue expectation from the same throw.

---

## 6. Our layer: re-scoring to half-PPR

nflverse publishes its own `total_fantasy_points_exp`. **We do not use it.** From
`ep_summarize.R`:

```r
rec:  0.1*yards + 1*receptions + 6*TD + 2*two_point
pass: 0.04*yards - 2*INT      + 4*TD + 2*two_point
```

That is **full PPR**, and **−2 per interception**. Mega Bowl is half-PPR at **−1**. Two
divergences, not one.

Verified empirically as well as from source — rebuilding their column under each scoring
assumption gives mean absolute differences of 2.09 (standard), 1.09 (half) and **0.10**
(full). Their mean is 8.09 pts/game against our half-PPR 7.11.

So [`mega/intel.py`](../mega/intel.py) applies the league's own `SCORING` dict to the
component expectations:

```python
df["half_ppr_exp"] = (
    s["pass_yd"] * pass_yards_gained_exp + s["pass_td"] * pass_touchdown_exp
    + s["pass_int"] * pass_interception_exp
    + s["rush_yd"] * rush_yards_gained_exp + s["rush_td"] * rush_touchdown_exp
    + s["rec"] * receptions_exp + s["rec_yd"] * rec_yards_gained_exp
    + s["rec_td"] * rec_touchdown_exp
)
```

Same dict that scores actual points in `score_half_ppr`, so the two sides are comparable by
construction.

---

## 7. Known gaps

**The two sides do not score identical terms.** Actual points include fumbles lost, 2-point
conversions and return touchdowns; `half_ppr_exp` includes none of them.

| Term | Expected counterpart? | 2025 effect |
|---|---|---|
| Fumbles lost | **None** — no fumble model exists | −0.071 pts/g |
| 2-pt conversions | Yes (`*_two_point_conv_exp`), unused | +0.033 pts/g |
| Return TDs | **None** — special teams is out of scope | +0.019 pts/g |
| | **Net** | **−0.018 pts/g** |

That is 0.28% of average scoring. Only 9 of 384 regulars are distorted by more than 0.5/g,
and they are return specialists (Chimere Dike, Kene Nwangwu) who read as mild
over-performers on volume alone. Wiring in the 2-pt columns would move 0.039 pts/g and is
not worth the code.

**Coverage.** 2.0% of 2025 skill player-games (37 players, mostly low-volume WR/TE) have no
ff_opportunity row at all. `nflverse_estimate` falls back to season points-per-game there.

**Week alignment is fine.** `ff_opportunity` and `weekly()` both run weeks 1–22, so actual
and expected span the same games including the postseason. Worth stating because the
rolling-form window had exactly this bug — see `intel._recent_form`.

---

## 8. Where we consume it

| Consumer | Use |
|---|---|
| `intel.buy_low_sell_high` | actual − expected per game; ≤ −2.0 → BUY LOW, ≥ +2.5 → SELL HIGH |
| `lookup.py` | the "Vs expected" column in the player game log |
| `projections.nflverse_estimate` | 25% of `est`, 50% of the prior (with `season_pg`) |

---

## 9. Auditing it yourself

```bash
# play-level rows, with every *_exp column
.venv/bin/python -c "
import nflreadpy as nfl
d = nfl.load_ff_opportunity(seasons=[2025], stat_type='pbp_pass', model_version='latest')
print(d.shape)"

# feature count straight from a model header
curl -sL -o m.xgb https://raw.githubusercontent.com/ffverse/ffopportunity/main/modelling/v1.0.0/pass_touchdown.xgb
python3 -c "
import struct; b=open('m.xgb','rb').read(16)
print(struct.unpack('<fIi', b[4:16]))"   # (base_score, num_feature, num_class)
```

Source worth reading, in order: `R/ep_preprocess.R` (features), `R/ep_predict.R` (the chain),
`R/ep_summarize.R` (the formulas).
