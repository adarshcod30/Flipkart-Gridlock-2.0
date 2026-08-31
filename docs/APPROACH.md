# Approach — Flipkart Gridlock 2.0

## 1. The actual shape of the data

The dataset spans exactly two days:

- **Day 48** — 69,427 rows, fully labeled, across 1,241 geohashes.
- **Day 49** — 7,872 labeled rows in `train.csv`, plus all 41,778 rows of `test.csv` (unlabeled).

`test.csv` is entirely day 49. That single fact drives the whole design: the
only information a model can legitimately use about "yesterday" is day 48,
and the only rows worth training on are day-49 rows, because those are the
only ones whose feature distribution — and specifically, whose relationship
to the day-48 profile — matches what the model will see at inference time.

An earlier iteration of this project trained on the full `train.csv` (days 48
and 49 combined) without any day-48 profile features, while the inference
notebook it shipped alongside expected a model trained with 101 profile-based
features. The two were never reconciled, so the checked-in model and the
checked-in inference code were silently incompatible. This rewrite fixes that
by making `gridlock.features.build_feature_pipeline` / `transform` the single
place both `train.py` and `predict.py` get their features from.

## 2. Feature engineering

**Geohash decoding.** Geohashes are base32-encoded (`0-9`, `b-z` minus `a`,
`i`, `l`, `o`) interleaved bit-strings over longitude and latitude. Decoding
each character narrows a `[lat_min, lat_max] x [lon_min, lon_max]` box by
half on every bit; the final box's center is the decoded coordinate. This is
implemented with no external geo library.

**Cyclical time index.** `H:M` timestamps become `hour`, `minute`, and
`time_idx = hour * 4 + minute // 15`, a monotonic 0–95 index over the day's
96 fifteen-minute slots.

**Day-48 historical profile.** Day-48 rows are pivoted into a
`geohash x time_idx` table of mean demand, then linearly interpolated across
any gaps in a geohash's 96-slot day. This produces 96 columns (`d48_t0`
through `d48_t95`) that get merged onto every day-49 row — train and test
alike — by `geohash`. A `d48_same_time` column additionally extracts just the
slot matching each row's own `time_idx`, i.e. "what was demand at this exact
location and time, yesterday." Geohashes present in `test.csv` but absent
from day 48 (10 of 1,190) fall back to the global day-48 profile average
rather than `NaN`.

**Road, vehicle, and weather features.** `RoadType` and `Weather` are
label-encoded from the union of train and test *feature* values (never test
labels) so a category never gets a different code depending on which file it
came from. `LargeVehicles` and `Landmarks` become binary flags.
`NumberofLanes` is used as-is. `Temperature` is imputed with the training
median and paired with a `temperature_missing` indicator.

## 3. Why training is restricted to day 49

Merging the day-48 profile onto a day-48 row and training on it would be
leakage: for a given `(geohash, time_idx)` pair there is typically exactly
one day-48 observation, so `d48_t{that slot}` for that row would be trivially
close to (often identical to) its own label. Restricting training to day-49
rows — where the day-48 profile is a genuine *prior*, not a restatement of
the label — keeps the feature honest, and it exactly mirrors the structure
of `test.csv`, which is also entirely day 49.

The tradeoff is a small training set: 7,872 rows. The model (`XGBRegressor`,
`max_depth=6`, `n_estimators=600`, `learning_rate=0.05`, with L1/L2
regularization and column/row subsampling) is deliberately shallower than
the `max_depth=16` trees used in earlier iterations of this project, which
were tuned against a training set nearly ten times larger and would
overfit badly at this scale.

## 4. Validation

5-fold `KFold` cross-validation (shuffled, `random_state=42`) on the 7,872
day-49 rows, scored with R² — the same metric the competition itself uses
(`score = max(0, R² * 100)`). This is the only validation signal available
locally, since `test.csv` carries no labels; it is reported in full,
fold-by-fold, in `artifacts/metrics.json` after every training run, rather
than citing a single unverifiable leaderboard number.

Current result: **R² = 0.9577 ± 0.0031** across 5 folds.

## 5. What actually drives predictions

Feature importance is dominated by `RoadType` (~51% of total gain) — genuine
signal, since `Highway` rows average 0.57 demand versus 0.06 for
`Residential` — followed by the day-48 profile slots. Raw `lat`/`lon`
contribute comparatively little on their own, which is expected: the profile
feature already encodes location-specific behavior more directly than raw
coordinates can.
