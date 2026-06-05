<p align="center">
  <img src="https://img.shields.io/badge/R²_Score-99.81%2F100-00C853?style=for-the-badge&logo=target&logoColor=white" alt="R² Score"/>
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LightGBM-4.6-9ACD32?style=for-the-badge" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/XGBoost-3.2-FF6600?style=for-the-badge" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/CatBoost-1.2-FFD700?style=for-the-badge" alt="CatBoost"/>
  <img src="https://img.shields.io/badge/Optuna-4.9-4B0082?style=for-the-badge" alt="Optuna"/>
</p>

<h1 align="center">🚦 Flipkart Gridlock 2.0</h1>

<p align="center">
  <strong>Traffic Demand Prediction | Supervised Regression | Geospatial + Temporal ML</strong>
</p>

<p align="center">
  <em>A competition-grade, end-to-end machine learning pipeline that predicts real-time traffic demand across geographic locations using geohash spatial indexing, cyclical temporal encoding, and a tri-model ensemble optimized with Optuna hyperparameter tuning.</em>
</p>

---

## 🏆 Results

| Model | OOF R² | Competition Score |
|:------|:------:|:-----------------:|
| LightGBM (baseline) | 0.9971 | 99.71 |
| XGBoost | 0.9972 | 99.72 |
| CatBoost | 0.9981 | 99.81 |
| LightGBM (Optuna-tuned) | 0.9966 | 99.66 |
| **🥇 Final Ensemble (optimized)** | **0.9981** | **99.81** |

> The final ensemble uses Nelder-Mead optimized weights: **7.6% XGBoost + 92.4% CatBoost**, selected via OOF R² maximization over a grid-search + scipy optimizer pipeline.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA (train.csv + test.csv)              │
├─────────────┬───────────────┬───────────────┬──────────────────┤
│  Geohash    │  Temporal     │  Road/Infra   │  Weather/Temp    │
│  Decoding   │  Parsing      │  Encoding     │  Features        │
├─────────────┴───────────────┴───────────────┴──────────────────┤
│              FEATURE ENGINEERING (154 features)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Geohash → lat/lon + truncated prefixes (p2–p5)        │  │
│  │ • Cyclical sin/cos encoding (hour, minute, day-of-week) │  │
│  │ • 16 group-level demand statistics (train-only)         │  │
│  │ • Interaction keys: geohash×time_bucket, road×hour      │  │
│  │ • Label encoding on combined train+test pools           │  │
│  └──────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────┤
│                    5-FOLD CROSS VALIDATION                     │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────────┐  │
│  │  LightGBM  │  │  XGBoost   │  │  CatBoost               │  │
│  │  (tuned)   │  │  (hist)    │  │  (2000 iter, depth=8)   │  │
│  └────────────┘  └────────────┘  └─────────────────────────┘  │
├────────────────────────────────────────────────────────────────┤
│               OPTUNA HPO (50 trials × 3-fold CV)               │
├────────────────────────────────────────────────────────────────┤
│          ENSEMBLE BLENDING (Nelder-Mead weight opt)            │
├────────────────────────────────────────────────────────────────┤
│                      submission.csv                            │
└────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Key Technical Innovations

### 1. Geohash Spatial Decomposition
The geohash column encodes precise geographic locations using a base-32 spatial indexing system. We unlock its full potential by:
- **Decoding** each 6-character geohash to `(latitude, longitude)` using a pure-Python base-32 decoder (no external dependencies)
- **Multi-resolution truncation** at precisions 2–5, capturing geographic hierarchies from country zones to city blocks
- **Cyclical geographic encoding** via `sin(lat_rad)`, `cos(lat_rad)`, `sin(lon_rad)`, `cos(lon_rad)` — preventing the model from treating coordinates as linear scalars
- **Centroid distance** — Euclidean distance from dataset center, capturing urban-core vs. suburban dynamics

### 2. Cyclical Temporal Encoding
Traffic follows strong daily and weekly rhythms. We capture these patterns by:
- Parsing the `H:M` timestamp format into `hour`, `minute`, and `total_minutes`
- Creating **15-min / 30-min / 60-min time buckets** (96 / 48 / 24 bins per day)
- Applying **sin/cos transformations** to hour, minute, and time bucket — eliminating the artificial gap between 23:45 and 00:00
- Engineering binary flags for **morning peak** (7–9), **evening peak** (17–20), **late night** (22–5), and **business hours** (9–18)
- Computing `day_of_week = day % 7` with cyclical sin/cos encoding and `is_weekend` flag

### 3. Train-Only Group Statistics (Anti-Leakage)
The single highest-impact feature block. For **16 grouping keys** (e.g., `geohash`, `geohash×time_bucket_15`, `RoadType×hour`), we compute:
- `mean`, `std`, `median`, `min`, `max`, `count` of `demand` — **using only training data**
- These statistics are merged onto both train and test sets, with missing test groups filled using the global training mean

The top feature, `gh_tb15_mean` (geohash × 15-minute bucket demand mean), effectively tells the model: *"At this location, at this time of day, demand is typically X."*

### 4. Optuna-Powered Hyperparameter Optimization
- **50 TPE-sampled trials** with 3-fold CV per trial
- Search space covers 10 hyperparameters: `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `reg_alpha`, `reg_lambda`, `min_split_gain`
- Fixed random seed for full reproducibility

### 5. Optimized Ensemble Blending
Rather than naive averaging, we find the **optimal linear combination** of OOF predictions that maximizes R²:
- Coarse grid search over weight triples → initial point
- `scipy.optimize.minimize` with Nelder-Mead → final weights
- Constraint: weights ≥ 0, sum to 1

---

## 📊 Feature Importance (Top 20)

| Rank | Feature | Importance |
|:----:|:--------|:----------:|
| 1 | `gh_tb15_min` | 1,751 |
| 2 | `gh_tb15_max` | 1,703 |
| 3 | `gh_tb15_mean` | 1,562 |
| 4 | `gh_dow_min` | 719 |
| 5 | `gh_tb15_std` | 658 |
| 6 | `roadtype_hour_enc` | 428 |
| 7 | `dow_tb15_std` | 396 |
| 8 | `gh_tb15_median` | 351 |
| 9 | `gh_dow_std` | 285 |
| 10 | `dow_tb15_enc` | 262 |
| 11 | `gh_tb30_max` | 257 |
| 12 | `roadtype_hour_count` | 254 |
| 13 | `gh_dow_mean` | 241 |
| 14 | `gh_dow_median` | 230 |
| 15 | `roadtype_hour_mean` | 222 |
| 16 | `gh_hour_std` | 178 |
| 17 | `gh_tb30_std` | 176 |
| 18 | `gh_dow_count` | 173 |
| 19 | `gh_dow_max` | 170 |
| 20 | `roadtype_hour_std` | 150 |

> As expected, **geohash × time-bucket group statistics dominate** — location-time demand patterns are the strongest signal in traffic prediction.

---

## 🗂️ Repository Structure

```
Flipkart-Gridlock-2.0/
├── dataset/
│   ├── train.csv              # 77,299 rows × 11 columns (with target)
│   ├── test.csv               # 41,778 rows × 10 columns (without target)
│   └── sample_submission.csv  # Submission format reference
├── solution.py                # Complete end-to-end ML pipeline
├── submission.csv             # Final predictions (41,778 rows)
├── feature_importance.png     # Top 20 feature importance bar chart
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install lightgbm xgboost catboost optuna scikit-learn pandas numpy scipy matplotlib
```

> **macOS users**: LightGBM requires `libomp`. Install via:
> ```bash
> brew install libomp
> ```

### Run the Pipeline
```bash
python3 solution.py
```

The script will:
1. Load and inspect both datasets
2. Decode geohashes → lat/lon coordinates
3. Engineer 154 features (geospatial, temporal, group statistics, interactions)
4. Train LightGBM, XGBoost, and CatBoost with 5-fold CV
5. Run 50-trial Optuna hyperparameter search for LightGBM
6. Find optimal ensemble blend weights via Nelder-Mead optimization
7. Generate `submission.csv` and `feature_importance.png`

**Expected runtime**: ~15–25 minutes on Apple Silicon (M-series), ~30–45 minutes on Intel.

---

## 📋 Dataset Overview

| Column | Type | Description |
|:-------|:-----|:------------|
| `Index` | int | Unique row identifier |
| `geohash` | str | 6-character geohash encoding geographic location |
| `day` | int | Day identifier (48–54, maps to day-of-week via mod 7) |
| `timestamp` | str | Time in `H:M` format (e.g., `0:0`, `14:30`) |
| `RoadType` | str | `Residential`, `Street`, `Highway`, or NaN |
| `NumberofLanes` | int | Number of lanes (1–5) |
| `LargeVehicles` | str | `Allowed` or `Not Allowed` |
| `Landmarks` | str | `Yes` or `No` |
| `Temperature` | float | Temperature at location (°C, some NaN) |
| `Weather` | str | `Sunny`, `Rainy`, `Foggy`, `Snowy`, or NaN |
| **`demand`** | **float** | **Target variable (0–1 scale, train only)** |

---

## 🔬 Methodology Deep Dive

### Data Leakage Prevention
All group-level demand statistics are computed **exclusively on training data**. The test set only receives these statistics via merge/join — it never contributes to the computation. This ensures that cross-validation scores genuinely reflect generalization performance.

### Handling Missing Values
- **RoadType/Weather**: ~5% missing → filled as `"Unknown"` category, then label-encoded
- **Temperature**: ~7% missing → filled with column median from training set
- **Group statistics**: Test geohashes unseen in training → filled with global training demand mean

### Evaluation Metric
The competition metric is **R² × 100**, floored at 0:
```
Score = max(0, R²(y_true, y_pred) × 100)
```
Where R² = 1 − (SS_res / SS_tot). A perfect model scores 100.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|:----------|:-----------|:--------|
| Core | Python 3.13, NumPy, Pandas | Data manipulation & feature engineering |
| Model 1 | LightGBM 4.6 | Gradient boosting (leaf-wise) |
| Model 2 | XGBoost 3.2 | Gradient boosting (histogram) |
| Model 3 | CatBoost 1.2 | Gradient boosting (ordered) |
| HPO | Optuna 4.9 (TPE sampler) | Bayesian hyperparameter optimization |
| Optimization | SciPy (Nelder-Mead) | Ensemble weight optimization |
| Validation | Scikit-learn (KFold, r2_score) | Cross-validation framework |
| Visualization | Matplotlib | Feature importance plots |

---

## 📄 License

This project was created for the **Flipkart Gridlock 2.0** competition on HackerEarth.

---

<p align="center">
  <strong>Built with ❤️ for competitive machine learning</strong>
</p>
