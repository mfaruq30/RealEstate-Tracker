# Real Estate Value Analyzer
**CS 506 Final Project — Check-In 2 (April)**

---

## TL;DR

We built a residual-based mispricing analyzer for Greater Boston residential real estate. Starting from 184,552 raw property records from the City of Boston FY2026 assessment data, we cleaned, filtered, and enriched the dataset with Census ACS demographics and Zillow ZHVI market trends, ending with **136,581 properties × 84 features**. We trained two models on these features:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | $262,947 | $416,006 | 0.525 |
| Random Forest (primary) | **$131,440** | **$213,770** | **0.875** |

The large jump from linear to RF indicates strong non-linear feature interactions. We use the Random Forest residuals (predicted − actual) to flag potentially mispriced properties, which is the project's main deliverable.

---

## 1. Project Goal

**Can we identify properties that are priced unusually high or low relative to comparable homes by combining structured property data with neighborhood-level context?**

We do not try to assert a property's "true" value. Instead, we use a regression model trained on observable features to estimate an *expected* price, and flag the largest positive and negative residuals as candidate mispricings.

---

## 2. Current Status

### What's done
- ✅ Real data collection from 6 sources (Boston Assessment, Parcels, Zillow, MBTA GTFS, BPD Crime, Census ACS)
- ✅ Cleaning + enrichment pipeline producing `boston_properties_enriched.csv` (136,581 rows × 84 cols)
- ✅ Feature engineering: 7 derived features (price_per_sqft, sqft_per_room, home_age, bath_to_bed_ratio, lot_to_living_ratio, is_remodeled, renovation_gap)
- ✅ Census + Zillow merges by ZIP (100% Census match rate, 97% Zillow match rate)
- ✅ Two models trained (linear + Random Forest), train/test split, full metrics
- ✅ 5 visualizations (price distribution, price vs sqft, predicted vs actual, feature importance, residual distribution)
- ✅ Per-property residuals saved to `outputs/checkpoint2/residuals.csv`
- ✅ Tests + GitHub Actions CI

### What's not yet done (planned for final report, May 1)
- ⏳ Per-property latitude/longitude — the Boston Parcels CSV does not include coordinates; we will extract them from the GeoJSON variant or geocode addresses
- ⏳ Transit proximity and crime density features (depend on lat/long)
- ⏳ Interactive map of residuals (depends on lat/long)
- ⏳ K-fold cross-validation (currently using a single 75/25 split)
- ⏳ Final report polish: Makefile, video demo, README rewrite

---

## 3. Data

### Sources used
| Dataset | Source | Used for |
|---|---|---|
| Property Assessment FY2026 | data.boston.gov | Primary dataset (price, sqft, beds, baths, year built, etc.) |
| Census ACS 5-Year (2019) | api.census.gov | ZIP-level median income, population, owner-occupancy |
| Zillow ZHVI by ZIP | zillow.com/research | ZIP-level home value index + YoY trend |

### Sources collected but not yet used
| Dataset | Status |
|---|---|
| Boston Parcels 2025 | Downloaded; CSV variant lacks coordinates — will switch to GeoJSON for final report |
| MBTA GTFS stops | Downloaded; awaiting lat/long to compute transit proximity |
| BPD Crime Incidents | Downloaded; awaiting lat/long to compute crime density |

### Cleaning pipeline (`src/real_estate_tracker/data_processing.py`)
1. Load raw assessment CSV (handles encoding fallbacks: utf-8 → latin-1 → cp1252)
2. Filter to residential land-use codes (R1, R2, R3, R4, CD, A) → 136,581 properties
3. Standardize column names (handles FY2024 and FY2026 column variants)
4. Coerce numeric types (strips comma-separators in city CSVs)
5. Combine `full_bathrooms` + `0.5 × half_bathrooms` → unified `bathrooms`
6. Pad ZIP codes to 5 digits
7. Trim top/bottom 1% of prices (IQR-style outlier removal)
8. Filter `sqft` to [200, 15000]
9. Impute missing year_built with median
10. Deduplicate on Parcel ID

### Enrichment pipeline (`src/real_estate_tracker/feature_enrichment.py`)
- Merge Census demographics by ZIP code
- Merge Zillow ZHVI by ZIP code
- Compute `price_vs_census_ratio` and `price_vs_zhvi_ratio` (property price normalized by ZIP medians)

---

## 4. Modeling

### Features used (16)
Structural: `sqft`, `lot_sqft`, `bedrooms`, `bathrooms`, `total_rooms`, `num_floors`, `year_built`, `fireplaces`
Engineered: `sqft_per_room`, `home_age`, `bath_to_bed_ratio`, `lot_to_living_ratio`, `is_remodeled`, `renovation_gap`
Neighborhood: `median_household_income`, `total_population`

> **Note on leakage:** an earlier version included `price_per_sqft` as a feature, which leaks the target. Removed before reporting metrics.

### Setup
- 75/25 train/test split (random_state=42)
- Linear Regression (baseline, sklearn defaults)
- Random Forest (n_estimators=150, max_depth=10, random_state=42)

### Results
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $262,947 | $416,006 | 0.525 |
| Random Forest | $131,440 | $213,770 | 0.875 |

The Random Forest reduces MAE by 50% over the linear baseline, indicating substantial non-linear structure in the data that linear regression cannot capture (e.g., feature interactions between sqft and neighborhood income).

### Random Forest feature importance (top 5)
1. `total_population` — 30% (acting as a neighborhood proxy in absence of lat/long)
2. `bathrooms` — 23%
3. `sqft` — 19%
4. `median_household_income` — 10%
5. `sqft_per_room` — 6%

These five features together account for ~88% of the model's decisions.

### Residual analysis
The percentage residual `(predicted − actual) / actual × 100` is approximately bell-shaped, centered near 0, with a slight right skew. This means the model is:
- Unbiased on average (no systematic over/under prediction)
- More likely to flag a property as underpriced than overpriced

Properties with large positive residuals (model predicts substantially more than the listed price) are our primary candidates for "potentially underpriced." See `outputs/checkpoint2/residuals.csv`.

---

## 5. Visualizations

All plots in `outputs/checkpoint2/figures/`:

| File | Shows |
|---|---|
| `price_distribution.png` | Right-skewed histogram of assessment prices; median ~$747K |
| `price_vs_sqft.png` | Scatter — high variance at any given sqft, motivating the need for location features |
| `predicted_vs_actual.png` | Tight diagonal cluster at low/mid prices, more scatter for luxury homes |
| `feature_importance.png` | Horizontal bar chart of RF Gini importances |
| `residual_distribution.png` | Histogram of percentage residuals, roughly normal around 0 |

---

## 6. Reproducing These Results

### Environment
Python 3.10+ (tested on 3.9 and 3.11).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1 — Download data
```bash
python scripts/download_datasets.py
python scripts/fetch_api_data.py
```

If data.boston.gov returns 403/404 errors (it occasionally blocks scripted requests), download manually:
- Boston Property Assessment FY2026 → save to `data/raw/boston_property_assessment_fy2026.csv`
- Boston Parcels 2025 (CSV) → `data/raw/boston_parcels_2025.csv`
- BPD Crime Incidents 2023+ → `data/raw/bpd_crime_incidents_2024.csv`

### Step 2 — Run the data pipeline
```bash
python scripts/run_pipeline.py
```
Produces `data/processed/boston_properties_enriched.csv` (~70 MB, 136K rows).

### Step 3 — Train the model
```bash
python scripts/run_model.py
```
Produces `outputs/checkpoint2/` with metrics, figures, residuals, and run summary. Random Forest training takes 2–5 minutes on 130K rows.

### Step 4 — Run tests
```bash
PYTHONPATH=src pytest -q
```

---

## 7. Repository Structure

```
RealEstate-Tracker/
├── data/
│   ├── raw/                              # Downloaded datasets (gitignored)
│   ├── processed/                        # Cleaned + enriched CSVs (gitignored)
│   ├── sample_listings.csv               # 30-row sample for CI
│   └── README.md                         # Column dictionary
├── outputs/
│   ├── checkpoint1/                      # Sample-data results (proposal stage)
│   └── checkpoint2/                      # Real Boston results (current)
│       ├── metrics.json
│       ├── residuals.csv
│       ├── run_summary.json
│       └── figures/
├── scripts/
│   ├── download_datasets.py              # Pull raw data from public sources
│   ├── fetch_api_data.py                 # Pull Census + FRED via API
│   ├── run_pipeline.py                   # Clean + enrich → boston_properties_enriched.csv
│   ├── run_checkpoint1.py                # Sample-data pipeline (proposal stage)
│   └── run_model.py                      # Train + evaluate models on enriched data
├── src/real_estate_tracker/
│   ├── data_processing.py                # Cleaning, feature engineering, column selection
│   ├── feature_enrichment.py             # Transit, crime, Census, Zillow merges
│   ├── modeling.py                       # train_and_evaluate, save_metrics
│   └── visualization.py                  # All plotting functions
├── tests/
│   ├── test_data_processing.py
│   ├── test_assessment_processing.py
│   ├── test_feature_enrichment.py
│   └── test_pipeline_smoke.py
├── .github/workflows/ci.yml              # GitHub Actions test runner
├── requirements.txt
└── README.md                             # This file
```

---

## 8. For Teammates Picking This Up

If you're joining this project mid-stream, here's the fastest way to be productive:

### To reproduce the current results (15 min)
1. Clone the repo, set up the venv, install requirements
2. Download the 3 critical files manually from data.boston.gov (assessment, parcels, crime) — see Step 1 above
3. Run `download_datasets.py` to grab the rest (Zillow, MBTA, etc.) — anything that 403s, download by hand
4. Run `fetch_api_data.py` for Census
5. Run `run_pipeline.py`, then `run_model.py`
6. Open `outputs/checkpoint2/figures/` to see all five plots

### What's safe to change
- `select_assessment_model_columns` in `data_processing.py` — add or remove features here
- Random Forest hyperparameters in `modeling.py`
- Anything in `visualization.py` — adding a new plot is straightforward (define a new function, call it from `run_model.py`)

### What to be careful with
- `clean_assessment_data` — column rename map handles two FY variants; don't break either
- `feature_enrichment.py` `iterrows()` loops — currently slow (10–30 min on 130K rows). Don't make the dataset larger without vectorizing first
- Adding any feature derived from `price` — that's target leakage. Always sanity check with `corr(feature, price)` before including

### Most useful next pieces of work, in priority order
1. **Get lat/long from the parcels GeoJSON** — unlocks transit, crime, and the residual map. Single biggest unblock for the final report.
2. **K-fold cross-validation** — 5 minutes of code, makes the metrics defensible.
3. **Hyperparameter search on the RF** — `GridSearchCV` or `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`. Could push R² past 0.90.
4. **Vectorize the haversine loops** in `feature_enrichment.py` using a KD-tree (`scipy.spatial.cKDTree`) — turns 30 min into 30 sec.
5. **Folium or Plotly residual map** — once lat/long is in, this is the project's final visual deliverable.

---

## 9. Known Limitations

- **No per-property location features.** ZIP-level features (Census, Zillow) are the spatial backbone. `total_population` is acting as a neighborhood proxy, which works in this dataset but won't generalize.
- **Census data is from 2019.** The 2020+ ACS endpoints we tried returned 400 errors; the 2019 ACS5 was the most recent usable. Demographics shift slowly, so this is acceptable for now but worth noting.
- **Single train/test split.** No cross-validation yet. Metrics could shift by a few points with a different random seed.
- **Outlier clipping (top/bottom 1%) drops some real luxury properties.** This biases the model toward typical residential homes, which is fine for the project goal but worth flagging.
- **Linear regression had numerical warnings during prediction.** The matmul overflow warnings are due to extreme values in some engineered features (e.g., `lot_to_living_ratio` for properties with tiny living areas). Random Forest is unaffected. Adding feature scaling would fix this if we want a cleaner linear baseline.

---

## 10. Project Timeline

| Week | Goal | Status |
|---|---|---|
| 1 | Repo setup, environment, scaffolding | ✅ Done |
| 2 | Data collection | ✅ Done |
| 3 (Check-In 1) | Initial cleaning, EDA, baseline | ✅ Done |
| 4 | Feature engineering (NLP + geospatial) | ⚠️ Partial — geospatial blocked on lat/long |
| 5 | Model training + comparisons | ✅ Done |
| 6 (Check-In 2) | Model evaluation + error analysis | ✅ **You are here** |
| 7 | Visualization + interactive map | ⏳ Pending |
| 8 | Final report + presentation | ⏳ Pending |