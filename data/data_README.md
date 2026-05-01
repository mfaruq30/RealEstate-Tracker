# Data Directory — Real Estate Value Analyzer

## Directory Structure

```
data/
├── raw/                             # Raw downloaded files (gitignored)
│   ├── boston_property_assessment_fy2026.csv
│   ├── zillow_zhvi_by_zip.csv
│   └── census_acs_boston_zips.csv
├── processed/                       # Cleaned + enriched files (gitignored)
│   ├── boston_properties_clean.csv
│   ├── boston_properties_enriched.csv
│   └── pipeline_summary.json
├── sample_listings.csv              # 30-row sample for CI tests (committed)
└── README.md                        # This file
```

## How to Get the Data

Run from the repo root:

```bash
make data        # Downloads all raw datasets
make pipeline    # Cleans + enriches into boston_properties_enriched.csv
```

Or run the underlying scripts directly:

```bash
python scripts/download_datasets.py    # Boston Assessment + Zillow ZHVI
python scripts/fetch_api_data.py       # Census ACS via API
python scripts/run_pipeline.py         # Cleaning + enrichment
```

If `data.boston.gov` returns a 403 on the Boston Property Assessment file (it occasionally blocks scripted requests), download manually from <https://data.boston.gov/dataset/property-assessment> and save it to `data/raw/boston_property_assessment_fy2026.csv`.

## Data Sources

| Dataset | Source | Update Frequency |
|---------|--------|------------------|
| Property Assessment FY2026 | [data.boston.gov](https://data.boston.gov/dataset/property-assessment) | Annual |
| Zillow ZHVI by ZIP | [Zillow Research](https://www.zillow.com/research/data/) | Monthly |
| Census ACS 5-Year (2019) | [Census API](https://api.census.gov/) | Annual |

The Census API uses the 2019 ACS5 endpoint because 2020+ endpoints returned HTTP 400 errors during collection (changed ZCTA geography requirements). 2019 demographics are stable enough for ZIP-level signal.

## Column Dictionary — `boston_properties_enriched.csv`

The enriched dataset has 136,581 rows and 84 columns. The columns below are the ones used as model features or as identifiers; the dataset also contains many raw passthrough columns from the original Boston Assessment CSV that we keep for traceability but do not feed into the model.

### Identifiers and location

| Column | Type | Description |
|--------|------|-------------|
| `pid` | str | Parcel ID — unique identifier, used for deduplication |
| `gis_id` | str | GIS identifier (kept for future spatial work) |
| `street_num` | str | Street number |
| `street_name` | str | Street name |
| `zip_code` | str | 5-digit ZIP code (zero-padded) |
| `land_use` | str | R1 = single family, R2 = 2-family, R3 = 3-family, R4 = 4+ family, CD = condo, A = apartment |

### Target variable

| Column | Type | Description |
|--------|------|-------------|
| `price` | float | Total assessed value in USD (renamed from `TOTAL_VALUE` in source) |

### Structural features (used in model)

| Column | Type | Description |
|--------|------|-------------|
| `sqft` | float | Living area in square feet (renamed from `LIVING_AREA`) |
| `lot_sqft` | float | Lot area in square feet (renamed from `LAND_SF`) |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Full + 0.5 × half bathrooms |
| `total_rooms` | int | Total room count |
| `num_floors` | float | Number of floors |
| `year_built` | int | Year of construction |
| `year_remodeled` | int | Year of last remodel (0 if never remodeled) |
| `fireplaces` | int | Number of fireplaces |

### Engineered features (used in model)

Derived in `add_assessment_features` in `data_processing.py`:

| Column | Formula | Purpose |
|--------|---------|---------|
| `home_age` | `2026 - year_built` | Interpretable transformation of year_built |
| `sqft_per_room` | `sqft / total_rooms` | Layout efficiency proxy |
| `bath_to_bed_ratio` | `bathrooms / bedrooms` | Luxury indicator (modern builds approach 1.0) |
| `lot_to_living_ratio` | `lot_sqft / sqft` | Urban-vs-suburban signal |
| `is_remodeled` | `1 if year_remodeled > 0 else 0` | Renovation flag |
| `renovation_gap` | `year_remodeled - year_built` | Years between build and remodel |
| `price_per_sqft` | `price / sqft` | EDA only — **excluded from model** (target leakage) |

### Census ACS features (ZIP-level merge)

Merged in `merge_census_demographics`. All values are at the ZIP-code level, identical for every property in the same ZIP.

| Column | Type | Description |
|--------|------|-------------|
| `median_household_income` | float | Median household income in the ZIP (USD) — **used in model** |
| `total_population` | int | Total population in the ZIP — **used in model** |
| `pct_owner_occupied` | float | Percent of housing units owner-occupied |
| `census_median_home_value` | float | Census-reported median home value in the ZIP |
| `price_vs_census_ratio` | float | `price / census_median_home_value` (>1 = above ZIP median) |

### Zillow ZHVI features (ZIP-level merge)

Merged in `merge_zillow_trends`. ~97% of properties matched a Zillow ZIP.

| Column | Type | Description |
|--------|------|-------------|
| `zhvi_latest` | float | Most recent Zillow Home Value Index for the ZIP |
| `zhvi_1yr_ago` | float | ZHVI 12 months prior |
| `zhvi_yoy_change_pct` | float | Year-over-year ZHVI change |
| `price_vs_zhvi_ratio` | float | `price / zhvi_latest` |

### Other passthrough columns

The cleaned CSV preserves many additional columns from the original Boston Assessment (e.g. `OWNER`, `MAIL_ADDRESSEE`, `BLDG_TYPE`, `INT_WALL`, `KITCHEN_STYLE1`, etc.) for traceability. These are **not** used as model features. The full list of 84 columns is recorded in `data/processed/pipeline_summary.json` after running the pipeline.

## Sample Data

`data/sample_listings.csv` contains 30 fictional listings with a different schema (price, sqft, beds, baths, year_built, zip_code, lat, lon, description). It exists for CI tests in `tests/test_pipeline_smoke.py` and is **not** used in the production pipeline. The real Boston pipeline uses the assessment CSV, which has no listing description text.

## Cleaning Summary

The pipeline produces these files (all gitignored):

- `boston_properties_clean.csv` — after cleaning, before enrichment (~136K rows)
- `boston_properties_enriched.csv` — final output (~136K rows × 84 columns)
- `pipeline_summary.json` — row counts at each stage, price stats, missing value counts per column

Raw 184,552 records → 136,581 after cleaning (filter to residential land use + outlier removal + sqft bounds + dedup on Parcel ID). Of those, 100% match Census demographics by ZIP and ~97% match Zillow ZHVI.

See the main project [README.md](../README.md) for full results and methodology.
