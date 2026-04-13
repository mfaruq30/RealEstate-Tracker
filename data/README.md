# Data Directory — Real Estate Value Analyzer

## Directory Structure

```
data/
├── raw/                          # Untouched downloads (git-ignored)
│   ├── boston_property_assessment_fy2026.csv
│   ├── boston_parcels_2025.csv
│   ├── zillow_zhvi_by_zip.csv
│   ├── redfin_zip_market_tracker.tsv
│   ├── stops.txt (MBTA GTFS)
│   ├── bpd_crime_incidents_2024.csv
│   ├── census_acs_boston_zips.csv
│   └── fred_mortgage_rates.csv
├── processed/                    # Cleaned + enriched (git-ignored)
│   ├── boston_properties_clean.csv
│   ├── boston_properties_enriched.csv
│   └── pipeline_summary.json
├── sample_listings.csv           # 31-row sample (checked into git)
└── README.md                     # This file
```

## How to Get the Data

```bash
# Step 1: Download static datasets
python scripts/download_datasets.py

# Step 2: Fetch API data (Census, FRED)
python scripts/fetch_api_data.py

# Step 3: Run the processing pipeline
python scripts/run_pipeline.py
```

**Optional environment variables** (for API access):
- `CENSUS_API_KEY` — Free from https://api.census.gov/data/key_signup.html
- `FRED_API_KEY` — Free from https://fred.stlouisfed.org/docs/api/fred/

## Data Sources

| Dataset | Source | URL | Update Freq |
|---------|--------|-----|-------------|
| Property Assessment FY2026 | City of Boston (Analyze Boston) | https://data.boston.gov/dataset/property-assessment | Annual |
| Parcels 2025 | City of Boston (Analyze Boston) | https://data.boston.gov/dataset/parcels-2025 | Annual |
| Zillow ZHVI by ZIP | Zillow Research | https://www.zillow.com/research/data/ | Monthly |
| Redfin Market Tracker | Redfin Data Center | https://www.redfin.com/news/data-center/ | Weekly |
| MBTA GTFS | MBTA Developers | https://www.mbta.com/developers | Monthly |
| Crime Incidents | Boston Police Dept | https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system | Daily |
| Census ACS 5-Year | US Census Bureau | https://api.census.gov/ | Annual |
| Mortgage Rates | FRED (St. Louis Fed) | https://fred.stlouisfed.org/ | Weekly |

## Column Dictionary — `boston_properties_enriched.csv`

### Property Characteristics (from Assessment)
| Column | Type | Description |
|--------|------|-------------|
| `pid` | str | Parcel ID (unique identifier) |
| `gis_id` | str | GIS identifier for geographic joins |
| `street_num` | str | Street number |
| `street_name` | str | Street name |
| `zip_code` | str | 5-digit ZIP code |
| `land_use` | str | Land use code (R1=single family, R2=2-family, R3=3-family, R4=4+ family, CD=condo) |
| `price` | float | Total assessed value ($) |
| `land_value` | float | Assessed land value ($) |
| `building_value` | float | Assessed building value ($) |
| `sqft` | float | Living area in square feet |
| `lot_sqft` | float | Lot area in square feet |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms (full + 0.5*half) |
| `total_rooms` | int | Total number of rooms |
| `num_floors` | float | Number of floors |
| `year_built` | int | Year of construction |
| `year_remodeled` | int | Year of last remodel (0 if never) |
| `overall_condition` | str | Overall condition rating |
| `exterior_condition` | str | Exterior condition rating |
| `interior_condition` | str | Interior condition rating |
| `fireplaces` | int | Number of fireplaces |
| `heating_type` | str | Heating system type |

### Engineered Features
| Column | Type | Description |
|--------|------|-------------|
| `price_per_sqft` | float | Price / living area |
| `sqft_per_room` | float | Living area / total rooms |
| `home_age` | int | 2026 - year_built |
| `bath_to_bed_ratio` | float | Bathrooms / bedrooms |
| `lot_to_living_ratio` | float | Lot sqft / living sqft |
| `is_remodeled` | int | 1 if property was remodeled, 0 otherwise |
| `renovation_gap` | int | Years between build and remodel |

### Geographic (from Parcels)
| Column | Type | Description |
|--------|------|-------------|
| `latitude` | float | Property latitude |
| `longitude` | float | Property longitude |

### Transit Proximity (from MBTA GTFS)
| Column | Type | Description |
|--------|------|-------------|
| `distance_to_nearest_transit_mi` | float | Miles to nearest MBTA stop |
| `nearest_transit_stop` | str | Name of nearest stop |
| `transit_proximity_band` | str | Category: <0.25mi, 0.25-0.5mi, 0.5-1mi, >1mi |

### Safety (from BPD Crime Data)
| Column | Type | Description |
|--------|------|-------------|
| `crime_density` | int | Crime incidents within 0.25 miles (2024 data) |

### Demographics (from Census ACS)
| Column | Type | Description |
|--------|------|-------------|
| `median_household_income` | float | Median household income in ZIP ($) |
| `total_population` | int | Total population in ZIP |
| `pct_owner_occupied` | float | % of housing units that are owner-occupied |
| `census_median_home_value` | float | Census-reported median home value in ZIP ($) |
| `price_vs_census_ratio` | float | Property price / Census median (>1 = above average) |

### Market Trends (from Zillow ZHVI)
| Column | Type | Description |
|--------|------|-------------|
| `zhvi_latest` | float | Latest Zillow Home Value Index for ZIP ($) |
| `zhvi_yoy_change_pct` | float | Year-over-year ZHVI change (%) |
| `price_vs_zhvi_ratio` | float | Property price / ZHVI (>1 = above market) |
