"""Enrich property data with external sources.

Merges transit proximity, crime density, Census demographics, and market
trends onto the cleaned property assessment dataset.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Haversine distance (miles)
# ---------------------------------------------------------------------------

_EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in miles between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MI * math.asin(math.sqrt(a))




# ---------------------------------------------------------------------------
# Census demographics merge
# ---------------------------------------------------------------------------

def load_census_data(census_path: str) -> pd.DataFrame:
    """Load the Census ACS CSV fetched by fetch_api_data.py."""
    df = pd.read_csv(census_path, dtype={"zcta": str})

    numeric_cols = [
        "median_home_value", "median_household_income",
        "total_population", "total_housing_units",
        "owner_occupied_units", "renter_occupied_units",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_census_demographics(
    properties: pd.DataFrame,
    census: pd.DataFrame,
    zip_col: str = "zip_code",
) -> pd.DataFrame:
    """Merge Census ACS demographics onto properties by ZIP code.

    Adds: median_household_income, total_population, pct_owner_occupied,
          census_median_home_value, price_vs_census_ratio.
    """
    out = properties.copy()
    out[zip_col] = out[zip_col].astype(str).str.strip().str[:5]
    census = census.copy()
    census["zcta"] = census["zcta"].astype(str).str.strip().str[:5]

    # Compute derived columns before merging
    census["pct_owner_occupied"] = (
        census["owner_occupied_units"] / census["total_housing_units"].clip(lower=1) * 100
    )

    merge_cols = [
        "zcta", "median_household_income", "total_population",
        "pct_owner_occupied", "median_home_value",
    ]
    census_subset = census[[c for c in merge_cols if c in census.columns]].copy()
    census_subset = census_subset.rename(columns={
        "zcta": zip_col,
        "median_home_value": "census_median_home_value",
    })

    out = out.merge(census_subset, on=zip_col, how="left")

    if "census_median_home_value" in out.columns:
        out["price_vs_census_ratio"] = (
            out["price"] / out["census_median_home_value"].clip(lower=1)
        )

    return out


# ---------------------------------------------------------------------------
# Zillow market trends merge
# ---------------------------------------------------------------------------

def load_zillow_zhvi(zhvi_path: str, boston_zips: set[str] | None = None) -> pd.DataFrame:
    """Load Zillow ZHVI by ZIP and filter to Boston-area ZIPs.

    Returns a DataFrame with columns: zip_code, zhvi_latest, zhvi_1yr_ago.
    """
    df = pd.read_csv(zhvi_path)
    # RegionName may be int or str — normalize to zero-padded 5-digit string
    df["RegionName"] = (
        pd.to_numeric(df["RegionName"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    if boston_zips:
        df = df[df["RegionName"].isin(boston_zips)]

    # Date columns are the monthly values; take the last two for latest + 1yr ago
    date_cols = [c for c in df.columns if c.startswith("20")]
    if len(date_cols) < 2:
        return pd.DataFrame(columns=["zip_code", "zhvi_latest"])

    latest_col = date_cols[-1]
    year_ago_col = date_cols[-13] if len(date_cols) >= 13 else date_cols[0]

    result = pd.DataFrame({
        "zip_code": df["RegionName"],
        "zhvi_latest": pd.to_numeric(df[latest_col], errors="coerce"),
        "zhvi_1yr_ago": pd.to_numeric(df[year_ago_col], errors="coerce"),
    })
    result["zhvi_yoy_change_pct"] = (
        (result["zhvi_latest"] - result["zhvi_1yr_ago"]) / result["zhvi_1yr_ago"].clip(lower=1) * 100
    )
    return result.dropna(subset=["zhvi_latest"])


def merge_zillow_trends(
    properties: pd.DataFrame,
    zhvi: pd.DataFrame,
    zip_col: str = "zip_code",
) -> pd.DataFrame:
    """Merge Zillow ZHVI onto properties by ZIP.

    Adds: zhvi_latest, zhvi_yoy_change_pct, price_vs_zhvi_ratio.
    """
    out = properties.copy()
    out[zip_col] = out[zip_col].astype(str).str.strip().str[:5]

    out = out.merge(zhvi, on=zip_col, how="left")

    if "zhvi_latest" in out.columns:
        out["price_vs_zhvi_ratio"] = out["price"] / out["zhvi_latest"].clip(lower=1)

    return out


# ---------------------------------------------------------------------------
# Full enrichment pipeline
# ---------------------------------------------------------------------------

def run_enrichment_pipeline(
    properties: pd.DataFrame,
    data_dir: str | Path,
    boston_zips: set[str] | None = None,
) -> pd.DataFrame:
    """Run the full enrichment pipeline on a cleaned property DataFrame.

    Expects data files in data_dir (typically data/raw/):
        - census_acs_boston_zips.csv
        - zillow_zhvi_by_zip.csv

    Returns the enriched DataFrame.
    """
    data_dir = Path(data_dir)
    enriched = properties.copy()

    # --- Census demographics ---
    census_path = data_dir / "census_acs_boston_zips.csv"
    if census_path.exists():
        print("[Enrichment] Merging Census demographics...")
        census = load_census_data(str(census_path))
        enriched = merge_census_demographics(enriched, census)
        matched = enriched["median_household_income"].notna().sum()
        print(f"  Properties with Census data: {matched}/{len(enriched)}")
    else:
        print(f"[Enrichment] SKIP Census — {census_path.name} not found")

    # --- Zillow market trends ---
    zhvi_path = data_dir / "zillow_zhvi_by_zip.csv"
    if zhvi_path.exists():
        print("[Enrichment] Merging Zillow ZHVI trends...")
        zhvi = load_zillow_zhvi(str(zhvi_path), boston_zips=boston_zips)
        enriched = merge_zillow_trends(enriched, zhvi)
        matched = enriched["zhvi_latest"].notna().sum()
        print(f"  Properties with ZHVI data: {matched}/{len(enriched)}")
    else:
        print(f"[Enrichment] SKIP Zillow — {zhvi_path.name} not found")

    return enriched