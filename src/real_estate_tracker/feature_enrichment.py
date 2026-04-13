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


def _haversine_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    """Vectorized haversine for broadcasting one point against many."""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Transit proximity
# ---------------------------------------------------------------------------

def load_mbta_stops(stops_path: str) -> pd.DataFrame:
    """Load MBTA stops.txt from the GTFS feed.

    Returns DataFrame with columns: stop_id, stop_name, stop_lat, stop_lon.
    """
    df = pd.read_csv(stops_path)
    df = df.rename(columns={"stop_lat": "stop_lat", "stop_lon": "stop_lon"})
    df = df.dropna(subset=["stop_lat", "stop_lon"])
    return df[["stop_id", "stop_name", "stop_lat", "stop_lon"]]


def add_transit_proximity(
    properties: pd.DataFrame,
    stops: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Add distance to nearest MBTA stop for each property.

    Adds columns:
        - distance_to_nearest_transit_mi: float
        - nearest_transit_stop: str (stop name)
        - transit_proximity_band: categorical (<0.25mi, 0.25-0.5, 0.5-1, >1)
    """
    out = properties.copy()

    if lat_col not in out.columns or lon_col not in out.columns:
        print(f"  WARNING: {lat_col}/{lon_col} not found, skipping transit proximity.")
        return out

    valid_mask = out[lat_col].notna() & out[lon_col].notna()
    distances = np.full(len(out), np.nan)
    nearest_names = np.full(len(out), "", dtype=object)

    stop_lats = stops["stop_lat"].values
    stop_lons = stops["stop_lon"].values
    stop_names = stops["stop_name"].values

    for i, (idx, row) in enumerate(out[valid_mask].iterrows()):
        prop_lat, prop_lon = row[lat_col], row[lon_col]
        dists = _haversine_vectorized(stop_lats, stop_lons, prop_lat, prop_lon)
        min_idx = np.argmin(dists)
        distances[out.index.get_loc(idx)] = dists[min_idx]
        nearest_names[out.index.get_loc(idx)] = stop_names[min_idx]

        if (i + 1) % 5000 == 0:
            print(f"  Transit proximity: {i+1}/{valid_mask.sum()} properties processed")

    out["distance_to_nearest_transit_mi"] = distances
    out["nearest_transit_stop"] = nearest_names

    out["transit_proximity_band"] = pd.cut(
        out["distance_to_nearest_transit_mi"],
        bins=[0, 0.25, 0.5, 1.0, float("inf")],
        labels=["<0.25mi", "0.25-0.5mi", "0.5-1mi", ">1mi"],
    )

    return out


# ---------------------------------------------------------------------------
# Crime density
# ---------------------------------------------------------------------------

def load_crime_data(crime_path: str) -> pd.DataFrame:
    """Load BPD crime incident CSV. Handles common column name variations."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(crime_path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {crime_path}")

    # Standardize lat/lon column names (BPD data uses different names across years)
    lat_candidates = ["Lat", "LATITUDE", "latitude", "lat"]
    lon_candidates = ["Long", "LONGITUDE", "longitude", "lon", "Lng"]
    for c in lat_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "crime_lat"})
            break
    for c in lon_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "crime_lon"})
            break

    df["crime_lat"] = pd.to_numeric(df.get("crime_lat"), errors="coerce")
    df["crime_lon"] = pd.to_numeric(df.get("crime_lon"), errors="coerce")
    df = df.dropna(subset=["crime_lat", "crime_lon"])
    # Remove invalid coordinates (0,0 or clearly wrong)
    df = df[(df["crime_lat"] > 40) & (df["crime_lat"] < 43)]
    df = df[(df["crime_lon"] < -70) & (df["crime_lon"] > -72)]
    return df


def add_crime_density(
    properties: pd.DataFrame,
    crime: pd.DataFrame,
    radius_mi: float = 0.25,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Count crime incidents within a radius of each property.

    Adds column: crime_density (count of incidents within radius_mi).
    """
    out = properties.copy()

    if lat_col not in out.columns or lon_col not in out.columns:
        print(f"  WARNING: {lat_col}/{lon_col} not found, skipping crime density.")
        return out

    crime_lats = crime["crime_lat"].values
    crime_lons = crime["crime_lon"].values

    valid_mask = out[lat_col].notna() & out[lon_col].notna()
    densities = np.zeros(len(out), dtype=int)

    for i, (idx, row) in enumerate(out[valid_mask].iterrows()):
        dists = _haversine_vectorized(crime_lats, crime_lons, row[lat_col], row[lon_col])
        densities[out.index.get_loc(idx)] = int(np.sum(dists <= radius_mi))

        if (i + 1) % 5000 == 0:
            print(f"  Crime density: {i+1}/{valid_mask.sum()} properties processed")

    out["crime_density"] = densities
    return out


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
# Parcels join (lat/long from GeoJSON or CSV)
# ---------------------------------------------------------------------------

def load_parcels_csv(parcels_path: str) -> pd.DataFrame:
    """Load Boston Parcels 2025 CSV with lat/long coordinates."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(parcels_path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {parcels_path}")

    # Try to find lat/long columns (names vary across releases)
    lat_candidates = ["Latitude", "latitude", "LAT", "Y", "y", "POINT_Y"]
    lon_candidates = ["Longitude", "longitude", "LON", "LONG", "X", "x", "POINT_X"]

    for c in lat_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "latitude"})
            break
    for c in lon_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "longitude"})
            break

    # Try to find a join key (GIS_ID, MAP_PAR_ID, etc.)
    join_candidates = ["GIS_ID", "gis_id", "MAP_PAR_ID", "PID_LONG", "PID"]
    join_col = None
    for c in join_candidates:
        if c in df.columns:
            join_col = c
            break

    cols = ["latitude", "longitude"]
    if join_col:
        cols.insert(0, join_col)

    available = [c for c in cols if c in df.columns]
    return df[available].dropna()


def merge_parcels_coords(
    properties: pd.DataFrame,
    parcels: pd.DataFrame,
) -> pd.DataFrame:
    """Merge parcel lat/long onto properties by GIS_ID or similar key.

    If no common join key exists, returns properties unchanged.
    """
    out = properties.copy()

    # Find common join column
    common_cols = set(out.columns) & set(parcels.columns) - {"latitude", "longitude"}
    if not common_cols:
        print("  WARNING: No common join key between properties and parcels.")
        return out

    join_col = common_cols.pop()
    out[join_col] = out[join_col].astype(str).str.strip()
    parcels = parcels.copy()
    parcels[join_col] = parcels[join_col].astype(str).str.strip()

    # Drop lat/lon from properties if they already exist (parcels are more accurate)
    for col in ["latitude", "longitude"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    out = out.merge(parcels, on=join_col, how="left")
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
        - boston_parcels_2025.csv (optional — for lat/long)
        - stops.txt (MBTA GTFS)
        - bpd_crime_incidents_2024.csv
        - census_acs_boston_zips.csv
        - zillow_zhvi_by_zip.csv

    Returns the enriched DataFrame.
    """
    data_dir = Path(data_dir)
    enriched = properties.copy()

    # --- Parcels (lat/long) ---
    parcels_path = data_dir / "boston_parcels_2025.csv"
    if parcels_path.exists():
        print("[Enrichment] Merging parcel coordinates...")
        parcels = load_parcels_csv(str(parcels_path))
        enriched = merge_parcels_coords(enriched, parcels)
        if "latitude" in enriched.columns:
            print(f"  Properties with lat/long: {enriched['latitude'].notna().sum()}/{len(enriched)}")
        else:
            print("  WARNING: No lat/long columns after parcels merge")
    else:
        print(f"[Enrichment] SKIP parcels — {parcels_path.name} not found")

    # --- Transit proximity ---
    stops_path = data_dir / "stops.txt"
    if stops_path.exists() and "latitude" in enriched.columns:
        print("[Enrichment] Computing transit proximity...")
        stops = load_mbta_stops(str(stops_path))
        enriched = add_transit_proximity(enriched, stops)
        med_dist = enriched["distance_to_nearest_transit_mi"].median()
        print(f"  Median distance to transit: {med_dist:.2f} mi")
    else:
        print(f"[Enrichment] SKIP transit — stops.txt or lat/long not available")

    # --- Crime density ---
    crime_path = data_dir / "bpd_crime_incidents_2024.csv"
    if crime_path.exists() and "latitude" in enriched.columns:
        print("[Enrichment] Computing crime density...")
        crime = load_crime_data(str(crime_path))
        enriched = add_crime_density(enriched, crime)
        print(f"  Median crime density (0.25mi): {enriched['crime_density'].median()}")
    else:
        print(f"[Enrichment] SKIP crime — crime CSV or lat/long not available")

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
