"""Tests for the feature enrichment module."""

import numpy as np
import pandas as pd
import pytest

from real_estate_tracker.feature_enrichment import (
    add_crime_density,
    add_transit_proximity,
    haversine_miles,
    load_zillow_zhvi,
    merge_census_demographics,
    merge_zillow_trends,
)


class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_miles(42.36, -71.06, 42.36, -71.06) == pytest.approx(0.0)

    def test_boston_to_cambridge_approx(self):
        # Boston City Hall to Harvard Square is roughly 3 miles
        dist = haversine_miles(42.3601, -71.0589, 42.3736, -71.1190)
        assert 2.5 < dist < 4.0

    def test_symmetry(self):
        d1 = haversine_miles(42.36, -71.06, 42.37, -71.12)
        d2 = haversine_miles(42.37, -71.12, 42.36, -71.06)
        assert d1 == pytest.approx(d2)


class TestTransitProximity:
    def _make_properties(self):
        return pd.DataFrame({
            "pid": ["A", "B", "C"],
            "latitude": [42.3601, 42.3736, np.nan],
            "longitude": [-71.0589, -71.1190, np.nan],
            "price": [500000, 600000, 700000],
        })

    def _make_stops(self):
        return pd.DataFrame({
            "stop_id": ["S1", "S2"],
            "stop_name": ["Park Street", "Harvard"],
            "stop_lat": [42.3564, 42.3736],
            "stop_lon": [-71.0624, -71.1190],
        })

    def test_adds_distance_column(self):
        props = self._make_properties()
        stops = self._make_stops()
        result = add_transit_proximity(props, stops)
        assert "distance_to_nearest_transit_mi" in result.columns
        assert "nearest_transit_stop" in result.columns
        assert "transit_proximity_band" in result.columns

    def test_nan_coordinates_handled(self):
        props = self._make_properties()
        stops = self._make_stops()
        result = add_transit_proximity(props, stops)
        assert np.isnan(result.iloc[2]["distance_to_nearest_transit_mi"])

    def test_nearest_stop_correct(self):
        props = self._make_properties()
        stops = self._make_stops()
        result = add_transit_proximity(props, stops)
        # Property B is at Harvard Square coords, so Harvard should be nearest
        assert result.iloc[1]["nearest_transit_stop"] == "Harvard"


class TestCrimeDensity:
    def test_counts_nearby_crimes(self):
        props = pd.DataFrame({
            "latitude": [42.36],
            "longitude": [-71.06],
            "price": [500000],
        })
        crimes = pd.DataFrame({
            "crime_lat": [42.3601, 42.3602, 42.50],  # 2 near, 1 far
            "crime_lon": [-71.0589, -71.0590, -71.20],
        })
        result = add_crime_density(props, crimes, radius_mi=0.5)
        assert result.iloc[0]["crime_density"] == 2


class TestCensusMerge:
    def test_merges_by_zip(self):
        props = pd.DataFrame({
            "zip_code": ["02118", "02119", "99999"],
            "price": [500000, 600000, 700000],
        })
        census = pd.DataFrame({
            "zcta": ["02118", "02119"],
            "median_household_income": [75000, 55000],
            "total_population": [30000, 25000],
            "total_housing_units": [15000, 12000],
            "owner_occupied_units": [5000, 3000],
            "median_home_value": [600000, 400000],
        })
        result = merge_census_demographics(props, census)
        assert result.iloc[0]["median_household_income"] == 75000
        assert np.isnan(result.iloc[2]["median_household_income"])

    def test_adds_derived_columns(self):
        props = pd.DataFrame({
            "zip_code": ["02118"],
            "price": [500000],
        })
        census = pd.DataFrame({
            "zcta": ["02118"],
            "median_household_income": [75000],
            "total_population": [30000],
            "total_housing_units": [15000],
            "owner_occupied_units": [5000],
            "median_home_value": [600000],
        })
        result = merge_census_demographics(props, census)
        assert "pct_owner_occupied" in result.columns
        assert "price_vs_census_ratio" in result.columns


class TestZillowMerge:
    def test_merges_zhvi(self):
        props = pd.DataFrame({
            "zip_code": ["02118", "02119"],
            "price": [500000, 600000],
        })
        zhvi = pd.DataFrame({
            "zip_code": ["02118", "02119"],
            "zhvi_latest": [550000, 450000],
            "zhvi_1yr_ago": [500000, 420000],
            "zhvi_yoy_change_pct": [10.0, 7.1],
        })
        result = merge_zillow_trends(props, zhvi)
        assert "zhvi_latest" in result.columns
        assert "price_vs_zhvi_ratio" in result.columns
        assert result.iloc[0]["price_vs_zhvi_ratio"] == pytest.approx(500000 / 550000, abs=0.01)
