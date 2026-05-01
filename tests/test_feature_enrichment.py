"""Tests for the feature enrichment module."""

import numpy as np
import pandas as pd
import pytest

from real_estate_tracker.feature_enrichment import (
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
