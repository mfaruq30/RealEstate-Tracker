"""Tests for Boston Assessment data cleaning and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from real_estate_tracker.data_processing import (
    add_assessment_features,
    clean_assessment_data,
    select_assessment_model_columns,
)


def _make_assessment_df(n: int = 5, **overrides) -> pd.DataFrame:
    """Create a minimal assessment DataFrame for testing."""
    data = {
        "PID": [f"PID{i}" for i in range(n)],
        "GIS_ID": [f"GIS{i}" for i in range(n)],
        "TOTAL_VALUE": [500000 + i * 100000 for i in range(n)],
        "LAND_SF": [3000 + i * 500 for i in range(n)],
        "LIVING_AREA": [1200 + i * 200 for i in range(n)],
        "R_BDRMS": [3] * n,
        "R_FULL_BTH": [2] * n,
        "R_HALF_BTH": [1] * n,
        "R_TOTAL_RMS": [7] * n,
        "NUM_FLOORS": [2] * n,
        "YR_BUILT": [1990 + i * 5 for i in range(n)],
        "YR_REMOD": [0, 2010, 0, 2015, 0],
        "ZIPCODE": ["02118", "02119", "02120", "02121", "02122"],
        "LU": ["R1", "R2", "CD", "R3", "R1"],
        "LU_DESC": ["Single Family"] * n,
        "R_OVRALL_CND": ["A", "A", "G", "G", "F"],
        "R_EXT_CND": ["A", "A", "G", "G", "F"],
        "R_INT_CND": ["A", "A", "G", "G", "F"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestCleanAssessmentData:
    def test_filters_to_residential(self):
        df = _make_assessment_df(LU=["R1", "C", "I", "R2", "E"])
        result = clean_assessment_data(df)
        assert all(lu in {"R1", "R2"} for lu in result["land_use"])

    def test_renames_columns(self):
        df = _make_assessment_df()
        result = clean_assessment_data(df)
        assert "price" in result.columns
        assert "sqft" in result.columns
        assert "zip_code" in result.columns
        assert "TOTAL_VALUE" not in result.columns

    def test_combines_bathrooms(self):
        df = _make_assessment_df(R_FULL_BTH=[2] * 5, R_HALF_BTH=[1] * 5)
        result = clean_assessment_data(df)
        assert (result["bathrooms"] == 2.5).all()

    def test_removes_zero_price(self):
        df = _make_assessment_df(TOTAL_VALUE=[0, 500000, 600000, 700000, 800000])
        result = clean_assessment_data(df)
        assert (result["price"] > 0).all()

    def test_deduplicates_on_pid(self):
        df = _make_assessment_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = clean_assessment_data(df)
        assert result["pid"].nunique() == len(result)

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required"):
            clean_assessment_data(df)

    def test_fills_missing_year_built(self):
        df = _make_assessment_df(YR_BUILT=[1990, None, 2000, None, 2010])
        result = clean_assessment_data(df)
        assert result["year_built"].isna().sum() == 0

    def test_adds_description_placeholder(self):
        df = _make_assessment_df()
        result = clean_assessment_data(df)
        assert "description" in result.columns


class TestAddAssessmentFeatures:
    def test_creates_expected_columns(self):
        df = _make_assessment_df()
        clean = clean_assessment_data(df)
        featured = add_assessment_features(clean)
        expected = [
            "price_per_sqft", "home_age", "sqft_per_room",
            "bath_to_bed_ratio", "lot_to_living_ratio",
            "is_remodeled", "renovation_gap",
        ]
        for col in expected:
            assert col in featured.columns, f"Missing column: {col}"

    def test_price_per_sqft_positive(self):
        df = _make_assessment_df()
        clean = clean_assessment_data(df)
        featured = add_assessment_features(clean)
        assert (featured["price_per_sqft"] > 0).all()

    def test_is_remodeled_binary(self):
        df = _make_assessment_df(YR_REMOD=[0, 2010, 0, 2015, 0])
        clean = clean_assessment_data(df)
        featured = add_assessment_features(clean)
        assert set(featured["is_remodeled"].unique()).issubset({0, 1})

    def test_home_age_reasonable(self):
        df = _make_assessment_df(YR_BUILT=[2000] * 5)
        clean = clean_assessment_data(df)
        featured = add_assessment_features(clean)
        assert (featured["home_age"] == 26).all()


class TestSelectAssessmentModelColumns:
    def test_returns_x_y(self):
        df = _make_assessment_df()
        clean = clean_assessment_data(df)
        featured = add_assessment_features(clean)
        x, y = select_assessment_model_columns(featured)
        assert len(x) == len(y)
        assert "price" not in x.columns
        assert y.name == "price"

    def test_no_missing_in_x(self):
        df = _make_assessment_df()
        clean = clean_assessment_data(df)
        featured = add_assessment_features(clean)
        x, _ = select_assessment_model_columns(featured)
        assert x.isna().sum().sum() == 0
