import pandas as pd

from real_estate_tracker.data_processing import add_features, clean_data, contains_any_term


def test_contains_any_term_matches_phrase() -> None:
    text = "This unit is recently renovated and move-in ready."
    assert contains_any_term(text.lower(), ["renovated"]) is True
    assert contains_any_term(text.lower(), ["fixer"]) is False


def test_clean_data_filters_invalid_rows() -> None:
    df = pd.DataFrame(
        {
            "price": [600000, 0],
            "sqft": [1200, 100],
            "bedrooms": [3, 2],
            "bathrooms": [2, 1],
            "year_built": [2000, 1990],
            "zip_code": ["02118", "02118"],
            "description": ["updated kitchen", "needs work"],
        }
    )
    cleaned = clean_data(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["price"] == 600000


def test_add_features_creates_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "price": [700000],
            "sqft": [1400],
            "bedrooms": [3],
            "bathrooms": [2],
            "year_built": [2005],
            "zip_code": ["02118"],
            "description": ["Modern renovated unit."],
        }
    )
    featured = add_features(df)
    for col in ["sqft_per_room", "is_renovated_signal", "needs_work_signal", "home_age", "price_per_sqft"]:
        assert col in featured.columns
    assert int(featured.iloc[0]["is_renovated_signal"]) == 1

