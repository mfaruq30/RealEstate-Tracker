from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "price",
    "sqft",
    "bedrooms",
    "bathrooms",
    "year_built",
    "zip_code",
    "description",
]


POSITIVE_TERMS = ("renovated", "updated", "modern", "new", "move-in ready")
NEGATIVE_TERMS = ("fixer", "as-is", "needs work", "investor special", "tlc")


def load_data(path: str) -> pd.DataFrame:
    """Load listing data from a CSV file."""
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Raise if required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning used for checkpoint 1."""
    validate_columns(df)
    out = df.copy()

    numeric_cols = ["price", "sqft", "bedrooms", "bathrooms", "year_built"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["description"] = out["description"].fillna("")
    out["zip_code"] = out["zip_code"].astype(str).str.strip()

    # Remove impossible values and obvious outliers for an initial pass.
    out = out[(out["price"] > 50000) & (out["price"] < 5000000)]
    out = out[(out["sqft"] > 200) & (out["sqft"] < 10000)]
    out = out.dropna(subset=["price", "sqft", "bedrooms", "bathrooms"])

    # Lightweight fill for year built.
    out["year_built"] = out["year_built"].fillna(out["year_built"].median())

    out = out.drop_duplicates(subset=["price", "sqft", "bedrooms", "bathrooms", "zip_code"])
    return out.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add checkpoint-level engineered features."""
    out = df.copy()

    # Layout efficiency proxy.
    total_rooms_proxy = out["bedrooms"] + out["bathrooms"]
    out["sqft_per_room"] = out["sqft"] / total_rooms_proxy.clip(lower=1)

    desc_lower = out["description"].str.lower()
    out["is_renovated_signal"] = desc_lower.apply(lambda s: contains_any_term(s, POSITIVE_TERMS)).astype(int)
    out["needs_work_signal"] = desc_lower.apply(lambda s: contains_any_term(s, NEGATIVE_TERMS)).astype(int)

    out["home_age"] = 2026 - out["year_built"]
    out["price_per_sqft"] = out["price"] / out["sqft"]
    return out


def contains_any_term(text: str, terms: Iterable[str]) -> bool:
    """Return True if any term appears as a phrase in text."""
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def select_model_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare model matrix and target for training."""
    feature_cols = [
        "sqft",
        "bedrooms",
        "bathrooms",
        "year_built",
        "sqft_per_room",
        "is_renovated_signal",
        "needs_work_signal",
        "home_age",
    ]
    x = df[feature_cols]
    y = df["price"]
    return x, y

