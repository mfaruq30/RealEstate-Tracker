from __future__ import annotations

import re
from typing import Iterable

import numpy as np
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

# Columns required in the Boston Property Assessment FY2026 CSV.
# Column names vary by fiscal year — we check for both variants.
ASSESSMENT_REQUIRED = [
    "TOTAL_VALUE",
    "LAND_SF",
    "LIVING_AREA",
    "YR_BUILT",
    "LU",  # Land use code
]

# Residential land-use codes in the Boston Assessment data.
RESIDENTIAL_LU_CODES = {"R1", "R2", "R3", "R4", "CD", "A"}

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


# ---------------------------------------------------------------------------
# Sample data cleaning (checkpoint 1 format)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Boston Assessment data cleaning (real data pipeline)
# ---------------------------------------------------------------------------

def load_assessment_data(path: str) -> pd.DataFrame:
    """Load the Boston Property Assessment CSV.

    Handles encoding quirks common in city-exported CSVs.
    """
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with any attempted encoding")


def clean_assessment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Boston Property Assessment FY2026 dataset.

    Steps:
        1. Validate required columns exist
        2. Filter to residential properties only (LU codes R1-R4, CD, A)
        3. Rename columns to project-standard snake_case
        4. Coerce numeric types
        5. Remove outliers (IQR on total_value)
        6. Handle missing values
        7. Deduplicate on PID (parcel ID)

    Returns a cleaned DataFrame with standardized column names.
    """
    missing = [c for c in ASSESSMENT_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required assessment columns: {missing}")

    out = df.copy()

    # Strip whitespace from column names (city CSVs often have trailing spaces)
    out.columns = out.columns.str.strip()

    # --- Step 2: Filter to residential properties ---
    out["LU"] = out["LU"].astype(str).str.strip().str.upper()
    out = out[out["LU"].isin(RESIDENTIAL_LU_CODES)]

    # --- Step 3: Rename to standard names ---
    # Handles both FY2024-style (R_BDRMS) and FY2026-style (BED_RMS) column names.
    rename_map = {
        "PID": "pid",
        "CM_ID": "cm_id",
        "GIS_ID": "gis_id",
        "ST_NUM": "street_num",
        "ST_NAME": "street_name",
        "ST_NAME_SUF": "street_suffix",
        "UNIT_NUM": "unit_num",
        "ZIPCODE": "zip_code",
        "ZIP_CODE": "zip_code",
        "LU": "land_use",
        "LU_DESC": "land_use_desc",
        "BLDG_TYPE": "building_type",
        "OWN_OCC": "owner_occupied",
        "TOTAL_VALUE": "price",
        "LAND_VALUE": "land_value",
        "BLDG_VALUE": "building_value",
        "LAND_SF": "lot_sqft",
        "LIVING_AREA": "sqft",
        # Bedroom variants
        "R_BDRMS": "bedrooms",
        "BED_RMS": "bedrooms",
        # Bathroom variants
        "R_FULL_BTH": "full_bathrooms",
        "FULL_BTH": "full_bathrooms",
        "R_HALF_BTH": "half_bathrooms",
        "HLF_BTH": "half_bathrooms",
        # Other room/feature variants
        "R_KITCH": "kitchens",
        "KITCHENS": "kitchens",
        "R_FPLACE": "fireplaces",
        "FIREPLACES": "fireplaces",
        "R_TOTAL_RMS": "total_rooms",
        "TT_RMS": "total_rooms",
        # System variants
        "R_AC": "air_conditioning",
        "AC_TYPE": "air_conditioning",
        "R_HEAT_TYP": "heating_type",
        "HEAT_TYPE": "heating_type",
        # Condition variants
        "R_EXT_CND": "exterior_condition",
        "EXT_COND": "exterior_condition",
        "R_OVRALL_CND": "overall_condition",
        "OVERALL_COND": "overall_condition",
        "R_INT_CND": "interior_condition",
        "INT_COND": "interior_condition",
        # Other
        "R_VIEW": "view_rating",
        "PROP_VIEW": "view_rating",
        "NUM_FLOORS": "num_floors",
        "RES_FLOOR": "num_floors",
        "STRUCTURE_CLASS": "structure_class",
        "YR_BUILT": "year_built",
        "YR_REMOD": "year_remodeled",
        "YR_REMODEL": "year_remodeled",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    # --- Step 4: Numeric coercion ---
    # City CSVs often use comma-separated numbers (e.g., "822,900")
    numeric_cols = [
        "price", "land_value", "building_value", "lot_sqft", "sqft",
        "bedrooms", "full_bathrooms", "half_bathrooms", "total_rooms",
        "num_floors", "year_built", "year_remodeled", "fireplaces", "kitchens",
    ]
    for col in numeric_cols:
        if col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype(str).str.replace(",", "", regex=False)
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Combine full + half baths into a single bathrooms column
    out["bathrooms"] = out.get("full_bathrooms", 0).fillna(0) + out.get("half_bathrooms", 0).fillna(0) * 0.5

    # ZIP codes may be stored as floats (e.g., 2127.0) — convert to zero-padded strings
    out["zip_code"] = (
        pd.to_numeric(out["zip_code"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    # --- Step 5: Remove outliers ---
    # Drop rows with missing or zero price/sqft
    out = out.dropna(subset=["price", "sqft"])
    out = out[(out["price"] > 0) & (out["sqft"] > 0)]

    # IQR-based outlier removal on price
    q1 = out["price"].quantile(0.01)
    q3 = out["price"].quantile(0.99)
    out = out[(out["price"] >= q1) & (out["price"] <= q3)]

    # Reasonable sqft range
    out = out[(out["sqft"] >= 200) & (out["sqft"] <= 15000)]

    # --- Step 6: Handle missing values ---
    out["year_built"] = out["year_built"].fillna(out["year_built"].median())
    if "year_remodeled" in out.columns:
        out["year_remodeled"] = out["year_remodeled"].fillna(0).astype(int)
    out["bedrooms"] = out["bedrooms"].fillna(0).astype(int)
    out["bathrooms"] = out["bathrooms"].fillna(0)

    # Fill condition codes with mode
    for cond_col in ["exterior_condition", "overall_condition", "interior_condition"]:
        if cond_col in out.columns:
            mode_val = out[cond_col].mode()
            if len(mode_val) > 0:
                out[cond_col] = out[cond_col].fillna(mode_val.iloc[0])

    # --- Step 7: Deduplicate ---
    if "pid" in out.columns:
        out = out.drop_duplicates(subset=["pid"], keep="first")
    else:
        out = out.drop_duplicates(subset=["price", "sqft", "zip_code"], keep="first")

    # Add a description placeholder (assessment data doesn't have descriptions)
    if "description" not in out.columns:
        out["description"] = ""

    return out.reset_index(drop=True)


def add_assessment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to cleaned assessment data.

    Features:
        - price_per_sqft: Price normalized by living area
        - sqft_per_room: Layout efficiency proxy
        - home_age: Years since construction
        - renovation_gap: Years between build and remodel (0 if never remodeled)
        - bath_to_bed_ratio: Bathrooms per bedroom
        - lot_to_living_ratio: Lot size vs living area
        - is_remodeled: Binary flag for whether property was remodeled
    """
    out = df.copy()

    out["price_per_sqft"] = out["price"] / out["sqft"].clip(lower=1)
    out["home_age"] = 2026 - out["year_built"]

    total_rooms = out.get("total_rooms", out["bedrooms"] + out["bathrooms"])
    out["sqft_per_room"] = out["sqft"] / total_rooms.clip(lower=1)

    out["bath_to_bed_ratio"] = out["bathrooms"] / out["bedrooms"].clip(lower=1)

    if "lot_sqft" in out.columns:
        out["lot_to_living_ratio"] = out["lot_sqft"] / out["sqft"].clip(lower=1)

    if "year_remodeled" in out.columns:
        out["is_remodeled"] = (out["year_remodeled"] > 0).astype(int)
        out["renovation_gap"] = np.where(
            out["year_remodeled"] > 0,
            out["year_remodeled"] - out["year_built"],
            0,
        )

    return out


# ---------------------------------------------------------------------------
# Original checkpoint 1 features (kept for backward compatibility)
# ---------------------------------------------------------------------------

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


def select_assessment_model_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare model matrix from assessment data.

    Uses all available numeric features, filtering to columns that exist.
    """
    candidate_cols = [
        "sqft", "lot_sqft", "bedrooms", "bathrooms", "total_rooms",
        "num_floors", "year_built", "fireplaces",
        "sqft_per_room", "home_age",
        "bath_to_bed_ratio", "lot_to_living_ratio",
        "is_remodeled", "renovation_gap",
        # Enrichment columns (added by feature_enrichment.py)
        "distance_to_nearest_transit_mi",
        "crime_density",
        "median_household_income",
        "total_population",
    ]
    available = [c for c in candidate_cols if c in df.columns]
    x = df[available].fillna(0)
    y = df["price"]
    return x, y

