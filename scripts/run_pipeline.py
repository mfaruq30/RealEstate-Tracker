"""Run the full data processing pipeline for the Boston Real Estate Value Analyzer.

Usage:
    python scripts/run_pipeline.py [--raw-dir data/raw] [--output-dir data/processed]

Steps:
    1. Load + clean Boston Property Assessment FY2026
    2. Add engineered features
    3. Enrich with external data (transit, crime, Census, Zillow)
    4. Save enriched dataset + summary statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from real_estate_tracker.data_processing import (
    add_assessment_features,
    clean_assessment_data,
    load_assessment_data,
)
from real_estate_tracker.feature_enrichment import run_enrichment_pipeline


# Boston-area ZIP codes for filtering Zillow/Redfin data
BOSTON_AREA_ZIPS = {
    "02101", "02102", "02103", "02108", "02109", "02110", "02111", "02113",
    "02114", "02115", "02116", "02118", "02119", "02120", "02121", "02122",
    "02124", "02125", "02126", "02127", "02128", "02129", "02130", "02131",
    "02132", "02134", "02135", "02136",
    "02138", "02139", "02140", "02141", "02142",  # Cambridge
    "02143", "02144", "02145",  # Somerville
    "02445", "02446", "02467",  # Brookline
    "02458", "02459", "02460", "02461", "02462", "02464", "02465", "02466",  # Newton
    "02169", "02170", "02171",  # Quincy
    "02452", "02453", "02472", "02478",  # Watertown/Waltham/Belmont
    "02149", "02150", "02151",  # Chelsea/Revere
    "02148", "02155",  # Medford/Malden
    "02420", "02421", "02474", "02476",  # Arlington/Lexington
    "02026", "02186", "02492", "02494",  # Dedham/Milton/Needham
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline.")
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "data" / "raw"),
        help="Directory containing raw downloaded data",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed"),
        help="Directory for processed output files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assessment_path = raw_dir / "boston_property_assessment_fy2026.csv"
    if not assessment_path.exists():
        print(f"ERROR: Assessment data not found at {assessment_path}")
        print("Run `python scripts/download_datasets.py` first.")
        sys.exit(1)

    # --- Step 1: Load + Clean ---
    print("=" * 60)
    print("STEP 1: Loading and cleaning assessment data")
    print("=" * 60)
    raw = load_assessment_data(str(assessment_path))
    print(f"  Raw records loaded: {len(raw):,}")

    clean = clean_assessment_data(raw)
    print(f"  After cleaning: {len(clean):,} residential properties")

    # Save clean (pre-enrichment) version
    clean_path = output_dir / "boston_properties_clean.csv"
    clean.to_csv(clean_path, index=False)
    print(f"  Saved: {clean_path.name}")

    # --- Step 2: Feature Engineering ---
    print("\n" + "=" * 60)
    print("STEP 2: Adding engineered features")
    print("=" * 60)
    featured = add_assessment_features(clean)
    new_cols = set(featured.columns) - set(clean.columns)
    print(f"  New features: {sorted(new_cols)}")

    # --- Step 3: Enrichment ---
    print("\n" + "=" * 60)
    print("STEP 3: Enriching with external data")
    print("=" * 60)
    enriched = run_enrichment_pipeline(
        featured,
        data_dir=str(raw_dir),
        boston_zips=BOSTON_AREA_ZIPS,
    )

    # --- Step 4: Save ---
    print("\n" + "=" * 60)
    print("STEP 4: Saving enriched dataset")
    print("=" * 60)
    enriched_path = output_dir / "boston_properties_enriched.csv"
    enriched.to_csv(enriched_path, index=False)
    print(f"  Saved: {enriched_path.name} ({len(enriched):,} rows x {len(enriched.columns)} cols)")

    # --- Summary Statistics ---
    summary = {
        "raw_records": int(raw.shape[0]),
        "clean_records": int(clean.shape[0]),
        "enriched_records": int(enriched.shape[0]),
        "total_columns": int(enriched.shape[1]),
        "columns": list(enriched.columns),
        "price_stats": {
            "min": float(enriched["price"].min()),
            "median": float(enriched["price"].median()),
            "mean": float(enriched["price"].mean()),
            "max": float(enriched["price"].max()),
        },
        "zip_codes_covered": int(enriched["zip_code"].nunique()),
        "missing_values": {
            col: int(enriched[col].isna().sum())
            for col in enriched.columns
            if enriched[col].isna().sum() > 0
        },
    }

    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path.name}")

    # Print key stats
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total properties: {enriched.shape[0]:,}")
    print(f"  Total features:   {enriched.shape[1]}")
    print(f"  ZIP codes:        {enriched['zip_code'].nunique()}")
    print(f"  Price range:      ${enriched['price'].min():,.0f} - ${enriched['price'].max():,.0f}")
    print(f"  Median price:     ${enriched['price'].median():,.0f}")

    if "distance_to_nearest_transit_mi" in enriched.columns:
        print(f"  Median transit dist: {enriched['distance_to_nearest_transit_mi'].median():.2f} mi")
    if "crime_density" in enriched.columns:
        print(f"  Median crime density: {enriched['crime_density'].median():.0f} incidents")
    if "median_household_income" in enriched.columns:
        matched = enriched["median_household_income"].notna().sum()
        print(f"  Census match rate:   {matched}/{len(enriched)} ({matched/len(enriched)*100:.0f}%)")


if __name__ == "__main__":
    main()
