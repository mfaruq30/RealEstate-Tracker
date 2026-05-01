"""Train models on the enriched Boston real estate dataset (Check-In 2).

Usage:
    python scripts/run_model.py [--input data/processed/boston_properties_enriched.csv]
                                [--output outputs/checkpoint2]

Loads the enriched dataset, trains baseline (Linear Regression) and tree-based
(Random Forest) models, and saves metrics + visualizations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure local package imports work without manual PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from real_estate_tracker.data_processing import select_assessment_model_columns
from real_estate_tracker.modeling import cross_validate_models, save_metrics, train_and_evaluate
from real_estate_tracker.visualization import (
    save_preliminary_figures,
    save_feature_importance_plot,
    save_residual_distribution_plot,
    save_residuals_vs_predicted_plot,
    save_error_by_price_tier_plot,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models on the enriched Boston dataset.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "processed" / "boston_properties_enriched.csv"),
        help="Path to enriched CSV produced by run_pipeline.py",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "checkpoint2"),
        help="Output directory for metrics + figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Enriched dataset not found at {input_path}")
        print("Run `python scripts/run_pipeline.py` first.")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load enriched data ---
    print("=" * 60)
    print("STEP 1: Loading enriched dataset")
    print("=" * 60)
    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df):,} rows x {len(df.columns)} columns")

    # --- Step 2: Select model columns ---
    print("\n" + "=" * 60)
    print("STEP 2: Selecting model features")
    print("=" * 60)
    x, y = select_assessment_model_columns(df)
    print(f"  Features used ({len(x.columns)}): {list(x.columns)}")
    print(f"  Target: price (median = ${y.median():,.0f})")

    # --- Step 3: Train + evaluate ---
    print("\n" + "=" * 60)
    print("STEP 3: Training models (this may take a few minutes on 130k rows)")
    print("=" * 60)
    result = train_and_evaluate(x, y)

    # --- Step 4: Print metrics ---
    print("\n" + "=" * 60)
    print("STEP 4: Model performance")
    print("=" * 60)
    for model_name, metrics in result["metrics"].items():
        print(f"\n  {model_name}:")
        print(f"    MAE:        ${metrics['mae']:>12,.0f}")
        print(f"    RMSE:       ${metrics['rmse']:>12,.0f}")
        print(f"    R²:         {metrics['r2']:>13.4f}")
        print(f"    MAPE:        {metrics['mape']:>12.2f}%")
        print(f"    Median APE:  {metrics['median_ape']:>12.2f}%")

    # --- Step 4b: Cross-validation ---
    print("\n" + "=" * 60)
    print("STEP 4b: 5-fold cross-validation")
    print("=" * 60)
    print("  Running 5-fold CV (this will take a few minutes)...")
    cv_results = cross_validate_models(x, y, n_folds=5)

    for model_name, cv_metrics in cv_results.items():
        print(f"\n  {model_name} (5-fold CV):")
        print(f"    R²:    {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
        print(f"    MAE:   ${cv_metrics['mae_mean']:>12,.0f} ± ${cv_metrics['mae_std']:>10,.0f}")
        print(f"    RMSE:  ${cv_metrics['rmse_mean']:>12,.0f} ± ${cv_metrics['rmse_std']:>10,.0f}")
        print(f"    MAPE:  {cv_metrics['mape_mean']:.2f}% ± {cv_metrics['mape_std']:.2f}%")

    # --- Step 5: Save metrics (single split + CV combined) ---
    combined_metrics = {
        "single_split": result["metrics"],
        "cross_validation": cv_results,
    }
    metrics_path = save_metrics(combined_metrics, str(output_dir))
    print(f"\n  Saved metrics: {metrics_path}")

    # --- Step 6: Save figures (uses random forest predictions) ---
    figure_dir = output_dir / "figures"
    saved_figures = save_preliminary_figures(
        df,
        result["y_test"],
        result["predictions"]["random_forest"],
        str(figure_dir),
    )
    print(f"  Saved figures:")
    for fig_path in saved_figures:
        print(f"    - {fig_path}")


    # --- Additional plots: feature importance + residual distribution ---
    rf_model = result["models"]["random_forest"]
    feat_importance_path = save_feature_importance_plot(
        feature_names=list(x.columns),
        importances=rf_model.feature_importances_,
        output_dir=str(figure_dir),
    )
    print(f"    - {feat_importance_path}")
    saved_figures.append(feat_importance_path)

    # --- Step 7: Save residuals for mispricing analysis ---
    # Residuals = predicted - actual (positive = model thinks property is overpriced)
    residuals_df = pd.DataFrame({
        "actual_price": result["y_test"].values,
        "predicted_price_lr": result["predictions"]["linear_regression"],
        "predicted_price_rf": result["predictions"]["random_forest"],
    })
    residuals_df["residual_rf"] = residuals_df["predicted_price_rf"] - residuals_df["actual_price"]
    residuals_df["residual_pct_rf"] = (
        residuals_df["residual_rf"] / residuals_df["actual_price"] * 100
    )
    # --- Price-tier breakdown: how does error vary by price band? ---
    print("\n" + "=" * 60)
    print("STEP 4c: Error breakdown by price tier")
    print("=" * 60)
    bins = [0, 500_000, 1_000_000, 2_000_000, float("inf")]
    labels = ["<$500K", "$500K-$1M", "$1M-$2M", "$2M+"]
    residuals_df["price_tier"] = pd.cut(residuals_df["actual_price"], bins=bins, labels=labels)
    tier_breakdown = []
    for tier in labels:
        tier_data = residuals_df[residuals_df["price_tier"] == tier]
        if len(tier_data) == 0:
            continue
        mae_tier = tier_data["residual_rf"].abs().mean()
        mape_tier = (tier_data["residual_rf"].abs() / tier_data["actual_price"] * 100).mean()
        tier_breakdown.append({
            "tier": tier,
            "n_properties": int(len(tier_data)),
            "mae": float(mae_tier),
            "mape_pct": float(mape_tier),
        })
        print(f"  {tier:>12}: n={len(tier_data):>6,}, MAE=${mae_tier:>10,.0f}, MAPE={mape_tier:>5.1f}%")
    residual_dist_path = save_residual_distribution_plot(
        residuals_pct=residuals_df["residual_pct_rf"],
        output_dir=str(figure_dir),
    )
    print(f"  Saved residual distribution: {residual_dist_path}")
    saved_figures.append(residual_dist_path)

    residuals_vs_pred_path = save_residuals_vs_predicted_plot(
        predicted=residuals_df["predicted_price_rf"],
        residuals=residuals_df["residual_rf"],
        output_dir=str(figure_dir),
    )
    print(f"  Saved residuals vs predicted: {residuals_vs_pred_path}")
    saved_figures.append(residuals_vs_pred_path)

    # Bar chart: MAE / MAPE by price tier — visualizes the table from STEP 4c
    tier_plot_path = save_error_by_price_tier_plot(
        tier_breakdown=tier_breakdown,
        output_dir=str(figure_dir),
    )
    print(f"  Saved error-by-tier plot: {tier_plot_path}")
    saved_figures.append(tier_plot_path)

    residuals_path = output_dir / "residuals.csv"
    residuals_df.to_csv(residuals_path, index=False)
    print(f"  Saved residuals: {residuals_path}")

    # --- Step 8: Run summary ---
    summary = {
        "input_file": str(input_path),
        "n_rows": int(len(df)),
        "n_features": len(x.columns),
        "features_used": list(x.columns),
        "metrics_single_split": result["metrics"],
        "metrics_cv_5fold": cv_results,
        "price_tier_breakdown": tier_breakdown,
        "figures": saved_figures,
    }
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
