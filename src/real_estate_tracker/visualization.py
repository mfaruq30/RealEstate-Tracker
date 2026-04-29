from __future__ import annotations

import os

import matplotlib
import pandas as pd

# Use a non-interactive backend for local/CI reproducibility.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_preliminary_figures(df: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, output_dir: str) -> list[str]:
    """Create checkpoint-ready preliminary visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    # 1) Distribution plot.
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(df["price"], bins=20)
    ax1.set_title("Listing Price Distribution")
    ax1.set_xlabel("Price (USD)")
    ax1.set_ylabel("Count")
    p1 = os.path.join(output_dir, "price_distribution.png")
    fig1.tight_layout()
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    paths.append(p1)

    # 2) Price vs sqft.
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(df["sqft"], df["price"], alpha=0.7)
    ax2.set_title("Price vs Square Footage")
    ax2.set_xlabel("Square Feet")
    ax2.set_ylabel("Price (USD)")
    p2 = os.path.join(output_dir, "price_vs_sqft.png")
    fig2.tight_layout()
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    paths.append(p2)

    # 3) Predicted vs actual.
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.scatter(y_true, y_pred, alpha=0.7)
    lim_min = min(float(y_true.min()), float(y_pred.min()))
    lim_max = max(float(y_true.max()), float(y_pred.max()))
    ax3.plot([lim_min, lim_max], [lim_min, lim_max], "--")
    ax3.set_title("Predicted vs Actual Price")
    ax3.set_xlabel("Actual Price (USD)")
    ax3.set_ylabel("Predicted Price (USD)")
    p3 = os.path.join(output_dir, "predicted_vs_actual.png")
    fig3.tight_layout()
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    paths.append(p3)

    return paths

def save_feature_importance_plot(
    feature_names: list[str],
    importances,
    output_dir: str,
) -> str:
    """Save a horizontal bar chart of Random Forest feature importances."""
    os.makedirs(output_dir, exist_ok=True)

    # Sort features by importance (descending), then reverse for horizontal display
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])
    sorted_names = [p[0] for p in pairs]
    sorted_importances = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sorted_names, sorted_importances)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance (Gini)")
    fig.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_residual_distribution_plot(
    residuals_pct,
    output_dir: str,
) -> str:
    """Save a histogram of percentage residuals (predicted - actual) / actual * 100.

    The tails of this distribution represent properties the model believes are
    mispriced relative to their features (the project's main deliverable).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clip extreme outliers for readable axis (a few residuals can exceed 500%)
    clipped = residuals_pct.clip(-100, 100)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clipped, bins=50, edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_title("Distribution of Residuals (% of Actual Price)")
    ax.set_xlabel("Residual % (predicted − actual) / actual × 100")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "residual_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def save_residuals_vs_predicted_plot(
    predicted: pd.Series,
    residuals: pd.Series,
    output_dir: str,
) -> str:
    """Save a residuals-vs-predicted scatter plot — the standard model diagnostic.

    A well-fit model should show residuals randomly scattered around 0 across
    the full range of predicted values. Patterns (fanning, curvature, bias)
    indicate model weaknesses.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, residuals, alpha=0.15, s=8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_title("Residuals vs Predicted Price (Random Forest)")
    ax.set_xlabel("Predicted Price (USD)")
    ax.set_ylabel("Residual = Predicted − Actual (USD)")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "residuals_vs_predicted.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path