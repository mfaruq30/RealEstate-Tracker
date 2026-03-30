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

