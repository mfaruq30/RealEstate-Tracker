from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(x: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict:
    """Train baseline and tree models, then return metrics and predictions."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=random_state,
        ),
    }

    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred
        # Percentage errors — better than MAE when prices span a wide range ($100K to $5M)
        # Avoid divide-by-zero by clipping y_test at $1
        ape = np.abs(y_pred - y_test.values) / np.clip(y_test.values, 1, None) * 100
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(np.mean(ape)),
            "median_ape": float(np.median(ape)),
        }

    return {
        "x_test": x_test,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "models": models,
    }


def save_metrics(metrics: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path

def cross_validate_models(
    x: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Run K-fold cross-validation for both models and return per-fold + summary stats.

    Returns a dict shaped like:
    {
        "linear_regression": {
            "r2_scores": [...],   # one per fold
            "mae_scores": [...],
            "rmse_scores": [...],
            "r2_mean": float, "r2_std": float,
            "mae_mean": float, "mae_std": float,
            "rmse_mean": float, "rmse_std": float,
        },
        "random_forest": { ... same shape ... }
    }
    """
    from sklearn.model_selection import KFold, cross_val_score

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results: dict[str, dict] = {}
    for name, model in models.items():
        # sklearn returns negative MAE/MSE so we flip signs to get positive numbers
        r2 = cross_val_score(model, x, y, cv=kf, scoring="r2")
        neg_mae = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_absolute_error")
        neg_mse = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_squared_error")
        # MAPE wasn't in sklearn's default scoring list until late versions; compute manually per fold
        neg_mape = cross_val_score(
            model, x, y, cv=kf,
            scoring="neg_mean_absolute_percentage_error",
        )

        mae = -neg_mae
        rmse = np.sqrt(-neg_mse)
        mape = -neg_mape * 100  # convert from fraction to percentage

        results[name] = {
            "r2_scores": r2.tolist(),
            "mae_scores": mae.tolist(),
            "rmse_scores": rmse.tolist(),
            "mape_scores": mape.tolist(),
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std()),
            "mae_mean": float(mae.mean()),
            "mae_std": float(mae.std()),
            "rmse_mean": float(rmse.mean()),
            "rmse_std": float(rmse.std()),
            "mape_mean": float(mape.mean()),
            "mape_std": float(mape.std()),
        }

    return results