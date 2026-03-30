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
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    return {
        "x_test": x_test,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
    }


def save_metrics(metrics: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path

