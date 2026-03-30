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

from real_estate_tracker.data_processing import add_features, clean_data, load_data, select_model_columns
from real_estate_tracker.modeling import save_metrics, train_and_evaluate
from real_estate_tracker.visualization import save_preliminary_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint 1 pipeline.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    raw = load_data(args.input)
    clean = clean_data(raw)
    featured = add_features(clean)
    x, y = select_model_columns(featured)

    result = train_and_evaluate(x, y)
    metrics_path = save_metrics(result["metrics"], args.output)

    figure_dir = os.path.join(args.output, "figures")
    saved_figures = save_preliminary_figures(
        featured,
        result["y_test"],
        result["predictions"]["random_forest"],
        figure_dir,
    )

    summary = {
        "input_rows": int(raw.shape[0]),
        "clean_rows": int(clean.shape[0]),
        "metrics_file": metrics_path,
        "figures": saved_figures,
    }
    summary_path = os.path.join(args.output, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

