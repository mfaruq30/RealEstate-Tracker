from pathlib import Path

from real_estate_tracker.data_processing import add_features, clean_data, load_data, select_model_columns
from real_estate_tracker.modeling import train_and_evaluate


def test_training_pipeline_smoke() -> None:
    csv_path = Path("data/sample_listings.csv")
    raw = load_data(str(csv_path))
    cleaned = clean_data(raw)
    featured = add_features(cleaned)
    x, y = select_model_columns(featured)

    result = train_and_evaluate(x, y)
    assert "linear_regression" in result["metrics"]
    assert "random_forest" in result["metrics"]
    assert result["metrics"]["linear_regression"]["mae"] >= 0
    assert result["metrics"]["random_forest"]["rmse"] >= 0

