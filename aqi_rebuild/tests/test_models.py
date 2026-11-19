"""
Test Suite: Model Inference Verification.
Tests 1-hour and 3-hour models on engineered features, checking predictions, classes, and probabilities.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aqi_rebuild.config.settings import ORDERED_CATEGORIES
from aqi_rebuild.pipeline.preprocessor import TimeseriesPreprocessor
from aqi_rebuild.pipeline.feature_engineer import AQIFeatureEngineer
from aqi_rebuild.pipeline.predictor import AQIPredictor


def test_model_inference():
    preprocessor = TimeseriesPreprocessor()
    engineer = AQIFeatureEngineer()
    predictor = AQIPredictor()

    buffer_df = preprocessor.load_buffer()
    latest_rows = engineer.get_latest_station_features(buffer_df)
    X = engineer.extract_model_features(latest_rows)

    preds_1h, probs_1h, preds_3h, probs_3h = predictor.predict_features(X)

    # 1. Output length matches input rows
    assert len(preds_1h) == len(X)
    assert len(probs_1h) == len(X)
    assert len(preds_3h) == len(X)
    assert len(probs_3h) == len(X)

    # 2. Probability vectors sum to 1.0
    for prob in probs_1h:
        assert np.isclose(np.sum(prob), 1.0, atol=1e-3), f"1h probabilities do not sum to 1: {np.sum(prob)}"

    for prob in probs_3h:
        assert np.isclose(np.sum(prob), 1.0, atol=1e-3), f"3h probabilities do not sum to 1: {np.sum(prob)}"

    # 3. Classes are valid AQI categories
    valid_categories = set(ORDERED_CATEGORIES)
    for p in preds_1h:
        assert p in valid_categories, f"Invalid predicted class for 1h: {p}"
    for p in preds_3h:
        assert p in valid_categories, f"Invalid predicted class for 3h: {p}"

    print(f"✅ Test Passed: 1-hour and 3-hour model inference validated across {len(X)} stations.")


if __name__ == "__main__":
    test_model_inference()
