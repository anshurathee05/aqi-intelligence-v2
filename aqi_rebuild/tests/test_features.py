"""
Test Suite: Feature Engineering Verification.
Verifies that AQIFeatureEngineer produces the EXACT 45 features expected by the trained ML pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Path configuration
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aqi_rebuild.config.settings import MODEL_FEATURE_NAMES, BEST_MODEL_1H_PATH, BEST_MODEL_3H_PATH
from aqi_rebuild.pipeline.feature_engineer import AQIFeatureEngineer
from aqi_rebuild.pipeline.preprocessor import TimeseriesPreprocessor
import joblib


def test_feature_count_and_names():
    engineer = AQIFeatureEngineer()
    preprocessor = TimeseriesPreprocessor()
    buffer_df = preprocessor.load_buffer()

    assert not buffer_df.empty, "Timeseries buffer should not be empty"

    transformed_df = engineer.transform_timeseries(buffer_df)
    model_features_df = engineer.extract_model_features(transformed_df)

    # 1. Feature count must equal 45
    assert model_features_df.shape[1] == 45, f"Expected 45 features, got {model_features_df.shape[1]}"
    assert len(MODEL_FEATURE_NAMES) == 45, f"Expected 45 MODEL_FEATURE_NAMES, got {len(MODEL_FEATURE_NAMES)}"

    # 2. Column names and order must match MODEL_FEATURE_NAMES exactly
    assert list(model_features_df.columns) == MODEL_FEATURE_NAMES, "Column names or ordering do not match MODEL_FEATURE_NAMES"

    # 3. Model pipeline feature_names_in_ must match
    m1 = joblib.load(BEST_MODEL_1H_PATH)
    assert list(m1.feature_names_in_) == MODEL_FEATURE_NAMES, "Model 1h feature_names_in_ does not match MODEL_FEATURE_NAMES"

    m3 = joblib.load(BEST_MODEL_3H_PATH)
    assert list(m3.feature_names_in_) == MODEL_FEATURE_NAMES, "Model 3h feature_names_in_ does not match MODEL_FEATURE_NAMES"

    print("✅ Test Passed: Exactly 45 features produced, strictly matching 1h and 3h model requirements.")


if __name__ == "__main__":
    test_feature_count_and_names()
