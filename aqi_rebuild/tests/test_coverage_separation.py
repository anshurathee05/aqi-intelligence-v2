"""
Test Suite: ML Forecasting Separation and Live Monitoring Coverage.
Verifies:
1. All 13 trained stations receive genuine 1h and 3h ML predictions and probabilities.
2. All 12 unseen stations receive strictly 'Forecast unavailable — station not in training' with no fake confidence or probabilities.
3. All 25 stations remain available for live monitoring with actual sensor values.
4. Live vs delayed vs offline status is correctly preserved.
5. Alerts: measurement alerts trigger for any station; forecast deterioration alerts trigger only for trained stations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aqi_rebuild.config.settings import TRAINED_STATIONS, ORDERED_CATEGORIES
from aqi_rebuild.pipeline.preprocessor import TimeseriesPreprocessor
from aqi_rebuild.pipeline.predictor import AQIPredictor
from aqi_rebuild.services.alert_service import PollutionAlertService


def test_station_coverage_separation():
    preprocessor = TimeseriesPreprocessor()
    buffer_df = preprocessor.load_buffer()
    assert not buffer_df.empty, "Buffer must not be empty"

    all_stations_in_buffer = sorted(buffer_df["candidate_station"].dropna().unique().tolist())
    assert len(all_stations_in_buffer) == 25, f"Expected 25 stations in buffer, got {len(all_stations_in_buffer)}"

    predictor = AQIPredictor()
    result = predictor.predict_latest_stations(buffer_df)
    stations_output = result.get("stations", [])

    assert len(stations_output) == 25, f"Expected 25 stations in prediction output, got {len(stations_output)}"

    trained_set = set(TRAINED_STATIONS)
    assert len(trained_set) == 13, f"Expected 13 trained stations, got {len(trained_set)}"

    seen_count = 0
    unseen_count = 0

    valid_categories = set(ORDERED_CATEGORIES) | {"Offline"}

    for s in stations_output:
        st_name = s["candidate_station"]
        is_seen = s["is_seen_in_training"]
        f1 = s["forecast_1h"]
        f3 = s["forecast_3h"]

        # Check basic monitoring fields exist for all 25 stations
        assert "station_name" in s
        assert "region" in s
        assert "latitude" in s
        assert "longitude" in s
        assert "freshness_status" in s
        assert "current_pm25" in s

        if st_name in trained_set:
            seen_count += 1
            assert is_seen is True, f"Station {st_name} must have is_seen_in_training=True"
            assert s["training_status"] == "Seen in Training"
            assert s["training_warning"] is None

            # Trained stations must receive genuine predictions (or 'Offline' if sensor dead)
            p1_cat = f1.get("predicted_category")
            p3_cat = f3.get("predicted_category")
            assert p1_cat in valid_categories, f"Invalid 1h category for {st_name}: {p1_cat}"
            assert p3_cat in valid_categories, f"Invalid 3h category for {st_name}: {p3_cat}"

            if s["freshness_status"] in ["CURRENT_LIVE", "DELAYED_LIVE"]:
                assert f1.get("confidence") is not None, f"Confidence missing for live trained station {st_name}"
                assert 0.0 <= f1["confidence"] <= 1.0
                assert len(f1.get("probabilities", {})) == 6, f"Expected 6 class probabilities for {st_name}"
                assert f3.get("confidence") is not None
                assert 0.0 <= f3["confidence"] <= 1.0
                assert len(f3.get("probabilities", {})) == 6
        else:
            unseen_count += 1
            assert is_seen is False, f"Station {st_name} must have is_seen_in_training=False"
            assert s["training_status"] == "Unseen Station (Monitoring Only)"
            assert s["training_warning"] is not None

            # Unseen stations MUST receive "Forecast unavailable — station not in training"
            assert f1.get("predicted_category") == "Forecast unavailable — station not in training", \
                f"Expected unavailable forecast for unseen station {st_name}, got {f1.get('predicted_category')}"
            assert f1.get("confidence") is None, f"Unseen station {st_name} must have confidence=None"
            assert f1.get("probabilities") == {}, f"Unseen station {st_name} must have empty probabilities"

            assert f3.get("predicted_category") == "Forecast unavailable — station not in training", \
                f"Expected unavailable forecast for unseen station {st_name}, got {f3.get('predicted_category')}"
            assert f3.get("confidence") is None
            assert f3.get("probabilities") == {}

    assert seen_count == 13, f"Expected exactly 13 trained stations evaluated, got {seen_count}"
    assert unseen_count == 12, f"Expected exactly 12 unseen stations evaluated, got {unseen_count}"

    # Verify Alert Separation
    alert_service = PollutionAlertService()

    # 1. Unseen station with high PM2.5 triggers live measurement alert
    unseen_alert_st = {
        "candidate_station": "Aya Nagar",
        "current_pm25": 290.0,
        "current_aqi_category": "Severe",
        "is_seen_in_training": False,
        "forecast_1h": {"predicted_category": "Forecast unavailable — station not in training", "confidence": None},
        "forecast_3h": {"predicted_category": "Forecast unavailable — station not in training", "confidence": None},
    }
    alerts = alert_service.evaluate_station_alerts(unseen_alert_st)
    assert any(a["type"] == "EMERGENCY_SEVERE" for a in alerts), "Unseen station must trigger live severe alert"
    assert not any(a["type"] == "EARLY_WARNING_DETERIORATION" for a in alerts), "Unseen station must not trigger forecast alert"

    # 2. Trained station with forecast deterioration triggers early warning
    trained_alert_st = {
        "candidate_station": "Anand Vihar",
        "current_pm25": 80.0,
        "current_aqi_category": "Moderate",
        "is_seen_in_training": True,
        "forecast_1h": {"predicted_category": "Very Poor", "confidence": 0.88},
        "forecast_3h": {"predicted_category": "Severe", "confidence": 0.91},
    }
    trained_alerts = alert_service.evaluate_station_alerts(trained_alert_st)
    assert any(a["type"] == "EARLY_WARNING_DETERIORATION" for a in trained_alerts), "Trained station must trigger forecast alert"

    print(f"✅ Coverage Separation Test Passed: 13 trained stations received ML forecasts; 12 unseen stations constrained to live monitoring.")


if __name__ == "__main__":
    test_station_coverage_separation()
