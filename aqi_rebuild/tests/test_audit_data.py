"""
Test Suite: Data Audit Verification.
Verifies that:
1. Historical seed records are marked is_live=False and data_source='historical_context'.
2. Live OpenAQ records are marked is_live=True and data_source='openaq_live'.
3. Outdated/historical observations are NEVER presented as fresh current live readings.
4. Offline stations are flagged with current_pm25=None and category='Offline / Inactive Sensor'.
5. Freshness logic: 3-hour live threshold vs delayed vs 48-hour severely stale.
6. Unseen station detection: identifies stations not in training, warns, and does NOT block prediction.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aqi_rebuild.config.settings import TRAINED_STATIONS, LIVE_FRESHNESS_HOURS, SEVERELY_STALE_HOURS, TIMEZONE
from aqi_rebuild.pipeline.preprocessor import TimeseriesPreprocessor
from aqi_rebuild.pipeline.predictor import AQIPredictor


def test_data_audit():
    preprocessor = TimeseriesPreprocessor()
    buffer_df = preprocessor.load_buffer()

    assert not buffer_df.empty, "Buffer should not be empty"

    # 1. Verify provenance columns exist
    assert "data_source" in buffer_df.columns, "data_source column missing in buffer"
    assert "is_live" in buffer_df.columns, "is_live column missing in buffer"

    # 2. Verify that historical seed rows have is_live == False
    hist_rows = buffer_df[buffer_df["data_source"] == "historical_context"]
    if not hist_rows.empty:
        assert (hist_rows["is_live"] == False).all(), "Historical seed rows must have is_live == False"

    # 3. Test that if a station has ONLY historical data (no live data), current_pm25 is None
    dummy_hist_only = pd.DataFrame([{
        "observed_at": "2026-11-14 00:00:00+00:00",
        "observed_at_ist": "2026-11-14 05:30:00+05:30",
        "location_id": 99999,
        "station_name": "Test Offline Station",
        "candidate_station": "Alipur",
        "region": "North Delhi",
        "latitude": 28.8,
        "longitude": 77.1,
        "pm25": 350.0,
        "co": 1.0, "no": 10.0, "no2": 20.0, "nox": 30.0, "o3": 15.0, "pm10": 400.0,
        "relative_humidity": 60.0, "so2": 10.0, "temperature": 25.0, "wind_speed": 1.0, "wind_direction": 180.0,
        "data_source": "historical_context",
        "is_live": False
    }])

    predictor = AQIPredictor()
    result = predictor.predict_latest_stations(dummy_hist_only)
    station_res = result["stations"][0]

    assert station_res["is_current_live"] == False, "Historical station must not be current live"
    assert station_res["current_pm25"] is None, "Historical station current_pm25 must be None, NOT historical value"
    assert "Offline" in station_res["current_aqi_category"], "Category must indicate Offline"
    assert station_res["historical_context_pm25"] == 350.0, "Old value can only be stored under historical_context_pm25"

    # 4. Test Freshness Logic: Fresh (<3h) vs Delayed (4.5h) vs Severely Stale (>48h)
    now_ist = pd.Timestamp.now(tz=TIMEZONE)

    # 4a. Fresh reading: 1.5 hours ago
    fresh_time = (now_ist - pd.Timedelta(hours=1.5)).strftime("%Y-%m-%d %H:%M:%S+05:30")
    dummy_fresh = pd.DataFrame([{
        "observed_at": (now_ist - pd.Timedelta(hours=1.5)).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "observed_at_ist": fresh_time,
        "location_id": 11111,
        "station_name": "Alipur, Delhi - DPCC",
        "candidate_station": "Alipur",
        "region": "North Delhi",
        "latitude": 28.8, "longitude": 77.1,
        "pm25": 180.0,
        "co": 1.0, "no": 10.0, "no2": 20.0, "nox": 30.0, "o3": 15.0, "pm10": 300.0,
        "relative_humidity": 60.0, "so2": 10.0, "temperature": 25.0, "wind_speed": 1.0, "wind_direction": 180.0,
        "data_source": "openaq_live", "is_live": True
    }])
    res_fresh = predictor.predict_latest_stations(dummy_fresh)["stations"][0]
    assert res_fresh["freshness_status"] == "CURRENT_LIVE", f"Expected CURRENT_LIVE, got {res_fresh['freshness_status']}"
    assert res_fresh["is_current_live"] == True
    assert res_fresh["current_pm25"] == 180.0

    # 4b. Delayed reading: 5.0 hours ago (> 3h but <= 48h)
    delayed_time = (now_ist - pd.Timedelta(hours=5.0)).strftime("%Y-%m-%d %H:%M:%S+05:30")
    dummy_delayed = pd.DataFrame([{
        "observed_at": (now_ist - pd.Timedelta(hours=5.0)).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "observed_at_ist": delayed_time,
        "location_id": 11111,
        "station_name": "Alipur, Delhi - DPCC",
        "candidate_station": "Alipur",
        "region": "North Delhi",
        "latitude": 28.8, "longitude": 77.1,
        "pm25": 180.0,
        "co": 1.0, "no": 10.0, "no2": 20.0, "nox": 30.0, "o3": 15.0, "pm10": 300.0,
        "relative_humidity": 60.0, "so2": 10.0, "temperature": 25.0, "wind_speed": 1.0, "wind_direction": 180.0,
        "data_source": "openaq_live", "is_live": True
    }])
    res_delayed = predictor.predict_latest_stations(dummy_delayed)["stations"][0]
    assert res_delayed["freshness_status"] == "DELAYED_LIVE", f"Expected DELAYED_LIVE, got {res_delayed['freshness_status']}"
    assert res_delayed["is_current_live"] == False, "Delayed reading must NOT be flagged as current live"
    assert res_delayed["is_delayed"] == True

    # 4c. Severely stale reading: 60 hours ago (> 48h)
    stale_time = (now_ist - pd.Timedelta(hours=60.0)).strftime("%Y-%m-%d %H:%M:%S+05:30")
    dummy_stale = pd.DataFrame([{
        "observed_at": (now_ist - pd.Timedelta(hours=60.0)).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "observed_at_ist": stale_time,
        "location_id": 11111,
        "station_name": "Alipur, Delhi - DPCC",
        "candidate_station": "Alipur",
        "region": "North Delhi",
        "latitude": 28.8, "longitude": 77.1,
        "pm25": 180.0,
        "co": 1.0, "no": 10.0, "no2": 20.0, "nox": 30.0, "o3": 15.0, "pm10": 300.0,
        "relative_humidity": 60.0, "so2": 10.0, "temperature": 25.0, "wind_speed": 1.0, "wind_direction": 180.0,
        "data_source": "openaq_live", "is_live": True
    }])
    res_stale = predictor.predict_latest_stations(dummy_stale)["stations"][0]
    assert res_stale["freshness_status"] == "SEVERELY_STALE"
    assert res_stale["current_pm25"] is None, "Severely stale reading must have current_pm25=None"
    assert "Inactive" in res_stale["current_aqi_category"] or "Offline" in res_stale["current_aqi_category"]

    # 5. Test Unseen Station Handling (Live monitoring active, ML forecast unavailable)
    dummy_unseen = pd.DataFrame([{
        "observed_at": fresh_time,
        "observed_at_ist": fresh_time,
        "location_id": 88888,
        "station_name": "Aya Nagar, New Delhi - IMD",
        "candidate_station": "Aya Nagar",  # Not in 13 trained stations!
        "region": "South Delhi",
        "latitude": 28.47, "longitude": 77.13,
        "pm25": 140.0,
        "co": 1.0, "no": 10.0, "no2": 20.0, "nox": 30.0, "o3": 15.0, "pm10": 250.0,
        "relative_humidity": 60.0, "so2": 10.0, "temperature": 25.0, "wind_speed": 1.0, "wind_direction": 180.0,
        "data_source": "openaq_live", "is_live": True
    }])
    res_unseen = predictor.predict_latest_stations(dummy_unseen)["stations"][0]
    assert res_unseen["is_seen_in_training"] == False, "Aya Nagar should be detected as unseen station"
    assert res_unseen["training_warning"] is not None, "Warning message must be present for unseen station"
    assert res_unseen["forecast_1h"]["predicted_category"] == "Forecast unavailable — station not in training"
    assert res_unseen["forecast_1h"]["confidence"] is None, "Unseen station must not have fake confidence"
    assert res_unseen["forecast_1h"]["probabilities"] == {}, "Unseen station must not have fake probabilities"
    assert res_unseen["forecast_3h"]["predicted_category"] == "Forecast unavailable — station not in training"
    assert res_unseen["forecast_3h"]["confidence"] is None
    assert res_unseen["forecast_3h"]["probabilities"] == {}
    assert res_unseen["current_pm25"] == 140.0, "Live PM2.5 measurement must still be retained for live monitoring"
    assert res_unseen["current_aqi_category"] == "Very Poor"

    # 6. Test Seen Station Handling (ML forecasts active with probabilities)
    dummy_seen = pd.DataFrame([{
        "observed_at": fresh_time,
        "observed_at_ist": fresh_time,
        "location_id": 235,
        "station_name": "Anand Vihar, New Delhi - DPCC",
        "candidate_station": "Anand Vihar",  # In trained stations!
        "region": "East Delhi",
        "latitude": 28.64, "longitude": 77.31,
        "pm25": 190.0,
        "co": 1.0, "no": 10.0, "no2": 20.0, "nox": 30.0, "o3": 15.0, "pm10": 350.0,
        "relative_humidity": 60.0, "so2": 10.0, "temperature": 25.0, "wind_speed": 1.0, "wind_direction": 180.0,
        "data_source": "openaq_live", "is_live": True
    }])
    res_seen = predictor.predict_latest_stations(dummy_seen)["stations"][0]
    assert res_seen["is_seen_in_training"] == True, "Anand Vihar must be detected as seen station"
    assert res_seen["training_warning"] is None
    assert res_seen["forecast_1h"]["predicted_category"] in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    assert res_seen["forecast_1h"]["confidence"] is not None
    assert len(res_seen["forecast_1h"]["probabilities"]) == 6
    assert res_seen["forecast_3h"]["predicted_category"] in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    assert res_seen["forecast_3h"]["confidence"] is not None
    assert len(res_seen["forecast_3h"]["probabilities"]) == 6

    print("✅ All Data Audit Tests Passed: 3h freshness, 48h stale, and strict training coverage separation validated.")


if __name__ == "__main__":
    test_data_audit()
