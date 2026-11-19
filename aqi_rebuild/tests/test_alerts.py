"""
Test Suite: Pollution Alert & Health Advisory Verification.
Tests triggering of severe alerts, rapid pollution surge detection, and deterioration early warnings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aqi_rebuild.services.alert_service import PollutionAlertService


def test_alert_generation():
    service = PollutionAlertService()

    # 1. Test Severe Emergency Alert
    severe_station = {
        "candidate_station": "Test_Severe_Station",
        "current_pm25": 280.0,
        "current_aqi_category": "Severe",
        "forecast_1h": {"predicted_category": "Severe", "confidence": 0.95},
        "forecast_3h": {"predicted_category": "Severe", "confidence": 0.90},
    }
    alerts = service.evaluate_station_alerts(severe_station, previous_pm25=270.0)
    assert any(a["type"] == "EMERGENCY_SEVERE" for a in alerts), "Expected EMERGENCY_SEVERE alert"

    # 2. Test Rapid Surge Spike Alert (+40 ug/m3 increase in 1 hour)
    spike_station = {
        "candidate_station": "Test_Spike_Station",
        "current_pm25": 160.0,
        "current_aqi_category": "Very Poor",
        "forecast_1h": {"predicted_category": "Very Poor", "confidence": 0.85},
        "forecast_3h": {"predicted_category": "Very Poor", "confidence": 0.80},
    }
    alerts = service.evaluate_station_alerts(spike_station, previous_pm25=110.0)
    assert any(a["type"] == "RAPID_SPIKE" for a in alerts), "Expected RAPID_SPIKE alert"

    # 3. Test Forecast Deterioration Alert for Model-Trained Station (Moderate -> Very Poor)
    deteriorate_station = {
        "candidate_station": "Anand Vihar",
        "current_pm25": 75.0,
        "current_aqi_category": "Moderate",
        "is_seen_in_training": True,
        "forecast_1h": {"predicted_category": "Poor", "confidence": 0.78},
        "forecast_3h": {"predicted_category": "Very Poor", "confidence": 0.82},
    }
    alerts = service.evaluate_station_alerts(deteriorate_station, previous_pm25=70.0)
    assert any(a["type"] == "EARLY_WARNING_DETERIORATION" for a in alerts), "Expected EARLY_WARNING_DETERIORATION alert for trained station"

    # 4. Test Unseen Station Forecast Alert Suppression (No fake forecast alerts)
    unseen_station = {
        "candidate_station": "Aya Nagar",
        "current_pm25": 75.0,
        "current_aqi_category": "Moderate",
        "is_seen_in_training": False,
        "forecast_1h": {"predicted_category": "Forecast unavailable — station not in training", "confidence": None},
        "forecast_3h": {"predicted_category": "Forecast unavailable — station not in training", "confidence": None},
    }
    unseen_alerts = service.evaluate_station_alerts(unseen_station, previous_pm25=70.0)
    assert not any(a["type"] == "EARLY_WARNING_DETERIORATION" for a in unseen_alerts), "Unseen stations must NEVER generate forecast-based alerts"

    # 5. Test Unseen Station Live Measurement Alerts (Severe & Spike still work)
    unseen_live_alert_station = {
        "candidate_station": "Aya Nagar",
        "current_pm25": 280.0,
        "current_aqi_category": "Severe",
        "is_seen_in_training": False,
        "forecast_1h": {"predicted_category": "Forecast unavailable — station not in training", "confidence": None},
        "forecast_3h": {"predicted_category": "Forecast unavailable — station not in training", "confidence": None},
    }
    unseen_live_alerts = service.evaluate_station_alerts(unseen_live_alert_station, previous_pm25=220.0)
    assert any(a["type"] == "EMERGENCY_SEVERE" for a in unseen_live_alerts), "Unseen stations MUST still receive live measurement-based severe alerts"
    assert any(a["type"] == "RAPID_SPIKE" for a in unseen_live_alerts), "Unseen stations MUST still receive live rapid spike alerts"

    print("✅ Test Passed: Measurement alerts verified across all stations; forecast alerts strictly constrained to trained stations.")


if __name__ == "__main__":
    test_alert_generation()
