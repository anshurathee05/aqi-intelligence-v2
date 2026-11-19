"""
End-to-End Pipeline Runner for AQI Rebuild.
Orchestrates live OpenAQ data fetching, buffer preprocessing, 45-feature engineering,
multi-horizon ML inference, and real-time pollution alert evaluation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aqi_rebuild.config.settings import (
    LIVE_READINGS_BUFFER,
    LATEST_PREDICTIONS_JSON,
    ALERT_HISTORY_JSON,
)
from aqi_rebuild.pipeline.openaq_fetcher import OpenAQFetcher
from aqi_rebuild.pipeline.preprocessor import TimeseriesPreprocessor
from aqi_rebuild.pipeline.feature_engineer import AQIFeatureEngineer
from aqi_rebuild.pipeline.predictor import AQIPredictor
from aqi_rebuild.services.alert_service import PollutionAlertService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aqi_rebuild.runner")


def run_full_pipeline(fetch_live: bool = True) -> dict:
    """Executes the full pipeline workflow."""
    logger.info("=" * 65)
    logger.info("STARTING AQI INTELLIGENCE PIPELINE EXECUTION")
    logger.info("=" * 65)

    preprocessor = TimeseriesPreprocessor()
    predictor = AQIPredictor()
    alert_service = PollutionAlertService()

    # Step 1: Live Fetch (if requested and possible)
    live_records = []
    if fetch_live:
        logger.info("Step 1: Contacting OpenAQ API v3 for latest Delhi sensor data...")
        try:
            fetcher = OpenAQFetcher()
            live_records = fetcher.fetch_all_stations_latest()
            logger.info(f"OpenAQ returned verified live records for {len(live_records)} stations.")
        except Exception as e:
            logger.warning(f"Could not reach OpenAQ API ({e}). Continuing with existing historical buffer.")

    # Step 2: Buffer Ingestion & Preprocessing
    logger.info("Step 2: Ingesting and sanitizing timeseries buffer...")
    buffer_df = preprocessor.ingest_live_readings(live_records)
    logger.info(f"Buffer contains {len(buffer_df)} hourly records across {buffer_df['station_name'].nunique()} stations.")

    # Step 3: Feature Engineering & Model Inference
    logger.info("Step 3: Generating 45 model features and predicting 1h & 3h AQI categories...")
    predictions = predictor.predict_latest_stations(buffer_df)

    # Step 4: Pollution Alerts Evaluation
    logger.info("Step 4: Evaluating emergency alerts, pollution surges, and early warnings...")
    alerts = alert_service.evaluate_all_stations(predictions)
    logger.info(f"Generated {len(alerts)} active alert(s) across monitoring network.")

    # Step 5: Summary
    stations = predictions.get("stations", [])
    forecasted_count = sum(1 for s in stations if s.get("is_seen_in_training"))
    monitoring_count = len(stations) - forecasted_count

    logger.info("=" * 65)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info(f"Stations monitored: {len(stations)}")
    logger.info(f"Stations forecasted: {forecasted_count} (Trained ML reference stations)")
    logger.info(f"Stations monitoring-only: {monitoring_count} (Unseen in training — ML forecast disabled)")
    logger.info(f"Active alerts: {len(alerts)}")
    logger.info(f"Predictions saved: {LATEST_PREDICTIONS_JSON}")
    logger.info("=" * 65)

    # Print summary table
    if stations:
        print("\n" + "=" * 135)
        print(f"DELHI CAAQMS NETWORK: {len(stations)} STATIONS MONITORED ({forecasted_count} FORECASTED, {monitoring_count} MONITORING-ONLY)")
        print("=" * 135)
        print(f"{'Station':18s} | {'Freshness Tier':22s} | {'Observed (IST)':18s} | {'Age':10s} | {'PM2.5':14s} | {'1h Forecast':26s} | {'Training Status':22s}")
        print("=" * 135)
        for s in stations:
            tier = s.get("freshness_label", "Unknown")[:20]
            obs = str(s.get("last_observed_at_ist", ""))[:16]
            age = s.get("observation_age_text", "N/A")
            pm25 = f"{s['current_pm25']:.1f} ug/m3" if s.get('current_pm25') is not None else "Offline (N/A)"
            p1 = s['forecast_1h']['predicted_category']
            p1_display = p1 if len(p1) <= 25 else (p1[:22] + "...")
            seen = "✅ ML Forecast (13)" if s.get("is_seen_in_training") else "📊 Monitoring Only"
            print(f"{s['candidate_station']:18s} | {tier:22s} | {obs:18s} | {age:10s} | {pm25:14s} | {p1_display:26s} | {seen:22s}")
        print("=" * 135 + "\n")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Intelligence Pipeline Runner")
    parser.add_argument("--no-fetch", action="store_true", help="Skip live OpenAQ API query and run on local buffer")
    args = parser.parse_args()

    run_full_pipeline(fetch_live=not args.no_fetch)
