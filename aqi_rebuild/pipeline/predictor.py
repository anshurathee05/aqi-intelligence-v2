"""
Prediction Engine for AQI Rebuild.
Loads the 45-feature 1-hour and 3-hour machine learning pipelines.
Generates multi-class probabilistic forecasts, confidence levels, and trajectory analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from aqi_rebuild.config.settings import (
    BEST_MODEL_1H_PATH,
    BEST_MODEL_3H_PATH,
    LATEST_PREDICTIONS_JSON,
    MODEL_FEATURE_NAMES,
    AQI_CATEGORIES,
    ORDERED_CATEGORIES,
    TIMEZONE,
    LIVE_FRESHNESS_HOURS,
    SEVERELY_STALE_HOURS,
    TRAINED_STATIONS,
)
from aqi_rebuild.pipeline.feature_engineer import AQIFeatureEngineer

logger = logging.getLogger("aqi_rebuild.predictor")


class AQIPredictor:
    """Inference engine for 1-hour and 3-hour AQI category forecasts."""

    def __init__(self):
        self.model_1h = None
        self.model_3h = None
        self.feature_engineer = AQIFeatureEngineer()
        self._load_models()

    def _load_models(self):
        """Loads the saved 45-feature models into memory."""
        if not BEST_MODEL_1H_PATH.exists():
            raise FileNotFoundError(f"1-hour model not found at {BEST_MODEL_1H_PATH}")
        if not BEST_MODEL_3H_PATH.exists():
            raise FileNotFoundError(f"3-hour model not found at {BEST_MODEL_3H_PATH}")

        logger.info(f"Loading 1-hour model from {BEST_MODEL_1H_PATH.name}...")
        self.model_1h = joblib.load(BEST_MODEL_1H_PATH)

        logger.info(f"Loading 3-hour model from {BEST_MODEL_3H_PATH.name}...")
        self.model_3h = joblib.load(BEST_MODEL_3H_PATH)

        # Sanity check feature count
        if getattr(self.model_1h, "n_features_in_", None) != 45:
            raise ValueError(
                f"Model 1h expects {getattr(self.model_1h, 'n_features_in_', None)} features, expected 45!"
            )
        logger.info("Successfully loaded 45-feature 1-hour and 3-hour models.")

    def predict_features(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes inference on a DataFrame strictly adhering to the 45 input features.
        Returns:
            (preds_1h, probs_1h, preds_3h, probs_3h)
        """
        if list(X_df.columns) != MODEL_FEATURE_NAMES:
            # Reorder or select
            X_df = X_df[MODEL_FEATURE_NAMES]

        preds_1h = self.model_1h.predict(X_df)
        probs_1h = self.model_1h.predict_proba(X_df)

        preds_3h = self.model_3h.predict(X_df)
        probs_3h = self.model_3h.predict_proba(X_df)

        return preds_1h, probs_1h, preds_3h, probs_3h

    def _determine_trajectory(self, current_cat: str, pred_1h: str, pred_3h: str) -> str:
        """Determines whether air quality is improving, deteriorating, or remaining stable."""
        curr_sev = AQI_CATEGORIES.get(current_cat, {}).get("severity", 3)
        h1_sev = AQI_CATEGORIES.get(pred_1h, {}).get("severity", 3)
        h3_sev = AQI_CATEGORIES.get(pred_3h, {}).get("severity", 3)

        if h3_sev > curr_sev or h1_sev > curr_sev:
            return "Deteriorating"
        elif h3_sev < curr_sev or h1_sev < curr_sev:
            return "Improving"
        else:
            return "Stable"

    def predict_latest_stations(self, timeseries_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates forecasts for the latest reading of every station in the timeseries buffer.
        Saves results to LATEST_PREDICTIONS_JSON and returns the structured dictionary.
        """
        if timeseries_df.empty:
            logger.warning("Empty timeseries buffer passed to predictor.")
            return {}

        latest_df = self.feature_engineer.get_latest_station_features(timeseries_df)
        X_df = self.feature_engineer.extract_model_features(latest_df)

        # Filter strictly to the 13 model-trained stations for ML inference
        trained_mask = latest_df["candidate_station"].isin(TRAINED_STATIONS)
        preds_1h_map = {}
        probs_1h_map = {}
        preds_3h_map = {}
        probs_3h_map = {}

        classes_1h = list(self.model_1h.classes_)
        classes_3h = list(self.model_3h.classes_)

        if trained_mask.any():
            X_trained = X_df.loc[trained_mask]
            preds_1h_arr, probs_1h_arr, preds_3h_arr, probs_3h_arr = self.predict_features(X_trained)
            for idx_t, row_idx in enumerate(X_trained.index):
                preds_1h_map[row_idx] = preds_1h_arr[idx_t]
                probs_1h_map[row_idx] = probs_1h_arr[idx_t]
                preds_3h_map[row_idx] = preds_3h_arr[idx_t]
                probs_3h_map[row_idx] = probs_3h_arr[idx_t]

        station_forecasts = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for idx, row in latest_df.iterrows():
            st_name = row["station_name"]
            cand_name = row["candidate_station"]
            obs_ist = str(row.get("observed_at_ist", row.get("dt_ist", "")))

            # 1. Freshness & Inactivity Evaluation
            is_openaq_feed = bool(row.get("is_live", False)) and str(row.get("data_source", "")) == "openaq_live"
            obs_time = pd.to_datetime(obs_ist, errors="coerce")
            now_ist = pd.Timestamp.now(tz=TIMEZONE)
            if pd.notna(obs_time):
                if obs_time.tzinfo is None:
                    obs_time = obs_time.tz_localize(TIMEZONE)
                age_hours = round((now_ist - obs_time).total_seconds() / 3600.0, 2)
                age_text = f"{age_hours:.1f} hours ago"
            else:
                age_hours = 9999.0
                age_text = "Unknown age"

            # Freshness tier classification
            if is_openaq_feed and age_hours <= LIVE_FRESHNESS_HOURS:
                freshness_status = "CURRENT_LIVE"
                freshness_label = f"🟢 Current Live (<{LIVE_FRESHNESS_HOURS}h)"
                is_current = True
                is_delayed = False
                is_severely_stale = False
            elif is_openaq_feed and age_hours <= SEVERELY_STALE_HOURS:
                freshness_status = "DELAYED_LIVE"
                freshness_label = f"🟡 Delayed ({age_text} - Not Real-Time)"
                is_current = False
                is_delayed = True
                is_severely_stale = False
            elif age_hours > SEVERELY_STALE_HOURS:
                freshness_status = "SEVERELY_STALE"
                freshness_label = f"🔴 Severely Stale ({age_text}) / Inactive Sensor"
                is_current = False
                is_delayed = False
                is_severely_stale = True
            else:
                freshness_status = "OFFLINE"
                freshness_label = "⚪ Offline / No Feed"
                is_current = False
                is_delayed = False
                is_severely_stale = False

            # 2. Training Station Coverage Inspection
            is_seen_in_training = cand_name in TRAINED_STATIONS
            if is_seen_in_training:
                training_status = "Seen in Training"
                training_warning = None
            else:
                training_status = "Unseen Station (Monitoring Only)"
                training_warning = (
                    f"Station '{cand_name}' was not among the 13 monitoring stations in the model training dataset "
                    f"({', '.join(TRAINED_STATIONS)}). ML forecasting is disabled for this station to maintain "
                    "strict model integrity. Real-time sensor monitoring, historical trends, and measurement alerts remain fully active."
                )

            raw_pm25 = float(row["pm25"]) if pd.notna(row["pm25"]) else None

            # 3. Live Monitoring vs Offline Segregation
            if freshness_status in ["CURRENT_LIVE", "DELAYED_LIVE"]:
                current_pm25 = raw_pm25
                current_category = "Unavailable"
                if current_pm25 is not None:
                    for cat, meta in AQI_CATEGORIES.items():
                        low, high = meta["range_pm25"]
                        if low <= current_pm25 < high or (cat == "Severe" and current_pm25 >= low):
                            current_category = cat
                            break

                meas_out = {
                    "pm10": float(row["pm10"]) if pd.notna(row.get("pm10")) else None,
                    "no2": float(row["no2"]) if pd.notna(row.get("no2")) else None,
                    "so2": float(row["so2"]) if pd.notna(row.get("so2")) else None,
                    "co": float(row["co"]) if pd.notna(row.get("co")) else None,
                    "o3": float(row["o3"]) if pd.notna(row.get("o3")) else None,
                    "temperature": float(row["temperature"]) if pd.notna(row.get("temperature")) else None,
                    "relative_humidity": float(row["relative_humidity"]) if pd.notna(row.get("relative_humidity")) else None,
                    "wind_speed": float(row["wind_speed"]) if pd.notna(row.get("wind_speed")) else None,
                    "wind_direction": float(row["wind_direction"]) if pd.notna(row.get("wind_direction")) else None,
                }

                # 4. ML Forecasting (Strictly for the 13 trained stations only)
                if is_seen_in_training and idx in preds_1h_map:
                    p1_prob = probs_1h_map[idx]
                    p1_cat = str(preds_1h_map[idx])
                    p1_conf = round(float(np.max(p1_prob)), 4)
                    p1_dict = {cls_name: round(float(p1_prob[i]), 4) for i, cls_name in enumerate(classes_1h)}

                    p3_prob = probs_3h_map[idx]
                    p3_cat = str(preds_3h_map[idx])
                    p3_conf = round(float(np.max(p3_prob)), 4)
                    p3_dict = {cls_name: round(float(p3_prob[i]), 4) for i, cls_name in enumerate(classes_3h)}

                    trajectory = self._determine_trajectory(current_category, p1_cat, p3_cat)
                    forecast_status = "Operational Forecast" if is_current else f"Delayed Forecast ({age_hours:.1f}h lag)"
                    forecast_1h_out = {
                        "predicted_category": p1_cat,
                        "confidence": p1_conf,
                        "probabilities": p1_dict,
                    }
                    forecast_3h_out = {
                        "predicted_category": p3_cat,
                        "confidence": p3_conf,
                        "probabilities": p3_dict,
                    }
                else:
                    # Unseen station: No ML prediction, no fake confidence or probabilities
                    trajectory = "Live Monitoring (No Forecast)"
                    forecast_status = "Forecast unavailable — station not in training"
                    forecast_1h_out = {
                        "predicted_category": "Forecast unavailable — station not in training",
                        "confidence": None,
                        "probabilities": {},
                    }
                    forecast_3h_out = {
                        "predicted_category": "Forecast unavailable — station not in training",
                        "confidence": None,
                        "probabilities": {},
                    }
            else:
                # Severely stale (>48h) or Offline: Never present old data as live
                current_pm25 = None
                current_category = "Offline / Inactive Sensor" if is_severely_stale else "Offline / No Feed"
                meas_out = {}
                trajectory = "Station Offline"

                if is_seen_in_training:
                    forecast_status = "Unavailable (No Recent Feed)"
                    forecast_1h_out = {
                        "predicted_category": "Offline",
                        "confidence": None,
                        "probabilities": {},
                    }
                    forecast_3h_out = {
                        "predicted_category": "Offline",
                        "confidence": None,
                        "probabilities": {},
                    }
                else:
                    forecast_status = "Forecast unavailable — station not in training"
                    forecast_1h_out = {
                        "predicted_category": "Forecast unavailable — station not in training",
                        "confidence": None,
                        "probabilities": {},
                    }
                    forecast_3h_out = {
                        "predicted_category": "Forecast unavailable — station not in training",
                        "confidence": None,
                        "probabilities": {},
                    }

            station_forecasts.append({
                "station_name": st_name,
                "candidate_station": cand_name,
                "region": row.get("region", "Delhi"),
                "latitude": float(row.get("latitude", 0.0)),
                "longitude": float(row.get("longitude", 0.0)),
                "last_observed_at_ist": obs_ist,
                "observation_age_hours": age_hours,
                "observation_age_text": age_text,
                "freshness_status": freshness_status,
                "freshness_label": freshness_label,
                "is_current_live": is_current,
                "is_delayed": is_delayed,
                "is_severely_stale": is_severely_stale,
                "is_seen_in_training": is_seen_in_training,
                "training_status": training_status,
                "training_warning": training_warning,
                "forecast_status": forecast_status,
                "current_pm25": current_pm25,
                "current_aqi_category": current_category,
                "forecast_1h": forecast_1h_out,
                "forecast_3h": forecast_3h_out,
                "trajectory": trajectory,
                "historical_context_pm25": raw_pm25 if not (is_current or is_delayed) else None,
                "measurements": meas_out,
            })

        forecasted_count = sum(1 for s in station_forecasts if s["is_seen_in_training"])
        monitoring_count = len(station_forecasts) - forecasted_count

        output = {
            "generated_at": now_str,
            "model_1h": "GradientBoostingClassifier (45-feature)",
            "model_3h": "GradientBoostingClassifier (45-feature)",
            "station_count": len(station_forecasts),
            "forecasted_station_count": forecasted_count,
            "monitoring_only_count": monitoring_count,
            "stations": station_forecasts
        }

        # Save to disk
        LATEST_PREDICTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(LATEST_PREDICTIONS_JSON, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        logger.info(
            f"Processed {len(station_forecasts)} stations: "
            f"{forecasted_count} forecasted with ML, "
            f"{monitoring_count} monitoring-only -> {LATEST_PREDICTIONS_JSON.name}"
        )
        return output
