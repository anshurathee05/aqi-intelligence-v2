"""
Pollution Alert & Early Warning Service.
Evaluates live sensor readings and multi-hour forecasts to trigger alerts:
- Severe/Hazardous AQI Threshold Alerts
- Rapid Pollution Spike Detection (Rate of Rise)
- Station-Level Deterioration Forecast Alerts
- Vulnerable Population Health Advisories (Elderly, Asthmatic, Children)
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from aqi_rebuild.config.settings import (
    ALERT_HISTORY_JSON,
    AQI_CATEGORIES,
    ORDERED_CATEGORIES,
    TRAINED_STATIONS,
)

logger = logging.getLogger("aqi_rebuild.alert_service")


class PollutionAlertService:
    """Evaluates sensor metrics and model forecasts to generate real-time alerts."""

    RAPID_SPIKE_THRESHOLD = 35.0  # PM2.5 rise of 35 ug/m3 in an hour is an acute spike

    def __init__(self):
        self.history_path = ALERT_HISTORY_JSON

    def evaluate_station_alerts(self, station_forecast: Dict[str, Any], previous_pm25: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Analyzes a single station's current and forecasted state to generate actionable alerts.
        """
        alerts = []
        name = station_forecast["candidate_station"]
        pm25 = station_forecast.get("current_pm25")
        current_cat = station_forecast.get("current_aqi_category", "Unavailable")
        f1h = station_forecast.get("forecast_1h", {})
        f3h = station_forecast.get("forecast_3h", {})
        pred_1h = f1h.get("predicted_category", "Unavailable")
        pred_3h = f3h.get("predicted_category", "Unavailable")
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Severe / Emergency Alert
        if pm25 is not None and pm25 >= 250:
            alerts.append({
                "id": f"{name}_severe_{now_ts}",
                "station": name,
                "type": "EMERGENCY_SEVERE",
                "severity": "CRITICAL",
                "level_color": "#7E0023",
                "title": f"🚨 Severe Air Pollution Warning at {name}",
                "message": f"PM2.5 concentration has reached {pm25:.1f} µg/m³ (Severe category).",
                "impact": "Hazardous to all citizens. Triggers acute respiratory symptoms even in healthy individuals.",
                "action": "Avoid outdoor activities completely. Keep windows sealed and run indoor air purifiers on high.",
                "vulnerable_groups": "High risk for children, elderly, and cardiac/respiratory patients.",
                "timestamp": now_ts
            })
        elif pm25 is not None and pm25 >= 120:
            alerts.append({
                "id": f"{name}_very_poor_{now_ts}",
                "station": name,
                "type": "HEALTH_WARNING",
                "severity": "HIGH",
                "level_color": "#FF0000",
                "title": f"⚠️ Very Poor Air Quality at {name}",
                "message": f"PM2.5 is at {pm25:.1f} µg/m³ (Very Poor category).",
                "impact": "Can cause respiratory illness on prolonged exposure.",
                "action": "Wear N95/FFP2 masks outdoors. Minimize strenuous aerobic exercise.",
                "vulnerable_groups": "Asthma and heart patients should remain indoors.",
                "timestamp": now_ts
            })

        # 2. Rapid Pollution Spike Alert
        if pm25 is not None and previous_pm25 is not None:
            delta = pm25 - previous_pm25
            if delta >= self.RAPID_SPIKE_THRESHOLD:
                alerts.append({
                    "id": f"{name}_spike_{now_ts}",
                    "station": name,
                    "type": "RAPID_SPIKE",
                    "severity": "WARNING",
                    "level_color": "#FF7E00",
                    "title": f"⚡ Rapid Pollution Surge Detected at {name}",
                    "message": f"PM2.5 surged by +{delta:.1f} µg/m³ over the past hour (from {previous_pm25:.1f} to {pm25:.1f} µg/m³).",
                    "impact": "Sudden accumulation of particulate matter due to calm winds or localized emissions.",
                    "action": "Immediate curtailment of nearby burning/dust generation; alert local DPCC enforcement.",
                    "vulnerable_groups": "Acute trigger for asthmatics.",
                    "timestamp": now_ts
                })

        # 3. Forecast Deterioration Early Warning (Strictly for stations with verified ML forecasts)
        is_seen = station_forecast.get("is_seen_in_training")
        if is_seen is None:
            is_seen = name in TRAINED_STATIONS

        # Only evaluate forecast-based alerts for training-covered stations with valid forecasts
        if is_seen and pred_1h not in ["Forecast unavailable — station not in training", "Offline", "Unavailable"]:
            curr_sev = AQI_CATEGORIES.get(current_cat, {}).get("severity", 0)
            h1_sev = AQI_CATEGORIES.get(pred_1h, {}).get("severity", 0)
            h3_sev = AQI_CATEGORIES.get(pred_3h, {}).get("severity", 0)

            if (h1_sev > curr_sev and h1_sev >= 4) or (h3_sev > curr_sev and h3_sev >= 4):
                worst_pred = pred_1h if h1_sev >= h3_sev else pred_3h
                worst_horizon = "1 hour" if h1_sev >= h3_sev else "3 hours"
                raw_conf = f1h.get("confidence") if h1_sev >= h3_sev else f3h.get("confidence")
                conf = float(raw_conf) if raw_conf is not None else 0.0

                alerts.append({
                    "id": f"{name}_deterioration_{now_ts}",
                    "station": name,
                    "type": "EARLY_WARNING_DETERIORATION",
                    "severity": "ADVISORY",
                    "level_color": "#FF7E00" if h3_sev < 6 else "#7E0023",
                    "title": f"🔮 Early Warning: Air Quality Projected to Deteriorate at {name}",
                    "message": f"ML model forecasts transition from '{current_cat}' to '{worst_pred}' within {worst_horizon} (Confidence: {conf*100:.1f}%).",
                    "impact": "Impending risk of severe exposure for commuters, school children, and outdoor workers.",
                    "action": "Reschedule outdoor activities before the deterioration window sets in.",
                    "vulnerable_groups": "Proactive protection advised before pollution peak.",
                    "timestamp": now_ts
                })

        return alerts

    def evaluate_all_stations(self, latest_predictions: Dict[str, Any], previous_readings_map: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Runs comprehensive alert checks across all stations in the prediction batch.
        """
        all_alerts = []
        stations = latest_predictions.get("stations", [])
        prev_map = previous_readings_map or {}

        for st in stations:
            cand = st.get("candidate_station")
            prev_val = prev_map.get(cand)
            st_alerts = self.evaluate_station_alerts(st, previous_pm25=prev_val)
            all_alerts.extend(st_alerts)

        # Persist alert history
        self._record_alerts(all_alerts)
        return all_alerts

    def _record_alerts(self, alerts: List[Dict[str, Any]]):
        """Appends generated alerts to the local alert history JSON."""
        if not alerts:
            return
        existing = []
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        # Prepend new alerts and keep most recent 200
        combined = alerts + existing
        deduped = []
        seen_ids = set()
        for a in combined:
            aid = a.get("id")
            if aid not in seen_ids:
                seen_ids.add(aid)
                deduped.append(a)

        deduped = deduped[:200]
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2)

    def load_active_alerts(self) -> List[Dict[str, Any]]:
        """Loads recent alert history."""
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
