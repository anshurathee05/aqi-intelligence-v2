"""
Configuration and settings for the AQI Rebuild platform.
Manages file paths, OpenAQ API credentials, AQI breakpoints, and dashboard styling.
"""

from pathlib import Path
import os
import re

# Base directory for the rebuild project
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Paths to external assets (existing models, reports, and raw data - READ ONLY)
EXTERNAL_REBUILD_DIR = PROJECT_ROOT / "datanew" / "aqi_rebuild"
EXTERNAL_MODELS_DIR = EXTERNAL_REBUILD_DIR / "models"
EXTERNAL_RESULTS_DIR = EXTERNAL_REBUILD_DIR / "results"
EXTERNAL_DATA_DIR = EXTERNAL_REBUILD_DIR / "data"

# Internal rebuild directories
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
PIPELINE_DIR = BASE_DIR / "pipeline"
SERVICES_DIR = BASE_DIR / "services"
DASHBOARD_DIR = BASE_DIR / "dashboard"
TESTS_DIR = BASE_DIR / "tests"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File paths
STATIONS_JSON = CONFIG_DIR / "stations.json"
LIVE_READINGS_BUFFER = DATA_DIR / "live_readings_buffer.csv"
LATEST_PREDICTIONS_JSON = DATA_DIR / "latest_predictions.json"
ALERT_HISTORY_JSON = DATA_DIR / "alert_history.json"

# Models (45-feature pipeline)
BEST_MODEL_1H_PATH = EXTERNAL_MODELS_DIR / "best_model_1h.joblib"
BEST_MODEL_3H_PATH = EXTERNAL_MODELS_DIR / "best_model_3h.joblib"

# Evaluation results & artifacts
FINAL_METRICS_PATH = EXTERNAL_RESULTS_DIR / "final_model_metrics.csv"
SIX_MODEL_METRICS_PATH = EXTERNAL_RESULTS_DIR / "six_model_comparison.csv"
FINAL_COST_PATH = EXTERNAL_RESULTS_DIR / "final_cost_comparison.csv"
CONFUSION_GB_1H = EXTERNAL_RESULTS_DIR / "gradient_boosting_1h_confusion.png"
CONFUSION_GB_3H = EXTERNAL_RESULTS_DIR / "gradient_boosting_3h_confusion.png"
CONFUSION_RF_1H = EXTERNAL_RESULTS_DIR / "random_forest_1h_confusion.png"
CONFUSION_RF_3H = EXTERNAL_RESULTS_DIR / "random_forest_3h_confusion.png"

# Historical reference dataset for buffer initialization & lag computation
REFERENCE_DATA_PATH = EXTERNAL_DATA_DIR / "feature_engineered_dataset.csv"
RECONSTRUCTED_HOURLY_PATH = EXTERNAL_DATA_DIR / "reconstructed_hourly_data.csv"

# Freshness thresholds (Hours)
LIVE_FRESHNESS_HOURS = 3.0       # Max delay for a reading to be classified as current real-time
SEVERELY_STALE_HOURS = 48.0      # Inactivity threshold: sensors older than this are discarded as stale

# Monitoring stations present during model training (One-Hot Encoded in ML pipeline)
TRAINED_STATIONS = [
    'Alipur', 'Anand Vihar', 'Bawana', 'Dwarka', 'ITO', 'Mandir Marg',
    'Mundka', 'Patparganj', 'Punjabi Bagh', 'Pusa', 'RK Puram', 'Rohini', 'Vivek Vihar'
]

# OpenAQ API configuration
def get_openaq_api_key() -> str:
    """Retrieve the OpenAQ API key from environment variable or .env file."""
    api_key = os.getenv("OPENAQ_API_KEY", "")
    if not api_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    match = re.search(r"OPENAQ_API_KEY\s*=\s*([^\r\n]+)", f.read())
                    if match:
                        api_key = match.group(1).strip()
            except Exception:
                pass
    return api_key

OPENAQ_API_KEY = get_openaq_api_key()
OPENAQ_BASE_URL = os.getenv("OPENAQ_BASE_URL", "https://api.openaq.org/v3")

# Timezone
TIMEZONE = "Asia/Kolkata"

# The 45 exact feature names expected by the 1h and 3h models
MODEL_FEATURE_NAMES = [
    'co', 'no', 'no2', 'nox', 'o3', 'pm10', 'pm25', 'relative_humidity', 'so2', 'temperature',
    'wind_speed', 'wind_direction_sin', 'wind_direction_cos',
    'PM25_Missing', 'co_missing', 'no_missing', 'no2_missing', 'nox_missing', 'o3_missing',
    'pm10_missing', 'relative_humidity_missing', 'so2_missing', 'temperature_missing',
    'wind_direction_missing', 'wind_speed_missing',
    'hour', 'day', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'pm25_lag_1', 'pm25_lag_3', 'pm25_lag_6', 'pm25_lag_12', 'pm25_lag_24',
    'pm25_rolling_mean_6h', 'pm25_rolling_mean_12h', 'pm25_rolling_mean_24h',
    'pm25_rolling_std_24h', 'pm25_24h_avg', 'candidate_station'
]

# AQI Categories, CPCB 24h PM2.5 Breakpoints (ug/m3), Colors, and Health Advisories
AQI_CATEGORIES = {
    "Good": {
        "range_pm25": (0, 30),
        "color": "#00E400",
        "bg_color": "rgba(0, 228, 0, 0.15)",
        "border_color": "#00A000",
        "severity": 1,
        "description": "Minimal health impact. Air quality is considered satisfactory.",
        "advisory": "Air quality is good. Perfect condition for outdoor activities and exercise."
    },
    "Satisfactory": {
        "range_pm25": (30, 60),
        "color": "#92D050",
        "bg_color": "rgba(146, 208, 80, 0.15)",
        "border_color": "#6E9C3E",
        "severity": 2,
        "description": "Minor breathing discomfort to sensitive people.",
        "advisory": "Air quality is acceptable. Sensitive individuals should monitor mild symptoms."
    },
    "Moderate": {
        "range_pm25": (60, 90),
        "color": "#FFFF00",
        "bg_color": "rgba(255, 255, 0, 0.18)",
        "border_color": "#C4C400",
        "severity": 3,
        "description": "Breathing discomfort to people with lungs, asthma and heart diseases.",
        "advisory": "Sensitive groups (children, elderly, asthmatics) should limit prolonged outdoor exertion."
    },
    "Poor": {
        "range_pm25": (90, 120),
        "color": "#FF7E00",
        "bg_color": "rgba(255, 126, 0, 0.2)",
        "border_color": "#D06000",
        "severity": 4,
        "description": "Breathing discomfort to most people on prolonged exposure.",
        "advisory": "Everyone should reduce heavy outdoor exertion. Wear N95 masks when outside."
    },
    "Very Poor": {
        "range_pm25": (120, 250),
        "color": "#FF0000",
        "bg_color": "rgba(255, 0, 0, 0.22)",
        "border_color": "#B80000",
        "severity": 5,
        "description": "Respiratory illness on prolonged exposure. Pronounced effects on lung/heart patients.",
        "advisory": "Avoid strenuous outdoor activities. Keep windows closed and operate indoor air purifiers."
    },
    "Severe": {
        "range_pm25": (250, 1000),
        "color": "#7E0023",
        "bg_color": "rgba(126, 0, 35, 0.28)",
        "border_color": "#520017",
        "severity": 6,
        "description": "Affects healthy people and seriously impacts those with existing diseases.",
        "advisory": "Emergency warning: remain indoors. Minimize all outdoor physical activity."
    }
}

ORDERED_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

__all__ = [
    "BASE_DIR",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "PIPELINE_DIR",
    "SERVICES_DIR",
    "DASHBOARD_DIR",
    "TESTS_DIR",
    "STATIONS_JSON",
    "LIVE_READINGS_BUFFER",
    "LATEST_PREDICTIONS_JSON",
    "ALERT_HISTORY_JSON",
    "BEST_MODEL_1H_PATH",
    "BEST_MODEL_3H_PATH",
    "FINAL_METRICS_PATH",
    "SIX_MODEL_METRICS_PATH",
    "FINAL_COST_PATH",
    "CONFUSION_GB_1H",
    "CONFUSION_GB_3H",
    "CONFUSION_RF_1H",
    "CONFUSION_RF_3H",
    "REFERENCE_DATA_PATH",
    "RECONSTRUCTED_HOURLY_PATH",
    "LIVE_FRESHNESS_HOURS",
    "SEVERELY_STALE_HOURS",
    "TRAINED_STATIONS",
    "OPENAQ_API_KEY",
    "OPENAQ_BASE_URL",
    "TIMEZONE",
    "MODEL_FEATURE_NAMES",
    "AQI_CATEGORIES",
    "ORDERED_CATEGORIES",
]
