"""
Model Leaderboard & Diagnostics Service.
Provides comprehensive comparison of models:
- 1-Hour & 3-Hour Leaderboards (Accuracy, Precision, Recall, Macro F1, Weighted F1)
- Feature Importance ranking & grouping (Memory vs Weather vs Pollutants vs Cycles)
- Confusion Matrix references
- Cost-Sensitive / Asymmetric Penalty Comparison
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import joblib
import numpy as np
import pandas as pd

from aqi_rebuild.config.settings import (
    FINAL_METRICS_PATH,
    SIX_MODEL_METRICS_PATH,
    FINAL_COST_PATH,
    BEST_MODEL_1H_PATH,
    BEST_MODEL_3H_PATH,
    CONFUSION_GB_1H,
    CONFUSION_GB_3H,
    CONFUSION_RF_1H,
    CONFUSION_RF_3H,
)

logger = logging.getLogger("aqi_rebuild.leaderboard_service")


class LeaderboardService:
    """Service providing model evaluation metrics, feature importances, and diagnostic plots."""

    def __init__(self):
        self._metrics_cache = None
        self._feature_importance_cache = {}

    def get_model_leaderboard(self) -> pd.DataFrame:
        """
        Loads the final evaluated metrics for all 6 models across 1h and 3h horizons.
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        metrics_file = FINAL_METRICS_PATH if FINAL_METRICS_PATH.exists() else SIX_MODEL_METRICS_PATH
        if not metrics_file.exists():
            logger.warning(f"Metrics file not found at {metrics_file}")
            return pd.DataFrame()

        df = pd.read_csv(metrics_file)
        # Format numeric columns as percentages or rounded floats
        numeric_cols = ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        self._metrics_cache = df
        return df

    def get_leaderboard_by_horizon(self, horizon: str = "1-hour") -> pd.DataFrame:
        """Returns models sorted by Macro F1 for a given forecast horizon."""
        df = self.get_model_leaderboard()
        if df.empty:
            return df
        filtered = df[df["Horizon"].str.lower() == horizon.lower()].copy()
        return filtered.sort_values("Macro F1", ascending=False).reset_index(drop=True)

    def get_cost_comparison(self) -> pd.DataFrame:
        """Loads cost-sensitive misclassification analysis if available."""
        if FINAL_COST_PATH.exists():
            try:
                return pd.read_csv(FINAL_COST_PATH)
            except Exception:
                pass
        return pd.DataFrame()

    def get_feature_importances(self, horizon: str = "1h", top_n: int = 20) -> pd.DataFrame:
        """
        Extracts feature importances directly from the trained Gradient Boosting pipeline.
        Groups features by cognitive domain (Memory/Lags, Meteorology, Co-Pollutants, Cycles, Missingness).
        """
        if horizon in self._feature_importance_cache:
            return self._feature_importance_cache[horizon].head(top_n)

        model_path = BEST_MODEL_1H_PATH if horizon == "1h" else BEST_MODEL_3H_PATH
        if not model_path.exists():
            return pd.DataFrame()

        try:
            pipeline = joblib.load(model_path)
            preprocessor = pipeline.named_steps.get("preprocessor")
            classifier = pipeline.named_steps.get("classifier")

            if not hasattr(classifier, "feature_importances_"):
                return pd.DataFrame()

            feature_names = preprocessor.get_feature_names_out()
            importances = classifier.feature_importances_

            fi_df = pd.DataFrame({
                "raw_feature": feature_names,
                "importance": importances
            })

            # Clean feature names (strip transformer prefix 'numeric__', 'categorical__')
            fi_df["feature"] = fi_df["raw_feature"].apply(
                lambda x: x.replace("numeric__", "").replace("categorical__", "")
            )

            # Assign feature domain category
            def assign_domain(name: str) -> str:
                n = name.lower()
                if "lag" in n or "rolling" in n or "24h_avg" in n:
                    return "PM2.5 Temporal Memory"
                elif any(w in n for w in ["wind", "temp", "humidity"]):
                    return "Meteorological"
                elif any(p in n for p in ["co", "no", "no2", "nox", "o3", "pm10", "so2"]):
                    return "Co-Pollutants"
                elif any(t in n for t in ["hour", "month", "day", "weekend"]):
                    return "Diurnal & Seasonal Cycles"
                elif "missing" in n:
                    return "Sensor Quality / Missingness"
                elif "candidate_station" in n:
                    return "Spatial / Station Location"
                return "Other"

            fi_df["domain"] = fi_df["feature"].apply(assign_domain)
            fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
            fi_df["relative_pct"] = (fi_df["importance"] / fi_df["importance"].sum()) * 100
            fi_df["cumulative_pct"] = fi_df["relative_pct"].cumsum()

            self._feature_importance_cache[horizon] = fi_df
            return fi_df.head(top_n)
        except Exception as e:
            logger.error(f"Error computing feature importances: {e}")
            return pd.DataFrame()

    def get_confusion_matrix_path(self, model_name: str = "gradient_boosting", horizon: str = "1h") -> Optional[Path]:
        """Returns the file path for the requested confusion matrix image."""
        m = model_name.lower()
        if "gradient" in m or "gb" in m:
            return CONFUSION_GB_1H if horizon == "1h" else CONFUSION_GB_3H
        elif "forest" in m or "rf" in m:
            return CONFUSION_RF_1H if horizon == "1h" else CONFUSION_RF_3H
        return None
