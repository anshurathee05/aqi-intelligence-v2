"""
Feature Engineering Engine for AQI Rebuild.
Transforms raw timeseries data into the EXACT 45 features required by the 1-hour and 3-hour ML models.
Handles missingness indicators, circular conversions, temporal cycles, lag variables, and rolling statistics.
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd

from aqi_rebuild.config.settings import MODEL_FEATURE_NAMES, TIMEZONE

logger = logging.getLogger("aqi_rebuild.feature_engineer")


class AQIFeatureEngineer:
    """Engineers the exact 45 input features for the AQI ML pipeline."""

    PREDICTOR_VARS = [
        "pm25", "co", "no", "no2", "nox", "o3", "pm10",
        "relative_humidity", "so2", "temperature", "wind_direction", "wind_speed"
    ]

    LAG_PERIODS = [1, 3, 6, 12, 24]

    def __init__(self):
        self.feature_names = MODEL_FEATURE_NAMES

    def transform_timeseries(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a continuous timeseries buffer (sorted by station_name and observed_at)
        into a DataFrame containing all engineered features.
        """
        if timeseries_df.empty:
            return pd.DataFrame(columns=self.feature_names)

        df = timeseries_df.copy()

        # 1. Parse timestamps to IST
        if "observed_at_ist" in df.columns:
            df["dt_ist"] = pd.to_datetime(df["observed_at_ist"])
        elif "observed_at" in df.columns:
            df["observed_at_dt"] = pd.to_datetime(df["observed_at"], utc=True)
            df["dt_ist"] = df["observed_at_dt"].dt.tz_convert(TIMEZONE)
        else:
            raise ValueError("Dataframe must contain 'observed_at_ist' or 'observed_at'.")

        # Ensure sorted chronologically per station
        df = df.sort_values(["station_name", "dt_ist"]).reset_index(drop=True)

        # 2. Missingness indicators
        # Note: PM25 is capitalized 'PM25_Missing' to match the trained pipeline exactly
        df["PM25_Missing"] = df["pm25"].isna().astype(int)
        for col in ["co", "no", "no2", "nox", "o3", "pm10", "relative_humidity", "so2", "temperature", "wind_direction", "wind_speed"]:
            if col in df.columns:
                df[f"{col}_missing"] = df[col].isna().astype(int)
            else:
                df[f"{col}_missing"] = 1
                df[col] = np.nan

        # 3. Circular wind direction
        if "wind_direction" in df.columns:
            wind_rad = np.deg2rad(df["wind_direction"] % 360)
            df["wind_direction_sin"] = np.sin(wind_rad)
            df["wind_direction_cos"] = np.cos(wind_rad)
        else:
            df["wind_direction_sin"] = np.nan
            df["wind_direction_cos"] = np.nan

        # 4. Temporal and cyclical calendar features
        df["hour"] = df["dt_ist"].dt.hour
        df["day"] = df["dt_ist"].dt.day
        df["day_of_week"] = df["dt_ist"].dt.dayofweek
        df["month"] = df["dt_ist"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

        # 5. Historical PM2.5 Lag features (station-wise)
        grouped_pm25 = df.groupby("station_name")["pm25"]
        for lag in self.LAG_PERIODS:
            df[f"pm25_lag_{lag}"] = grouped_pm25.shift(lag)

        # 6. PM2.5 Rolling features (station-wise)
        # 6h, 12h, 24h rolling means of prior observations (shift(1))
        df["pm25_rolling_mean_6h"] = grouped_pm25.transform(
            lambda s: s.shift(1).rolling(6, min_periods=4).mean()
        )
        df["pm25_rolling_mean_12h"] = grouped_pm25.transform(
            lambda s: s.shift(1).rolling(12, min_periods=8).mean()
        )
        df["pm25_rolling_mean_24h"] = grouped_pm25.transform(
            lambda s: s.shift(1).rolling(24, min_periods=16).mean()
        )
        df["pm25_rolling_std_24h"] = grouped_pm25.transform(
            lambda s: s.shift(1).rolling(24, min_periods=16).std()
        )

        # Trailing 24-hour PM2.5 average including the current reading
        df["pm25_24h_avg"] = grouped_pm25.transform(
            lambda s: s.rolling(24, min_periods=16).mean()
        )

        # Fallbacks for sparse periods: if rolling mean or lag is NaN due to gaps,
        # backfill with recent available pm25 or current reading
        for lag in self.LAG_PERIODS:
            df[f"pm25_lag_{lag}"] = df[f"pm25_lag_{lag}"].fillna(df["pm25"])
        df["pm25_rolling_mean_6h"] = df["pm25_rolling_mean_6h"].fillna(df["pm25"])
        df["pm25_rolling_mean_12h"] = df["pm25_rolling_mean_12h"].fillna(df["pm25"])
        df["pm25_rolling_mean_24h"] = df["pm25_rolling_mean_24h"].fillna(df["pm25"])
        df["pm25_rolling_std_24h"] = df["pm25_rolling_std_24h"].fillna(0.0)
        df["pm25_24h_avg"] = df["pm25_24h_avg"].fillna(df["pm25"])

        # 7. Candidate station column (categorical)
        if "candidate_station" not in df.columns:
            # Map from station_name if candidate_station is missing
            df["candidate_station"] = df["station_name"].apply(self._clean_candidate_name)

        return df

    def _clean_candidate_name(self, name: str) -> str:
        """Standardizes station name to candidate station string."""
        s = str(name).split(",")[0].strip()
        if "Dwarka" in s:
            return "Dwarka"
        if "R K Puram" in s or "RK Puram" in s:
            return "RK Puram"
        return s

    def extract_model_features(self, df_with_features: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts only the exact 45 model features in the required order.
        Guarantees that input shape and column names match what the scikit-learn pipeline expects.
        """
        missing_cols = [c for c in self.feature_names if c not in df_with_features.columns]
        if missing_cols:
            raise KeyError(f"Missing required model features: {missing_cols}")

        return df_with_features[self.feature_names].copy()

    def get_latest_station_features(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes timeseries and returns the single latest row for each station,
        with metadata and all 45 model features.
        """
        feat_df = self.transform_timeseries(timeseries_df)
        latest_rows = (
            feat_df.sort_values(["station_name", "dt_ist"])
            .groupby("station_name", as_index=False)
            .last()
        )
        return latest_rows
