"""
Preprocessor and Timeseries Buffer Manager.
Maintains continuous hourly records per station to support lag & rolling feature generation.
Merges live OpenAQ readings with verified historical buffer without modifying raw data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from aqi_rebuild.config.settings import (
    LIVE_READINGS_BUFFER,
    REFERENCE_DATA_PATH,
    RECONSTRUCTED_HOURLY_PATH,
    TIMEZONE,
)

logger = logging.getLogger("aqi_rebuild.preprocessor")


class TimeseriesPreprocessor:
    """Manages the historical timeseries buffer and ingestion of live readings."""

    CORE_COLUMNS = [
        "observed_at",
        "observed_at_ist",
        "location_id",
        "station_name",
        "candidate_station",
        "region",
        "latitude",
        "longitude",
        "co",
        "no",
        "no2",
        "nox",
        "o3",
        "pm10",
        "pm25",
        "relative_humidity",
        "so2",
        "temperature",
        "wind_direction",
        "wind_speed",
        "data_source",
        "is_live",
    ]

    NUMERIC_POLLUTANTS = [
        "co", "no", "no2", "nox", "o3", "pm10", "pm25",
        "relative_humidity", "so2", "temperature", "wind_direction", "wind_speed"
    ]

    def __init__(self, buffer_path: Optional[Path] = None):
        self.buffer_path = buffer_path or LIVE_READINGS_BUFFER
        self.ensure_buffer_initialized()

    def ensure_buffer_initialized(self, lookback_hours: int = 168):
        """
        Initializes the buffer from the verified reference dataset if it doesn't already exist.
        Keeps the trailing lookback_hours (default 7 days) of real data for each station.
        """
        if self.buffer_path.exists() and self.buffer_path.stat().st_size > 100:
            return

        logger.info(f"Initializing timeseries buffer at {self.buffer_path} from reference data...")
        ref_df = None

        if RECONSTRUCTED_HOURLY_PATH.exists():
            ref_df = pd.read_csv(RECONSTRUCTED_HOURLY_PATH)
        elif REFERENCE_DATA_PATH.exists():
            ref_df = pd.read_csv(REFERENCE_DATA_PATH)

        if ref_df is not None and not ref_df.empty:
            # Standardize columns
            if "observed_at_ist" not in ref_df.columns and "observed_at" in ref_df.columns:
                ref_df["observed_at"] = pd.to_datetime(ref_df["observed_at"], utc=True)
                ref_df["observed_at_ist"] = ref_df["observed_at"].dt.tz_convert(TIMEZONE)
            else:
                ref_df["observed_at"] = pd.to_datetime(ref_df["observed_at"], utc=True)
                ref_df["observed_at_ist"] = pd.to_datetime(ref_df["observed_at_ist"])

            # Filter to available core columns
            available_cols = [c for c in self.CORE_COLUMNS if c in ref_df.columns]
            trimmed_df = ref_df[available_cols].copy()

            # Keep the last lookback_hours per station
            trimmed_df = (
                trimmed_df.sort_values(["station_name", "observed_at"])
                .groupby("station_name", as_index=False)
                .tail(lookback_hours)
            )

            trimmed_df["data_source"] = "historical_context"
            trimmed_df["is_live"] = False

            trimmed_df["observed_at"] = trimmed_df["observed_at"].astype(str)
            trimmed_df["observed_at_ist"] = trimmed_df["observed_at_ist"].astype(str)
            trimmed_df.to_csv(self.buffer_path, index=False)
            logger.info(f"Initialized timeseries buffer with {len(trimmed_df)} historical records.")
        else:
            logger.warning("No reference data found to initialize buffer. Created empty buffer.")
            empty_df = pd.DataFrame(columns=self.CORE_COLUMNS)
            empty_df.to_csv(self.buffer_path, index=False)

    def load_buffer(self) -> pd.DataFrame:
        """Loads and formats the current buffer."""
        self.ensure_buffer_initialized()
        df = pd.read_csv(self.buffer_path)
        if not df.empty:
            if "data_source" not in df.columns:
                df["data_source"] = "historical_context"
            if "is_live" not in df.columns:
                df["is_live"] = False
            df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, errors="coerce")
            df["observed_at_ist"] = pd.to_datetime(df["observed_at_ist"], errors="coerce")
            for col in self.NUMERIC_POLLUTANTS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def validate_reading_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizes an incoming reading dictionary, ensuring plausible physical boundaries."""
        sanitized = dict(record)
        # Concentration lower bounds (cannot be negative)
        for col in ["pm25", "pm10", "co", "no", "no2", "nox", "o3", "so2", "relative_humidity", "wind_speed"]:
            val = sanitized.get(col)
            if val is not None and not pd.isna(val):
                try:
                    fval = float(val)
                    if fval < 0:
                        sanitized[col] = 0.0
                    else:
                        sanitized[col] = fval
                except (ValueError, TypeError):
                    sanitized[col] = np.nan

        # Wind direction modulo 360
        wd = sanitized.get("wind_direction")
        if wd is not None and not pd.isna(wd):
            try:
                sanitized["wind_direction"] = float(wd) % 360.0
            except (ValueError, TypeError):
                sanitized["wind_direction"] = np.nan

        return sanitized

    def ingest_live_readings(self, live_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Merges new live readings into the buffer, deduplicating on (station_name, observed_at).
        Returns the updated complete buffer dataframe sorted chronologically.
        """
        buffer_df = self.load_buffer()

        if not live_records:
            logger.info("No new live records provided for ingestion.")
            return buffer_df

        valid_records = []
        for r in live_records:
            san = self.validate_reading_record(r)
            if "data_source" not in san:
                san["data_source"] = "openaq_live"
            if "is_live" not in san:
                san["is_live"] = True
            valid_records.append(san)
        new_df = pd.DataFrame(valid_records)

        # Standardize timestamps in new_df
        new_df["observed_at"] = pd.to_datetime(new_df["observed_at"], utc=True, errors="coerce")
        new_df["observed_at_ist"] = pd.to_datetime(new_df["observed_at_ist"], errors="coerce")

        for col in self.NUMERIC_POLLUTANTS:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

        if buffer_df.empty:
            combined_df = new_df
        else:
            # Concatenate and deduplicate by (station_name, observed_at), keeping last
            # This ensures fresh verified live readings update existing keys with is_live=True
            combined_df = pd.concat([buffer_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["station_name", "observed_at"], keep="last")
            logger.info(f"Synchronized {len(new_df)} live observations into buffer (total buffer rows: {len(combined_df)}).")

        # Sort and maintain reasonable buffer size (keep last 300 hours per station)
        combined_df = (
            combined_df.sort_values(["station_name", "observed_at"])
            .groupby("station_name", as_index=False)
            .tail(300)
        )

        # Save back to CSV
        save_df = combined_df.copy()
        save_df["observed_at"] = save_df["observed_at"].astype(str)
        save_df["observed_at_ist"] = save_df["observed_at_ist"].astype(str)
        save_df.to_csv(self.buffer_path, index=False)

        return combined_df
