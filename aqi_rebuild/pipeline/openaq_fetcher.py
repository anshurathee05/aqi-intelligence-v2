"""
OpenAQ Live Data Fetcher.
Fetches verified latest measurements directly from OpenAQ API v3 for Delhi monitoring stations.
Handles rate limits, network timeouts, parameter resolution, and timestamp standardization to IST.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import requests

from aqi_rebuild.config.settings import (
    OPENAQ_API_KEY,
    OPENAQ_BASE_URL,
    STATIONS_JSON,
    TIMEZONE,
    LIVE_FRESHNESS_HOURS,
    SEVERELY_STALE_HOURS,
)

logger = logging.getLogger("aqi_rebuild.openaq_fetcher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Parameter name mapping: OpenAQ parameter name -> standard dataframe column name
OPENAQ_PARAM_MAP = {
    "pm25": "pm25",
    "pm10": "pm10",
    "co": "co",
    "no": "no",
    "no2": "no2",
    "nox": "nox",
    "o3": "o3",
    "so2": "so2",
    "relativehumidity": "relative_humidity",
    "relative_humidity": "relative_humidity",
    "temperature": "temperature",
    "wind_speed": "wind_speed",
    "wind_direction": "wind_direction",
}


class OpenAQFetcher:
    """Client for fetching live air quality readings from OpenAQ API v3."""

    MAX_STALENESS_HOURS = 48  # Discard readings older than 48 hours to prevent stale readings as live data

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAQ_API_KEY
        self.base_url = OPENAQ_BASE_URL.rstrip("/")
        self.headers = {"X-API-Key": self.api_key} if self.api_key else {}
        self.stations = self._load_stations()
        self.sensor_to_param = self._build_sensor_lookup()

    def _load_stations(self) -> List[Dict[str, Any]]:
        """Load station configurations from stations.json."""
        if not STATIONS_JSON.exists():
            logger.error(f"Stations configuration not found at {STATIONS_JSON}")
            return []
        with open(STATIONS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("stations", [])

    def _build_sensor_lookup(self) -> Dict[int, tuple]:
        """
        Build lookup map: sensor_id -> (location_id, param_col_name).
        Checks both active_sensors and all_sensors across historical & modern catalogs.
        """
        lookup = {}
        for st in self.stations:
            loc_id = st["location_id"]
            # Active sensors map
            for param, sid in st.get("active_sensors", {}).items():
                col_name = OPENAQ_PARAM_MAP.get(param.lower(), param.lower())
                lookup[sid] = (loc_id, col_name)
            # All sensors map (including secondary / legacy IDs)
            for param, sids in st.get("all_sensors", {}).items():
                col_name = OPENAQ_PARAM_MAP.get(param.lower(), param.lower())
                if isinstance(sids, list):
                    for sid in sids:
                        lookup[sid] = (loc_id, col_name)
                elif isinstance(sids, int):
                    lookup[sids] = (loc_id, col_name)
        return lookup

    def fetch_station_latest(self, location_id: int, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        Fetch latest readings for a single station from /v3/locations/{id}/latest.
        """
        url = f"{self.base_url}/locations/{location_id}/latest"
        try:
            resp = requests.get(url, headers=self.headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json().get("results", [])
                return {"location_id": location_id, "readings": data}
            else:
                logger.warning(f"OpenAQ returned status {resp.status_code} for location {location_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching location {location_id} from OpenAQ: {e}")
            return None

    def fetch_all_stations_latest(self) -> List[Dict[str, Any]]:
        """
        Fetch the latest observations for all configured Delhi monitoring stations.
        Returns a list of structured row dictionaries ready for dataframe ingestion.
        """
        station_by_id = {st["location_id"]: st for st in self.stations}
        fetched_rows = []

        logger.info(f"Initiating live fetch for {len(self.stations)} Delhi monitoring stations...")

        for st in self.stations:
            loc_id = st["location_id"]
            result = self.fetch_station_latest(loc_id)
            if not result or not result.get("readings"):
                logger.info(f"No recent live readings for {st['candidate_name']} (ID: {loc_id}).")
                continue

            raw_readings = result["readings"]
            station_measurements: Dict[str, float] = {}
            param_timestamps: Dict[str, pd.Timestamp] = {}
            latest_dt_utc = None
            latest_dt_ist = None

            for item in raw_readings:
                sid = item.get("sensorsId")
                val = item.get("value")
                dt_dict = item.get("datetime") or {}

                utc_str = dt_dict.get("utc")
                cur_dt_utc = None
                if utc_str:
                    try:
                        cur_dt_utc = pd.to_datetime(utc_str, utc=True)
                    except Exception:
                        cur_dt_utc = None

                # Resolve parameter by sensor ID
                if sid in self.sensor_to_param and val is not None:
                    _, col_name = self.sensor_to_param[sid]
                    try:
                        fval = float(val)
                        # Prefer reading with the latest timestamp for each parameter
                        if col_name not in param_timestamps or (cur_dt_utc and cur_dt_utc > param_timestamps[col_name]):
                            station_measurements[col_name] = fval
                            if cur_dt_utc:
                                param_timestamps[col_name] = cur_dt_utc
                    except (ValueError, TypeError):
                        pass

                if cur_dt_utc:
                    if latest_dt_utc is None or cur_dt_utc > latest_dt_utc:
                        latest_dt_utc = cur_dt_utc

            if latest_dt_utc is not None:
                # Freshness validation: reject readings older than SEVERELY_STALE_HOURS
                now_utc = pd.Timestamp.now(tz="UTC")
                age_hours = (now_utc - latest_dt_utc).total_seconds() / 3600.0
                if age_hours > SEVERELY_STALE_HOURS:
                    logger.warning(
                        f"Readings for {st['candidate_name']} are severely stale (Observed: {latest_dt_utc}, Age: {age_hours:.1f}h > {SEVERELY_STALE_HOURS}h). "
                        f"Discarded from live feed to prevent presenting outdated observations as fresh live data."
                    )
                    continue
                latest_dt_ist = latest_dt_utc.tz_convert(TIMEZONE)

            if latest_dt_ist and station_measurements:
                # Format observed_at matching historical standard (floored/rounded to hour or exact)
                hourly_ist = latest_dt_ist.floor("h")
                hourly_utc = latest_dt_utc.floor("h") if latest_dt_utc is not None else latest_dt_ist.tz_convert("UTC").floor("h")

                now_utc = pd.Timestamp.now(tz="UTC")
                final_age_hours = round((now_utc - latest_dt_utc).total_seconds() / 3600.0, 2)
                is_fresh = final_age_hours <= LIVE_FRESHNESS_HOURS

                row = {
                    "observed_at": hourly_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                    "observed_at_ist": hourly_ist.strftime("%Y-%m-%d %H:%M:%S+05:30"),
                    "location_id": loc_id,
                    "station_name": st["station_name"],
                    "candidate_station": st["candidate_name"],
                    "region": st["region"],
                    "latitude": st["latitude"],
                    "longitude": st["longitude"],
                    "data_source": "openaq_live",
                    "is_live": True,
                    "is_current_live": is_fresh,
                    "is_delayed": not is_fresh,
                    "reading_age_hours": final_age_hours,
                }

                # Populate all measurement columns (or NaN if missing in this poll)
                for col in ["co", "no", "no2", "nox", "o3", "pm10", "pm25", "relative_humidity", "so2", "temperature", "wind_direction", "wind_speed"]:
                    row[col] = station_measurements.get(col, None)

                fetched_rows.append(row)
                logger.info(f"Retrieved live data for {st['candidate_name']}: PM2.5 = {row.get('pm25')} ug/m3 at {row['observed_at_ist']}")

            # Sleep slightly to respect rate limits
            time.sleep(0.15)

        logger.info(f"Successfully collected live readings for {len(fetched_rows)} / {len(self.stations)} stations.")
        return fetched_rows


if __name__ == "__main__":
    fetcher = OpenAQFetcher()
    results = fetcher.fetch_all_stations_latest()
    print(f"Total stations fetched: {len(results)}")
    if results:
        print("Sample live record:")
        print(json.dumps(results[0], indent=2))
