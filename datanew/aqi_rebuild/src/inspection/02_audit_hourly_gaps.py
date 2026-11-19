
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_DIR.parent / "hourly_air_quality_raw.csv"
REPORT_DIR = PROJECT_DIR / "reports"

df = pd.read_csv(RAW_FILE, low_memory=False)

# Parse timestamps as UTC, then convert to Indian Standard Time.
df["timestamp_utc"] = pd.to_datetime(
    df["observed_at"], errors="coerce", utc=True
)
df["timestamp_ist"] = df["timestamp_utc"].dt.tz_convert(
    "Asia/Kolkata"
)

station_reports = []
interval_reports = []
alignment_reports = []

for (station_id, station_name), group in df.groupby(
    ["location_id", "station_name"], dropna=False
):
    times = (
        group["timestamp_ist"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(times) < 2:
        continue

    # Difference between consecutive observations.
    differences = times.diff().dropna()
    difference_hours = differences.dt.total_seconds() / 3600

    # Count intervals by duration.
    interval_counts = difference_hours.value_counts().sort_index()

    for hours, count in interval_counts.items():
        interval_reports.append({
            "location_id": station_id,
            "station_name": station_name,
            "interval_hours": hours,
            "count": count
        })

    # Expected hourly timestamps between actual first and last readings.
    expected = pd.date_range(
        start=times.iloc[0],
        end=times.iloc[-1],
        freq="h"
    )

    actual_set = set(times)
    missing = [t for t in expected if t not in actual_set]

    # Longest gap between consecutive observations.
    longest_gap = differences.max()

    station_reports.append({
        "location_id": station_id,
        "station_name": station_name,
        "actual_observations": len(times),
        "expected_hourly_observations": len(expected),
        "missing_hourly_timestamps": len(missing),
        "coverage_percent": round(
            len(times) / len(expected) * 100, 2
        ),
        "median_interval_hours": difference_hours.median(),
        "longest_gap_hours": longest_gap.total_seconds() / 3600,
        "first_timestamp_ist": times.iloc[0],
        "last_timestamp_ist": times.iloc[-1]
    })

    # Check whether readings occur at consistent minutes/seconds.
    alignment_reports.append({
        "location_id": station_id,
        "station_name": station_name,
        "minute_values": sorted(times.dt.minute.unique().tolist()),
        "second_values": sorted(times.dt.second.unique().tolist()),
        "hour_values_count": times.dt.hour.nunique()
    })

station_report = pd.DataFrame(station_reports)
interval_report = pd.DataFrame(interval_reports)
alignment_report = pd.DataFrame(alignment_reports)

station_report.to_csv(
    REPORT_DIR / "hourly_station_coverage.csv", index=False
)
interval_report.to_csv(
    REPORT_DIR / "hourly_interval_distribution.csv", index=False
)
alignment_report.to_csv(
    REPORT_DIR / "timestamp_alignment.csv", index=False
)

print("\n========== STATION HOURLY COVERAGE ==========")
print(station_report.to_string(index=False))

print("\n========== INTERVAL DISTRIBUTION ==========")
for station_id, group in interval_report.groupby("location_id"):
    name = station_report.loc[
        station_report["location_id"] == station_id,
        "station_name"
    ].iloc[0]

    print(f"\n{station_id} - {name}")
    print(group[["interval_hours", "count"]].to_string(index=False))

print("\n========== TIMESTAMP ALIGNMENT ==========")
print(alignment_report.to_string(index=False))

print("\nReports saved in:", REPORT_DIR.resolve())
