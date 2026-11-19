
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_DIR.parent / "hourly_air_quality_raw.csv"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

print("Loading original dataset...")
df = pd.read_csv(RAW_FILE, low_memory=False)

print("\nDataset loaded successfully.")

# --------------------------------------------------
# 1. BASIC INFORMATION
# --------------------------------------------------

print("\n" + "=" * 65)
print("1. DATASET OVERVIEW")
print("=" * 65)

print("File:", RAW_FILE)
print("Shape:", df.shape)
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
for col in df.columns:
    print("-", col)

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nLast 5 rows:")
print(df.tail().to_string())

# --------------------------------------------------
# 2. MISSING VALUES
# --------------------------------------------------

print("\n" + "=" * 65)
print("2. MISSING VALUE ANALYSIS")
print("=" * 65)

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
})

missing = missing.sort_values(
    "missing_percent", ascending=False
)

print(missing.to_string())

missing.to_csv(REPORT_DIR / "raw_missing_values.csv")

# --------------------------------------------------
# 3. DUPLICATE ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 65)
print("3. DUPLICATE ANALYSIS")
print("=" * 65)

exact_duplicates = df.duplicated().sum()

print("Exact duplicate rows:", exact_duplicates)

if "location_id" in df.columns and "observed_at" in df.columns:
    duplicate_keys = df.duplicated(
        subset=["location_id", "observed_at"],
        keep=False
    )

    print(
        "Rows sharing station ID and timestamp:",
        duplicate_keys.sum()
    )

    print(
        "Duplicate station-timestamp groups:",
        df.loc[duplicate_keys].groupby(
            ["location_id", "observed_at"],
            dropna=False
        ).ngroups
    )

# --------------------------------------------------
# 4. TIMESTAMP ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 65)
print("4. TIMESTAMP ANALYSIS")
print("=" * 65)

timestamps = pd.to_datetime(
    df["observed_at"],
    errors="coerce",
    utc=True
)

print("Unparseable timestamps:", timestamps.isna().sum())
print("Earliest timestamp (UTC):", timestamps.min())
print("Latest timestamp (UTC):", timestamps.max())
print("Unique timestamps:", timestamps.nunique())

# --------------------------------------------------
# 5. STATION ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 65)
print("5. STATION ANALYSIS")
print("=" * 65)

station_cols = ["location_id"]

if "station_name" in df.columns:
    station_cols.append("station_name")

station_summary = df.groupby(
    station_cols,
    dropna=False
).agg(
    rows=("observed_at", "size"),
    unique_timestamps=("observed_at", "nunique"),
    first_timestamp=("observed_at", "min"),
    last_timestamp=("observed_at", "max")
).reset_index()

print(station_summary.to_string(index=False))

station_summary.to_csv(
    REPORT_DIR / "raw_station_summary.csv",
    index=False
)

# --------------------------------------------------
# 6. NUMERIC COLUMN ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 65)
print("6. NUMERIC DATA ANALYSIS")
print("=" * 65)

numeric_df = df.select_dtypes(include="number")

print("Numeric columns:", numeric_df.columns.tolist())

if not numeric_df.empty:
    numeric_summary = numeric_df.describe(
        percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]
    ).T

    print(numeric_summary.to_string())

    numeric_summary.to_csv(
        REPORT_DIR / "raw_numeric_summary.csv"
    )

# --------------------------------------------------
# 7. INVALID / NEGATIVE VALUES
# --------------------------------------------------

print("\n" + "=" * 65)
print("7. NEGATIVE NUMERIC VALUES")
print("=" * 65)

negative_counts = (numeric_df < 0).sum().sort_values(
    ascending=False
)

print(negative_counts.to_string())

# --------------------------------------------------
# 8. STATION-WISE TIMESTAMP COVERAGE
# --------------------------------------------------

print("\n" + "=" * 65)
print("8. STATION-WISE TIMESTAMP COVERAGE")
print("=" * 65)

temp = df.copy()
temp["_timestamp_utc"] = timestamps

coverage = temp.groupby(
    station_cols,
    dropna=False
).agg(
    rows=("_timestamp_utc", "size"),
    valid_timestamps=("_timestamp_utc", "count"),
    unique_valid_timestamps=("_timestamp_utc", "nunique"),
    first_seen=("_timestamp_utc", "min"),
    last_seen=("_timestamp_utc", "max")
).reset_index()

coverage["duplicate_timestamp_rows"] = (
    coverage["valid_timestamps"]
    - coverage["unique_valid_timestamps"]
)

print(coverage.to_string(index=False))

coverage.to_csv(
    REPORT_DIR / "raw_timestamp_coverage.csv",
    index=False
)

# --------------------------------------------------
# 9. SAVE TEXT REPORT
# --------------------------------------------------

report = [
    f"Dataset shape: {df.shape}",
    f"Exact duplicate rows: {exact_duplicates}",
    f"Invalid timestamps: {timestamps.isna().sum()}",
    f"Unique timestamps: {timestamps.nunique()}",
    f"Earliest UTC: {timestamps.min()}",
    f"Latest UTC: {timestamps.max()}",
    f"Station count: {df['location_id'].nunique()}",
    f"Columns: {df.columns.tolist()}"
]

(REPORT_DIR / "raw_inspection_summary.txt").write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("INSPECTION COMPLETED")
print("=" * 65)

print("Reports saved in:", REPORT_DIR)
print("Original CSV was not modified.")
