from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIRMS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "firms_2022_2024"
    / "fire_archive_SV-C2_799789.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "experiments"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

JSON_REPORT = REPORT_DIR / "firms_2022_2024_audit.json"
CSV_REPORT = REPORT_DIR / "firms_numeric_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_value_counts(series, top_n=20):
    counts = series.value_counts(dropna=False).head(top_n)

    result = {}

    for key, value in counts.items():
        if pd.isna(key):
            key = "MISSING"
        else:
            key = str(key)

        result[key] = int(value)

    return result


# ============================================================
# LOAD DATA
# ============================================================

print_section("FIREWATCH FIRMS DATA AUDIT")

print(f"Loading:\n{FIRMS_FILE}")

if not FIRMS_FILE.exists():
    raise FileNotFoundError(
        f"\nFIRMS file not found:\n{FIRMS_FILE}\n"
        "Check the folder/file name."
    )

df = pd.read_csv(FIRMS_FILE)

print(f"\nLoaded {len(df):,} observations")
print(f"Columns: {len(df.columns)}")


# ============================================================
# BASIC STRUCTURE
# ============================================================

print_section("1. DATASET STRUCTURE")

print("Shape:", df.shape)

print("\nColumns:")
for i, col in enumerate(df.columns):
    print(f"{i:2d}. {col}")

print("\nData types:")
print(df.dtypes)


# ============================================================
# DATE PARSING
# ============================================================

print_section("2. TEMPORAL COVERAGE")

df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

invalid_dates = int(df["acq_date"].isna().sum())

print("Invalid dates:", invalid_dates)

if invalid_dates < len(df):

    print(
        "Date range:",
        df["acq_date"].min(),
        "to",
        df["acq_date"].max()
    )

    df["year"] = df["acq_date"].dt.year
    df["month"] = df["acq_date"].dt.month

    print("\nObservations by year:")
    print(
        df["year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nObservations by month:")
    print(
        df["month"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# MISSING VALUES
# ============================================================

print_section("3. MISSING VALUES")

missing = df.isna().sum()

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": (missing / len(df)) * 100
})

print(
    missing_table
    .sort_values("missing_count", ascending=False)
    .to_string()
)


# ============================================================
# DUPLICATES
# ============================================================

print_section("4. DUPLICATE ANALYSIS")

full_duplicates = int(df.duplicated().sum())

print(f"Exact duplicate rows: {full_duplicates:,}")

possible_event_columns = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite"
]

available_event_columns = [
    c for c in possible_event_columns
    if c in df.columns
]

event_duplicates = int(
    df.duplicated(
        subset=available_event_columns
    ).sum()
)

print(
    "Potential duplicate observations "
    f"({', '.join(available_event_columns)}): "
    f"{event_duplicates:,}"
)


# ============================================================
# COORDINATE VALIDATION
# ============================================================

print_section("5. COORDINATE VALIDATION")

invalid_lat = (
    df["latitude"].isna()
    | (df["latitude"] < -90)
    | (df["latitude"] > 90)
)

invalid_lon = (
    df["longitude"].isna()
    | (df["longitude"] < -180)
    | (df["longitude"] > 180)
)

print(f"Invalid latitude rows: {invalid_lat.sum():,}")
print(f"Invalid longitude rows: {invalid_lon.sum():,}")

print("\nLatitude range:")
print(df["latitude"].min(), "to", df["latitude"].max())

print("\nLongitude range:")
print(df["longitude"].min(), "to", df["longitude"].max())


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

print_section("6. SATELLITE DISTRIBUTION")

print(
    df["satellite"]
    .value_counts(dropna=False)
    .to_string()
)


print_section("7. INSTRUMENT DISTRIBUTION")

print(
    df["instrument"]
    .value_counts(dropna=False)
    .to_string()
)


print_section("8. DAY / NIGHT DISTRIBUTION")

print(
    df["daynight"]
    .value_counts(dropna=False)
    .to_string()
)


print_section("9. FIRMS TYPE DISTRIBUTION")

print(
    df["type"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


print_section("10. CONFIDENCE DISTRIBUTION")

print(
    df["confidence"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


# ============================================================
# NUMERIC FEATURES
# ============================================================

numeric_features = [
    "brightness",
    "bright_t31",
    "frp",
    "scan",
    "track"
]

print_section("11. NUMERIC FEATURE SUMMARY")

numeric_summary = df[numeric_features].describe(
    percentiles=[
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99
    ]
).T

print(numeric_summary.to_string())

numeric_summary.to_csv(CSV_REPORT)


# ============================================================
# INVALID / SUSPICIOUS VALUES
# ============================================================

print_section("12. SUSPICIOUS NUMERIC VALUES")

checks = {}

checks["brightness_le_zero"] = int(
    (df["brightness"] <= 0).sum()
)

checks["bright_t31_le_zero"] = int(
    (df["bright_t31"] <= 0).sum()
)

checks["frp_negative"] = int(
    (df["frp"] < 0).sum()
)

checks["scan_le_zero"] = int(
    (df["scan"] <= 0).sum()
)

checks["track_le_zero"] = int(
    (df["track"] <= 0).sum()
)

for name, count in checks.items():
    print(f"{name}: {count:,}")


# ============================================================
# FRP ANALYSIS
# ============================================================

print_section("13. FRP DISTRIBUTION")

frp = df["frp"].dropna()

for percentile in [
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999
]:
    value = frp.quantile(percentile)

    print(
        f"P{percentile * 100:6.1f}: "
        f"{value:.4f}"
    )

print(f"\nMaximum FRP: {frp.max():.4f}")


# ============================================================
# BRIGHTNESS ANALYSIS
# ============================================================

print_section("14. BRIGHTNESS DISTRIBUTION")

brightness = df["brightness"].dropna()

for percentile in [
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999
]:
    value = brightness.quantile(percentile)

    print(
        f"P{percentile * 100:6.1f}: "
        f"{value:.4f}"
    )

print(
    f"\nMaximum brightness: "
    f"{brightness.max():.4f}"
)


# ============================================================
# ACQUISITION TIME VALIDATION
# ============================================================

print_section("15. ACQUISITION TIME")

acq_numeric = pd.to_numeric(
    df["acq_time"],
    errors="coerce"
)

hours = (acq_numeric // 100)
minutes = (acq_numeric % 100)

invalid_time = (
    acq_numeric.isna()
    | (hours < 0)
    | (hours > 23)
    | (minutes < 0)
    | (minutes > 59)
)

print(
    f"Invalid acquisition times: "
    f"{invalid_time.sum():,}"
)

print("\nAcquisition hour distribution:")

valid_hours = hours[~invalid_time]

print(
    valid_hours
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SENSOR × YEAR
# ============================================================

print_section("16. SATELLITE BY YEAR")

if "year" in df.columns:

    satellite_year = pd.crosstab(
        df["year"],
        df["satellite"]
    )

    print(satellite_year.to_string())


# ============================================================
# FIRMS TYPE × YEAR
# ============================================================

print_section("17. FIRMS TYPE BY YEAR")

if "year" in df.columns:

    type_year = pd.crosstab(
        df["year"],
        df["type"]
    )

    print(type_year.to_string())


# ============================================================
# REPORT
# ============================================================

print_section("18. SAVING AUDIT REPORT")

report = {
    "dataset": str(FIRMS_FILE),
    "rows": int(len(df)),
    "columns": int(len(df.columns)),

    "date_range": {
        "min": str(df["acq_date"].min()),
        "max": str(df["acq_date"].max())
    },

    "year_counts": {
        str(k): int(v)
        for k, v in
        df["year"].value_counts().sort_index().items()
    },

    "missing_values": {
        col: {
            "count": int(df[col].isna().sum()),
            "percent": float(
                df[col].isna().mean() * 100
            )
        }
        for col in df.columns
    },

    "duplicates": {
        "exact_rows": full_duplicates,
        "event_key_duplicates": event_duplicates
    },

    "coordinate_validation": {
        "invalid_latitude": int(invalid_lat.sum()),
        "invalid_longitude": int(invalid_lon.sum()),
        "latitude_min": float(df["latitude"].min()),
        "latitude_max": float(df["latitude"].max()),
        "longitude_min": float(df["longitude"].min()),
        "longitude_max": float(df["longitude"].max())
    },

    "satellite_distribution":
        safe_value_counts(df["satellite"]),

    "instrument_distribution":
        safe_value_counts(df["instrument"]),

    "daynight_distribution":
        safe_value_counts(df["daynight"]),

    "type_distribution":
        safe_value_counts(df["type"]),

    "confidence_distribution":
        safe_value_counts(df["confidence"]),

    "numeric_checks": checks,

    "acquisition_time": {
        "invalid_count": int(invalid_time.sum())
    }
}

with open(
    JSON_REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print(f"JSON report saved:\n{JSON_REPORT}")
print(f"\nNumeric summary saved:\n{CSV_REPORT}")


print_section("AUDIT COMPLETE")

print(
    "No data was modified.\n"
    "This script only inspected the raw FIRMS archive."
)