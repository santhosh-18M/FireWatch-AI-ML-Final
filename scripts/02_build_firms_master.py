from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "firms_2022_2024"
    / "fire_archive_SV-C2_799789.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "firms_master_2022_2024.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "firms_master_2022_2024_summary.json"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("FIREWATCH — BUILD FIRMS MASTER DATASET")
print("=" * 80)

print(f"\nLoading:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"\nRaw rows: {len(df):,}")


# ============================================================
# PARSE DATE / TIME
# ============================================================

print("\n[1/8] Parsing timestamps...")

df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="raise"
)

acq_time_str = (
    df["acq_time"]
    .astype(int)
    .astype(str)
    .str.zfill(4)
)

df["hour"] = acq_time_str.str[:2].astype("int8")
df["minute"] = acq_time_str.str[2:].astype("int8")

df["acquired_at"] = (
    df["acq_date"]
    + pd.to_timedelta(df["hour"], unit="h")
    + pd.to_timedelta(df["minute"], unit="m")
)

df["year"] = df["acq_date"].dt.year.astype("int16")
df["month"] = df["acq_date"].dt.month.astype("int8")
df["day_of_year"] = df["acq_date"].dt.dayofyear.astype("int16")


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

print("[2/8] Checking duplicates...")

before = len(df)

df = df.drop_duplicates().copy()

duplicates_removed = before - len(df)

print(f"      Removed: {duplicates_removed:,}")


# ============================================================
# CONFIDENCE
# ============================================================

print("[3/8] Encoding FIRMS confidence...")

confidence_map = {
    "l": 0,
    "n": 1,
    "h": 2
}

df["confidence_code"] = (
    df["confidence"]
    .str.lower()
    .map(confidence_map)
)

if df["confidence_code"].isna().any():
    unknown = (
        df.loc[
            df["confidence_code"].isna(),
            "confidence"
        ]
        .value_counts(dropna=False)
    )

    raise ValueError(
        f"Unexpected FIRMS confidence values:\n{unknown}"
    )

df["confidence_code"] = df["confidence_code"].astype("int8")


# ============================================================
# DAY / NIGHT
# ============================================================

print("[4/8] Encoding day/night...")

daynight_map = {
    "N": 0,
    "D": 1
}

df["daynight_code"] = df["daynight"].map(daynight_map)

if df["daynight_code"].isna().any():
    unknown = (
        df.loc[
            df["daynight_code"].isna(),
            "daynight"
        ]
        .value_counts(dropna=False)
    )

    raise ValueError(
        f"Unexpected day/night values:\n{unknown}"
    )

df["daynight_code"] = df["daynight_code"].astype("int8")


# ============================================================
# SAFE DERIVED THERMAL FEATURES
# ============================================================

print("[5/8] Building thermal features...")

# Difference between VIIRS M13 brightness temperature
# and the background/reference T31 temperature.
df["brightness_t31_delta"] = (
    df["brightness"] - df["bright_t31"]
).astype("float32")

# FRP is strongly right-skewed.
df["log_frp"] = np.log1p(df["frp"]).astype("float32")

# Brightness reaches 367 K frequently in this archive.
# Preserve the original value and record that the observation
# is at the archive's observed upper brightness boundary.
df["brightness_at_upper_bound"] = (
    df["brightness"] >= 367.0
).astype("int8")


# ============================================================
# CYCLICAL TIME FEATURES
# ============================================================

print("[6/8] Building time features...")

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24.0
).astype("float32")

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24.0
).astype("float32")

df["month_sin"] = np.sin(
    2 * np.pi * (df["month"] - 1) / 12.0
).astype("float32")

df["month_cos"] = np.cos(
    2 * np.pi * (df["month"] - 1) / 12.0
).astype("float32")


# ============================================================
# OBSERVATION ID
# ============================================================

print("[7/8] Creating stable observation IDs...")

df = df.sort_values(
    ["acquired_at", "latitude", "longitude"],
    kind="stable"
).reset_index(drop=True)

df["observation_id"] = (
    "FW_"
    + df.index.astype(str).str.zfill(8)
)


# ============================================================
# DATA TYPES
# ============================================================

float32_columns = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "bright_t31",
    "frp"
]

for col in float32_columns:
    df[col] = df[col].astype("float32")


# ============================================================
# COLUMN ORDER
# ============================================================

columns = [
    # identity / GIS metadata
    "observation_id",
    "latitude",
    "longitude",

    # temporal metadata
    "acq_date",
    "acq_time",
    "acquired_at",
    "year",
    "month",
    "day_of_year",
    "hour",
    "minute",

    # raw FIRMS measurements
    "brightness",
    "bright_t31",
    "frp",
    "scan",
    "track",
    "confidence",
    "daynight",
    "type",

    # encoded FIRMS fields
    "confidence_code",
    "daynight_code",

    # thermal derived features
    "brightness_t31_delta",
    "log_frp",
    "brightness_at_upper_bound",

    # cyclical temporal features
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    # provenance only
    "satellite",
    "instrument",
    "version"
]

df = df[columns]


# ============================================================
# SAVE
# ============================================================

print("[8/8] Saving master dataset...")

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

assert len(df) > 0
assert df["observation_id"].is_unique
assert df["latitude"].between(-90, 90).all()
assert df["longitude"].between(-180, 180).all()
assert df["acquired_at"].notna().all()
assert (df["frp"] >= 0).all()


# ============================================================
# REPORT
# ============================================================

report = {
    "source_file": str(INPUT_FILE),
    "output_file": str(OUTPUT_FILE),

    "raw_rows": int(before),
    "final_rows": int(len(df)),
    "duplicates_removed": int(duplicates_removed),

    "date_min": str(df["acquired_at"].min()),
    "date_max": str(df["acquired_at"].max()),

    "year_counts": {
        str(k): int(v)
        for k, v in
        df["year"]
        .value_counts()
        .sort_index()
        .items()
    },

    "confidence_counts": {
        str(k): int(v)
        for k, v in
        df["confidence"]
        .value_counts()
        .items()
    },

    "daynight_counts": {
        str(k): int(v)
        for k, v in
        df["daynight"]
        .value_counts()
        .items()
    },

    "type_counts": {
        str(k): int(v)
        for k, v in
        df["type"]
        .value_counts()
        .sort_index()
        .items()
    },

    "brightness_upper_bound_count": int(
        df["brightness_at_upper_bound"].sum()
    ),

    "columns": list(df.columns),

    "notes": [
        "Raw latitude and longitude are retained for GIS, spatial splitting, OSM and satellite lookup.",
        "Latitude and longitude are not automatically intended as model predictors.",
        "FIRMS type is retained as source metadata and is not the FireWatch source-class target.",
        "Satellite and instrument are retained as provenance even though this archive contains only SNPP.",
        "No source-class labels were generated in this stage.",
        "No 2025 observations were used."
    ]
}

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=2)


print("\n" + "=" * 80)
print("MASTER DATASET COMPLETE")
print("=" * 80)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print(f"\nDate range:")
print(
    df["acquired_at"].min(),
    "to",
    df["acquired_at"].max()
)

print("\nYear counts:")
print(
    df["year"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nConfidence:")
print(
    df["confidence"]
    .value_counts()
    .to_string()
)

print(
    "\nBrightness observations at "
    ">=367 K:",
    f"{df['brightness_at_upper_bound'].sum():,}"
)

print(f"\nSaved:\n{OUTPUT_FILE}")
print(f"\nReport:\n{REPORT_FILE}")

print(
    "\nIMPORTANT:"
    "\n2025 holdout has not been read or modified."
    "\nNo ML labels have been created yet."
)