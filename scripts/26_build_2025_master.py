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
    / "firms_2025_holdout"
    / "fire_archive_SV-C2_791643.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "firms_master_2025.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "firms_master_2025_summary.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 26")
print("BUILD 2025 TEMPORAL HOLDOUT FIRMS MASTER")
print("=" * 80)

print(
    "\nIMPORTANT:"
    "\n- 2025 holdout is now UNLOCKED."
    "\n- No model is executed."
    "\n- No labels are generated."
    "\n- Feature definitions are frozen."
)


# ============================================================
# LOAD
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"2025 FIRMS file not found:\n{INPUT_FILE}"
    )

print(
    f"\nLoading:\n{INPUT_FILE}"
)

df = pd.read_csv(
    INPUT_FILE
)

raw_rows = len(df)

print(
    f"\nRaw rows: {raw_rows:,}"
)


# ============================================================
# REQUIRED RAW SCHEMA
# ============================================================

required_columns = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
    "type",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        "2025 FIRMS schema does not match "
        f"development archive. Missing: {missing}"
    )


# ============================================================
# PARSE DATE / TIME
# ============================================================

print(
    "\n[1/8] Parsing timestamps..."
)

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

df["hour"] = (
    acq_time_str
    .str[:2]
    .astype("int8")
)

df["minute"] = (
    acq_time_str
    .str[2:]
    .astype("int8")
)

df["acquired_at"] = (
    df["acq_date"]
    + pd.to_timedelta(
        df["hour"],
        unit="h"
    )
    + pd.to_timedelta(
        df["minute"],
        unit="m"
    )
)

df["year"] = (
    df["acq_date"]
    .dt.year
    .astype("int16")
)

df["month"] = (
    df["acq_date"]
    .dt.month
    .astype("int8")
)

df["day_of_year"] = (
    df["acq_date"]
    .dt.dayofyear
    .astype("int16")
)


# ============================================================
# HOLDOUT DATE VALIDATION
# ============================================================

if not (
    df["year"] == 2025
).all():

    bad_years = (
        df["year"]
        .value_counts()
        .sort_index()
    )

    raise RuntimeError(
        "Holdout file contains non-2025 data:\n"
        f"{bad_years}"
    )


# ============================================================
# EXACT DUPLICATES
# ============================================================

print(
    "[2/8] Checking duplicates..."
)

before = len(df)

df = (
    df
    .drop_duplicates()
    .copy()
)

duplicates_removed = (
    before - len(df)
)

print(
    f"      Removed: "
    f"{duplicates_removed:,}"
)


# ============================================================
# CONFIDENCE
# ============================================================

print(
    "[3/8] Encoding FIRMS confidence..."
)

confidence_map = {
    "l": 0,
    "n": 1,
    "h": 2,
}

df["confidence_code"] = (
    df["confidence"]
    .str.lower()
    .map(confidence_map)
)

if df[
    "confidence_code"
].isna().any():

    unknown = (
        df.loc[
            df["confidence_code"].isna(),
            "confidence"
        ]
        .value_counts(
            dropna=False
        )
    )

    raise RuntimeError(
        "Unexpected 2025 confidence values:\n"
        f"{unknown}"
    )

df["confidence_code"] = (
    df["confidence_code"]
    .astype("int8")
)


# ============================================================
# DAY / NIGHT
# ============================================================

print(
    "[4/8] Encoding day/night..."
)

daynight_map = {
    "N": 0,
    "D": 1,
}

df["daynight_code"] = (
    df["daynight"]
    .map(daynight_map)
)

if df[
    "daynight_code"
].isna().any():

    unknown = (
        df.loc[
            df["daynight_code"].isna(),
            "daynight"
        ]
        .value_counts(
            dropna=False
        )
    )

    raise RuntimeError(
        "Unexpected 2025 day/night values:\n"
        f"{unknown}"
    )

df["daynight_code"] = (
    df["daynight_code"]
    .astype("int8")
)


# ============================================================
# THERMAL FEATURES
# ============================================================

print(
    "[5/8] Building frozen thermal features..."
)

df[
    "brightness_t31_delta"
] = (
    df["brightness"]
    - df["bright_t31"]
).astype(
    "float32"
)

df["log_frp"] = (
    np.log1p(
        df["frp"]
    )
    .astype("float32")
)

df[
    "brightness_at_upper_bound"
] = (
    df["brightness"]
    >= 367.0
).astype(
    "int8"
)


# ============================================================
# CYCLICAL FEATURES
# ============================================================

print(
    "[6/8] Building frozen time features..."
)

df["hour_sin"] = np.sin(
    2
    * np.pi
    * df["hour"]
    / 24.0
).astype(
    "float32"
)

df["hour_cos"] = np.cos(
    2
    * np.pi
    * df["hour"]
    / 24.0
).astype(
    "float32"
)

df["month_sin"] = np.sin(
    2
    * np.pi
    * (
        df["month"] - 1
    )
    / 12.0
).astype(
    "float32"
)

df["month_cos"] = np.cos(
    2
    * np.pi
    * (
        df["month"] - 1
    )
    / 12.0
).astype(
    "float32"
)


# ============================================================
# STABLE 2025 OBSERVATION IDS
# ============================================================

print(
    "[7/8] Creating 2025 observation IDs..."
)

df = (
    df
    .sort_values(
        [
            "acquired_at",
            "latitude",
            "longitude",
        ],
        kind="stable",
    )
    .reset_index(
        drop=True
    )
)

# Deliberately different prefix from development data.
# This prevents ID collisions when 2024 history and
# 2025 observations are temporarily combined.

df["observation_id"] = (
    "FW25_"
    + df.index
    .astype(str)
    .str.zfill(8)
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
    "frp",
]

for column in float32_columns:

    df[column] = (
        df[column]
        .astype("float32")
    )


# ============================================================
# EXACT FROZEN COLUMN ORDER
# ============================================================

columns = [
    "observation_id",
    "latitude",
    "longitude",

    "acq_date",
    "acq_time",
    "acquired_at",
    "year",
    "month",
    "day_of_year",
    "hour",
    "minute",

    "brightness",
    "bright_t31",
    "frp",
    "scan",
    "track",
    "confidence",
    "daynight",
    "type",

    "confidence_code",
    "daynight_code",

    "brightness_t31_delta",
    "log_frp",
    "brightness_at_upper_bound",

    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    "satellite",
    "instrument",
    "version",
]

df = df[
    columns
]


# ============================================================
# VALIDATION
# ============================================================

print(
    "[8/8] Validating and saving..."
)

assert len(df) > 0

assert (
    df["observation_id"]
    .is_unique
)

assert (
    df["latitude"]
    .between(
        -90,
        90
    )
    .all()
)

assert (
    df["longitude"]
    .between(
        -180,
        180
    )
    .all()
)

assert (
    df["acquired_at"]
    .notna()
    .all()
)

assert (
    df["frp"] >= 0
).all()

assert (
    df["year"] == 2025
).all()


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report = {
    "stage":
        "2025_holdout_master",

    "holdout_status":
        "UNLOCKED_FOR_FINAL_EVALUATION",

    "source_file":
        str(INPUT_FILE),

    "output_file":
        str(OUTPUT_FILE),

    "raw_rows":
        int(raw_rows),

    "final_rows":
        int(len(df)),

    "duplicates_removed":
        int(
            duplicates_removed
        ),

    "date_min":
        str(
            df["acquired_at"]
            .min()
        ),

    "date_max":
        str(
            df["acquired_at"]
            .max()
        ),

    "columns":
        list(
            df.columns
        ),

    "model_used":
        False,

    "labels_generated":
        False,

    "feature_policy_changed":
        False,

    "notes": [
        (
            "Feature definitions are identical "
            "to the frozen 2022-2024 FIRMS "
            "master feature pipeline."
        ),
        (
            "FW25 observation IDs are used "
            "to prevent collision with "
            "development observation IDs."
        ),
        (
            "No model predictions were made."
        ),
        (
            "No labels were generated."
        ),
        (
            "No model or feature decisions "
            "may be changed based on 2025."
        ),
    ],
}

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 26 COMPLETE"
)

print(
    "=" * 80
)

print(
    f"\nRows: "
    f"{len(df):,}"
)

print(
    "\nDate range:"
)

print(
    df["acquired_at"].min(),
    "to",
    df["acquired_at"].max()
)

print(
    "\nConfidence:"
)

print(
    df["confidence"]
    .value_counts()
    .to_string()
)

print(
    "\nFIRMS type:"
)

print(
    df["type"]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    f"\nSaved:\n{OUTPUT_FILE}"
)

print(
    f"\nReport:\n{REPORT_FILE}"
)

print(
    "\nIMPORTANT:"
    "\n- 2025 is now unlocked."
    "\n- Model has NOT been run."
    "\n- Labels have NOT been generated."
    "\n- Frozen development decisions remain unchanged."
)