from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = (
    ROOT / "data" / "satellite" / "exports" / "full"
)

SOURCE_SAMPLE = (
    ROOT / "data" / "satellite" / "sentinel_sampling_2022_2024.parquet"
)

OUTPUT_PARQUET = (
    ROOT / "data" / "satellite" / "sentinel_evidence_2022_2024.parquet"
)

REPORT_FILE = (
    ROOT / "reports" / "experiments" / "sentinel_full_validation.json"
)

EXPECTED_TOTAL = 20_000
EXPECTED_BATCH_ROWS = 5_000
EXPECTED_BATCHES = 4

SPECTRAL_COLUMNS = [
    "B4",
    "B8",
    "B11",
    "B12",
]

INDEX_COLUMNS = [
    "NDVI",
    "NBR",
    "NDMI",
]

ALL_SATELLITE_COLUMNS = SPECTRAL_COLUMNS + INDEX_COLUMNS


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print("\n" + "=" * 80)
    print("VALIDATION FAILED")
    print("=" * 80)
    print(message)
    sys.exit(1)


def parse_ee_time(series):
    """
    Earth Engine CSV exports may contain timestamps as:
      - ISO strings
      - epoch milliseconds
      - numeric-looking strings

    Return UTC-naive pandas timestamps.
    """

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    numeric_mask = numeric.notna()

    if numeric_mask.any():

        result.loc[numeric_mask] = (
            pd.to_datetime(
                numeric.loc[numeric_mask],
                unit="ms",
                utc=True,
                errors="coerce"
            )
            .dt.tz_localize(None)
        )

    text_mask = ~numeric_mask & series.notna()

    if text_mask.any():

        result.loc[text_mask] = (
            pd.to_datetime(
                series.loc[text_mask],
                utc=True,
                errors="coerce"
            )
            .dt.tz_localize(None)
        )

    return result


def normalize_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )


def safe_quantiles(series):
    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if clean.empty:
        return {}

    q = clean.quantile(
        [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]
    )

    return {
        str(k): float(v)
        for k, v in q.items()
    }


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — SENTINEL FULL MERGE + VALIDATION")
print("=" * 80)

print(
    "\nThis stage will:"
    "\n  1. Read all four Earth Engine exports"
    "\n  2. Verify batch integrity"
    "\n  3. Verify exact 20,000 observation coverage"
    "\n  4. Detect temporal leakage"
    "\n  5. Validate missing satellite evidence"
    "\n  6. Audit spectral/index values"
    "\n  7. Merge validated evidence"
)

print("\n2025 holdout remains untouched.")


# ============================================================
# 1. LOCATE FILES
# ============================================================

print("\n[1/9] Locating four Sentinel exports...")


batch_files = []

for batch_number in range(1, EXPECTED_BATCHES + 1):

    expected_name = (
        f"firewatch_sentinel_full_batch_{batch_number:02d}.csv"
    )

    path = EXPORT_DIR / expected_name

    if not path.exists():
        fail(
            f"Missing Batch {batch_number}:\n{path}"
        )

    batch_files.append(
        (batch_number, path)
    )

    print(
        f"      Batch {batch_number}: {path.name}"
    )


if not SOURCE_SAMPLE.exists():
    fail(
        f"Original sampling dataset missing:\n{SOURCE_SAMPLE}"
    )


# ============================================================
# 2. READ + VALIDATE EACH BATCH
# ============================================================

print("\n[2/9] Reading individual batches...")


frames = []
batch_report = {}


for expected_batch, path in batch_files:

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print(
        f"      Batch {expected_batch}: {len(df):,} rows"
    )

    if len(df) != EXPECTED_BATCH_ROWS:
        fail(
            f"Batch {expected_batch} has {len(df):,} rows. "
            f"Expected {EXPECTED_BATCH_ROWS:,}."
        )

    if "observation_id" not in df.columns:
        fail(
            f"Batch {expected_batch} has no observation_id column."
        )

    if df["observation_id"].isna().any():
        fail(
            f"Batch {expected_batch} contains missing observation IDs."
        )

    if df["observation_id"].duplicated().any():
        duplicates = int(
            df["observation_id"].duplicated().sum()
        )

        fail(
            f"Batch {expected_batch} contains "
            f"{duplicates} duplicate observation IDs."
        )

    if "full_export_batch" not in df.columns:
        fail(
            f"Batch {expected_batch} is missing full_export_batch."
        )

    batch_values = set(
        pd.to_numeric(
            df["full_export_batch"],
            errors="coerce"
        ).dropna().astype(int)
    )

    if batch_values != {expected_batch}:
        fail(
            f"Batch {expected_batch} contains unexpected "
            f"batch identifiers: {batch_values}"
        )

    batch_report[
        f"batch_{expected_batch:02d}"
    ] = {
        "rows": int(len(df)),
        "unique_observation_ids": int(
            df["observation_id"].nunique()
        ),
        "filename": path.name,
    }

    frames.append(df)


# ============================================================
# 3. MERGE
# ============================================================

print("\n[3/9] Merging four batches...")


merged = pd.concat(
    frames,
    ignore_index=True
)


print(
    f"      Combined rows: {len(merged):,}"
)


if len(merged) != EXPECTED_TOTAL:
    fail(
        f"Merged dataset has {len(merged):,} rows. "
        f"Expected {EXPECTED_TOTAL:,}."
    )


duplicate_count = int(
    merged["observation_id"].duplicated().sum()
)


if duplicate_count != 0:
    fail(
        f"{duplicate_count} cross-batch duplicate IDs detected."
    )


if merged["observation_id"].nunique() != EXPECTED_TOTAL:
    fail(
        "Merged observation ID count is not exactly 20,000."
    )


print("      Cross-batch duplicates: 0")


# ============================================================
# 4. VERIFY AGAINST ORIGINAL 20K SAMPLE
# ============================================================

print("\n[4/9] Comparing against original sampling dataset...")


source = pd.read_parquet(
    SOURCE_SAMPLE
)


if len(source) != EXPECTED_TOTAL:
    fail(
        f"Original sampling dataset contains "
        f"{len(source):,} rows instead of 20,000."
    )


source_ids = set(
    source["observation_id"].astype(str)
)

export_ids = set(
    merged["observation_id"].astype(str)
)


missing_ids = source_ids - export_ids
unexpected_ids = export_ids - source_ids


print(
    f"      Missing expected IDs : {len(missing_ids):,}"
)

print(
    f"      Unexpected IDs       : {len(unexpected_ids):,}"
)


if missing_ids:
    fail(
        f"{len(missing_ids)} expected observations are missing."
    )


if unexpected_ids:
    fail(
        f"{len(unexpected_ids)} unexpected observations exist."
    )


print(
    "      Exact observation coverage: PASS"
)


# ============================================================
# 5. TEMPORAL LEAKAGE
# ============================================================

print("\n[5/9] Checking temporal integrity...")


merged["event_time_parsed"] = parse_ee_time(
    merged["event_time"]
)


merged["sentinel_latest_time_parsed"] = parse_ee_time(
    merged["sentinel_latest_time"]
)


if merged["event_time_parsed"].isna().any():

    bad = int(
        merged["event_time_parsed"].isna().sum()
    )

    fail(
        f"{bad} event timestamps could not be parsed."
    )


has_latest = (
    merged["sentinel_latest_time_parsed"].notna()
)


future_mask = (
    has_latest
    & (
        merged["sentinel_latest_time_parsed"]
        >
        merged["event_time_parsed"]
        + pd.Timedelta(seconds=1)
    )
)


future_count = int(
    future_mask.sum()
)


print(
    f"      Future Sentinel scenes: {future_count}"
)


if future_count != 0:
    fail(
        f"Temporal leakage detected in {future_count} observations."
    )


image_age_calculated = (
    merged["event_time_parsed"]
    - merged["sentinel_latest_time_parsed"]
).dt.total_seconds() / 86400.0


valid_age = image_age_calculated[
    has_latest
]


too_old = int(
    (valid_age > 14.1).sum()
)


negative_age = int(
    (valid_age < -0.001).sum()
)


print(
    f"      Latest scenes >14.1 days old: {too_old}"
)

print(
    f"      Negative image ages: {negative_age}"
)


if too_old != 0:
    fail(
        "One or more latest Sentinel scenes fall outside "
        "the intended 14-day past window."
    )


if negative_age != 0:
    fail(
        "Negative Sentinel image ages detected."
    )


# ============================================================
# 6. SATELLITE AVAILABILITY / MISSINGNESS
# ============================================================

print("\n[6/9] Validating satellite availability...")


required_columns = [
    "sentinel_scene_count",
    "sentinel_valid_pixel_count",
    "satellite_available",
    "satellite_extraction_status",
] + ALL_SATELLITE_COLUMNS


missing_columns = [
    col
    for col in required_columns
    if col not in merged.columns
]


if missing_columns:
    fail(
        "Required Sentinel columns missing:\n"
        + "\n".join(missing_columns)
    )


merged["satellite_available_bool"] = normalize_boolean(
    merged["satellite_available"]
)


if merged["satellite_available_bool"].isna().any():

    bad = int(
        merged["satellite_available_bool"].isna().sum()
    )

    fail(
        f"{bad} satellite_available values could not be parsed."
    )


merged["sentinel_scene_count"] = pd.to_numeric(
    merged["sentinel_scene_count"],
    errors="coerce"
)


merged["sentinel_valid_pixel_count"] = pd.to_numeric(
    merged["sentinel_valid_pixel_count"],
    errors="coerce"
)


available = merged["satellite_available_bool"]

unavailable = ~available


available_count = int(
    available.sum()
)

unavailable_count = int(
    unavailable.sum()
)


status_counts = (
    merged["satellite_extraction_status"]
    .fillna("MISSING_STATUS")
    .value_counts()
    .to_dict()
)


print(
    f"      Available   : {available_count:,} "
    f"({available_count / EXPECTED_TOTAL * 100:.2f}%)"
)

print(
    f"      Unavailable : {unavailable_count:,} "
    f"({unavailable_count / EXPECTED_TOTAL * 100:.2f}%)"
)


for status, count in status_counts.items():

    print(
        f"      {status}: {count:,}"
    )


# Available must have scenes and valid pixels.

bad_available_scene = (
    available
    & (
        merged["sentinel_scene_count"] <= 0
    )
)


bad_available_pixels = (
    available
    & (
        merged["sentinel_valid_pixel_count"] <= 0
    )
)


if bad_available_scene.any():
    fail(
        "Satellite-available observations exist with zero scenes."
    )


if bad_available_pixels.any():
    fail(
        "Satellite-available observations exist with "
        "zero valid pixels."
    )


# Unavailable observations must NOT contain fabricated spectral data.

fake_values_mask = (
    unavailable
    & merged[
        ALL_SATELLITE_COLUMNS
    ].notna().any(axis=1)
)


fake_value_count = int(
    fake_values_mask.sum()
)


print(
    f"      Unavailable rows with spectral values: "
    f"{fake_value_count}"
)


if fake_value_count != 0:
    fail(
        "Unavailable Sentinel observations contain "
        "spectral/index values."
    )


# Available observations should normally contain all seven values.

available_missing = (
    available
    & merged[
        ALL_SATELLITE_COLUMNS
    ].isna().any(axis=1)
)


available_missing_count = int(
    available_missing.sum()
)


print(
    f"      Available rows with incomplete spectral evidence: "
    f"{available_missing_count}"
)


if available_missing_count != 0:
    fail(
        "One or more available observations contain "
        "incomplete spectral evidence."
    )


# ============================================================
# 7. SPECTRAL AUDIT
# ============================================================

print("\n[7/9] Auditing spectral values...")


spectral_report = {}


for column in ALL_SATELLITE_COLUMNS:

    merged[column] = pd.to_numeric(
        merged[column],
        errors="coerce"
    )

    values = merged.loc[
        available,
        column
    ].dropna()


    if values.empty:
        fail(
            f"No usable values found for {column}."
        )


    spectral_report[column] = {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": safe_quantiles(values),
    }


    print(
        f"\n      {column}"
    )

    print(
        f"        count : {len(values):,}"
    )

    print(
        f"        mean  : {values.mean():.6f}"
    )

    print(
        f"        min   : {values.min():.6f}"
    )

    print(
        f"        max   : {values.max():.6f}"
    )


# ------------------------------------------------------------
# Indices have a mathematical expected range [-1, 1].
# ------------------------------------------------------------

index_range_failures = {}


for column in INDEX_COLUMNS:

    values = merged.loc[
        available,
        column
    ].dropna()


    bad = (
        (values < -1.001)
        | (values > 1.001)
    )


    count = int(
        bad.sum()
    )


    index_range_failures[column] = count


    if count != 0:
        fail(
            f"{column} contains {count} values outside "
            "the expected [-1, 1] range."
        )


# ------------------------------------------------------------
# Reflectance:
#
# Do NOT require strict 0..1 because Sentinel processing can
# occasionally contain unusual values. We instead detect
# grossly suspicious values.
# ------------------------------------------------------------

reflectance_extreme_counts = {}


for column in SPECTRAL_COLUMNS:

    values = merged.loc[
        available,
        column
    ].dropna()


    extreme = (
        (values < -0.2)
        | (values > 2.0)
    )


    count = int(
        extreme.sum()
    )


    reflectance_extreme_counts[
        column
    ] = count


    print(
        f"      {column} gross extreme values: {count}"
    )


# ============================================================
# 8. COVERAGE AUDIT
# ============================================================

print("\n[8/9] Auditing sample coverage...")


coverage_columns = [
    "sample_year",
    "sample_context_group",
    "sample_temporal_group",
    "sample_frp_group",
]


coverage_report = {}


for column in coverage_columns:

    if column not in merged.columns:
        continue

    counts = (
        merged[column]
        .fillna("MISSING")
        .value_counts()
        .to_dict()
    )


    coverage_report[column] = {
        str(k): int(v)
        for k, v in counts.items()
    }


    print(
        f"\n      {column}:"
    )


    for key, value in counts.items():

        print(
            f"        {key}: {value:,}"
        )


# Availability by context

availability_by_context = {}


if "sample_context_group" in merged.columns:

    context_table = pd.crosstab(
        merged["sample_context_group"],
        merged["satellite_available_bool"]
    )


    print(
        "\n      Satellite availability by context:"
    )

    print(
        context_table.to_string()
    )


    availability_by_context = {
        str(index): {
            str(column): int(
                context_table.loc[
                    index,
                    column
                ]
            )
            for column in context_table.columns
        }
        for index in context_table.index
    }


# ============================================================
# 9. SAVE CLEAN DATASET + REPORT
# ============================================================

print("\n[9/9] Saving validated Sentinel evidence...")


# Keep original exported fields, but remove temporary
# parsing columns.

output = merged.drop(
    columns=[
        "event_time_parsed",
        "sentinel_latest_time_parsed",
        "satellite_available_bool",
    ],
    errors="ignore"
)


# Deterministic ordering.

if "full_export_row" in output.columns:

    output["full_export_row"] = pd.to_numeric(
        output["full_export_row"],
        errors="coerce"
    )

    output = (
        output
        .sort_values(
            "full_export_row"
        )
        .reset_index(
            drop=True
        )
    )


OUTPUT_PARQUET.parent.mkdir(
    parents=True,
    exist_ok=True
)


output.to_parquet(
    OUTPUT_PARQUET,
    index=False
)


report = {

    "validation_result":
        "PASS",

    "total_rows":
        int(len(output)),

    "unique_observation_ids":
        int(
            output["observation_id"].nunique()
        ),

    "expected_rows":
        EXPECTED_TOTAL,

    "missing_expected_ids":
        int(len(missing_ids)),

    "unexpected_ids":
        int(len(unexpected_ids)),

    "duplicate_observation_ids":
        duplicate_count,

    "future_scene_violations":
        future_count,

    "scene_older_than_14_1_days":
        too_old,

    "negative_image_age_count":
        negative_age,

    "satellite_available_count":
        available_count,

    "satellite_unavailable_count":
        unavailable_count,

    "satellite_availability_percentage":
        float(
            available_count
            / EXPECTED_TOTAL
            * 100
        ),

    "unavailable_rows_with_spectral_values":
        fake_value_count,

    "available_rows_with_missing_spectral_values":
        available_missing_count,

    "status_counts": {
        str(k): int(v)
        for k, v in status_counts.items()
    },

    "batch_validation":
        batch_report,

    "spectral_statistics":
        spectral_report,

    "index_range_failures":
        index_range_failures,

    "reflectance_gross_extreme_counts":
        reflectance_extreme_counts,

    "coverage":
        coverage_report,

    "availability_by_context":
        availability_by_context,

    "methodology": {

        "sentinel_collection":
            "COPERNICUS/S2_SR_HARMONIZED",

        "temporal_policy":
            "past-only 14-day window",

        "composite":
            "median of cloud-masked candidate scenes",

        "buffer_m":
            100,

        "scale_m":
            20,

        "bands": [
            "B4",
            "B8",
            "B11",
            "B12"
        ],

        "indices": [
            "NDVI",
            "NBR",
            "NDMI"
        ],

        "missing_evidence_policy":
            "explicit missing values; no zero fabrication",

        "satellite_role":
            "contextual evidence, not ground-truth fire label",

        "holdout_2025_accessed":
            False
    }
}


REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


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


print("\n" + "=" * 80)
print("SENTINEL FULL VALIDATION: PASS")
print("=" * 80)

print(
    f"\nValidated observations : {len(output):,}"
)

print(
    f"Satellite available    : {available_count:,} "
    f"({available_count / EXPECTED_TOTAL * 100:.2f}%)"
)

print(
    f"Satellite unavailable  : {unavailable_count:,}"
)

print(
    f"Future leakage         : {future_count}"
)

print(
    f"Duplicate IDs          : {duplicate_count}"
)

print(
    f"Missing expected IDs   : {len(missing_ids)}"
)

print(
    "\nValidated dataset:"
)

print(
    OUTPUT_PARQUET
)

print(
    "\nValidation report:"
)

print(
    REPORT_FILE
)

print(
    "\n2025 holdout remains LOCKED."
)

print(
    "\nNext stage:"
    "\nDefensible source-label construction."
)