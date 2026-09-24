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
    ROOT / "data" / "satellite"
    / "sentinel_sampling_2022_2024.parquet"
)

OUTPUT_PARQUET = (
    ROOT / "data" / "satellite"
    / "sentinel_evidence_2022_2024.parquet"
)

REPORT_FILE = (
    ROOT / "reports" / "experiments"
    / "sentinel_full_validation_final.json"
)

EXPECTED_TOTAL = 20_000
EXPECTED_BATCHES = 4
EXPECTED_BATCH_ROWS = 5_000

BANDS = [
    "B4",
    "B8",
    "B11",
    "B12",
]

INDICES = [
    "NDVI",
    "NBR",
    "NDMI",
]

SATELLITE_FEATURES = BANDS + INDICES


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

    text_mask = (
        ~numeric_mask
        & series.notna()
    )

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


def parse_bool(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )


def quantile_dict(series):

    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    q = clean.quantile(
        [
            0,
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            1
        ]
    )

    return {
        str(k): float(v)
        for k, v in q.items()
    }


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — FINAL SENTINEL EVIDENCE VALIDATION")
print("=" * 80)

print(
    "\nAvailability policy:"
    "\nSatellite evidence is usable only when ALL seven"
    "\nmodel satellite features are finite:"
    "\nB4, B8, B11, B12, NDVI, NBR, NDMI."
)

print(
    "\nThe original Earth Engine availability flag and"
    "\nvalid-pixel count are preserved for audit purposes."
)

print(
    "\nNo missing satellite values will be zero-filled."
)

print(
    "\n2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD FOUR EXPORTS
# ============================================================

print("\n[1/8] Loading four Earth Engine exports...")


frames = []


for batch_number in range(
    1,
    EXPECTED_BATCHES + 1
):

    path = (
        EXPORT_DIR
        / f"firewatch_sentinel_full_batch_{batch_number:02d}.csv"
    )

    if not path.exists():

        fail(
            f"Missing export:\n{path}"
        )


    df = pd.read_csv(
        path,
        low_memory=False
    )


    print(
        f"      Batch {batch_number}: "
        f"{len(df):,} rows"
    )


    if len(df) != EXPECTED_BATCH_ROWS:

        fail(
            f"Batch {batch_number} has incorrect row count."
        )


    if df["observation_id"].duplicated().any():

        fail(
            f"Duplicate IDs inside Batch {batch_number}."
        )


    frames.append(
        df
    )


df = pd.concat(
    frames,
    ignore_index=True
)


if len(df) != EXPECTED_TOTAL:

    fail(
        f"Expected 20,000 rows, found {len(df):,}."
    )


if df["observation_id"].duplicated().any():

    fail(
        "Cross-batch duplicate observation IDs detected."
    )


print(
    f"\n      Combined rows: {len(df):,}"
)

print(
    "      Duplicate IDs: 0"
)


# ============================================================
# 2. VERIFY EXACT SAMPLE COVERAGE
# ============================================================

print(
    "\n[2/8] Verifying against original 20,000 sample..."
)


source = pd.read_parquet(
    SOURCE_SAMPLE
)


source_ids = set(
    source["observation_id"].astype(str)
)

export_ids = set(
    df["observation_id"].astype(str)
)


missing_ids = (
    source_ids
    - export_ids
)

unexpected_ids = (
    export_ids
    - source_ids
)


print(
    f"      Missing IDs    : {len(missing_ids)}"
)

print(
    f"      Unexpected IDs : {len(unexpected_ids)}"
)


if missing_ids or unexpected_ids:

    fail(
        "Earth Engine export does not exactly match "
        "the original sample."
    )


# ============================================================
# 3. NUMERIC SATELLITE FEATURES
# ============================================================

print(
    "\n[3/8] Parsing satellite features..."
)


for column in SATELLITE_FEATURES:

    if column not in df.columns:

        fail(
            f"Missing satellite feature: {column}"
        )


    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Replace infinities with missing.

df[
    SATELLITE_FEATURES
] = df[
    SATELLITE_FEATURES
].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 4. PRESERVE RAW EE QA FIELDS
# ============================================================

print(
    "\n[4/8] Preserving Earth Engine QA metadata..."
)


df[
    "satellite_available_exported"
] = parse_bool(
    df[
        "satellite_available"
    ]
)


df[
    "satellite_extraction_status_exported"
] = df[
    "satellite_extraction_status"
].copy()


df[
    "sentinel_valid_pixel_count_exported"
] = pd.to_numeric(
    df[
        "sentinel_valid_pixel_count"
    ],
    errors="coerce"
)


# ============================================================
# 5. DEFINE FINAL EVIDENCE AVAILABILITY
# ============================================================

print(
    "\n[5/8] Deriving final evidence availability..."
)


# Complete evidence means ALL seven features are finite.

complete_evidence = (
    df[
        SATELLITE_FEATURES
    ]
    .notna()
    .all(
        axis=1
    )
)


partial_evidence = (
    df[
        SATELLITE_FEATURES
    ]
    .notna()
    .any(
        axis=1
    )
    & ~complete_evidence
)


no_evidence = (
    df[
        SATELLITE_FEATURES
    ]
    .isna()
    .all(
        axis=1
    )
)


if partial_evidence.any():

    fail(
        f"{int(partial_evidence.sum())} observations "
        "contain partial satellite evidence. "
        "This requires separate investigation."
    )


df[
    "satellite_evidence_available"
] = complete_evidence


df[
    "satellite_evidence_status"
] = np.where(
    complete_evidence,
    "complete",
    "unavailable"
)


usable_count = int(
    complete_evidence.sum()
)

unavailable_count = int(
    no_evidence.sum()
)


print(
    f"      Complete evidence : {usable_count:,} "
    f"({usable_count / EXPECTED_TOTAL * 100:.2f}%)"
)

print(
    f"      No evidence       : {unavailable_count:,} "
    f"({unavailable_count / EXPECTED_TOTAL * 100:.2f}%)"
)

print(
    f"      Partial evidence  : "
    f"{int(partial_evidence.sum()):,}"
)


# ============================================================
# 6. INVESTIGATE EXPORTED FLAG DISAGREEMENTS
# ============================================================

print(
    "\n[6/8] Auditing Earth Engine QA disagreement..."
)


exported_available = (
    df[
        "satellite_available_exported"
    ]
    .fillna(False)
)


usable_but_exported_false = (
    complete_evidence
    & ~exported_available
)


exported_true_but_incomplete = (
    exported_available
    & ~complete_evidence
)


qa_disagreement_count = int(
    usable_but_exported_false.sum()
)


bad_exported_true_count = int(
    exported_true_but_incomplete.sum()
)


print(
    "      Complete evidence but exported flag false: "
    f"{qa_disagreement_count}"
)

print(
    "      Exported flag true but incomplete evidence: "
    f"{bad_exported_true_count}"
)


# We expect the 16 already investigated cases.

if qa_disagreement_count != 16:

    fail(
        "Expected exactly 16 known QA disagreements, "
        f"found {qa_disagreement_count}."
    )


if bad_exported_true_count != 0:

    fail(
        "Earth Engine marked one or more incomplete "
        "observations as available."
    )


# Check known disagreement structure.

disagreement = df.loc[
    usable_but_exported_false
].copy()


if not (
    disagreement[
        "satellite_extraction_status_exported"
    ]
    == "scene_present_no_valid_pixels"
).all():

    fail(
        "Unexpected extraction status among "
        "the 16 known disagreements."
    )


if not (
    disagreement[
        "sentinel_valid_pixel_count_exported"
    ]
    == 0
).all():

    fail(
        "Unexpected valid-pixel count among "
        "the 16 known disagreements."
    )


print(
    "      Known 16-row QA discrepancy reproduced: PASS"
)


# ============================================================
# 7. TEMPORAL + RANGE VALIDATION
# ============================================================

print(
    "\n[7/8] Running temporal and spectral validation..."
)


event_time = parse_ee_time(
    df[
        "event_time"
    ]
)


latest_time = parse_ee_time(
    df[
        "sentinel_latest_time"
    ]
)


if event_time.isna().any():

    fail(
        "One or more FIRMS event timestamps "
        "could not be parsed."
    )


has_latest = (
    latest_time.notna()
)


future_scene = (
    has_latest
    & (
        latest_time
        >
        event_time
        + pd.Timedelta(
            seconds=1
        )
    )
)


future_count = int(
    future_scene.sum()
)


age_days = (
    event_time
    - latest_time
).dt.total_seconds() / 86400.0


too_old_count = int(
    (
        has_latest
        & (
            age_days > 14.1
        )
    ).sum()
)


negative_age_count = int(
    (
        has_latest
        & (
            age_days < -0.001
        )
    ).sum()
)


print(
    f"      Future scenes     : {future_count}"
)

print(
    f"      >14.1-day scenes  : {too_old_count}"
)

print(
    f"      Negative ages     : {negative_age_count}"
)


if future_count != 0:

    fail(
        "Future Sentinel imagery detected."
    )


if too_old_count != 0:

    fail(
        "Sentinel imagery outside 14-day "
        "lookback detected."
    )


if negative_age_count != 0:

    fail(
        "Negative image age detected."
    )


# ------------------------------------------------------------
# Index mathematical range
# ------------------------------------------------------------

for column in INDICES:

    values = df.loc[
        complete_evidence,
        column
    ]


    invalid = (
        (values < -1.001)
        | (values > 1.001)
    )


    count = int(
        invalid.sum()
    )


    print(
        f"      {column} outside [-1,1]: {count}"
    )


    if count != 0:

        fail(
            f"{column} contains mathematically "
            "invalid values."
        )


# ------------------------------------------------------------
# Gross reflectance sanity check
# ------------------------------------------------------------

reflectance_extremes = {}


for column in BANDS:

    values = df.loc[
        complete_evidence,
        column
    ]


    extreme = (
        (values < -0.2)
        | (values > 2.0)
    )


    count = int(
        extreme.sum()
    )


    reflectance_extremes[
        column
    ] = count


    print(
        f"      {column} gross extremes: {count}"
    )


# ============================================================
# 8. STATISTICS + SAVE
# ============================================================

print(
    "\n[8/8] Saving final validated satellite evidence..."
)


statistics = {}


for column in SATELLITE_FEATURES:

    values = df.loc[
        complete_evidence,
        column
    ]


    statistics[column] = {

        "count":
            int(
                values.count()
            ),

        "mean":
            float(
                values.mean()
            ),

        "std":
            float(
                values.std()
            ),

        "min":
            float(
                values.min()
            ),

        "max":
            float(
                values.max()
            ),

        "quantiles":
            quantile_dict(
                values
            )
    }


# ------------------------------------------------------------
# IMPORTANT:
#
# Keep the original exported QA columns.
# We do not overwrite history.
#
# The model-facing field is:
#
# satellite_evidence_available
# ------------------------------------------------------------


if "full_export_row" in df.columns:

    df[
        "full_export_row"
    ] = pd.to_numeric(
        df[
            "full_export_row"
        ],
        errors="coerce"
    )


    df = (
        df
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


df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)


report = {

    "validation_result":
        "PASS",

    "total_observations":
        int(
            len(df)
        ),

    "unique_observation_ids":
        int(
            df[
                "observation_id"
            ].nunique()
        ),

    "missing_expected_ids":
        len(
            missing_ids
        ),

    "unexpected_ids":
        len(
            unexpected_ids
        ),

    "usable_satellite_evidence":
        usable_count,

    "usable_satellite_evidence_pct":
        float(
            usable_count
            / EXPECTED_TOTAL
            * 100
        ),

    "unavailable_satellite_evidence":
        unavailable_count,

    "partial_satellite_evidence":
        int(
            partial_evidence.sum()
        ),

    "qa_disagreement": {

        "complete_evidence_but_exported_unavailable":
            qa_disagreement_count,

        "exported_available_but_incomplete":
            bad_exported_true_count,

        "interpretation":
            (
                "16 observations contain complete "
                "spectral/index reductions despite "
                "the auxiliary B8 count reduction "
                "returning zero. Raw Earth Engine QA "
                "fields are preserved. Model-facing "
                "availability is determined from actual "
                "complete finite feature evidence."
            )
    },

    "temporal_validation": {

        "future_scene_count":
            future_count,

        "older_than_14_1_days":
            too_old_count,

        "negative_age_count":
            negative_age_count,
    },

    "reflectance_gross_extremes":
        reflectance_extremes,

    "spectral_statistics":
        statistics,

    "availability_policy": (
        "All B4, B8, B11, B12, NDVI, NBR and NDMI "
        "must be finite."
    ),

    "missing_value_policy":
        "Missing satellite evidence remains missing; no zero filling.",

    "satellite_role":
        "Contextual model evidence, not ground-truth fire label.",

    "holdout_2025_accessed":
        False,
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


print(
    "\n" + "=" * 80
)

print(
    "FINAL SENTINEL VALIDATION: PASS"
)

print(
    "=" * 80
)


print(
    f"\nTotal observations       : {len(df):,}"
)

print(
    f"Usable satellite evidence: {usable_count:,} "
    f"({usable_count / EXPECTED_TOTAL * 100:.2f}%)"
)

print(
    f"Unavailable evidence     : {unavailable_count:,}"
)

print(
    f"Partial evidence         : "
    f"{int(partial_evidence.sum()):,}"
)

print(
    f"Known QA disagreements   : {qa_disagreement_count}"
)

print(
    f"Future leakage           : {future_count}"
)

print(
    f"Missing expected IDs     : {len(missing_ids)}"
)

print(
    "\nFinal dataset:"
)

print(
    OUTPUT_PARQUET
)

print(
    "\nReport:"
)

print(
    REPORT_FILE
)

print(
    "\n2025 HOLDOUT REMAINS LOCKED."
)

print(
    "\nSatellite evidence stage is complete."
)

print(
    "\nNEXT: defensible source-label construction."
)