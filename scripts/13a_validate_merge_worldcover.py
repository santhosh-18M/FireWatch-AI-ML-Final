from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = ROOT / "data" / "labels" / "worldcover" / "exports"

SOURCE_SAMPLE = (
    ROOT
    / "data"
    / "satellite"
    / "sentinel_sampling_2022_2024.parquet"
)

OUTPUT_DIR = ROOT / "data" / "labels" / "worldcover"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "worldcover_evidence_2022_2024.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "worldcover_validation.json"
)

EXPECTED_ROWS = 20_000
EXPECTED_BATCH_ROWS = 5_000

EXPECTED_FILES = [
    "firewatch_worldcover_batch_01.csv",
    "firewatch_worldcover_batch_02.csv",
    "firewatch_worldcover_batch_03.csv",
    "firewatch_worldcover_batch_04.csv",
]


# ============================================================
# WORLDCOVER CLASSES
# ============================================================

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

VALID_CODES = set(WORLDCOVER_CLASSES.keys())


# These are the fractions we actually exported.
FRACTION_TO_CLASS = {
    "worldcover_tree_fraction_100m": 10,
    "worldcover_shrub_fraction_100m": 20,
    "worldcover_grass_fraction_100m": 30,
    "worldcover_cropland_fraction_100m": 40,
    "worldcover_built_fraction_100m": 50,
    "worldcover_bare_fraction_100m": 60,
    "worldcover_water_fraction_100m": 80,
    "worldcover_wetland_fraction_100m": 90,
    "worldcover_mangrove_fraction_100m": 95,
}

FRACTION_COLUMNS = list(FRACTION_TO_CLASS.keys())


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 13A VALIDATION")
print("ESA WORLDCOVER MERGE + AUDIT — CORRECTED")
print("=" * 80)

print(
    "\nPurpose:"
    "\n- Merge four WorldCover exports."
    "\n- Verify exact 20,000 observation coverage."
    "\n- Validate categorical land-cover evidence."
    "\n- Derive dominant 100 m class from exported class fractions."
    "\n- Preserve raw Earth Engine mode separately for provenance."
)

print(
    "\nIMPORTANT:"
    "\n- No final source labels are assigned."
    "\n- Built-up does NOT mean industrial."
    "\n- Vegetation does NOT prove natural fire."
    "\n- 2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD EXPORTS
# ============================================================

print("\n[1/8] Loading WorldCover exports...")

frames = []
batch_summary = []

for batch_number, filename in enumerate(EXPECTED_FILES, start=1):

    path = EXPORT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing WorldCover export:\n{path}"
        )

    batch = pd.read_csv(path)

    print(
        f"      Batch {batch_number}: "
        f"{len(batch):,} rows"
    )

    if len(batch) != EXPECTED_BATCH_ROWS:
        raise RuntimeError(
            f"Batch {batch_number} expected "
            f"{EXPECTED_BATCH_ROWS:,} rows, "
            f"found {len(batch):,}."
        )

    batch["_validation_source_file"] = filename

    frames.append(batch)

    batch_summary.append(
        {
            "batch": batch_number,
            "file": filename,
            "rows": int(len(batch)),
        }
    )


df = pd.concat(
    frames,
    ignore_index=True
)

print(f"\n      Combined rows: {len(df):,}")

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} rows, "
        f"found {len(df):,}."
    )


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

print("\n[2/8] Checking required columns...")

required_columns = [
    "observation_id",
    "latitude",
    "longitude",
    "worldcover_point_class",
    "worldcover_point_available",
    "worldcover_mode_100m",
    "worldcover_total_pixels_100m",
] + FRACTION_COLUMNS

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("      Required columns present.")


# ============================================================
# 3. VERIFY IDS
# ============================================================

print("\n[3/8] Verifying observation IDs...")

duplicate_count = int(
    df["observation_id"].duplicated().sum()
)

print(f"      Duplicate IDs: {duplicate_count:,}")

if duplicate_count:
    raise RuntimeError(
        "Duplicate WorldCover observation IDs detected."
    )


source = pd.read_parquet(
    SOURCE_SAMPLE,
    columns=[
        "observation_id",
        "latitude",
        "longitude",
    ],
)

source["observation_id"] = (
    source["observation_id"].astype(str)
)

df["observation_id"] = (
    df["observation_id"].astype(str)
)

source_ids = set(source["observation_id"])
export_ids = set(df["observation_id"])

missing_ids = sorted(source_ids - export_ids)
unexpected_ids = sorted(export_ids - source_ids)

print(f"      Missing IDs   : {len(missing_ids):,}")
print(f"      Unexpected IDs: {len(unexpected_ids):,}")

if missing_ids or unexpected_ids:
    raise RuntimeError(
        "WorldCover IDs do not exactly match "
        "the source sample."
    )


# ============================================================
# 4. NORMALIZE NUMERIC COLUMNS
# ============================================================

print("\n[4/8] Normalizing values...")

numeric_columns = [
    "latitude",
    "longitude",
    "worldcover_point_class",
    "worldcover_mode_100m",
    "worldcover_total_pixels_100m",
] + FRACTION_COLUMNS

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Preserve the raw EE mode.
# We are NOT going to use it as our validated dominant class.

df[
    "worldcover_mode_100m_raw"
] = df["worldcover_mode_100m"]


# ============================================================
# 5. VALIDATE POINT CLASS
# ============================================================

print("\n[5/8] Validating point classes...")

point_non_null = (
    df["worldcover_point_class"]
    .dropna()
)

invalid_point_codes = sorted(
    set(point_non_null.astype(int))
    - VALID_CODES
)

print(
    f"      Point class available: "
    f"{len(point_non_null):,}"
)

print(
    f"      Point class missing  : "
    f"{df['worldcover_point_class'].isna().sum():,}"
)

print(
    f"      Invalid point codes  : "
    f"{invalid_point_codes}"
)

if invalid_point_codes:
    raise RuntimeError(
        "Invalid WorldCover point classes detected."
    )


df[
    "worldcover_point_class_name"
] = (
    df["worldcover_point_class"]
    .map(WORLDCOVER_CLASSES)
)


# ============================================================
# 6. VALIDATE FRACTIONS
# ============================================================

print("\n[6/8] Validating 100 m fractions...")

invalid_fraction_rows = pd.Series(
    False,
    index=df.index
)

for column in FRACTION_COLUMNS:

    invalid = (
        df[column].notna()
        &
        (
            (df[column] < -1e-9)
            |
            (df[column] > 1.000001)
        )
    )

    invalid_fraction_rows |= invalid


invalid_fraction_count = int(
    invalid_fraction_rows.sum()
)

print(
    f"      Fraction values outside [0,1]: "
    f"{invalid_fraction_count:,}"
)

if invalid_fraction_count:
    raise RuntimeError(
        "Invalid WorldCover fraction values detected."
    )


# ============================================================
# DERIVE VALIDATED DOMINANT CLASS
# ============================================================

print(
    "\n      Deriving dominant 100 m class "
    "from categorical fractions..."
)


fraction_matrix = (
    df[FRACTION_COLUMNS]
    .copy()
)


# Find the exported class with the largest fractional coverage.
#
# NOTE:
# Snow/Ice (70) and Moss/Lichen (100) were not exported as
# fractions in Step 13A. If none of the exported fractions
# contains evidence, we fall back to the valid point class.

fraction_values = (
    fraction_matrix
    .fillna(-1)
    .to_numpy()
)

max_positions = np.argmax(
    fraction_values,
    axis=1
)

max_values = np.max(
    fraction_values,
    axis=1
)

fraction_class_codes = np.array(
    [
        FRACTION_TO_CLASS[column]
        for column in FRACTION_COLUMNS
    ]
)


derived_codes = fraction_class_codes[
    max_positions
].astype(float)


# No valid exported fraction -> fallback to point class.
no_fraction_evidence = (
    max_values < 0
)

derived_codes[
    no_fraction_evidence
] = (
    df.loc[
        no_fraction_evidence,
        "worldcover_point_class"
    ]
    .to_numpy()
)


df[
    "worldcover_dominant_class_100m"
] = derived_codes


df[
    "worldcover_dominant_class_100m_name"
] = (
    df[
        "worldcover_dominant_class_100m"
    ]
    .map(WORLDCOVER_CLASSES)
)


# Maximum observed fraction = dominance strength.
df[
    "worldcover_dominant_fraction_100m"
] = np.where(
    max_values >= 0,
    max_values,
    np.nan
)


# Sum of exported class fractions.
df[
    "worldcover_selected_fraction_sum_100m"
] = (
    df[FRACTION_COLUMNS]
    .sum(
        axis=1,
        min_count=1
    )
)


fraction_sum_over_one = int(
    (
        df[
            "worldcover_selected_fraction_sum_100m"
        ]
        > 1.0001
    ).sum()
)


print(
    f"      Selected fraction sum > 1: "
    f"{fraction_sum_over_one:,}"
)

if fraction_sum_over_one:
    raise RuntimeError(
        "WorldCover class fractions exceed 1."
    )


# ============================================================
# EVIDENCE AVAILABILITY
# ============================================================

df[
    "worldcover_evidence_available"
] = (
    df[
        "worldcover_point_class"
    ].notna()
    |
    df[
        FRACTION_COLUMNS
    ].notna().any(axis=1)
)


available_count = int(
    df[
        "worldcover_evidence_available"
    ].sum()
)

unavailable_count = (
    len(df)
    - available_count
)


print(
    f"      Evidence available: "
    f"{available_count:,}"
)

print(
    f"      Evidence missing  : "
    f"{unavailable_count:,}"
)


# ============================================================
# 7. COORDINATE CHECK
# ============================================================

print("\n[7/8] Checking coordinate consistency...")


coordinate_check = (
    source.rename(
        columns={
            "latitude": "source_latitude",
            "longitude": "source_longitude",
        }
    )
    .merge(
        df[
            [
                "observation_id",
                "latitude",
                "longitude",
            ]
        ],
        on="observation_id",
        how="inner",
        validate="one_to_one",
    )
)


lat_diff = (
    coordinate_check["source_latitude"]
    - coordinate_check["latitude"]
).abs()

lon_diff = (
    coordinate_check["source_longitude"]
    - coordinate_check["longitude"]
).abs()


coordinate_mismatch = int(
    (
        (lat_diff > 1e-6)
        |
        (lon_diff > 1e-6)
    ).sum()
)


print(
    f"      Coordinate mismatches: "
    f"{coordinate_mismatch:,}"
)

if coordinate_mismatch:
    raise RuntimeError(
        "Coordinate mismatch detected."
    )


# ============================================================
# 8. AUDIT
# ============================================================

print("\n[8/8] Auditing WorldCover evidence...")


point_distribution = (
    df[
        "worldcover_point_class_name"
    ]
    .fillna("MISSING")
    .value_counts()
)


dominant_distribution = (
    df[
        "worldcover_dominant_class_100m_name"
    ]
    .fillna("MISSING")
    .value_counts()
)


print("\n      Point-class distribution:")

for name, count in point_distribution.items():

    percentage = (
        count / len(df) * 100
    )

    print(
        f"        {name:<28} "
        f"{count:>6,} "
        f"({percentage:>6.2f}%)"
    )


print(
    "\n      Derived dominant class within 100 m:"
)

for name, count in dominant_distribution.items():

    percentage = (
        count / len(df) * 100
    )

    print(
        f"        {name:<28} "
        f"{count:>6,} "
        f"({percentage:>6.2f}%)"
    )


print("\n      Mean 100 m fractions:")


fraction_means = {}

for column in FRACTION_COLUMNS:

    value = float(
        df[column].mean()
    )

    fraction_means[column] = value

    print(
        f"        {column:<42} "
        f"{value:.4f}"
    )


# ============================================================
# RAW MODE DIAGNOSTIC
# ============================================================

raw_mode_values = (
    df[
        "worldcover_mode_100m_raw"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


print(
    "\n      Raw Earth Engine mode values "
    "(diagnostic only):"
)

for value, count in raw_mode_values.items():

    print(
        f"        {str(value):<10} "
        f"{count:>6,}"
    )


# ============================================================
# PROVENANCE
# ============================================================

df[
    "worldcover_reference_year"
] = 2021

df[
    "worldcover_reference_product"
] = "ESA WorldCover v200"

df[
    "worldcover_role"
] = (
    "external_landcover_reference_evidence"
)

df[
    "worldcover_dominant_method"
] = (
    "maximum_exported_class_fraction_100m"
)


# ============================================================
# SAVE
# ============================================================

df = (
    df.sort_values(
        "observation_id"
    )
    .reset_index(drop=True)
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


report = {
    "stage":
        "worldcover_validation_corrected",

    "source":
        "ESA WorldCover v200",

    "reference_year":
        2021,

    "rows":
        int(len(df)),

    "duplicate_ids":
        duplicate_count,

    "missing_source_ids":
        len(missing_ids),

    "unexpected_ids":
        len(unexpected_ids),

    "coordinate_mismatches":
        coordinate_mismatch,

    "evidence_available":
        available_count,

    "evidence_missing":
        unavailable_count,

    "invalid_point_codes":
        invalid_point_codes,

    "invalid_fraction_rows":
        invalid_fraction_count,

    "selected_fraction_sum_over_one":
        fraction_sum_over_one,

    "point_class_distribution": {
        str(k): int(v)
        for k, v
        in point_distribution.items()
    },

    "derived_dominant_100m_distribution": {
        str(k): int(v)
        for k, v
        in dominant_distribution.items()
    },

    "mean_fractions_100m":
        fraction_means,

    "raw_ee_mode_note":
        (
            "Raw reduceRegion mode values are retained "
            "for provenance but are not used as the "
            "validated categorical dominant class."
        ),

    "dominant_class_method":
        (
            "Class having maximum exported WorldCover "
            "fraction within 100 m. Point class used only "
            "as fallback when fraction evidence is absent."
        ),

    "batch_summary":
        batch_summary,

    "final_source_labels_assigned":
        False,

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
) as file:

    json.dump(
        report,
        file,
        indent=2
    )


print("\n" + "=" * 80)
print("WORLDCOVER VALIDATION COMPLETE")
print("=" * 80)

print(
    f"\nValidated rows : {len(df):,}"
)

print(
    f"Evidence       : {available_count:,}"
)

print(
    f"Missing        : {unavailable_count:,}"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nReport:")
print(REPORT_FILE)

print(
    "\nFINAL SOURCE LABELS ASSIGNED: NO"
)

print(
    "2025 HOLDOUT ACCESSED: NO"
)

print(
    "\nNEXT:"
    "\nAdd independent industrial-facility evidence."
)