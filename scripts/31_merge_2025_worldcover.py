from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "worldcover_2025_exports"
)

SAMPLE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "evaluation_sample_2025.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "worldcover_evidence_2025.parquet"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "worldcover_evidence_2025.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "worldcover_evidence_2025_summary.json"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


EXPECTED_ROWS = 20_000
EXPECTED_FILES = 20

WORLD_COVER_CLASSES = {
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

FRACTION_COLUMNS = [
    f"wc_fraction_{code}"
    for code in WORLD_COVER_CLASSES
]


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 31")
print("MERGE + VALIDATE 2025 WORLDCOVER EVIDENCE")
print("=" * 80)

print(
    "\nPurpose:"
    "\n- Merge all Earth Engine WorldCover exports."
    "\n- Verify all 20,000 holdout observations."
    "\n- Detect duplicates and missing observations."
    "\n- Derive dominant 100 m WorldCover class."
    "\n- Join evidence back to frozen 2025 sample."
    "\n- Generate NO source labels."
    "\n- Execute NO classifier."
)


# ============================================================
# 1. FIND EXPORT FILES
# ============================================================

print("\n[1/9] Finding WorldCover export files...")


if not EXPORT_DIR.exists():
    raise FileNotFoundError(
        f"Export directory not found:\n{EXPORT_DIR}"
    )


csv_files = sorted(
    EXPORT_DIR.glob(
        "firewatch_worldcover_2025_batch_*.csv"
    )
)


print(f"      CSV files found: {len(csv_files)}")


for file in csv_files:
    print(f"      - {file.name}")


if len(csv_files) != EXPECTED_FILES:
    raise RuntimeError(
        f"Expected {EXPECTED_FILES} WorldCover CSV files, "
        f"but found {len(csv_files)}."
    )


# ============================================================
# 2. LOAD EXPORTS
# ============================================================

print("\n[2/9] Loading and combining exports...")


frames = []

for file in csv_files:

    part = pd.read_csv(file)

    part["export_source_file"] = file.name

    frames.append(part)


wc = pd.concat(
    frames,
    ignore_index=True
)


print(f"      Combined rows: {len(wc):,}")


if len(wc) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} exported rows, "
        f"found {len(wc):,}."
    )


# ============================================================
# 3. CLEAN EARTH ENGINE COLUMN NAMES
# ============================================================

print("\n[3/9] Validating export schema...")


# Earth Engine CSV exports can include system:index and .geo.
# They are not evidence features and can safely be ignored.

required_columns = [
    "observation_id",
    "latitude",
    "longitude",
    "worldcover_point_class",
    *FRACTION_COLUMNS,
]


missing_columns = [
    column
    for column in required_columns
    if column not in wc.columns
]


if missing_columns:

    print("\nColumns actually found:")

    for column in wc.columns:
        print(f"      {column}")

    raise RuntimeError(
        "\nMissing required WorldCover columns:\n"
        + "\n".join(missing_columns)
    )


print("      Required WorldCover fields present.")


# ============================================================
# 4. OBSERVATION ID VALIDATION
# ============================================================

print("\n[4/9] Validating observation IDs...")


wc["observation_id"] = (
    wc["observation_id"]
    .astype(str)
    .str.strip()
)


duplicate_mask = wc[
    "observation_id"
].duplicated(keep=False)


duplicate_count = int(
    duplicate_mask.sum()
)


unique_count = int(
    wc["observation_id"].nunique()
)


print(f"      Unique IDs: {unique_count:,}")
print(f"      Duplicate rows: {duplicate_count:,}")


if duplicate_count > 0:

    duplicate_examples = (
        wc.loc[
            duplicate_mask,
            [
                "observation_id",
                "export_source_file",
            ],
        ]
        .sort_values("observation_id")
        .head(20)
    )

    print("\nDuplicate examples:")
    print(
        duplicate_examples.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Duplicate WorldCover observation IDs detected."
    )


if unique_count != EXPECTED_ROWS:
    raise RuntimeError(
        "WorldCover exports do not contain exactly "
        "20,000 unique observations."
    )


# ============================================================
# 5. LOAD FROZEN EVALUATION SAMPLE
# ============================================================

print("\n[5/9] Loading frozen 2025 evaluation sample...")


if not SAMPLE_FILE.exists():
    raise FileNotFoundError(
        f"Frozen sample not found:\n{SAMPLE_FILE}"
    )


sample = pd.read_parquet(
    SAMPLE_FILE
)


print(f"      Sample rows: {len(sample):,}")


if len(sample) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} sample rows, "
        f"found {len(sample):,}."
    )


if "observation_id" not in sample.columns:
    raise RuntimeError(
        "observation_id missing from frozen sample."
    )


sample["observation_id"] = (
    sample["observation_id"]
    .astype(str)
    .str.strip()
)


if sample[
    "observation_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate observation IDs in frozen sample."
    )


sample_ids = set(
    sample["observation_id"]
)


wc_ids = set(
    wc["observation_id"]
)


missing_from_worldcover = (
    sample_ids - wc_ids
)

unexpected_worldcover = (
    wc_ids - sample_ids
)


print(
    f"      Missing WorldCover IDs: "
    f"{len(missing_from_worldcover):,}"
)

print(
    f"      Unexpected WorldCover IDs: "
    f"{len(unexpected_worldcover):,}"
)


if missing_from_worldcover:

    print(
        "\nExample missing IDs:"
    )

    for value in list(
        sorted(
            missing_from_worldcover
        )
    )[:10]:

        print(f"      {value}")


if unexpected_worldcover:

    print(
        "\nExample unexpected IDs:"
    )

    for value in list(
        sorted(
            unexpected_worldcover
        )
    )[:10]:

        print(f"      {value}")


if (
    missing_from_worldcover
    or unexpected_worldcover
):

    raise RuntimeError(
        "WorldCover IDs do not exactly match "
        "the frozen evaluation sample."
    )


# ============================================================
# 6. NORMALIZE WORLDCOVER VALUES
# ============================================================

print("\n[6/9] Normalizing WorldCover evidence...")


wc[
    "worldcover_point_class"
] = pd.to_numeric(
    wc["worldcover_point_class"],
    errors="coerce",
)


for column in FRACTION_COLUMNS:

    wc[column] = pd.to_numeric(
        wc[column],
        errors="coerce",
    )


wc[
    "worldcover_point_available"
] = (
    wc[
        "worldcover_point_class"
    ]
    .notna()
)


# ============================================================
# 7. DERIVE DOMINANT 100 M CLASS
# ============================================================

print(
    "\n[7/9] Deriving dominant 100 m WorldCover class..."
)


fraction_matrix = wc[
    FRACTION_COLUMNS
].to_numpy(
    dtype=float
)


all_missing = np.isnan(
    fraction_matrix
).all(
    axis=1
)


# Replace NaN temporarily only for argmax.
safe_fraction_matrix = np.where(
    np.isnan(
        fraction_matrix
    ),
    -np.inf,
    fraction_matrix,
)


dominant_indices = np.argmax(
    safe_fraction_matrix,
    axis=1,
)


class_codes = np.array(
    list(
        WORLD_COVER_CLASSES.keys()
    ),
    dtype=int,
)


dominant_classes = class_codes[
    dominant_indices
].astype(float)


dominant_classes[
    all_missing
] = np.nan


wc[
    "worldcover_dominant_100m"
] = dominant_classes


wc[
    "worldcover_100m_available"
] = ~all_missing


# ------------------------------------------------------------
# Dominant fraction
# ------------------------------------------------------------

dominant_fraction = np.max(
    safe_fraction_matrix,
    axis=1,
)


dominant_fraction[
    all_missing
] = np.nan


wc[
    "worldcover_dominant_fraction_100m"
] = dominant_fraction


# ------------------------------------------------------------
# Human-readable class names
# ------------------------------------------------------------

wc[
    "worldcover_point_name"
] = (
    wc[
        "worldcover_point_class"
    ]
    .map(
        WORLD_COVER_CLASSES
    )
)


wc[
    "worldcover_dominant_100m_name"
] = (
    wc[
        "worldcover_dominant_100m"
    ]
    .map(
        WORLD_COVER_CLASSES
    )
)


point_available = int(
    wc[
        "worldcover_point_available"
    ].sum()
)


buffer_available = int(
    wc[
        "worldcover_100m_available"
    ].sum()
)


print(
    f"      Point evidence available: "
    f"{point_available:,} "
    f"({point_available / len(wc) * 100:.2f}%)"
)


print(
    f"      100 m evidence available: "
    f"{buffer_available:,} "
    f"({buffer_available / len(wc) * 100:.2f}%)"
)


print(
    "\n      Dominant 100 m distribution:"
)


dominant_distribution = (
    wc[
        "worldcover_dominant_100m"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


print(
    dominant_distribution.to_string()
)


# ============================================================
# 8. JOIN BACK TO FROZEN SAMPLE
# ============================================================

print(
    "\n[8/9] Joining evidence to frozen sample..."
)


evidence_columns = [
    "observation_id",
    "worldcover_point_class",
    "worldcover_point_available",
    "worldcover_point_name",
    "worldcover_dominant_100m",
    "worldcover_dominant_100m_name",
    "worldcover_dominant_fraction_100m",
    "worldcover_100m_available",
    *FRACTION_COLUMNS,
]


evidence = wc[
    evidence_columns
].copy()


final = sample.merge(
    evidence,
    on="observation_id",
    how="left",
    validate="one_to_one",
)


if len(final) != EXPECTED_ROWS:
    raise RuntimeError(
        "Row count changed during WorldCover merge."
    )


missing_after_merge = int(
    final[
        "worldcover_point_class"
    ]
    .isna()
    .sum()
)


print(
    f"      Final rows: {len(final):,}"
)


print(
    f"      Missing point evidence after merge: "
    f"{missing_after_merge:,}"
)


# ============================================================
# 9. SAVE
# ============================================================

print(
    "\n[9/9] Saving final WorldCover evidence..."
)


final.to_parquet(
    OUTPUT_FILE,
    index=False,
)


final.to_csv(
    OUTPUT_CSV,
    index=False,
)


point_distribution = (
    final[
        "worldcover_point_class"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


dominant_distribution = (
    final[
        "worldcover_dominant_100m"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


def distribution_to_dict(series):

    result = {}

    for key, value in series.items():

        if pd.isna(key):
            label = "missing"
        else:
            label = str(int(key))

        result[label] = int(value)

    return result


report = {
    "stage":
        "2025_worldcover_merge_validation",

    "evaluation_rows":
        int(len(sample)),

    "export_files":
        int(len(csv_files)),

    "export_rows":
        int(len(wc)),

    "unique_export_observation_ids":
        unique_count,

    "duplicate_export_rows":
        duplicate_count,

    "missing_ids":
        int(len(missing_from_worldcover)),

    "unexpected_ids":
        int(len(unexpected_worldcover)),

    "point_evidence_available":
        point_available,

    "point_evidence_coverage_pct":
        float(
            point_available
            / len(final)
            * 100
        ),

    "buffer_evidence_available":
        buffer_available,

    "buffer_evidence_coverage_pct":
        float(
            buffer_available
            / len(final)
            * 100
        ),

    "worldcover_point_distribution":
        distribution_to_dict(
            point_distribution
        ),

    "worldcover_dominant_100m_distribution":
        distribution_to_dict(
            dominant_distribution
        ),

    "dataset":
        "ESA_WorldCover_2021_v200",

    "buffer_meters":
        100,

    "scale_meters":
        10,

    "source_labels_generated":
        False,

    "model_executed":
        False,

    "output_parquet":
        str(OUTPUT_FILE),

    "output_csv":
        str(OUTPUT_CSV),

    "notes": [
        (
            "WorldCover 2021 is static independent "
            "land-cover reference evidence."
        ),
        (
            "WorldCover class does not by itself "
            "confirm the cause of a 2025 FIRMS anomaly."
        ),
        (
            "Dominant 100 m class was derived from "
            "the exported class fractions."
        ),
        (
            "No model predictions were generated "
            "during this step."
        ),
        (
            "No source labels were generated "
            "during this step."
        ),
    ],
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


print(
    "\n" + "=" * 80
)

print(
    "STEP 31 COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\nEvaluation sample: "
    f"{len(sample):,}"
)

print(
    f"WorldCover exports: "
    f"{len(csv_files)}"
)

print(
    f"Export rows: "
    f"{len(wc):,}"
)

print(
    f"Unique observation IDs: "
    f"{unique_count:,}"
)

print(
    f"Point coverage: "
    f"{point_available / len(final) * 100:.2f}%"
)

print(
    f"100 m coverage: "
    f"{buffer_available / len(final) * 100:.2f}%"
)


print(
    f"\nSaved:"
    f"\n{OUTPUT_FILE}"
)


print(
    f"\nReport:"
    f"\n{REPORT_FILE}"
)


print(
    "\nIMPORTANT:"
    "\n- 2025 WorldCover evidence is now prepared."
    "\n- No source labels have been generated."
    "\n- No model has been executed."
    "\n- Next: frozen independent industrial-reference matching."
)