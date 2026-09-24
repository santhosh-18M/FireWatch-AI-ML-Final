from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_FILE = (
    ROOT / "data" / "labels"
    / "label_evidence_2022_2024.parquet"
)

WORLD_FILE = (
    ROOT / "data" / "labels"
    / "worldcover"
    / "worldcover_evidence_2022_2024.parquet"
)

INDUSTRIAL_FILE = (
    ROOT / "data" / "labels"
    / "industrial_reference"
    / "combined_industrial_matches_2022_2024.parquet"
)

OUTPUT_FILE = (
    ROOT / "data" / "labels"
    / "source_labels_v1_2022_2024.parquet"
)

OUTPUT_CSV = (
    ROOT / "data" / "labels"
    / "source_labels_v1_review.csv"
)

REPORT_FILE = (
    ROOT / "reports" / "experiments"
    / "source_labels_v1_report.json"
)

EXPECTED_ROWS = 20_000

INDUSTRIAL_STRONG_KM = 2.0
INDUSTRIAL_EXCLUSION_KM = 5.0


# ============================================================
# WORLD COVER CLASSES
# ============================================================

# ESA WorldCover 2021 codes
#
# 10 = Tree cover
# 20 = Shrubland
# 30 = Grassland
# 40 = Cropland
# 50 = Built-up
# 60 = Bare / sparse vegetation
# 70 = Snow and ice
# 80 = Permanent water
# 90 = Herbaceous wetland
# 95 = Mangroves
# 100 = Moss and lichen

VEGETATION_CODES = {
    10,
    20,
    30,
    40,
}

WORLD_CLASS_NAMES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangroves",
    100: "moss_lichen",
}


# ============================================================
# HELPERS
# ============================================================

def detect_column(df, candidates, description):

    for column in candidates:
        if column in df.columns:
            return column

    raise RuntimeError(
        f"Could not find {description}.\n"
        f"Tried:\n"
        + "\n".join(candidates)
    )


def safe_int_code(value):

    if pd.isna(value):
        return np.nan

    try:
        return int(round(float(value)))

    except Exception:
        return np.nan


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 14")
print("BUILD SOURCE LABELS V1")
print("=" * 80)

print(
    "\nTarget classes:"
    "\n  0. industrial_fire_candidate"
    "\n  1. natural_vegetation_fire"
    "\n  2. other_uncertain"
)

print(
    "\nImportant:"
    "\n- Source class and temporal behavior remain separate."
    "\n- Persistence does NOT create an industrial label."
    "\n- OSM does NOT create the target."
    "\n- Sentinel spectral values do NOT create the target."
    "\n- FIRMS FRP/brightness do NOT create the target."
    "\n- No A/verified labels are automatically assigned."
    "\n- 2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/9] Loading datasets...")

base = pd.read_parquet(BASE_FILE)
world = pd.read_parquet(WORLD_FILE)
industrial = pd.read_parquet(INDUSTRIAL_FILE)

print(f"      Base       : {len(base):,}")
print(f"      WorldCover : {len(world):,}")
print(f"      Industrial : {len(industrial):,}")


for name, frame in [
    ("base", base),
    ("worldcover", world),
    ("industrial", industrial),
]:

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_ROWS:,} rows, "
            f"found {len(frame):,}"
        )

    if "observation_id" not in frame.columns:
        raise RuntimeError(
            f"{name}: observation_id missing."
        )

    if frame["observation_id"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicate observation IDs."
        )


# ============================================================
# 2. DETECT WORLDCOVER COLUMNS
# ============================================================

print("\n[2/9] Detecting WorldCover columns...")

point_col = detect_column(
    world,
    [
        "worldcover_point_class",
        "worldcover_point",
        "point_class",
        "worldcover_class",
    ],
    "WorldCover point class column",
)

dominant_col = detect_column(
    world,
    [
        "worldcover_dominant_100m",
        "worldcover_dominant_class_100m",
        "dominant_class_100m",
        "worldcover_mode_100m_derived",
        "derived_dominant_100m",
    ],
    "WorldCover derived dominant 100m class",
)

print(f"      Point class   : {point_col}")
print(f"      Dominant 100m : {dominant_col}")


# ============================================================
# 3. REDUCE INPUT TABLES
# ============================================================

print("\n[3/9] Preparing evidence columns...")

world_small = world[
    [
        "observation_id",
        point_col,
        dominant_col,
    ]
].copy()

world_small = world_small.rename(
    columns={
        point_col:
            "label_worldcover_point",

        dominant_col:
            "label_worldcover_dominant_100m",
    }
)


industrial_required = [
    "observation_id",
    "independent_industrial_distance_km",
    "independent_industrial_reference_name",
    "independent_industrial_reference_type",
    "independent_industrial_reference_subtype",
    "independent_industrial_reference_source",
]

missing = [
    c for c in industrial_required
    if c not in industrial.columns
]

if missing:
    raise RuntimeError(
        "Industrial evidence missing columns:\n"
        + "\n".join(missing)
    )

industrial_small = industrial[
    industrial_required
].copy()


# ============================================================
# 4. MERGE
# ============================================================

print("\n[4/9] Merging evidence...")

data = base.merge(
    world_small,
    on="observation_id",
    how="left",
    validate="one_to_one",
)

data = data.merge(
    industrial_small,
    on="observation_id",
    how="left",
    validate="one_to_one",
)

if len(data) != EXPECTED_ROWS:
    raise RuntimeError(
        "Row count changed after merge."
    )

print(f"      Merged rows: {len(data):,}")


# ============================================================
# 5. NORMALIZE WORLDCOVER
# ============================================================

print("\n[5/9] Normalizing WorldCover evidence...")

data["label_worldcover_point"] = (
    data[
        "label_worldcover_point"
    ]
    .apply(safe_int_code)
)

data["label_worldcover_dominant_100m"] = (
    data[
        "label_worldcover_dominant_100m"
    ]
    .apply(safe_int_code)
)


data["worldcover_point_name"] = (
    data[
        "label_worldcover_point"
    ]
    .map(WORLD_CLASS_NAMES)
    .fillna("unknown")
)

data["worldcover_dominant_100m_name"] = (
    data[
        "label_worldcover_dominant_100m"
    ]
    .map(WORLD_CLASS_NAMES)
    .fillna("unknown")
)


data["worldcover_point_vegetation"] = (
    data[
        "label_worldcover_point"
    ]
    .isin(VEGETATION_CODES)
)

data["worldcover_dominant_vegetation"] = (
    data[
        "label_worldcover_dominant_100m"
    ]
    .isin(VEGETATION_CODES)
)


# Stronger vegetation evidence requires agreement between
# the FIRMS point class and dominant 100 m neighborhood class.

data["independent_vegetation_evidence"] = (
    data["worldcover_point_vegetation"]
    &
    data["worldcover_dominant_vegetation"]
)


print(
    "      Point vegetation       : "
    f"{data['worldcover_point_vegetation'].sum():,}"
)

print(
    "      Dominant vegetation    : "
    f"{data['worldcover_dominant_vegetation'].sum():,}"
)

print(
    "      Both vegetation        : "
    f"{data['independent_vegetation_evidence'].sum():,}"
)


# ============================================================
# 6. INDUSTRIAL EVIDENCE BANDS
# ============================================================

print("\n[6/9] Creating industrial evidence bands...")


distance = pd.to_numeric(
    data[
        "independent_industrial_distance_km"
    ],
    errors="coerce",
)


data["industrial_reference_le_1km"] = (
    distance <= 1.0
)

data["industrial_reference_le_2km"] = (
    distance <= INDUSTRIAL_STRONG_KM
)

data["industrial_reference_2_to_3km"] = (
    (distance > 2.0)
    &
    (distance <= 3.0)
)

data["industrial_reference_le_5km"] = (
    distance <= INDUSTRIAL_EXCLUSION_KM
)


print(
    "      <= 1 km : "
    f"{data['industrial_reference_le_1km'].sum():,}"
)

print(
    "      <= 2 km : "
    f"{data['industrial_reference_le_2km'].sum():,}"
)

print(
    "      2-3 km  : "
    f"{data['industrial_reference_2_to_3km'].sum():,}"
)

print(
    "      <= 5 km : "
    f"{data['industrial_reference_le_5km'].sum():,}"
)


# ============================================================
# 7. BUILD SOURCE LABEL
# ============================================================

print("\n[7/9] Constructing conservative source labels...")


# Default: ambiguous / insufficient evidence.
data["source_class"] = (
    "other_uncertain"
)

data["label_quality"] = (
    "C_AMBIGUOUS"
)

data["label_source"] = (
    "insufficient_or_conflicting_evidence"
)

data["label_evidence"] = (
    "No sufficiently strong independent source evidence."
)


# ------------------------------------------------------------
# NATURAL WEAK LABEL
# ------------------------------------------------------------
#
# Requirements:
# - WorldCover point is vegetation/agriculture.
# - Dominant 100m class is vegetation/agriculture.
# - No independent industrial reference within 5 km.
#
# Note:
# WorldCover describes land cover, not fire cause.
# Therefore this remains a B weak label.

natural_mask = (
    data[
        "independent_vegetation_evidence"
    ]
    &
    ~data[
        "industrial_reference_le_5km"
    ]
)


data.loc[
    natural_mask,
    "source_class"
] = "natural_vegetation_fire"


data.loc[
    natural_mask,
    "label_quality"
] = "B_STRONG_WEAK_LABEL"


data.loc[
    natural_mask,
    "label_source"
] = "esa_worldcover_2021"


data.loc[
    natural_mask,
    "label_evidence"
] = (
    "WorldCover point and dominant 100m neighborhood "
    "both indicate vegetation/agricultural land cover, "
    "with no independent industrial reference within 5 km."
)


# ------------------------------------------------------------
# INDUSTRIAL WEAK LABEL
# ------------------------------------------------------------
#
# Primary positive evidence:
# independent industrial facility <= 2 km.
#
# This intentionally overrides the natural weak label if
# WorldCover says vegetation around an independently known
# industrial facility. That case is recorded as a conflict
# rather than silently discarded.

industrial_mask = (
    data[
        "industrial_reference_le_2km"
    ]
)


data.loc[
    industrial_mask,
    "source_class"
] = "industrial_fire_candidate"


data.loc[
    industrial_mask,
    "label_quality"
] = "B_STRONG_WEAK_LABEL"


data.loc[
    industrial_mask,
    "label_source"
] = (
    "independent_industrial_reference"
)


data.loc[
    industrial_mask,
    "label_evidence"
] = (
    "FIRMS observation lies within 2 km of an "
    "independently catalogued industrial thermal facility."
)


# Strongest subset gets a more specific B tier.
industrial_1km = (
    data[
        "industrial_reference_le_1km"
    ]
)


data.loc[
    industrial_1km,
    "label_quality"
] = "B_HIGH_CONFIDENCE_WEAK_LABEL"


data.loc[
    industrial_1km,
    "label_evidence"
] = (
    "FIRMS observation lies within 1 km of an "
    "independently catalogued industrial thermal facility."
)


# ------------------------------------------------------------
# CONFLICT FLAGS
# ------------------------------------------------------------

data["label_conflict_industrial_vs_vegetation"] = (
    data[
        "industrial_reference_le_2km"
    ]
    &
    data[
        "independent_vegetation_evidence"
    ]
)


data["label_review_priority"] = "normal"


data.loc[
    data[
        "label_conflict_industrial_vs_vegetation"
    ],
    "label_review_priority"
] = "high"


data.loc[
    data[
        "industrial_reference_2_to_3km"
    ],
    "label_review_priority"
] = "high"


# 2-3 km is intentionally NOT automatically industrial.
# It stays OTHER_UNCERTAIN for now unless manually reviewed.


# ============================================================
# 8. TEMPORAL STATUS — SEPARATE TARGET DIMENSION
# ============================================================

print("\n[8/9] Building separate temporal status...")


if (
    "detections_30d" not in data.columns
    or
    "distinct_active_days_30d" not in data.columns
):
    raise RuntimeError(
        "Temporal features missing."
    )


detections = pd.to_numeric(
    data["detections_30d"],
    errors="coerce",
).fillna(0)


active_days = pd.to_numeric(
    data["distinct_active_days_30d"],
    errors="coerce",
).fillna(0)


data["temporal_status"] = "EPISODIC"


recurrent_mask = (
    (detections >= 3)
    &
    (active_days < 5)
)

persistent_mask = (
    (detections >= 3)
    &
    (active_days >= 5)
)


data.loc[
    recurrent_mask,
    "temporal_status"
] = "RECURRENT"


data.loc[
    persistent_mask,
    "temporal_status"
] = "PERSISTENT"


# ============================================================
# LABEL AUDIT
# ============================================================

print("\n      Source-class distribution:")

class_counts = (
    data[
        "source_class"
    ]
    .value_counts()
)

print(
    class_counts.to_string()
)


print("\n      Label-quality distribution:")

quality_counts = (
    data[
        "label_quality"
    ]
    .value_counts()
)

print(
    quality_counts.to_string()
)


print("\n      Temporal-status distribution:")

temporal_counts = (
    data[
        "temporal_status"
    ]
    .value_counts()
)

print(
    temporal_counts.to_string()
)


conflict_count = int(
    data[
        "label_conflict_industrial_vs_vegetation"
    ].sum()
)

review_2_3 = int(
    data[
        "industrial_reference_2_to_3km"
    ].sum()
)


print(
    "\n      Industrial-vs-vegetation conflicts: "
    f"{conflict_count:,}"
)

print(
    "      Industrial references 2-3 km "
    "(review, not auto-label): "
    f"{review_2_3:,}"
)


# Cross-tab: source × temporal
cross_tab = pd.crosstab(
    data["source_class"],
    data["temporal_status"],
)

print(
    "\n      Source × temporal:"
)

print(
    cross_tab.to_string()
)


# ============================================================
# 9. SAVE
# ============================================================

print("\n[9/9] Saving label dataset...")


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


data.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# REVIEW FILE
# ------------------------------------------------------------
#
# Include:
# 1. every industrial label
# 2. every 2-3 km industrial candidate
# 3. every industrial/vegetation conflict
# 4. sample of natural labels
# 5. sample of uncertain labels

industrial_review = data[
    data[
        "source_class"
    ].eq(
        "industrial_fire_candidate"
    )
].copy()


near_review = data[
    data[
        "industrial_reference_2_to_3km"
    ]
].copy()


conflict_review = data[
    data[
        "label_conflict_industrial_vs_vegetation"
    ]
].copy()


natural_pool = data[
    data[
        "source_class"
    ].eq(
        "natural_vegetation_fire"
    )
]


uncertain_pool = data[
    data[
        "source_class"
    ].eq(
        "other_uncertain"
    )
]


natural_review = natural_pool.sample(
    n=min(
        250,
        len(natural_pool)
    ),
    random_state=42,
)


uncertain_review = uncertain_pool.sample(
    n=min(
        250,
        len(uncertain_pool)
    ),
    random_state=42,
)


review = pd.concat(
    [
        industrial_review,
        near_review,
        conflict_review,
        natural_review,
        uncertain_review,
    ],
    ignore_index=True,
)


review = review.drop_duplicates(
    subset=["observation_id"]
)


review_columns = [
    "observation_id",
    "latitude",
    "longitude",
    "source_class",
    "label_quality",
    "label_source",
    "label_evidence",
    "label_review_priority",
    "independent_industrial_distance_km",
    "independent_industrial_reference_name",
    "independent_industrial_reference_type",
    "independent_industrial_reference_subtype",
    "worldcover_point_name",
    "worldcover_dominant_100m_name",
    "label_conflict_industrial_vs_vegetation",
    "detections_30d",
    "distinct_active_days_30d",
    "temporal_status",
]


# Add OSM columns for AUDIT ONLY.
for column in [
    "osm_evidence_relation_1_5km",
    "osm_evidence_relation_3km",
]:

    if column in review.columns:
        review_columns.append(column)


review[
    review_columns
].to_csv(
    OUTPUT_CSV,
    index=False
)


report = {
    "stage":
        "source_label_construction_v1",

    "rows":
        int(len(data)),

    "source_classes": {
        str(k): int(v)
        for k, v
        in class_counts.items()
    },

    "label_quality": {
        str(k): int(v)
        for k, v
        in quality_counts.items()
    },

    "temporal_status": {
        str(k): int(v)
        for k, v
        in temporal_counts.items()
    },

    "industrial_vegetation_conflicts":
        conflict_count,

    "industrial_2_to_3km_review_rows":
        review_2_3,

    "review_file_rows":
        int(len(review)),

    "policy": {
        "industrial": (
            "Independent industrial reference <=2 km."
        ),

        "industrial_high_confidence_weak": (
            "Independent industrial reference <=1 km."
        ),

        "natural": (
            "WorldCover point and dominant 100m "
            "both vegetation/agriculture and no "
            "independent industrial reference <=5 km."
        ),

        "other_uncertain": (
            "Insufficient/conflicting evidence or "
            "industrial reference only in 2-3 km "
            "review band."
        ),

        "osm_used_to_create_label":
            False,

        "sentinel_used_to_create_label":
            False,

        "temporal_recurrence_used_to_create_source_label":
            False,
    },

    "worldcover_note": (
        "ESA WorldCover describes land cover, not "
        "the confirmed cause of a thermal anomaly."
    ),

    "industrial_reference_note": (
        "Facility proximity provides industrial context "
        "but does not independently prove an industrial fire."
    ),

    "verified_A_labels":
        0,

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


print("\n" + "=" * 80)
print("SOURCE LABEL V1 BUILD COMPLETE")
print("=" * 80)

print(
    f"\nRows: "
    f"{len(data):,}"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nReview CSV:")
print(OUTPUT_CSV)

print("\nReport:")
print(REPORT_FILE)

print(
    "\nA VERIFIED LABELS: 0"
)

print(
    "2025 HOLDOUT ACCESSED: NO"
)

print(
    "\nIMPORTANT:"
    "\nThese labels remain contextual/weak labels."
    "\nDo not report them as confirmed real-world"
    "\nindustrial or natural fire ground truth."
)

print(
    "\nNEXT:"
    "\nAudit the resulting class balance and conflicts."
    "\nIf acceptable, freeze the source-label policy and"
    "\nprepare leakage-safe model features/splits."
)