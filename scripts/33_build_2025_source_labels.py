from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "worldcover_evidence_2025.parquet"
)

INDUSTRIAL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "industrial_reference_matches_2025.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "source_labels_v1_2025.parquet"
)

OUTPUT_CSV = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "source_labels_v1_2025_review.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "source_labels_v1_2025_report.json"
)

EXPECTED_ROWS = 20_000

# FROZEN DEVELOPMENT POLICY
INDUSTRIAL_STRONG_KM = 2.0
INDUSTRIAL_EXCLUSION_KM = 5.0


# ============================================================
# FROZEN WORLDCOVER DEFINITIONS
# ============================================================

VEGETATION_CODES = {
    10,  # Tree cover
    20,  # Shrubland
    30,  # Grassland
    40,  # Cropland
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
print("FIREWATCH — STEP 33")
print("2025 FINAL HOLDOUT — APPLY FROZEN SOURCE LABEL POLICY V1")
print("=" * 80)

print(
    "\nTarget classes:"
    "\n  0. industrial_fire_candidate"
    "\n  1. natural_vegetation_fire"
    "\n  2. other_uncertain"
)

print(
    "\nFROZEN POLICY:"
    "\n- Industrial: independent industrial reference <= 2 km."
    "\n- Industrial high-confidence weak tier: <= 1 km."
    "\n- Natural: WorldCover point AND dominant 100 m are"
    "\n  vegetation/agriculture AND no industrial reference <= 5 km."
    "\n- Remaining observations: other_uncertain."
)

print(
    "\nImportant:"
    "\n- Policy is NOT changed using 2025 results."
    "\n- Industrial overrides natural when evidence conflicts."
    "\n- OSM does NOT create source labels."
    "\n- FIRMS thermal values do NOT create source labels."
    "\n- Temporal recurrence does NOT create source labels."
    "\n- No classifier is executed."
)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/8] Loading frozen holdout evidence...")


if not BASE_FILE.exists():
    raise FileNotFoundError(
        f"WorldCover evidence missing:\n{BASE_FILE}"
    )

if not INDUSTRIAL_FILE.exists():
    raise FileNotFoundError(
        f"Industrial evidence missing:\n{INDUSTRIAL_FILE}"
    )


base = pd.read_parquet(BASE_FILE)

industrial = pd.read_parquet(
    INDUSTRIAL_FILE
)


print(f"      Base/WorldCover : {len(base):,}")
print(f"      Industrial      : {len(industrial):,}")


for name, frame in [
    ("base", base),
    ("industrial", industrial),
]:

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_ROWS:,} rows, "
            f"found {len(frame):,}."
        )

    if "observation_id" not in frame.columns:
        raise RuntimeError(
            f"{name}: observation_id missing."
        )

    frame["observation_id"] = (
        frame["observation_id"]
        .astype(str)
        .str.strip()
    )

    if frame["observation_id"].duplicated().any():
        raise RuntimeError(
            f"{name}: duplicate observation IDs."
        )


# ============================================================
# 2. VALIDATE WORLDCOVER
# ============================================================

print("\n[2/8] Validating WorldCover evidence...")


required_world = [
    "worldcover_point_class",
    "worldcover_dominant_100m",
]


missing_world = [
    column
    for column in required_world
    if column not in base.columns
]


if missing_world:
    raise RuntimeError(
        "Missing WorldCover fields:\n"
        + "\n".join(missing_world)
    )


base[
    "label_worldcover_point"
] = (
    base[
        "worldcover_point_class"
    ]
    .apply(safe_int_code)
)


base[
    "label_worldcover_dominant_100m"
] = (
    base[
        "worldcover_dominant_100m"
    ]
    .apply(safe_int_code)
)


base[
    "worldcover_point_name_label"
] = (
    base[
        "label_worldcover_point"
    ]
    .map(WORLD_CLASS_NAMES)
    .fillna("unknown")
)


base[
    "worldcover_dominant_100m_name_label"
] = (
    base[
        "label_worldcover_dominant_100m"
    ]
    .map(WORLD_CLASS_NAMES)
    .fillna("unknown")
)


base[
    "worldcover_point_vegetation"
] = (
    base[
        "label_worldcover_point"
    ]
    .isin(VEGETATION_CODES)
)


base[
    "worldcover_dominant_vegetation"
] = (
    base[
        "label_worldcover_dominant_100m"
    ]
    .isin(VEGETATION_CODES)
)


base[
    "independent_vegetation_evidence"
] = (
    base[
        "worldcover_point_vegetation"
    ]
    &
    base[
        "worldcover_dominant_vegetation"
    ]
)


print(
    "      Point vegetation    : "
    f"{base['worldcover_point_vegetation'].sum():,}"
)

print(
    "      Dominant vegetation : "
    f"{base['worldcover_dominant_vegetation'].sum():,}"
)

print(
    "      Both vegetation     : "
    f"{base['independent_vegetation_evidence'].sum():,}"
)


# ============================================================
# 3. PREPARE INDUSTRIAL EVIDENCE
# ============================================================

print("\n[3/8] Preparing industrial evidence...")


industrial_required = [
    "observation_id",
    "independent_industrial_distance_km",
    "independent_industrial_reference_name",
    "independent_industrial_reference_type",
    "independent_industrial_reference_subtype",
    "independent_industrial_reference_source",
]


missing_industrial = [
    column
    for column in industrial_required
    if column not in industrial.columns
]


if missing_industrial:
    raise RuntimeError(
        "Industrial evidence missing columns:\n"
        + "\n".join(missing_industrial)
    )


industrial_small = industrial[
    industrial_required
].copy()


# ============================================================
# 4. MERGE
# ============================================================

print("\n[4/8] Merging frozen evidence...")


data = base.merge(
    industrial_small,
    on="observation_id",
    how="left",
    validate="one_to_one",
)


if len(data) != EXPECTED_ROWS:
    raise RuntimeError(
        "Row count changed after industrial merge."
    )


if data[
    "independent_industrial_distance_km"
].isna().any():

    raise RuntimeError(
        "Missing industrial-reference distance after merge."
    )


print(f"      Merged rows: {len(data):,}")


# ============================================================
# 5. FROZEN INDUSTRIAL BANDS
# ============================================================

print("\n[5/8] Applying frozen industrial evidence bands...")


distance = pd.to_numeric(
    data[
        "independent_industrial_distance_km"
    ],
    errors="coerce",
)


data[
    "industrial_reference_le_1km"
] = (
    distance <= 1.0
)


data[
    "industrial_reference_le_2km"
] = (
    distance <= INDUSTRIAL_STRONG_KM
)


data[
    "industrial_reference_2_to_3km"
] = (
    (distance > 2.0)
    &
    (distance <= 3.0)
)


data[
    "industrial_reference_le_5km"
] = (
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
# 6. APPLY EXACT FROZEN SOURCE-LABEL POLICY
# ============================================================

print("\n[6/8] Applying FROZEN V1 source-label policy...")


# ------------------------------------------------------------
# DEFAULT: OTHER / UNCERTAIN
# ------------------------------------------------------------

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
    "source_class",
] = "natural_vegetation_fire"


data.loc[
    natural_mask,
    "label_quality",
] = "B_STRONG_WEAK_LABEL"


data.loc[
    natural_mask,
    "label_source",
] = "esa_worldcover_2021"


data.loc[
    natural_mask,
    "label_evidence",
] = (
    "WorldCover point and dominant 100m neighborhood "
    "both indicate vegetation/agricultural land cover, "
    "with no independent industrial reference within 5 km."
)


# ------------------------------------------------------------
# INDUSTRIAL WEAK LABEL
# ------------------------------------------------------------
# Industrial is intentionally applied AFTER natural,
# preserving the frozen development override behavior.

industrial_mask = (
    data[
        "industrial_reference_le_2km"
    ]
)


data.loc[
    industrial_mask,
    "source_class",
] = "industrial_fire_candidate"


data.loc[
    industrial_mask,
    "label_quality",
] = "B_STRONG_WEAK_LABEL"


data.loc[
    industrial_mask,
    "label_source",
] = "independent_industrial_reference"


data.loc[
    industrial_mask,
    "label_evidence",
] = (
    "FIRMS observation lies within 2 km of an "
    "independently catalogued industrial thermal facility."
)


# ------------------------------------------------------------
# <=1 KM INDUSTRIAL TIER
# ------------------------------------------------------------

industrial_1km = (
    data[
        "industrial_reference_le_1km"
    ]
)


data.loc[
    industrial_1km,
    "label_quality",
] = "B_HIGH_CONFIDENCE_WEAK_LABEL"


data.loc[
    industrial_1km,
    "label_evidence",
] = (
    "FIRMS observation lies within 1 km of an "
    "independently catalogued industrial thermal facility."
)


# ------------------------------------------------------------
# CONFLICTS
# ------------------------------------------------------------

data[
    "label_conflict_industrial_vs_vegetation"
] = (
    data[
        "industrial_reference_le_2km"
    ]
    &
    data[
        "independent_vegetation_evidence"
    ]
)


data[
    "label_review_priority"
] = "normal"


data.loc[
    data[
        "label_conflict_industrial_vs_vegetation"
    ],
    "label_review_priority",
] = "high"


data.loc[
    data[
        "industrial_reference_2_to_3km"
    ],
    "label_review_priority",
] = "high"


# ============================================================
# 7. TEMPORAL STATUS — SEPARATE DIMENSION
# ============================================================

print("\n[7/8] Building frozen separate temporal status...")


required_temporal = [
    "detections_30d",
    "distinct_active_days_30d",
]


missing_temporal = [
    column
    for column in required_temporal
    if column not in data.columns
]


if missing_temporal:
    raise RuntimeError(
        "Temporal features missing:\n"
        + "\n".join(missing_temporal)
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
    "temporal_status",
] = "RECURRENT"


data.loc[
    persistent_mask,
    "temporal_status",
] = "PERSISTENT"


# ============================================================
# AUDIT BEFORE SAVE
# ============================================================

class_counts = (
    data[
        "source_class"
    ]
    .value_counts()
)


quality_counts = (
    data[
        "label_quality"
    ]
    .value_counts()
)


temporal_counts = (
    data[
        "temporal_status"
    ]
    .value_counts()
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


print("\n      Source-class distribution:")
print(class_counts.to_string())


print("\n      Label-quality distribution:")
print(quality_counts.to_string())


print("\n      Temporal-status distribution:")
print(temporal_counts.to_string())


print(
    "\n      Industrial-vs-vegetation conflicts: "
    f"{conflict_count:,}"
)


print(
    "      Industrial references 2-3 km "
    "(review, not auto-label): "
    f"{review_2_3:,}"
)


cross_tab = pd.crosstab(
    data["source_class"],
    data["temporal_status"],
)


print("\n      Source × temporal:")
print(cross_tab.to_string())


# ============================================================
# 8. SAVE
# ============================================================

print("\n[8/8] Saving frozen-policy 2025 labels...")


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


data.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# REVIEW FILE — SAME DEVELOPMENT POLICY
# ------------------------------------------------------------

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
        len(natural_pool),
    ),
    random_state=42,
)


uncertain_review = uncertain_pool.sample(
    n=min(
        250,
        len(uncertain_pool),
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
    subset=[
        "observation_id"
    ]
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
    "worldcover_point_name_label",
    "worldcover_dominant_100m_name_label",
    "label_conflict_industrial_vs_vegetation",
    "detections_30d",
    "distinct_active_days_30d",
    "temporal_status",
]


# OSM is AUDIT ONLY.
for column in [
    "osm_evidence_relation_1_5km",
    "osm_evidence_relation_3km",
    "sample_context_group",
]:

    if column in review.columns:
        review_columns.append(column)


review[
    review_columns
].to_csv(
    OUTPUT_CSV,
    index=False,
)


report = {
    "stage":
        "2025_final_holdout_source_labels_v1",

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

        "industrial":
            "Independent industrial reference <=2 km.",

        "industrial_high_confidence_weak":
            "Independent industrial reference <=1 km.",

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

        "industrial_overrides_natural":
            True,

        "osm_used_to_create_label":
            False,

        "firms_thermal_used_to_create_label":
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

    "weak_labels":
        True,

    "model_executed":
        False,

    "policy_changed_after_holdout_access":
        False,
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


print("\n" + "=" * 80)
print("STEP 33 COMPLETE")
print("=" * 80)


print(f"\nRows: {len(data):,}")


print("\nFINAL 2025 SOURCE LABEL COUNTS")

for class_name in [
    "industrial_fire_candidate",
    "natural_vegetation_fire",
    "other_uncertain",
]:

    count = int(
        (
            data["source_class"]
            == class_name
        ).sum()
    )

    percentage = (
        count
        / len(data)
        * 100
    )

    print(
        f"  {class_name:<30} "
        f"{count:>6,} "
        f"({percentage:>6.2f}%)"
    )


print(
    f"\nIndustrial <=1 km: "
    f"{int(data['industrial_reference_le_1km'].sum()):,}"
)


print(
    f"Industrial <=2 km: "
    f"{int(data['industrial_reference_le_2km'].sum()):,}"
)


print(
    f"Industrial/vegetation conflicts: "
    f"{conflict_count:,}"
)


print(
    f"2-3 km review band: "
    f"{review_2_3:,}"
)


print(
    "\nTEMPORAL STATUS"
)

print(
    temporal_counts.to_string()
)


print(
    f"\nOutput:"
    f"\n{OUTPUT_FILE}"
)


print(
    f"\nReview CSV:"
    f"\n{OUTPUT_CSV}"
)


print(
    f"\nReport:"
    f"\n{REPORT_FILE}"
)


print(
    "\nIMPORTANT:"
    "\n- Frozen V1 label policy applied without modification."
    "\n- These are contextual WEAK LABELS, not confirmed fire causes."
    "\n- 2025 model predictions have NOT been generated."
    "\n- Do NOT tune anything from these label counts."
    "\n- Next step: execute the frozen production classifier ONCE."
)