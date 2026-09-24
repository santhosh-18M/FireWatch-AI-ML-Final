from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "labels"
    / "source_labels_v1_2022_2024.parquet"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "training_dataset_v1.parquet"
)

FEATURE_FILE = (
    ROOT / "reports" / "experiments"
    / "training_features_v1.json"
)

REPORT_FILE = (
    ROOT / "reports" / "experiments"
    / "training_dataset_v1_report.json"
)

RANDOM_STATE = 42

# We retain every industrial example.
#
# Natural/other sampling prevents the small industrial
# class from being numerically overwhelmed.
NATURAL_TARGET = 5000
OTHER_TARGET = 2500

SPATIAL_GRID_DEGREES = 0.5


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 15")
print("PREPARE LEAKAGE-SAFE TRAINING DATASET V1")
print("=" * 80)

print(
    "\nImportant:"
    "\n- V1 label policy is now frozen."
    "\n- Latitude/longitude are metadata, not predictors."
    "\n- WorldCover is NOT a predictor."
    "\n- Independent industrial distance is NOT a predictor."
    "\n- Label metadata is NOT a predictor."
    "\n- 2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/8] Loading labeled development data...")

df = pd.read_parquet(INPUT_FILE)

print(f"      Rows: {len(df):,}")

required = [
    "observation_id",
    "latitude",
    "longitude",
    "source_class",
    "label_quality",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

if df["observation_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate observation IDs detected."
    )


# ============================================================
# 2. SOURCE CLASS AUDIT
# ============================================================

print("\n[2/8] Auditing frozen labels...")

print(
    df["source_class"]
    .value_counts()
    .to_string()
)


expected_classes = {
    "industrial_fire_candidate",
    "natural_vegetation_fire",
    "other_uncertain",
}

actual_classes = set(
    df["source_class"]
    .dropna()
    .unique()
)

if actual_classes != expected_classes:
    raise RuntimeError(
        f"Unexpected classes: {actual_classes}"
    )


# ============================================================
# 3. SPATIAL GROUPS
# ============================================================

print("\n[3/8] Creating 0.5-degree spatial groups...")

lat_bin = np.floor(
    df["latitude"]
    / SPATIAL_GRID_DEGREES
).astype(int)

lon_bin = np.floor(
    df["longitude"]
    / SPATIAL_GRID_DEGREES
).astype(int)

df["spatial_group"] = (
    lat_bin.astype(str)
    + "_"
    + lon_bin.astype(str)
)

print(
    f"      Spatial groups: "
    f"{df['spatial_group'].nunique():,}"
)


# ============================================================
# 4. STRATIFIED GEOGRAPHIC SAMPLING
# ============================================================

print("\n[4/8] Building training candidate set...")


industrial = df[
    df["source_class"]
    .eq("industrial_fire_candidate")
].copy()

natural = df[
    df["source_class"]
    .eq("natural_vegetation_fire")
].copy()

other = df[
    df["source_class"]
    .eq("other_uncertain")
].copy()


def geographic_sample(frame, target, seed):

    if len(frame) <= target:
        return frame.copy()

    rng = np.random.default_rng(seed)

    groups = list(
        frame.groupby(
            "spatial_group",
            sort=False
        )
    )

    selected_indices = []

    # First preserve geographic coverage:
    # one observation per available spatial group.
    for _, group in groups:

        idx = rng.choice(
            group.index.to_numpy(),
            size=1,
            replace=False,
        )

        selected_indices.extend(
            idx.tolist()
        )

    selected_indices = list(
        dict.fromkeys(selected_indices)
    )

    # If number of groups itself exceeds target,
    # select target groups deterministically.
    if len(selected_indices) >= target:

        chosen = rng.choice(
            np.array(selected_indices),
            size=target,
            replace=False,
        )

        return frame.loc[
            chosen
        ].copy()

    remaining_needed = (
        target
        - len(selected_indices)
    )

    remaining_pool = frame[
        ~frame.index.isin(
            selected_indices
        )
    ]

    if remaining_needed > 0:

        additional = rng.choice(
            remaining_pool.index.to_numpy(),
            size=min(
                remaining_needed,
                len(remaining_pool),
            ),
            replace=False,
        )

        selected_indices.extend(
            additional.tolist()
        )

    return frame.loc[
        selected_indices
    ].copy()


natural_sample = geographic_sample(
    natural,
    NATURAL_TARGET,
    RANDOM_STATE,
)

other_sample = geographic_sample(
    other,
    OTHER_TARGET,
    RANDOM_STATE + 1,
)


training = pd.concat(
    [
        industrial,
        natural_sample,
        other_sample,
    ],
    ignore_index=True,
)


training = training.sample(
    frac=1.0,
    random_state=RANDOM_STATE,
).reset_index(drop=True)


print(
    f"      Industrial : "
    f"{len(industrial):,}"
)

print(
    f"      Natural    : "
    f"{len(natural_sample):,}"
)

print(
    f"      Other      : "
    f"{len(other_sample):,}"
)

print(
    f"      Total      : "
    f"{len(training):,}"
)


# ============================================================
# 5. DEFINE FORBIDDEN PREDICTORS
# ============================================================

print("\n[5/8] Defining leakage exclusions...")


# Explicit columns that must never become predictors.
forbidden_exact = {
    # Identity / GIS metadata
    "observation_id",
    "latitude",
    "longitude",
    "spatial_group",

    # Target
    "source_class",

    # Label metadata
    "label_quality",
    "label_source",
    "label_evidence",
    "label_review_priority",

    # Temporal target/output
    "temporal_status",

    # Direct WorldCover label evidence
    "label_worldcover_point",
    "label_worldcover_dominant_100m",
    "worldcover_point_name",
    "worldcover_dominant_100m_name",
    "worldcover_point_vegetation",
    "worldcover_dominant_vegetation",
    "independent_vegetation_evidence",

    # Independent industrial label evidence
    "independent_industrial_distance_km",
    "independent_industrial_reference_id",
    "independent_industrial_reference_name",
    "independent_industrial_reference_type",
    "independent_industrial_reference_subtype",
    "independent_industrial_reference_source",
    "independent_industrial_location_accuracy",
    "independent_industrial_capacity_mw",
    "independent_industrial_reference_latitude",
    "independent_industrial_reference_longitude",

    # Conflict/review fields
    "label_conflict_industrial_vs_vegetation",
}


# Prefixes generated directly from the independent
# industrial label evidence.
forbidden_prefixes = [
    "independent_industrial_within_",
    "industrial_reference_",
]


# ============================================================
# 6. SELECT MODEL FEATURES
# ============================================================

print("\n[6/8] Selecting model predictors...")


# We intentionally use an allow-list strategy.
#
# This is safer than "all columns except..." because the
# dataset contains many audit/provenance fields.

candidate_features = [
    # --------------------------------------------------------
    # FIRMS thermal
    # --------------------------------------------------------
    "brightness",
    "bright_t31",
    "frp",
    "scan",
    "track",
    "confidence_code",
    "daynight_num",
    "hour",
    "month",
    "brightness_t31_delta",
    "log_frp",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    # --------------------------------------------------------
    # Causal historical temporal features
    # --------------------------------------------------------
    "detections_3d",
    "detections_7d",
    "detections_10d",
    "detections_30d",
    "distinct_active_days_30d",
    "hours_since_prev",
    "mean_prev_30d_frp",
    "median_prev_30d_frp",
    "max_prev_30d_frp",
    "frp_ratio_to_history",
    "active_days_fraction_30d",

    # --------------------------------------------------------
    # OSM MODEL EVIDENCE
    #
    # OSM did NOT create source_class, so it may be used
    # as a predictor.
    # --------------------------------------------------------
    "distance_to_strong_industrial_km",
    "distance_to_supporting_industrial_km",
    "distance_to_infrastructure_km",
    "distance_to_vegetation_km",
    "strong_industrial_vs_vegetation_distance_km",

    # --------------------------------------------------------
    # Sentinel-2 model evidence
    #
    # Sentinel values did NOT directly create source_class.
    # --------------------------------------------------------
    "sentinel_b4",
    "sentinel_b8",
    "sentinel_b11",
    "sentinel_b12",
    "sentinel_ndvi",
    "sentinel_nbr",
    "sentinel_ndmi",
    "satellite_evidence_available",
]


# The exact satellite column names may differ from the
# semantic names above, so add safe aliases if present.

alias_groups = {
    "sentinel_b4": [
        "sentinel_b4",
        "B4",
        "b4",
    ],

    "sentinel_b8": [
        "sentinel_b8",
        "B8",
        "b8",
    ],

    "sentinel_b11": [
        "sentinel_b11",
        "B11",
        "b11",
    ],

    "sentinel_b12": [
        "sentinel_b12",
        "B12",
        "b12",
    ],

    "sentinel_ndvi": [
        "sentinel_ndvi",
        "NDVI",
        "ndvi",
    ],

    "sentinel_nbr": [
        "sentinel_nbr",
        "NBR",
        "nbr",
    ],

    "sentinel_ndmi": [
        "sentinel_ndmi",
        "NDMI",
        "ndmi",
    ],
}


resolved_features = []


for feature in candidate_features:

    if feature in alias_groups:

        found = None

        for alias in alias_groups[
            feature
        ]:

            if alias in training.columns:
                found = alias
                break

        if found is not None:
            resolved_features.append(
                found
            )

    else:

        if feature in training.columns:
            resolved_features.append(
                feature
            )


# Remove duplicates while preserving order.
resolved_features = list(
    dict.fromkeys(
        resolved_features
    )
)


# Safety check.
for feature in resolved_features:

    if feature in forbidden_exact:
        raise RuntimeError(
            f"Forbidden predictor selected: {feature}"
        )

    for prefix in forbidden_prefixes:

        if feature.startswith(prefix):
            raise RuntimeError(
                f"Forbidden predictor selected: {feature}"
            )


print(
    f"      Selected predictors: "
    f"{len(resolved_features)}"
)

for feature in resolved_features:
    print(f"        - {feature}")


# ============================================================
# 7. FEATURE AUDIT
# ============================================================

print("\n[7/8] Auditing selected predictors...")


feature_audit = []


for feature in resolved_features:

    series = training[
        feature
    ]

    missing_count = int(
        series.isna().sum()
    )

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    feature_audit.append({
        "feature":
            feature,

        "dtype":
            str(series.dtype),

        "missing_count":
            missing_count,

        "missing_percentage":
            (
                missing_count
                / len(training)
                * 100
            ),

        "unique_count":
            unique_count,
    })


audit_df = pd.DataFrame(
    feature_audit
)


print(
    audit_df.to_string(
        index=False
    )
)


constant_features = (
    audit_df[
        audit_df[
            "unique_count"
        ]
        <= 1
    ]["feature"]
    .tolist()
)


if constant_features:

    print(
        "\n      Removing constant features:"
    )

    for feature in constant_features:
        print(f"        - {feature}")

    resolved_features = [
        f for f in resolved_features
        if f not in constant_features
    ]


# ============================================================
# 8. SAVE TRAINING DATA
# ============================================================

print("\n[8/8] Saving training dataset...")


# Keep metadata required for spatial splitting and auditing,
# plus model features.

metadata_columns = [
    "observation_id",
    "latitude",
    "longitude",
    "spatial_group",
    "source_class",
    "label_quality",
]


training_columns = (
    metadata_columns
    + resolved_features
)


training_output = training[
    training_columns
].copy()


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


training_output.to_parquet(
    OUTPUT_FILE,
    index=False
)


FEATURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "target":
                "source_class",

            "features":
                resolved_features,

            "metadata_not_predictors":
                metadata_columns,

            "label_policy_version":
                "v1_frozen",

            "worldcover_predictor":
                False,

            "independent_industrial_predictor":
                False,

            "latitude_longitude_predictor":
                False,

            "2025_holdout_used":
                False,
        },
        f,
        indent=2,
    )


class_counts = (
    training_output[
        "source_class"
    ]
    .value_counts()
    .to_dict()
)


spatial_counts = (
    training_output[
        "source_class"
    ]
    .groupby(
        training_output[
            "source_class"
        ]
    )
    .size()
    .to_dict()
)


report = {
    "stage":
        "training_dataset_preparation_v1",

    "label_policy":
        "v1_frozen",

    "input_rows":
        int(len(df)),

    "training_rows":
        int(len(training_output)),

    "class_distribution": {
        str(k): int(v)
        for k, v
        in class_counts.items()
    },

    "spatial_groups":
        int(
            training_output[
                "spatial_group"
            ].nunique()
        ),

    "spatial_grid_degrees":
        SPATIAL_GRID_DEGREES,

    "features":
        resolved_features,

    "feature_count":
        len(resolved_features),

    "excluded_from_predictors": {
        "latitude_longitude":
            True,

        "worldcover_label_evidence":
            True,

        "independent_industrial_label_evidence":
            True,

        "label_metadata":
            True,

        "temporal_status":
            True,
    },

    "sampling": {
        "industrial":
            "all",

        "natural_target":
            NATURAL_TARGET,

        "other_target":
            OTHER_TARGET,

        "strategy":
            "geographic coverage then deterministic random fill",

        "random_state":
            RANDOM_STATE,
    },

    "holdout_2025_accessed":
        False,
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


print("\n" + "=" * 80)
print("TRAINING DATASET V1 READY")
print("=" * 80)

print(
    f"\nTraining rows : "
    f"{len(training_output):,}"
)

print(
    f"Features      : "
    f"{len(resolved_features)}"
)

print(
    f"Spatial groups: "
    f"{training_output['spatial_group'].nunique():,}"
)

print("\nClass distribution:")

print(
    training_output[
        "source_class"
    ]
    .value_counts()
    .to_string()
)

print("\nDataset:")
print(OUTPUT_FILE)

print("\nFeature specification:")
print(FEATURE_FILE)

print("\nReport:")
print(REPORT_FILE)

print(
    "\n2025 HOLDOUT ACCESSED: NO"
)

print(
    "\nNEXT:"
    "\nCreate spatially disjoint train / validation /"
    "\ncalibration splits and audit class coverage."
)

print(
    "\nMODEL TRAINING HAS NOT STARTED."
)