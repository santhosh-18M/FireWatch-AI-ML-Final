from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FIRMS_OSM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "firms_with_osm_2022_2024.parquet"
)

SATELLITE_FILE = (
    ROOT
    / "data"
    / "satellite"
    / "sentinel_evidence_2022_2024.parquet"
)

SAMPLE_FILE = (
    ROOT
    / "data"
    / "satellite"
    / "sentinel_sampling_2022_2024.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "labels"
    / "label_evidence_2022_2024.parquet"
)

REVIEW_FILE = (
    ROOT
    / "data"
    / "labels"
    / "label_review_candidates.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "label_evidence_audit.json"
)

EXPECTED_SAMPLE_ROWS = 20_000

SATELLITE_FEATURES = [
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "NBR",
    "NDMI",
]


# ============================================================
# HELPERS
# ============================================================

def fail(message):

    print("\n" + "=" * 80)
    print("FAILED")
    print("=" * 80)

    print(message)

    sys.exit(1)


def first_existing(columns, candidates):

    for candidate in candidates:

        if candidate in columns:

            return candidate

    return None


def numeric(df, column):

    if column is None:
        return pd.Series(
            np.nan,
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def bool_from_numeric(series):

    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .fillna(0)
        .astype(float)
        > 0
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 12: BUILD LABEL-EVIDENCE DATASET")
print("=" * 80)

print(
    "\nPurpose:"
    "\nBuild an auditable evidence table for the 20,000"
    "\ndevelopment observations BEFORE assigning final"
    "\nsource-class labels."
)

print(
    "\nImportant:"
    "\n- This script DOES NOT train a model."
    "\n- This script DOES NOT create final ML labels."
    "\n- Persistence remains separate from source class."
    "\n- 2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/8] Loading development evidence...")


for path in [
    FIRMS_OSM_FILE,
    SATELLITE_FILE,
    SAMPLE_FILE,
]:

    if not path.exists():

        fail(
            f"Required file missing:\n{path}"
        )


firms = pd.read_parquet(
    FIRMS_OSM_FILE
)

satellite = pd.read_parquet(
    SATELLITE_FILE
)

sample = pd.read_parquet(
    SAMPLE_FILE
)


print(
    f"      FIRMS + temporal + OSM : {len(firms):,}"
)

print(
    f"      Sentinel evidence      : {len(satellite):,}"
)

print(
    f"      Development sample     : {len(sample):,}"
)


if len(sample) != EXPECTED_SAMPLE_ROWS:

    fail(
        f"Expected 20,000 sampled observations, "
        f"found {len(sample):,}."
    )


if sample["observation_id"].duplicated().any():

    fail(
        "Duplicate observation IDs in sampling dataset."
    )


if satellite["observation_id"].duplicated().any():

    fail(
        "Duplicate observation IDs in satellite dataset."
    )


if firms["observation_id"].duplicated().any():

    fail(
        "Duplicate observation IDs in FIRMS/OSM dataset."
    )


# ============================================================
# 2. SUBSET FIRMS TO EXACT 20K
# ============================================================

print("\n[2/8] Selecting exact 20,000 FIRMS observations...")


sample_ids = set(
    sample["observation_id"].astype(str)
)


firms[
    "observation_id"
] = firms[
    "observation_id"
].astype(str)


firms_sample = firms[
    firms["observation_id"].isin(
        sample_ids
    )
].copy()


print(
    f"      Matched FIRMS rows: {len(firms_sample):,}"
)


if len(firms_sample) != EXPECTED_SAMPLE_ROWS:

    fail(
        "Could not recover exactly 20,000 observations "
        "from the FIRMS evidence dataset."
    )


# ============================================================
# 3. JOIN SENTINEL
# ============================================================

print("\n[3/8] Joining validated Sentinel evidence...")


satellite[
    "observation_id"
] = satellite[
    "observation_id"
].astype(str)


# Only bring model-relevant satellite fields + QA.

satellite_keep = [
    "observation_id",
    "satellite_evidence_available",
    "satellite_evidence_status",
] + SATELLITE_FEATURES


missing_sat_columns = [
    column
    for column in satellite_keep
    if column not in satellite.columns
]


if missing_sat_columns:

    fail(
        "Missing Sentinel columns:\n"
        + "\n".join(
            missing_sat_columns
        )
    )


evidence = firms_sample.merge(
    satellite[
        satellite_keep
    ],
    on="observation_id",
    how="left",
    validate="one_to_one"
)


if len(evidence) != EXPECTED_SAMPLE_ROWS:

    fail(
        "Row count changed after Sentinel join."
    )


missing_satellite_join = int(
    evidence[
        "satellite_evidence_available"
    ].isna().sum()
)


print(
    f"      Joined rows               : {len(evidence):,}"
)

print(
    f"      Missing satellite records : "
    f"{missing_satellite_join}"
)


if missing_satellite_join != 0:

    fail(
        "One or more observations did not match "
        "the validated Sentinel dataset."
    )


# ============================================================
# 4. IDENTIFY EVIDENCE COLUMNS
# ============================================================

print("\n[4/8] Identifying evidence channels...")


columns = set(
    evidence.columns
)


strong_distance_col = first_existing(
    columns,
    [
        "distance_to_strong_industrial_km",
        "strong_industrial_distance_km",
        "nearest_strong_industrial_km",
    ]
)


vegetation_distance_col = first_existing(
    columns,
    [
        "distance_to_vegetation_km",
        "vegetation_distance_km",
        "nearest_vegetation_km",
        "distance_to_natural_km",
    ]
)


supporting_distance_col = first_existing(
    columns,
    [
        "distance_to_supporting_industrial_km",
        "supporting_industrial_distance_km",
        "nearest_supporting_industrial_km",
    ]
)


infrastructure_distance_col = first_existing(
    columns,
    [
        "distance_to_weak_infrastructure_km",
        "weak_infrastructure_distance_km",
        "distance_to_infrastructure_km",
        "nearest_weak_infrastructure_km",
    ]
)


detections_30_col = first_existing(
    columns,
    [
        "detections_30d",
        "prior_detections_30d",
    ]
)


distinct_days_col = first_existing(
    columns,
    [
        "distinct_active_days_30d",
        "active_days_30d",
    ]
)


print(
    f"      Strong industrial distance : "
    f"{strong_distance_col}"
)

print(
    f"      Vegetation distance        : "
    f"{vegetation_distance_col}"
)

print(
    f"      Supporting industrial      : "
    f"{supporting_distance_col}"
)

print(
    f"      Weak infrastructure        : "
    f"{infrastructure_distance_col}"
)

print(
    f"      30-day detections          : "
    f"{detections_30_col}"
)

print(
    f"      Distinct active days       : "
    f"{distinct_days_col}"
)


if strong_distance_col is None:

    fail(
        "Strong industrial distance column not found."
    )


if vegetation_distance_col is None:

    fail(
        "Vegetation distance column not found."
    )


if detections_30_col is None:

    fail(
        "30-day detection count column not found."
    )


if distinct_days_col is None:

    fail(
        "Distinct active-day column not found."
    )


# ============================================================
# 5. CREATE RAW EVIDENCE FLAGS
# ============================================================

print("\n[5/8] Building evidence flags...")


strong_distance = numeric(
    evidence,
    strong_distance_col
)

vegetation_distance = numeric(
    evidence,
    vegetation_distance_col
)

supporting_distance = numeric(
    evidence,
    supporting_distance_col
)

infrastructure_distance = numeric(
    evidence,
    infrastructure_distance_col
)

detections_30 = numeric(
    evidence,
    detections_30_col
).fillna(0)

distinct_days = numeric(
    evidence,
    distinct_days_col
).fillna(0)


# ------------------------------------------------------------
# OSM CONTEXT EVIDENCE
#
# These are evidence descriptors, NOT labels.
# ------------------------------------------------------------

evidence[
    "evidence_strong_industrial_1km"
] = (
    strong_distance <= 1.0
)


evidence[
    "evidence_strong_industrial_1_5km"
] = (
    strong_distance <= 1.5
)


evidence[
    "evidence_strong_industrial_3km"
] = (
    strong_distance <= 3.0
)


evidence[
    "evidence_vegetation_1km"
] = (
    vegetation_distance <= 1.0
)


evidence[
    "evidence_vegetation_1_5km"
] = (
    vegetation_distance <= 1.5
)


evidence[
    "evidence_vegetation_3km"
] = (
    vegetation_distance <= 3.0
)


if supporting_distance_col is not None:

    evidence[
        "evidence_supporting_industrial_1_5km"
    ] = (
        supporting_distance <= 1.5
    )

else:

    evidence[
        "evidence_supporting_industrial_1_5km"
    ] = False


if infrastructure_distance_col is not None:

    evidence[
        "evidence_weak_infrastructure_1_5km"
    ] = (
        infrastructure_distance <= 1.5
    )

else:

    evidence[
        "evidence_weak_infrastructure_1_5km"
    ] = False


# ------------------------------------------------------------
# TEMPORAL EVIDENCE
#
# This is deliberately NOT used as proof that the SOURCE
# is industrial or natural.
# ------------------------------------------------------------

evidence[
    "evidence_has_prior_30d"
] = (
    detections_30 >= 1
)


evidence[
    "evidence_repeated_30d"
] = (
    detections_30 >= 3
)


evidence[
    "evidence_many_active_days_30d"
] = (
    distinct_days >= 5
)


evidence[
    "temporal_evidence_group"
] = np.select(
    [
        detections_30.eq(0),

        (
            detections_30.between(
                1,
                2
            )
        ),

        (
            (detections_30 >= 3)
            & (distinct_days < 5)
        ),

        (
            (detections_30 >= 3)
            & (distinct_days >= 5)
        ),
    ],
    [
        "no_prior",
        "limited_prior",
        "repeated_low_active_days",
        "high_recurrence",
    ],
    default="unknown"
)


# ------------------------------------------------------------
# SATELLITE EVIDENCE
# ------------------------------------------------------------

evidence[
    "evidence_satellite_available"
] = (
    evidence[
        "satellite_evidence_available"
    ]
    .fillna(False)
    .astype(bool)
)


# Satellite indices are retained as measurements.
#
# DO NOT create:
#
# NDVI > x -> natural label
#
# or:
#
# NBR < x -> industrial label
#
# at this stage.
#
# We first audit their distributions.


# ============================================================
# 6. DEFINE EVIDENCE RELATIONSHIPS
# ============================================================

print("\n[6/8] Describing evidence relationships...")


industrial_close = evidence[
    "evidence_strong_industrial_1_5km"
]

vegetation_close = evidence[
    "evidence_vegetation_1_5km"
]


evidence[
    "osm_evidence_relation_1_5km"
] = np.select(
    [
        industrial_close
        & ~vegetation_close,

        vegetation_close
        & ~industrial_close,

        industrial_close
        & vegetation_close,
    ],
    [
        "industrial_only",
        "vegetation_only",
        "mixed",
    ],
    default="neither"
)


industrial_3km = evidence[
    "evidence_strong_industrial_3km"
]

vegetation_3km = evidence[
    "evidence_vegetation_3km"
]


evidence[
    "osm_evidence_relation_3km"
] = np.select(
    [
        industrial_3km
        & ~vegetation_3km,

        vegetation_3km
        & ~industrial_3km,

        industrial_3km
        & vegetation_3km,
    ],
    [
        "industrial_only",
        "vegetation_only",
        "mixed",
    ],
    default="neither"
)


# ------------------------------------------------------------
# REVIEW PRIORITY
#
# This does NOT assign a source class.
#
# It only tells us which cases are especially useful for
# reviewing the eventual labeling strategy.
# ------------------------------------------------------------

evidence[
    "review_priority"
] = np.select(
    [
        (
            evidence[
                "osm_evidence_relation_1_5km"
            ]
            == "mixed"
        ),

        (
            evidence[
                "osm_evidence_relation_1_5km"
            ]
            == "neither"
        ),

        (
            evidence[
                "osm_evidence_relation_1_5km"
            ]
            == "industrial_only"
        )
        & evidence[
            "evidence_satellite_available"
        ],

        (
            evidence[
                "osm_evidence_relation_1_5km"
            ]
            == "vegetation_only"
        )
        & evidence[
            "evidence_satellite_available"
        ],
    ],
    [
        "HIGH_MIXED_CONTEXT",
        "HIGH_NO_CLOSE_CONTEXT",
        "INDUSTRIAL_CONTEXT_REVIEW",
        "VEGETATION_CONTEXT_REVIEW",
    ],
    default="STANDARD"
)


# ------------------------------------------------------------
# LABEL PLACEHOLDERS
#
# IMPORTANT:
# These remain UNASSIGNED.
# ------------------------------------------------------------

evidence[
    "source_class"
] = "UNASSIGNED"


evidence[
    "label_quality"
] = "UNASSIGNED"


evidence[
    "label_source"
] = "UNASSIGNED"


evidence[
    "label_evidence"
] = ""


evidence[
    "review_status"
] = "NOT_REVIEWED"


# ============================================================
# 7. AUDIT EVIDENCE
# ============================================================

print("\n[7/8] Auditing evidence distribution...")


osm_15_counts = (
    evidence[
        "osm_evidence_relation_1_5km"
    ]
    .value_counts()
)


osm_3_counts = (
    evidence[
        "osm_evidence_relation_3km"
    ]
    .value_counts()
)


temporal_counts = (
    evidence[
        "temporal_evidence_group"
    ]
    .value_counts()
)


satellite_available_count = int(
    evidence[
        "evidence_satellite_available"
    ].sum()
)


print(
    "\n      OSM relation @ 1.5 km:"
)

for key, value in osm_15_counts.items():

    print(
        f"        {key:20s}: {value:,}"
    )


print(
    "\n      OSM relation @ 3 km:"
)

for key, value in osm_3_counts.items():

    print(
        f"        {key:20s}: {value:,}"
    )


print(
    "\n      Temporal evidence:"
)

for key, value in temporal_counts.items():

    print(
        f"        {key:28s}: {value:,}"
    )


print(
    "\n      Satellite evidence available:"
    f" {satellite_available_count:,}"
)


# ------------------------------------------------------------
# Cross-tab:
# OSM evidence vs temporal behavior.
# ------------------------------------------------------------

cross = pd.crosstab(
    evidence[
        "osm_evidence_relation_1_5km"
    ],
    evidence[
        "temporal_evidence_group"
    ]
)


print(
    "\n      OSM context × temporal evidence:"
)

print(
    cross.to_string()
)


# ------------------------------------------------------------
# Sentinel distribution by OSM context.
# ------------------------------------------------------------

sentinel_summary = (
    evidence[
        evidence[
            "evidence_satellite_available"
        ]
    ]
    .groupby(
        "osm_evidence_relation_1_5km"
    )[
        [
            "NDVI",
            "NBR",
            "NDMI"
        ]
    ]
    .agg(
        [
            "count",
            "mean",
            "median"
        ]
    )
)


print(
    "\n      Sentinel index summary by OSM evidence:"
)

print(
    sentinel_summary.to_string()
)


# ============================================================
# 8. CREATE REVIEW SAMPLE + SAVE
# ============================================================

print("\n[8/8] Building review candidates...")


# We want representation from:
#
# - industrial-only context
# - vegetation-only context
# - mixed context
# - neither context
#
# and different temporal patterns.
#
# This is NOT random training-label generation.
# It is an evidence-review dataset.


review_parts = []


relations = [
    "industrial_only",
    "vegetation_only",
    "mixed",
    "neither",
]


temporal_groups = [
    "no_prior",
    "limited_prior",
    "repeated_low_active_days",
    "high_recurrence",
]


for relation in relations:

    for temporal_group in temporal_groups:

        subset = evidence[
            (
                evidence[
                    "osm_evidence_relation_1_5km"
                ]
                == relation
            )
            &
            (
                evidence[
                    "temporal_evidence_group"
                ]
                == temporal_group
            )
        ]


        if len(subset) == 0:
            continue


        n = min(
            25,
            len(subset)
        )


        review_parts.append(
            subset.sample(
                n=n,
                random_state=42
            )
        )


review = pd.concat(
    review_parts,
    ignore_index=True
)


review = (
    review
    .drop_duplicates(
        subset=[
            "observation_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


# Useful review columns.

review_columns = [
    column
    for column in [
        "observation_id",
        "acquired_at",
        "latitude",
        "longitude",
        "brightness",
        "bright_t31",
        "frp",
        "confidence",
        "daynight",
        detections_30_col,
        distinct_days_col,
        strong_distance_col,
        supporting_distance_col,
        infrastructure_distance_col,
        vegetation_distance_col,
        "nearest_strong_industrial_category",
        "nearest_vegetation_category",
        "osm_evidence_relation_1_5km",
        "osm_evidence_relation_3km",
        "temporal_evidence_group",
        "satellite_evidence_available",
        "B4",
        "B8",
        "B11",
        "B12",
        "NDVI",
        "NBR",
        "NDMI",
        "review_priority",
        "source_class",
        "label_quality",
        "label_source",
        "label_evidence",
        "review_status",
    ]
    if column is not None
    and column in review.columns
]


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


evidence.to_parquet(
    OUTPUT_FILE,
    index=False
)


review.to_csv(
    REVIEW_FILE,
    columns=review_columns,
    index=False
)


report = {

    "stage":
        "label_evidence_only",

    "final_labels_assigned":
        False,

    "rows":
        int(
            len(evidence)
        ),

    "unique_observation_ids":
        int(
            evidence[
                "observation_id"
            ].nunique()
        ),

    "satellite_evidence_available":
        satellite_available_count,

    "osm_relation_1_5km": {
        str(k): int(v)
        for k, v in osm_15_counts.items()
    },

    "osm_relation_3km": {
        str(k): int(v)
        for k, v in osm_3_counts.items()
    },

    "temporal_evidence_groups": {
        str(k): int(v)
        for k, v in temporal_counts.items()
    },

    "review_candidate_rows":
        int(
            len(review)
        ),

    "source_classes_planned": [
        "INDUSTRIAL_FIRE_CANDIDATE",
        "NATURAL_VEGETATION_FIRE",
        "OTHER_UNCERTAIN",
    ],

    "temporal_status_separate":
        True,

    "planned_temporal_status": [
        "EPISODIC",
        "RECURRENT",
        "PERSISTENT",
    ],

    "label_quality_scheme": {

        "A":
            "independently verified/reviewed",

        "B":
            (
                "high-confidence evidence with "
                "documented provenance"
            ),

        "C":
            "ambiguous or insufficient evidence"
    },

    "methodological_warning": (
        "OSM, temporal and Sentinel evidence are not "
        "automatically converted into final labels. "
        "Using a predictor to deterministically create "
        "the target and then evaluating the model on "
        "that same predictor would create circularity."
    ),

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
    "LABEL-EVIDENCE DATASET CREATED"
)

print(
    "=" * 80
)


print(
    f"\nEvidence rows    : {len(evidence):,}"
)

print(
    f"Review candidates: {len(review):,}"
)

print(
    "\nFinal source labels assigned: NO"
)

print(
    "\nEvidence dataset:"
)

print(
    OUTPUT_FILE
)

print(
    "\nReview candidates:"
)

print(
    REVIEW_FILE
)

print(
    "\nAudit report:"
)

print(
    REPORT_FILE
)

print(
    "\n2025 HOLDOUT REMAINS LOCKED."
)

print(
    "\nNEXT:"
    "\nInspect evidence distributions and design"
    "\nthe defensible labeling strategy."
)