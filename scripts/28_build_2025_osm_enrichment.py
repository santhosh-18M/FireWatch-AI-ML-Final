from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "firms_with_temporal_2025.parquet"
)

INDUSTRIAL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "osm_industrial_features.parquet"
)

VEGETATION_FILE = (
    ROOT
    / "data"
    / "processed"
    / "osm_vegetation_features.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "firms_with_osm_2025.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "osm_enrichment_2025_summary.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# FROZEN OSM EVIDENCE CHANNELS
# ============================================================

STRONG_INDUSTRIAL = {
    "industrial_landuse",
    "mining",
    "power_plant",
    "manufacturing",
    "industrial_tag",
    "oil_gas",
    "kiln",
}

SUPPORTING_INDUSTRIAL = {
    "storage",
    "industrial_chimney",
    "power_generation",
}

WEAK_INFRASTRUCTURE = {
    "power_substation",
    "fuel_facility",
}


# ============================================================
# HELPERS
# ============================================================

def validate_coordinates(df, name):

    invalid = (
        ~df["latitude"].between(-90, 90)
        |
        ~df["longitude"].between(-180, 180)
    )

    count = int(
        invalid.sum()
    )

    if count:
        raise RuntimeError(
            f"{name} contains "
            f"{count:,} invalid coordinates."
        )


def build_tree(df):

    coordinates = np.radians(
        df[
            [
                "latitude",
                "longitude",
            ]
        ].to_numpy(
            dtype=np.float64
        )
    )

    return BallTree(
        coordinates,
        metric="haversine",
    )


def query_nearest(
    firms_coordinates_rad,
    context_df,
    tree,
    prefix,
):

    distance_rad, index = tree.query(
        firms_coordinates_rad,
        k=1,
    )

    distance_km = (
        distance_rad[:, 0]
        * EARTH_RADIUS_KM
    ).astype(
        np.float32
    )

    nearest_index = (
        index[:, 0]
    )

    nearest = (
        context_df
        .iloc[
            nearest_index
        ]
        .reset_index(
            drop=True
        )
    )

    return {
        f"distance_to_{prefix}_km":
            distance_km,

        f"nearest_{prefix}_category":
            nearest[
                "context_category"
            ]
            .astype(str)
            .to_numpy(),

        f"nearest_{prefix}_strength":
            nearest[
                "evidence_strength"
            ]
            .astype(np.int8)
            .to_numpy(),

        f"nearest_{prefix}_osm_type":
            nearest[
                "osm_type"
            ]
            .astype(str)
            .to_numpy(),

        f"nearest_{prefix}_osm_id":
            nearest[
                "osm_id"
            ]
            .astype(np.int64)
            .to_numpy(),
    }


def add_proximity_flags(
    df,
    distance_column,
    prefix,
):

    for threshold in [
        1.0,
        1.5,
        3.0,
        5.0,
    ]:

        suffix = (
            str(threshold)
            .replace(".", "_")
        )

        df[
            f"{prefix}_within_{suffix}km"
        ] = (
            df[
                distance_column
            ]
            <= threshold
        ).astype(
            np.int8
        )


def distance_summary(series):

    return {
        "min":
            float(series.min()),

        "p01":
            float(
                series.quantile(.01)
            ),

        "p05":
            float(
                series.quantile(.05)
            ),

        "p25":
            float(
                series.quantile(.25)
            ),

        "median":
            float(
                series.median()
            ),

        "p75":
            float(
                series.quantile(.75)
            ),

        "p90":
            float(
                series.quantile(.90)
            ),

        "p95":
            float(
                series.quantile(.95)
            ),

        "p99":
            float(
                series.quantile(.99)
            ),

        "max":
            float(series.max()),

        "mean":
            float(series.mean()),
    }


def proximity_summary(series):

    result = {}

    for threshold in [
        1.0,
        1.5,
        3.0,
        5.0,
        10.0,
        25.0,
    ]:

        count = int(
            (
                series
                <= threshold
            ).sum()
        )

        result[
            f"within_{threshold:g}km"
        ] = {
            "count":
                count,

            "percentage":
                float(
                    count
                    / len(series)
                    * 100
                ),
        }

    return result


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 28")
print("2025 HOLDOUT — FROZEN OSM ENRICHMENT")
print("=" * 80)

print(
    "\nIMPORTANT:"
    "\n- Exact OSM evidence groups from development."
    "\n- Same nearest-neighbour methodology."
    "\n- OSM does NOT generate source labels."
    "\n- Model is NOT executed."
)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    INPUT_FILE,
    INDUSTRIAL_FILE,
    VEGETATION_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


# ============================================================
# LOAD
# ============================================================

start = time.time()

print(
    "\n[1/8] Loading 2025 FIRMS + temporal..."
)

firms = pd.read_parquet(
    INPUT_FILE
)

original_rows = len(
    firms
)

print(
    f"      Rows: "
    f"{original_rows:,}"
)


if original_rows != 646_376:

    raise RuntimeError(
        "Unexpected 2025 row count."
    )


validate_coordinates(
    firms,
    "2025 FIRMS",
)


# ============================================================
# LOAD OSM
# ============================================================

print(
    "\n[2/8] Loading frozen OSM evidence..."
)

industrial = pd.read_parquet(
    INDUSTRIAL_FILE
)

vegetation = pd.read_parquet(
    VEGETATION_FILE
)


validate_coordinates(
    industrial,
    "Industrial OSM",
)

validate_coordinates(
    vegetation,
    "Vegetation OSM",
)


print(
    f"      Industrial: "
    f"{len(industrial):,}"
)

print(
    f"      Vegetation: "
    f"{len(vegetation):,}"
)


# ============================================================
# SAME OSM GROUPS
# ============================================================

print(
    "\n[3/8] Applying frozen evidence groups..."
)

strong_industrial = (
    industrial[
        industrial[
            "context_category"
        ].isin(
            STRONG_INDUSTRIAL
        )
    ]
    .reset_index(
        drop=True
    )
)


supporting_industrial = (
    industrial[
        industrial[
            "context_category"
        ].isin(
            SUPPORTING_INDUSTRIAL
        )
    ]
    .reset_index(
        drop=True
    )
)


weak_infrastructure = (
    industrial[
        industrial[
            "context_category"
        ].isin(
            WEAK_INFRASTRUCTURE
        )
    ]
    .reset_index(
        drop=True
    )
)


print(
    f"      Strong industrial: "
    f"{len(strong_industrial):,}"
)

print(
    f"      Supporting industrial: "
    f"{len(supporting_industrial):,}"
)

print(
    f"      Weak infrastructure: "
    f"{len(weak_infrastructure):,}"
)

print(
    f"      Vegetation/agriculture: "
    f"{len(vegetation):,}"
)


for name, frame in [
    (
        "strong industrial",
        strong_industrial,
    ),
    (
        "supporting industrial",
        supporting_industrial,
    ),
    (
        "weak infrastructure",
        weak_infrastructure,
    ),
    (
        "vegetation",
        vegetation,
    ),
]:

    if frame.empty:

        raise RuntimeError(
            f"{name} evidence is empty."
        )


# ============================================================
# TREES
# ============================================================

print(
    "\n[4/8] Building BallTrees..."
)

strong_tree = build_tree(
    strong_industrial
)

supporting_tree = build_tree(
    supporting_industrial
)

weak_tree = build_tree(
    weak_infrastructure
)

vegetation_tree = build_tree(
    vegetation
)


# ============================================================
# FIRMS COORDINATES
# ============================================================

coordinates = np.radians(
    firms[
        [
            "latitude",
            "longitude",
        ]
    ].to_numpy(
        dtype=np.float64
    )
)


# ============================================================
# QUERIES
# ============================================================

print(
    "\n[5/8] Querying nearest context..."
)


queries = [
    (
        strong_industrial,
        strong_tree,
        "strong_industrial",
    ),
    (
        supporting_industrial,
        supporting_tree,
        "supporting_industrial",
    ),
    (
        weak_infrastructure,
        weak_tree,
        "infrastructure",
    ),
    (
        vegetation,
        vegetation_tree,
        "vegetation",
    ),
]


for context_df, tree, prefix in queries:

    print(
        f"      {prefix}..."
    )

    values = query_nearest(
        coordinates,
        context_df,
        tree,
        prefix,
    )

    for column, data in (
        values.items()
    ):

        firms[column] = data


# ============================================================
# DERIVED FEATURES
# ============================================================

print(
    "\n[6/8] Building frozen derived OSM features..."
)


add_proximity_flags(
    firms,
    "distance_to_strong_industrial_km",
    "strong_industrial",
)

add_proximity_flags(
    firms,
    "distance_to_supporting_industrial_km",
    "supporting_industrial",
)

add_proximity_flags(
    firms,
    "distance_to_infrastructure_km",
    "infrastructure",
)

add_proximity_flags(
    firms,
    "distance_to_vegetation_km",
    "vegetation",
)


firms[
    "strong_industrial_vs_vegetation_distance_km"
] = (
    firms[
        "distance_to_vegetation_km"
    ]
    -
    firms[
        "distance_to_strong_industrial_km"
    ]
).astype(
    np.float32
)


strong_distance = (
    firms[
        "distance_to_strong_industrial_km"
    ]
)

vegetation_distance = (
    firms[
        "distance_to_vegetation_km"
    ]
)


conditions = [
    (
        (strong_distance <= 3.0)
        &
        (vegetation_distance > 3.0)
    ),
    (
        (vegetation_distance <= 3.0)
        &
        (strong_distance > 3.0)
    ),
    (
        (strong_distance <= 3.0)
        &
        (vegetation_distance <= 3.0)
    ),
]


choices = [
    "strong_industrial_context",
    "vegetation_context",
    "mixed_context",
]


firms[
    "osm_context_relation"
] = np.select(
    conditions,
    choices,
    default="no_close_context",
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n[7/8] Validating..."
)


if len(
    firms
) != original_rows:

    raise RuntimeError(
        "Row count changed."
    )


if not (
    firms["year"] == 2025
).all():

    raise RuntimeError(
        "Non-2025 row detected."
    )


if firms[
    "observation_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate observation ID."
    )


distance_columns = [
    "distance_to_strong_industrial_km",
    "distance_to_supporting_industrial_km",
    "distance_to_infrastructure_km",
    "distance_to_vegetation_km",
]


for column in distance_columns:

    if firms[
        column
    ].isna().any():

        raise RuntimeError(
            f"{column} contains NaN."
        )

    if (
        firms[
            column
        ]
        < 0
    ).any():

        raise RuntimeError(
            f"{column} contains negative values."
        )

    if not np.isfinite(
        firms[column]
    ).all():

        raise RuntimeError(
            f"{column} contains non-finite values."
        )


# ============================================================
# SAVE
# ============================================================

print(
    "\n[8/8] Saving..."
)

firms.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

report = {
    "stage":
        "2025_frozen_osm_enrichment",

    "rows":
        int(
            len(firms)
        ),

    "osm_feature_counts": {
        "all_industrial":
            int(
                len(industrial)
            ),

        "strong_industrial":
            int(
                len(strong_industrial)
            ),

        "supporting_industrial":
            int(
                len(supporting_industrial)
            ),

        "weak_infrastructure":
            int(
                len(weak_infrastructure)
            ),

        "vegetation":
            int(
                len(vegetation)
            ),
    },

    "distance_summaries": {
        column:
            distance_summary(
                firms[column]
            )
        for column in distance_columns
    },

    "proximity_summaries": {
        column:
            proximity_summary(
                firms[column]
            )
        for column in distance_columns
    },

    "osm_context_relation": {
        str(k): int(v)
        for k, v in (
            firms[
                "osm_context_relation"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "model_used":
        False,

    "source_labels_generated":
        False,

    "osm_policy_changed":
        False,

    "notes": [
        (
            "Exact frozen OSM context categories "
            "from development were reused."
        ),
        (
            "OSM evidence is contextual and does "
            "not itself prove a fire source."
        ),
        (
            "OSM way locations remain representative "
            "points rather than exact polygon boundaries."
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


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 28 COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\nRows: "
    f"{len(firms):,}"
)


print(
    "\nOSM CONTEXT RELATION"
)

print(
    firms[
        "osm_context_relation"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nSTRONG INDUSTRIAL DISTANCE"
)

print(
    firms[
        "distance_to_strong_industrial_km"
    ]
    .describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99,
        ]
    )
    .to_string()
)


print(
    "\nVEGETATION DISTANCE"
)

print(
    firms[
        "distance_to_vegetation_km"
    ]
    .describe(
        percentiles=[
            .01,
            .05,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99,
        ]
    )
    .to_string()
)


print(
    f"\nRuntime: "
    f"{(time.time() - start) / 60:.2f} minutes"
)


print(
    f"\nSaved:\n{OUTPUT_FILE}"
)


print(
    "\nIMPORTANT:"
    "\n- 646,376 2025 observations preserved."
    "\n- Same OSM feature definitions as development."
    "\n- No labels generated."
    "\n- Frozen model has NOT been executed."
)