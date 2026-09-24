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

FIRMS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "firms_with_temporal_2022_2024.parquet"
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
    / "reports"
    / "experiments"
    / "osm_context_validation_sample.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "osm_context_validation_summary.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Test only.
# Full enrichment comes later.
# ------------------------------------------------------------

SAMPLE_SIZE = 10_000

RANDOM_STATE = 42

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# HELPERS
# ============================================================

def validate_coordinates(df, name):

    invalid = (
        ~df["latitude"].between(-90, 90)
        |
        ~df["longitude"].between(-180, 180)
    )

    invalid_count = int(
        invalid.sum()
    )

    if invalid_count:

        raise RuntimeError(
            f"{name} contains "
            f"{invalid_count:,} invalid coordinates."
        )


def build_balltree(df):

    coordinates = np.radians(
        df[
            [
                "latitude",
                "longitude"
            ]
        ].to_numpy(
            dtype=np.float64
        )
    )

    return BallTree(
        coordinates,
        metric="haversine"
    )


def nearest_context(
    sample,
    context_df,
    tree,
    prefix
):

    query_coordinates = np.radians(
        sample[
            [
                "latitude",
                "longitude"
            ]
        ].to_numpy(
            dtype=np.float64
        )
    )

    distance_rad, index = tree.query(
        query_coordinates,
        k=1
    )

    distance_km = (
        distance_rad[:, 0]
        * EARTH_RADIUS_KM
    )

    nearest_index = index[:, 0]

    nearest = (
        context_df
        .iloc[nearest_index]
        .reset_index(drop=True)
    )

    result = pd.DataFrame(
        {
            f"distance_to_{prefix}_km":
                distance_km,

            f"nearest_{prefix}_category":
                nearest[
                    "context_category"
                ].astype(str),

            f"nearest_{prefix}_strength":
                nearest[
                    "evidence_strength"
                ].astype(int),

            f"nearest_{prefix}_osm_type":
                nearest[
                    "osm_type"
                ].astype(str),

            f"nearest_{prefix}_osm_id":
                nearest[
                    "osm_id"
                ].astype(np.int64),

            f"nearest_{prefix}_name":
                nearest[
                    "name"
                ].fillna("")
                .astype(str),

            f"nearest_{prefix}_latitude":
                nearest[
                    "latitude"
                ].astype(float),

            f"nearest_{prefix}_longitude":
                nearest[
                    "longitude"
                ].astype(float),
        }
    )

    return result


def distance_summary(series):

    return {
        "min": float(
            series.min()
        ),

        "p01": float(
            series.quantile(0.01)
        ),

        "p05": float(
            series.quantile(0.05)
        ),

        "p25": float(
            series.quantile(0.25)
        ),

        "median": float(
            series.median()
        ),

        "p75": float(
            series.quantile(0.75)
        ),

        "p90": float(
            series.quantile(0.90)
        ),

        "p95": float(
            series.quantile(0.95)
        ),

        "p99": float(
            series.quantile(0.99)
        ),

        "max": float(
            series.max()
        ),

        "mean": float(
            series.mean()
        )
    }


def within_counts(series):

    thresholds = [
        0.5,
        1.0,
        1.5,
        3.0,
        5.0,
        10.0,
        25.0
    ]

    result = {}

    total = len(series)

    for threshold in thresholds:

        count = int(
            (series <= threshold).sum()
        )

        result[
            f"within_{threshold:g}km"
        ] = {
            "count": count,

            "percentage": float(
                count / total * 100
            )
        }

    return result


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — OSM CONTEXT VALIDATION")
print("=" * 80)

print(
    "\nThis is a SMALL validation run."
)

print(
    f"Sample size: {SAMPLE_SIZE:,}"
)

print(
    "\nNo labels are created."
)

print(
    "2025 holdout is not accessed."
)


# ============================================================
# CHECK INPUTS
# ============================================================

for path in [
    FIRMS_FILE,
    INDUSTRIAL_FILE,
    VEGETATION_FILE
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file missing:\n{path}"
        )


# ============================================================
# LOAD FIRMS
# ============================================================

print(
    "\n[1/6] Loading FIRMS development data..."
)

start = time.time()

firms = pd.read_parquet(
    FIRMS_FILE,
    columns=[
        "observation_id",
        "latitude",
        "longitude",
        "acquired_at",
        "brightness",
        "frp",
        "confidence",
        "daynight",
        "type",
        "detections_30d",
        "distinct_active_days_30d"
    ]
)

print(
    f"      FIRMS rows: "
    f"{len(firms):,}"
)

validate_coordinates(
    firms,
    "FIRMS"
)


# ============================================================
# SAMPLE
# ============================================================

print(
    "\n[2/6] Creating reproducible FIRMS sample..."
)

if len(firms) < SAMPLE_SIZE:

    raise RuntimeError(
        "FIRMS dataset smaller than sample size."
    )

sample = (
    firms.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
    .sort_values(
        "acquired_at"
    )
    .reset_index(drop=True)
)

print(
    f"      Sample rows: "
    f"{len(sample):,}"
)

print(
    f"      Sample period: "
    f"{sample['acquired_at'].min()} "
    f"to "
    f"{sample['acquired_at'].max()}"
)


# ============================================================
# LOAD OSM
# ============================================================

print(
    "\n[3/6] Loading OSM evidence..."
)

industrial = pd.read_parquet(
    INDUSTRIAL_FILE
)

vegetation = pd.read_parquet(
    VEGETATION_FILE
)

validate_coordinates(
    industrial,
    "Industrial OSM"
)

validate_coordinates(
    vegetation,
    "Vegetation OSM"
)

print(
    f"      Industrial features: "
    f"{len(industrial):,}"
)

print(
    f"      Vegetation features: "
    f"{len(vegetation):,}"
)


# ============================================================
# BUILD SPATIAL INDEX
# ============================================================

print(
    "\n[4/6] Building BallTree spatial indexes..."
)

tree_start = time.time()

industrial_tree = build_balltree(
    industrial
)

vegetation_tree = build_balltree(
    vegetation
)

print(
    f"      BallTrees built in "
    f"{time.time() - tree_start:.2f} seconds."
)


# ============================================================
# QUERY
# ============================================================

print(
    "\n[5/6] Querying nearest context..."
)

query_start = time.time()

industrial_result = nearest_context(
    sample,
    industrial,
    industrial_tree,
    "industrial"
)

vegetation_result = nearest_context(
    sample,
    vegetation,
    vegetation_tree,
    "vegetation"
)

result = pd.concat(
    [
        sample.reset_index(drop=True),
        industrial_result,
        vegetation_result
    ],
    axis=1
)


# ============================================================
# DERIVED VALIDATION FEATURES
# ============================================================

result[
    "osm_distance_difference_km"
] = (
    result[
        "distance_to_vegetation_km"
    ]
    -
    result[
        "distance_to_industrial_km"
    ]
)


for threshold in [
    1.0,
    1.5,
    3.0,
    5.0
]:

    threshold_name = str(
        threshold
    ).replace(".", "_")

    result[
        f"industrial_within_{threshold_name}km"
    ] = (
        result[
            "distance_to_industrial_km"
        ]
        <= threshold
    ).astype(np.int8)

    result[
        f"vegetation_within_{threshold_name}km"
    ] = (
        result[
            "distance_to_vegetation_km"
        ]
        <= threshold
    ).astype(np.int8)


print(
    f"      Queries completed in "
    f"{time.time() - query_start:.2f} seconds."
)


# ============================================================
# SANITY CHECKS
# ============================================================

if (
    result[
        "distance_to_industrial_km"
    ] < 0
).any():

    raise RuntimeError(
        "Negative industrial distance found."
    )

if (
    result[
        "distance_to_vegetation_km"
    ] < 0
).any():

    raise RuntimeError(
        "Negative vegetation distance found."
    )

if not np.isfinite(
    result[
        "distance_to_industrial_km"
    ]
).all():

    raise RuntimeError(
        "Non-finite industrial distance found."
    )

if not np.isfinite(
    result[
        "distance_to_vegetation_km"
    ]
).all():

    raise RuntimeError(
        "Non-finite vegetation distance found."
    )


# ============================================================
# REPORT
# ============================================================

print(
    "\n[6/6] Generating validation report..."
)

industrial_distance = (
    result[
        "distance_to_industrial_km"
    ]
)

vegetation_distance = (
    result[
        "distance_to_vegetation_km"
    ]
)


report = {

    "sample_size":
        int(len(result)),

    "random_state":
        RANDOM_STATE,

    "sample_period": {
        "start": str(
            sample[
                "acquired_at"
            ].min()
        ),

        "end": str(
            sample[
                "acquired_at"
            ].max()
        )
    },

    "osm_feature_counts": {
        "industrial":
            int(len(industrial)),

        "vegetation":
            int(len(vegetation))
    },

    "industrial_distance_km":
        distance_summary(
            industrial_distance
        ),

    "vegetation_distance_km":
        distance_summary(
            vegetation_distance
        ),

    "industrial_proximity":
        within_counts(
            industrial_distance
        ),

    "vegetation_proximity":
        within_counts(
            vegetation_distance
        ),

    "nearest_industrial_categories": {
        str(k): int(v)
        for k, v in (
            result[
                "nearest_industrial_category"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "nearest_vegetation_categories": {
        str(k): int(v)
        for k, v in (
            result[
                "nearest_vegetation_category"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "methodology_notes": [

        (
            "This validation uses 10,000 "
            "reproducibly sampled FIRMS "
            "observations from 2022-2024."
        ),

        (
            "OSM way locations currently use "
            "representative mean-node points."
        ),

        (
            "Distances therefore represent "
            "distance to extracted OSM "
            "representative points and must not "
            "be described as exact distance to "
            "polygon boundaries."
        ),

        (
            "OSM context is evidence only and "
            "does not create fire labels."
        ),

        (
            "2025 FIRMS holdout remains untouched."
        )
    ]
}


result.to_csv(
    OUTPUT_FILE,
    index=False
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


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "OSM CONTEXT VALIDATION COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nINDUSTRIAL DISTANCE (km)"
)

print(
    industrial_distance
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
    .to_string()
)


print(
    "\nVEGETATION DISTANCE (km)"
)

print(
    vegetation_distance
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
    .to_string()
)


print(
    "\nINDUSTRIAL PROXIMITY"
)

for key, value in report[
    "industrial_proximity"
].items():

    print(
        f"  {key}: "
        f"{value['count']:,} "
        f"({value['percentage']:.2f}%)"
    )


print(
    "\nVEGETATION PROXIMITY"
)

for key, value in report[
    "vegetation_proximity"
].items():

    print(
        f"  {key}: "
        f"{value['count']:,} "
        f"({value['percentage']:.2f}%)"
    )


print(
    "\nTOP NEAREST INDUSTRIAL CATEGORIES"
)

print(
    result[
        "nearest_industrial_category"
    ]
    .value_counts()
    .head(15)
    .to_string()
)


print(
    "\nTOP NEAREST VEGETATION CATEGORIES"
)

print(
    result[
        "nearest_vegetation_category"
    ]
    .value_counts()
    .head(15)
    .to_string()
)


print(
    f"\nTotal runtime: "
    f"{time.time() - start:.2f} seconds"
)

print(
    f"\nValidation sample:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nReport:\n"
    f"{REPORT_FILE}"
)

print(
    "\nIMPORTANT:"
    "\nThis was validation only."
    "\nThe 1.72M-row FIRMS dataset has NOT "
    "been enriched yet."
)