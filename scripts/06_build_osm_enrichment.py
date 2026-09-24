from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
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
    / "data"
    / "processed"
    / "firms_with_osm_2022_2024.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "osm_enrichment_2022_2024_summary.json"
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# OSM EVIDENCE CHANNELS
# ============================================================
#
# We deliberately do NOT treat all "industrial" OSM
# objects as equally meaningful.
#
# These groups represent contextual evidence strength.
# They are NOT fire labels.
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

    invalid_count = int(
        invalid.sum()
    )

    if invalid_count:

        raise RuntimeError(
            f"{name} contains "
            f"{invalid_count:,} invalid coordinates."
        )


def build_tree(df):

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


def query_nearest(
    firms_coordinates_rad,
    context_df,
    tree,
    prefix,
    include_metadata=True
):

    distance_rad, index = tree.query(
        firms_coordinates_rad,
        k=1
    )

    distance_km = (
        distance_rad[:, 0]
        * EARTH_RADIUS_KM
    ).astype(np.float32)

    nearest_index = index[:, 0]

    output = {
        f"distance_to_{prefix}_km":
            distance_km
    }

    if include_metadata:

        nearest = (
            context_df
            .iloc[nearest_index]
            .reset_index(drop=True)
        )

        output[
            f"nearest_{prefix}_category"
        ] = (
            nearest[
                "context_category"
            ]
            .astype(str)
            .to_numpy()
        )

        output[
            f"nearest_{prefix}_strength"
        ] = (
            nearest[
                "evidence_strength"
            ]
            .astype(np.int8)
            .to_numpy()
        )

        output[
            f"nearest_{prefix}_osm_type"
        ] = (
            nearest[
                "osm_type"
            ]
            .astype(str)
            .to_numpy()
        )

        output[
            f"nearest_{prefix}_osm_id"
        ] = (
            nearest[
                "osm_id"
            ]
            .astype(np.int64)
            .to_numpy()
        )

    return output


def add_proximity_flags(
    df,
    distance_column,
    prefix
):

    for threshold in [
        1.0,
        1.5,
        3.0,
        5.0
    ]:

        suffix = str(
            threshold
        ).replace(".", "_")

        df[
            f"{prefix}_within_{suffix}km"
        ] = (
            df[distance_column]
            <= threshold
        ).astype(np.int8)


def distance_summary(series):

    return {
        "min":
            float(series.min()),

        "p01":
            float(series.quantile(0.01)),

        "p05":
            float(series.quantile(0.05)),

        "p25":
            float(series.quantile(0.25)),

        "median":
            float(series.median()),

        "p75":
            float(series.quantile(0.75)),

        "p90":
            float(series.quantile(0.90)),

        "p95":
            float(series.quantile(0.95)),

        "p99":
            float(series.quantile(0.99)),

        "max":
            float(series.max()),

        "mean":
            float(series.mean())
    }


def proximity_summary(
    series
):

    result = {}

    for threshold in [
        1.0,
        1.5,
        3.0,
        5.0,
        10.0,
        25.0
    ]:

        count = int(
            (series <= threshold).sum()
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
                )
        }

    return result


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — FULL OSM CONTEXT ENRICHMENT")
print("=" * 80)

print(
    "\nInput:"
    f"\n{INPUT_FILE}"
)

print(
    "\nThis stage:"
    "\n  FIRMS thermal evidence"
    "\n+ causal temporal evidence"
    "\n+ OSM contextual evidence"
)

print(
    "\nNo source/fire labels are created."
)

print(
    "2025 holdout is not accessed."
)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    INPUT_FILE,
    INDUSTRIAL_FILE,
    VEGETATION_FILE
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


# ============================================================
# LOAD FIRMS
# ============================================================

overall_start = time.time()

print(
    "\n[1/8] Loading FIRMS + temporal dataset..."
)

firms = pd.read_parquet(
    INPUT_FILE
)

original_rows = len(firms)

print(
    f"      Rows: {original_rows:,}"
)

print(
    f"      Existing columns: "
    f"{len(firms.columns):,}"
)

validate_coordinates(
    firms,
    "FIRMS"
)


# ============================================================
# LOAD OSM
# ============================================================

print(
    "\n[2/8] Loading OSM evidence..."
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
    f"      Industrial: "
    f"{len(industrial):,}"
)

print(
    f"      Vegetation/agriculture: "
    f"{len(vegetation):,}"
)


# ============================================================
# SPLIT INDUSTRIAL EVIDENCE
# ============================================================

print(
    "\n[3/8] Separating OSM evidence channels..."
)

strong_industrial = (
    industrial[
        industrial[
            "context_category"
        ].isin(
            STRONG_INDUSTRIAL
        )
    ]
    .reset_index(drop=True)
)

supporting_industrial = (
    industrial[
        industrial[
            "context_category"
        ].isin(
            SUPPORTING_INDUSTRIAL
        )
    ]
    .reset_index(drop=True)
)

weak_infrastructure = (
    industrial[
        industrial[
            "context_category"
        ].isin(
            WEAK_INFRASTRUCTURE
        )
    ]
    .reset_index(drop=True)
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


for name, frame in [
    (
        "strong industrial",
        strong_industrial
    ),
    (
        "supporting industrial",
        supporting_industrial
    ),
    (
        "weak infrastructure",
        weak_infrastructure
    ),
    (
        "vegetation",
        vegetation
    )
]:

    if frame.empty:

        raise RuntimeError(
            f"{name} context is empty."
        )


# ============================================================
# BUILD BALLTREES
# ============================================================

print(
    "\n[4/8] Building spatial indexes..."
)

tree_start = time.time()

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

print(
    f"      Spatial indexes built in "
    f"{time.time() - tree_start:.2f} seconds."
)


# ============================================================
# PREPARE FIRMS COORDINATES
# ============================================================

firms_coordinates_rad = np.radians(
    firms[
        [
            "latitude",
            "longitude"
        ]
    ].to_numpy(
        dtype=np.float64
    )
)


# ============================================================
# QUERY ALL FIRMS
# ============================================================

print(
    "\n[5/8] Querying "
    f"{original_rows:,} FIRMS observations..."
)

query_start = time.time()


# ------------------------------------------------------------
# Strong industrial
# ------------------------------------------------------------

print(
    "      Querying strong industrial context..."
)

strong_result = query_nearest(
    firms_coordinates_rad,
    strong_industrial,
    strong_tree,
    "strong_industrial",
    include_metadata=True
)

for column, values in (
    strong_result.items()
):

    firms[column] = values


# ------------------------------------------------------------
# Supporting industrial
# ------------------------------------------------------------

print(
    "      Querying supporting industrial context..."
)

support_result = query_nearest(
    firms_coordinates_rad,
    supporting_industrial,
    supporting_tree,
    "supporting_industrial",
    include_metadata=True
)

for column, values in (
    support_result.items()
):

    firms[column] = values


# ------------------------------------------------------------
# Weak infrastructure
# ------------------------------------------------------------

print(
    "      Querying weak infrastructure context..."
)

weak_result = query_nearest(
    firms_coordinates_rad,
    weak_infrastructure,
    weak_tree,
    "infrastructure",
    include_metadata=True
)

for column, values in (
    weak_result.items()
):

    firms[column] = values


# ------------------------------------------------------------
# Vegetation/agriculture
# ------------------------------------------------------------

print(
    "      Querying vegetation/agricultural context..."
)

vegetation_result = query_nearest(
    firms_coordinates_rad,
    vegetation,
    vegetation_tree,
    "vegetation",
    include_metadata=True
)

for column, values in (
    vegetation_result.items()
):

    firms[column] = values


print(
    f"      All nearest-neighbour queries "
    f"completed in "
    f"{time.time() - query_start:.2f} seconds."
)


# ============================================================
# DERIVED OSM FEATURES
# ============================================================

print(
    "\n[6/8] Building derived OSM features..."
)


# ------------------------------------------------------------
# Proximity flags
# ------------------------------------------------------------

add_proximity_flags(
    firms,
    "distance_to_strong_industrial_km",
    "strong_industrial"
)

add_proximity_flags(
    firms,
    "distance_to_supporting_industrial_km",
    "supporting_industrial"
)

add_proximity_flags(
    firms,
    "distance_to_infrastructure_km",
    "infrastructure"
)

add_proximity_flags(
    firms,
    "distance_to_vegetation_km",
    "vegetation"
)


# ------------------------------------------------------------
# Strong industrial vs vegetation comparison
# ------------------------------------------------------------

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
).astype(np.float32)


# Positive:
# strong industrial representative point is closer.
#
# Negative:
# vegetation representative point is closer.


# ------------------------------------------------------------
# Context relation
# ------------------------------------------------------------

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
    )
]


choices = [
    "strong_industrial_context",
    "vegetation_context",
    "mixed_context"
]


firms[
    "osm_context_relation"
] = np.select(
    conditions,
    choices,
    default="no_close_context"
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n[7/8] Validating enriched dataset..."
)

if len(firms) != original_rows:

    raise RuntimeError(
        "Row count changed during OSM enrichment."
    )


distance_columns = [
    "distance_to_strong_industrial_km",
    "distance_to_supporting_industrial_km",
    "distance_to_infrastructure_km",
    "distance_to_vegetation_km"
]


for column in distance_columns:

    if firms[column].isna().any():

        raise RuntimeError(
            f"{column} contains missing values."
        )

    if (
        firms[column] < 0
    ).any():

        raise RuntimeError(
            f"{column} contains negative distances."
        )

    if not np.isfinite(
        firms[column]
    ).all():

        raise RuntimeError(
            f"{column} contains non-finite values."
        )


if firms[
    "observation_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate observation_id detected."
    )


print(
    f"      Rows preserved: "
    f"{len(firms):,}"
)

print(
    "      Distance checks passed."
)

print(
    "      observation_id uniqueness passed."
)


# ============================================================
# SAVE
# ============================================================

print(
    "\n[8/8] Saving enriched dataset..."
)

save_start = time.time()

firms.to_parquet(
    OUTPUT_FILE,
    index=False
)

print(
    f"      Saved in "
    f"{time.time() - save_start:.2f} seconds."
)


# ============================================================
# REPORT
# ============================================================

report = {

    "input_file":
        str(INPUT_FILE),

    "output_file":
        str(OUTPUT_FILE),

    "rows":
        int(len(firms)),

    "columns":
        int(len(firms.columns)),

    "osm_feature_counts": {

        "all_industrial":
            int(len(industrial)),

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

        "vegetation_agriculture":
            int(len(vegetation))
    },

    "distance_summaries": {

        column:
            distance_summary(
                firms[column]
            )

        for column
        in distance_columns
    },

    "proximity_summaries": {

        column:
            proximity_summary(
                firms[column]
            )

        for column
        in distance_columns
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

    "nearest_strong_industrial_categories": {

        str(k): int(v)

        for k, v in (
            firms[
                "nearest_strong_industrial_category"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "nearest_vegetation_categories": {

        str(k): int(v)

        for k, v in (
            firms[
                "nearest_vegetation_category"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "methodology_notes": [

        (
            "OSM context is evidence only and "
            "does not create source/fire labels."
        ),

        (
            "Industrial OSM evidence was separated "
            "into strong, supporting and weak "
            "infrastructure channels."
        ),

        (
            "Power substations and fuel facilities "
            "are not treated as equivalent to "
            "industrial plants or industrial land."
        ),

        (
            "OSM ways currently use representative "
            "mean-node points for scalable proximity "
            "analysis."
        ),

        (
            "Distances therefore must not be "
            "described as exact polygon-boundary "
            "distances."
        ),

        (
            "Latitude and longitude remain available "
            "for GIS and spatial validation but "
            "should not automatically become model "
            "predictors."
        ),

        (
            "No labels were generated in this stage."
        ),

        (
            "2025 FIRMS holdout was not accessed."
        )
    ]
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


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "FULL OSM ENRICHMENT COMPLETE"
)

print(
    "=" * 80
)

print(
    f"\nRows: "
    f"{len(firms):,}"
)

print(
    f"Columns: "
    f"{len(firms.columns):,}"
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
    "\nSTRONG INDUSTRIAL DISTANCE (km)"
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
            .99
        ]
    )
    .to_string()
)


print(
    "\nVEGETATION DISTANCE (km)"
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
            .99
        ]
    )
    .to_string()
)


print(
    "\nNEAREST STRONG INDUSTRIAL CATEGORIES"
)

print(
    firms[
        "nearest_strong_industrial_category"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nNEAREST VEGETATION CATEGORIES"
)

print(
    firms[
        "nearest_vegetation_category"
    ]
    .value_counts()
    .to_string()
)


print(
    f"\nTotal runtime: "
    f"{(time.time() - overall_start) / 60:.2f} minutes"
)


print(
    f"\nSaved dataset:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nReport:\n"
    f"{REPORT_FILE}"
)


print(
    "\nIMPORTANT:"
    "\n- No fire/source labels created."
    "\n- Latitude/longitude are GIS metadata, "
    "not automatically model predictors."
    "\n- OSM distances are contextual evidence."
    "\n- 2025 holdout remains untouched."
)