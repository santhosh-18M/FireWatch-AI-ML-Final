from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OBS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "worldcover_evidence_2025.parquet"
)

REFERENCE_FILE = (
    ROOT
    / "data"
    / "labels"
    / "industrial_reference"
    / "combined_industrial_reference.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "industrial_reference_matches_2025.parquet"
)

CLOSE_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "industrial_reference_close_matches_2025.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "industrial_reference_matching_2025.json"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_KM = 6371.0088
EXPECTED_ROWS = 20_000

THRESHOLDS = [
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    25.0,
]


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 32")
print("2025 FROZEN INDEPENDENT INDUSTRIAL REFERENCE MATCHING")
print("=" * 80)

print(
    "\nPurpose:"
    "\n- Match the frozen 2025 evaluation observations"
    "\n  against the frozen development industrial reference."
    "\n- Use Haversine great-circle distance."
    "\n- Use the same BallTree methodology as development."
    "\n- Do NOT rebuild GEM/WRI references."
    "\n- Do NOT generate source labels."
    "\n- Do NOT execute the classifier."
)


# ============================================================
# 1. LOAD 2025 OBSERVATIONS
# ============================================================

print("\n[1/7] Loading frozen 2025 evaluation evidence...")


if not OBS_FILE.exists():
    raise FileNotFoundError(
        f"2025 WorldCover evidence not found:\n{OBS_FILE}"
    )


obs = pd.read_parquet(OBS_FILE)

print(f"      Observations: {len(obs):,}")


if len(obs) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} observations, "
        f"found {len(obs):,}."
    )


required_obs = [
    "observation_id",
    "latitude",
    "longitude",
]


missing_obs = [
    column
    for column in required_obs
    if column not in obs.columns
]


if missing_obs:
    raise RuntimeError(
        "Missing observation columns:\n"
        + "\n".join(missing_obs)
    )


obs["observation_id"] = (
    obs["observation_id"]
    .astype(str)
    .str.strip()
)


if obs["observation_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate observation IDs detected "
        "in the 2025 evaluation sample."
    )


if obs["latitude"].isna().any():
    raise RuntimeError(
        "Missing latitude detected."
    )


if obs["longitude"].isna().any():
    raise RuntimeError(
        "Missing longitude detected."
    )


# ============================================================
# 2. LOAD FROZEN INDUSTRIAL REFERENCE
# ============================================================

print("\n[2/7] Loading FROZEN industrial reference...")


if not REFERENCE_FILE.exists():
    raise FileNotFoundError(
        "Frozen combined industrial reference not found:\n"
        f"{REFERENCE_FILE}\n\n"
        "Do NOT rebuild it using 2025 information."
    )


references = pd.read_parquet(
    REFERENCE_FILE
)


print(
    f"      Frozen references: "
    f"{len(references):,}"
)


required_ref = [
    "reference_id",
    "reference_name",
    "reference_type",
    "reference_subtype",
    "latitude",
    "longitude",
    "reference_source",
    "location_accuracy",
    "capacity_mw",
]


missing_ref = [
    column
    for column in required_ref
    if column not in references.columns
]


if missing_ref:

    print("\nReference columns found:")

    for column in references.columns:
        print(f"      {column}")

    raise RuntimeError(
        "\nFrozen reference is missing columns:\n"
        + "\n".join(missing_ref)
    )


references["latitude"] = pd.to_numeric(
    references["latitude"],
    errors="coerce",
)

references["longitude"] = pd.to_numeric(
    references["longitude"],
    errors="coerce",
)


if references[
    ["latitude", "longitude"]
].isna().any().any():

    raise RuntimeError(
        "Frozen industrial reference contains "
        "missing coordinates."
    )


if len(references) == 0:
    raise RuntimeError(
        "Frozen industrial reference is empty."
    )


print("\n      Reference type counts:")

print(
    references[
        "reference_type"
    ]
    .value_counts()
    .to_string()
)


print("\n      Reference source counts:")

print(
    references[
        "reference_source"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 3. BUILD FROZEN SPATIAL INDEX
# ============================================================

print("\n[3/7] Building BallTree...")


ref_coords = np.radians(
    references[
        [
            "latitude",
            "longitude",
        ]
    ].to_numpy(
        dtype=float
    )
)


tree = BallTree(
    ref_coords,
    metric="haversine",
)


obs_coords = np.radians(
    obs[
        [
            "latitude",
            "longitude",
        ]
    ].to_numpy(
        dtype=float
    )
)


# ============================================================
# 4. NEAREST-REFERENCE MATCH
# ============================================================

print(
    "\n[4/7] Matching nearest frozen industrial reference..."
)


distance_rad, nearest_idx = tree.query(
    obs_coords,
    k=1,
)


distance_km = (
    distance_rad[:, 0]
    * EARTH_RADIUS_KM
)


nearest_idx = nearest_idx[:, 0]


nearest = (
    references
    .iloc[nearest_idx]
    .reset_index(drop=True)
)


result = pd.DataFrame({
    "observation_id":
        obs["observation_id"].to_numpy(),

    "independent_industrial_distance_km":
        distance_km,

    "independent_industrial_reference_id":
        nearest["reference_id"].to_numpy(),

    "independent_industrial_reference_name":
        nearest["reference_name"].to_numpy(),

    "independent_industrial_reference_type":
        nearest["reference_type"].to_numpy(),

    "independent_industrial_reference_subtype":
        nearest["reference_subtype"].to_numpy(),

    "independent_industrial_reference_source":
        nearest["reference_source"].to_numpy(),

    "independent_industrial_location_accuracy":
        nearest["location_accuracy"].to_numpy(),

    "independent_industrial_capacity_mw":
        nearest["capacity_mw"].to_numpy(),

    "independent_industrial_reference_latitude":
        nearest["latitude"].to_numpy(),

    "independent_industrial_reference_longitude":
        nearest["longitude"].to_numpy(),
})


# ============================================================
# 5. APPLY FROZEN DISTANCE FLAGS
# ============================================================

print("\n[5/7] Applying frozen distance thresholds...")


for threshold in THRESHOLDS:

    suffix = str(
        threshold
    ).replace(".", "_")

    result[
        f"independent_industrial_within_{suffix}km"
    ] = (
        result[
            "independent_industrial_distance_km"
        ]
        <= threshold
    )


if len(result) != EXPECTED_ROWS:
    raise RuntimeError(
        "Industrial matching changed row count."
    )


if result["observation_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate IDs produced by industrial matching."
    )


# ============================================================
# 6. AUDIT
# ============================================================

print("\n[6/7] Auditing frozen-reference matches...")


dist = result[
    "independent_industrial_distance_km"
]


print(
    f"      Minimum distance : "
    f"{dist.min():.3f} km"
)

print(
    f"      Median distance  : "
    f"{dist.median():.3f} km"
)

print(
    f"      Mean distance    : "
    f"{dist.mean():.3f} km"
)

print(
    f"      P90 distance     : "
    f"{dist.quantile(0.90):.3f} km"
)

print(
    f"      P95 distance     : "
    f"{dist.quantile(0.95):.3f} km"
)

print(
    f"      P99 distance     : "
    f"{dist.quantile(0.99):.3f} km"
)


threshold_report = {}


print(
    "\n      Independent industrial proximity:"
)


for threshold in THRESHOLDS:

    suffix = str(
        threshold
    ).replace(".", "_")

    column = (
        f"independent_industrial_within_"
        f"{suffix}km"
    )

    count = int(
        result[column].sum()
    )

    percentage = (
        count
        / len(result)
        * 100
    )

    threshold_report[
        str(threshold)
    ] = {
        "count": count,
        "percentage": percentage,
    }

    print(
        f"        <= {threshold:>4.1f} km : "
        f"{count:>6,} "
        f"({percentage:>6.3f}%)"
    )


print(
    "\n      Nearest reference type:"
)

print(
    result[
        "independent_industrial_reference_type"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n      Nearest reference source:"
)

print(
    result[
        "independent_industrial_reference_source"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 7. SAVE
# ============================================================

print("\n[7/7] Saving...")


result.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# Close-match audit only.
audit = result.merge(
    obs,
    on="observation_id",
    how="left",
    validate="one_to_one",
)


close = (
    audit[
        audit[
            "independent_industrial_distance_km"
        ]
        <= 10.0
    ]
    .copy()
    .sort_values(
        "independent_industrial_distance_km"
    )
)


close.to_csv(
    CLOSE_FILE,
    index=False,
)


report = {
    "stage":
        "2025_frozen_independent_industrial_matching",

    "evaluation_observations":
        int(len(result)),

    "frozen_reference_file":
        str(REFERENCE_FILE),

    "frozen_reference_count":
        int(len(references)),

    "reference_type_counts":
        {
            str(k): int(v)
            for k, v in (
                references[
                    "reference_type"
                ]
                .value_counts()
                .items()
            )
        },

    "reference_source_counts":
        {
            str(k): int(v)
            for k, v in (
                references[
                    "reference_source"
                ]
                .value_counts()
                .items()
            )
        },

    "distance_km": {
        "minimum":
            float(dist.min()),

        "median":
            float(dist.median()),

        "mean":
            float(dist.mean()),

        "p90":
            float(
                dist.quantile(0.90)
            ),

        "p95":
            float(
                dist.quantile(0.95)
            ),

        "p99":
            float(
                dist.quantile(0.99)
            ),
    },

    "threshold_coverage":
        threshold_report,

    "close_matches_within_10km":
        int(len(close)),

    "methodology": {
        "distance":
            "Haversine great-circle distance",

        "spatial_index":
            "BallTree",

        "reference_policy":
            (
                "Frozen development combined "
                "industrial reference reused "
                "without modification."
            ),

        "label_created":
            False,

        "interpretation":
            (
                "Spatial proximity is independent "
                "industrial-context evidence and "
                "does not prove FIRMS anomaly cause."
            ),
    },

    "worldcover_already_prepared":
        True,

    "source_labels_generated":
        False,

    "model_executed":
        False,

    "development_model_modified":
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


print(
    "\n" + "=" * 80
)

print(
    "STEP 32 COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\nFrozen industrial references : "
    f"{len(references):,}"
)

print(
    f"2025 evaluation observations : "
    f"{len(result):,}"
)

print(
    f"Within 2 km                 : "
    f"{int(result['independent_industrial_within_2_0km'].sum()):,}"
)

print(
    f"Within 5 km                 : "
    f"{int(result['independent_industrial_within_5_0km'].sum()):,}"
)

print(
    f"Within 10 km                : "
    f"{len(close):,}"
)


print(
    f"\nEvidence:"
    f"\n{OUTPUT_FILE}"
)


print(
    f"\nClose-match audit:"
    f"\n{CLOSE_FILE}"
)


print(
    f"\nReport:"
    f"\n{REPORT_FILE}"
)


print(
    "\nIMPORTANT:"
    "\n- Frozen industrial reference reused."
    "\n- No industrial reference was rebuilt."
    "\n- No source labels generated."
    "\n- No classifier executed."
    "\n- Next: apply the FROZEN V1 source-label policy."
)