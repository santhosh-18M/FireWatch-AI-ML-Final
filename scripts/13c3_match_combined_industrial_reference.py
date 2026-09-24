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
    ROOT / "data" / "labels"
    / "label_evidence_2022_2024.parquet"
)

OIL_GAS_FILE = (
    ROOT / "data" / "labels"
    / "industrial_reference"
    / "india_industrial_reference.parquet"
)

POWER_FILE = (
    ROOT / "data" / "labels"
    / "industrial_reference"
    / "india_powerplant_reference.parquet"
)

OUTPUT_DIR = (
    ROOT / "data" / "labels"
    / "industrial_reference"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "combined_industrial_matches_2022_2024.parquet"
)

CLOSE_FILE = (
    OUTPUT_DIR
    / "combined_industrial_close_matches.csv"
)

REFERENCE_FILE = (
    OUTPUT_DIR
    / "combined_industrial_reference.parquet"
)

REPORT_FILE = (
    ROOT / "reports" / "experiments"
    / "combined_industrial_matching.json"
)

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
print("FIREWATCH — STEP 13C-3")
print("COMBINED INDEPENDENT INDUSTRIAL MATCHING")
print("=" * 80)

print(
    "\nSources:"
    "\n- GEM Oil/Gas Extraction Tracker"
    "\n- WRI Global Power Plant Database"
)

print(
    "\nImportant:"
    "\n- These are independent industrial-context references."
    "\n- Distance does NOT prove the cause of a FIRMS detection."
    "\n- No source labels are created here."
    "\n- OSM is NOT used to create this evidence."
    "\n- 2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD OBSERVATIONS
# ============================================================

print("\n[1/8] Loading observations...")

obs = pd.read_parquet(OBS_FILE)

print(f"      Observations: {len(obs):,}")

if len(obs) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,}, found {len(obs):,}"
    )

required_obs = [
    "observation_id",
    "latitude",
    "longitude",
]

for column in required_obs:
    if column not in obs.columns:
        raise RuntimeError(
            f"Missing observation column: {column}"
        )

if obs["observation_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate observation IDs detected."
    )


# ============================================================
# 2. LOAD OIL/GAS
# ============================================================

print("\n[2/8] Loading oil/gas references...")

oil = pd.read_parquet(OIL_GAS_FILE)

oil = oil[
    oil["eligible_for_land_matching"]
    .fillna(False)
    .astype(bool)
].copy()

oil_ref = pd.DataFrame({
    "reference_id":
        oil["industrial_reference_id"].astype(str),

    "reference_name":
        oil["industrial_reference_name"].astype(str),

    "reference_type":
        "oil_gas_extraction",

    "reference_subtype":
        oil["fuel_type"].astype(str),

    "latitude":
        pd.to_numeric(
            oil["latitude"],
            errors="coerce"
        ),

    "longitude":
        pd.to_numeric(
            oil["longitude"],
            errors="coerce"
        ),

    "reference_source":
        "GEM",

    "location_accuracy":
        oil["location_accuracy"].astype(str),

    "capacity_mw":
        np.nan,
})

print(
    f"      Eligible oil/gas: "
    f"{len(oil_ref):,}"
)


# ============================================================
# 3. LOAD POWER PLANTS
# ============================================================

print("\n[3/8] Loading thermal power references...")

power = pd.read_parquet(POWER_FILE)

power = power[
    power["thermal_industrial_reference"]
    .fillna(False)
    .astype(bool)
].copy()

power_ref = pd.DataFrame({
    "reference_id":
        power["industrial_reference_id"].astype(str),

    "reference_name":
        power["industrial_reference_name"].astype(str),

    "reference_type":
        "thermal_power_plant",

    "reference_subtype":
        power["primary_fuel"].astype(str),

    "latitude":
        pd.to_numeric(
            power["latitude"],
            errors="coerce"
        ),

    "longitude":
        pd.to_numeric(
            power["longitude"],
            errors="coerce"
        ),

    "reference_source":
        "WRI",

    "location_accuracy":
        "database_coordinate",

    "capacity_mw":
        pd.to_numeric(
            power["capacity_mw"],
            errors="coerce"
        ),
})

print(
    f"      Thermal power plants: "
    f"{len(power_ref):,}"
)


# ============================================================
# 4. COMBINE REFERENCES
# ============================================================

print("\n[4/8] Combining references...")

references = pd.concat(
    [
        oil_ref,
        power_ref,
    ],
    ignore_index=True
)

references = references[
    references["latitude"].notna()
    &
    references["longitude"].notna()
].copy()

references = references[
    references["latitude"].between(6, 38)
    &
    references["longitude"].between(68, 98)
].copy()


# Avoid accidental cross-dataset exact duplicate points.
references["coordinate_key"] = (
    references["latitude"]
    .round(5)
    .astype(str)
    + "_"
    + references["longitude"]
    .round(5)
    .astype(str)
)


duplicate_coordinates = int(
    references[
        "coordinate_key"
    ].duplicated().sum()
)


print(
    f"      Combined references : "
    f"{len(references):,}"
)

print(
    f"      Duplicate coordinate keys: "
    f"{duplicate_coordinates:,}"
)


print("\n      Reference type counts:")

print(
    references[
        "reference_type"
    ]
    .value_counts()
    .to_string()
)


print("\n      Reference subtype counts:")

print(
    references[
        "reference_subtype"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 5. BUILD SPATIAL INDEX
# ============================================================

print("\n[5/8] Building BallTree...")

ref_coords = np.radians(
    references[
        [
            "latitude",
            "longitude",
        ]
    ].to_numpy(dtype=float)
)

tree = BallTree(
    ref_coords,
    metric="haversine"
)

obs_coords = np.radians(
    obs[
        [
            "latitude",
            "longitude",
        ]
    ].to_numpy(dtype=float)
)


distance_rad, nearest_idx = tree.query(
    obs_coords,
    k=1
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


# ============================================================
# 6. BUILD EVIDENCE TABLE
# ============================================================

print("\n[6/8] Building combined evidence...")

result = pd.DataFrame({
    "observation_id":
        obs["observation_id"].astype(str),

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


# ============================================================
# 7. AUDIT
# ============================================================

print("\n[7/8] Auditing coverage...")


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


threshold_report = {}

print("\n      Independent industrial proximity:")

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

    pct = (
        count / len(result)
        * 100
    )

    threshold_report[
        str(threshold)
    ] = {
        "count": count,
        "percentage": pct,
    }

    print(
        f"        <= {threshold:>4.1f} km : "
        f"{count:>6,} "
        f"({pct:>6.3f}%)"
    )


# ------------------------------------------------------------
# Nearest reference source/type
# ------------------------------------------------------------

print(
    "\n      Nearest-reference type:"
)

print(
    result[
        "independent_industrial_reference_type"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# Tight-match subtype distributions
# ------------------------------------------------------------

for threshold in [
    1.0,
    2.0,
    3.0,
    5.0,
]:

    subset = result[
        result[
            "independent_industrial_distance_km"
        ]
        <= threshold
    ]

    print(
        f"\n      <= {threshold:.1f} km "
        f"reference subtype:"
    )

    if len(subset) == 0:

        print("        No observations.")

    else:

        print(
            subset[
                "independent_industrial_reference_subtype"
            ]
            .value_counts()
            .to_string()
        )


# ------------------------------------------------------------
# Cross-check with OSM/temporal evidence
# ------------------------------------------------------------

audit = result.merge(
    obs,
    on="observation_id",
    how="left",
    validate="one_to_one"
)


tight = audit[
    audit[
        "independent_industrial_distance_km"
    ]
    <= 5.0
].copy()


print(
    f"\n      Observations within 5 km: "
    f"{len(tight):,}"
)


for column in [
    "osm_evidence_relation_1_5km",
    "osm_evidence_relation_3km",
    "temporal_evidence_group",
]:

    if column in tight.columns:

        print(
            f"\n      {column}:"
        )

        print(
            tight[column]
            .fillna("MISSING")
            .value_counts()
            .to_string()
        )


# ============================================================
# 8. SAVE
# ============================================================

print("\n[8/8] Saving...")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


references.drop(
    columns=["coordinate_key"]
).to_parquet(
    REFERENCE_FILE,
    index=False
)


result.to_parquet(
    OUTPUT_FILE,
    index=False
)


close = audit[
    audit[
        "independent_industrial_distance_km"
    ]
    <= 10.0
].copy()


close = close.sort_values(
    "independent_industrial_distance_km"
)


close.to_csv(
    CLOSE_FILE,
    index=False
)


report = {
    "stage":
        "combined_independent_industrial_matching",

    "development_observations":
        int(len(result)),

    "oil_gas_references":
        int(len(oil_ref)),

    "thermal_power_references":
        int(len(power_ref)),

    "combined_references":
        int(len(references)),

    "duplicate_coordinate_keys":
        duplicate_coordinates,

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

    "methodology": {
        "distance":
            "Haversine great-circle distance",

        "spatial_index":
            "BallTree",

        "industrial_sources": [
            "GEM Global Oil and Gas Extraction Tracker",
            "WRI Global Power Plant Database v1.3.0",
        ],

        "label_created":
            False,

        "interpretation":
            (
                "Spatial proximity is independent "
                "industrial-context evidence and "
                "does not prove FIRMS anomaly cause."
            ),
    },

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
print("COMBINED INDUSTRIAL MATCHING COMPLETE")
print("=" * 80)

print(
    f"\nIndependent references: "
    f"{len(references):,}"
)

print(
    f"Observations          : "
    f"{len(result):,}"
)

print(
    f"Within 5 km           : "
    f"{len(tight):,}"
)

print(
    f"Within 10 km          : "
    f"{len(close):,}"
)

print("\nEvidence:")
print(OUTPUT_FILE)

print("\nCombined reference:")
print(REFERENCE_FILE)

print("\nClose matches:")
print(CLOSE_FILE)

print("\nReport:")
print(REPORT_FILE)

print(
    "\nNO SOURCE LABELS CREATED."
)

print(
    "2025 HOLDOUT ACCESSED: NO"
)

print(
    "\nNEXT:"
    "\nUse independent industrial + WorldCover evidence"
    "\nto construct defensible source-label candidates."
)