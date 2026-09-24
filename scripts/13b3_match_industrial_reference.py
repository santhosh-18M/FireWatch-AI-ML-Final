from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OBSERVATIONS_FILE = (
    ROOT
    / "data"
    / "labels"
    / "label_evidence_2022_2024.parquet"
)

REFERENCE_FILE = (
    ROOT
    / "data"
    / "labels"
    / "industrial_reference"
    / "india_industrial_reference.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "labels"
    / "industrial_reference"
    / "industrial_reference_matches_2022_2024.parquet"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "labels"
    / "industrial_reference"
    / "industrial_reference_close_matches.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "industrial_reference_matching.json"
)

EARTH_RADIUS_KM = 6371.0088

EXPECTED_ROWS = 20_000


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 13B-3")
print("MATCH INDEPENDENT INDUSTRIAL REFERENCES")
print("=" * 80)

print(
    "\nPurpose:"
    "\nCalculate distance from each of the 20,000"
    "\ndevelopment FIRMS observations to the nearest"
    "\neligible independent oil/gas industrial reference."
)

print(
    "\nImportant:"
    "\n- Distance is evidence, not a final label."
    "\n- Non-match does NOT mean natural."
    "\n- Approximate reference coordinates remain marked."
    "\n- OSM is NOT used for this distance."
    "\n- 2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/7] Loading development observations...")

obs = pd.read_parquet(
    OBSERVATIONS_FILE
)

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

missing = [
    c for c in required_obs
    if c not in obs.columns
]

if missing:
    raise RuntimeError(
        "Missing observation columns:\n"
        + "\n".join(missing)
    )

if obs["observation_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate observation IDs detected."
    )


print("\n[2/7] Loading independent industrial references...")

ref = pd.read_parquet(
    REFERENCE_FILE
)

print(f"      India references: {len(ref):,}")

required_ref = [
    "industrial_reference_id",
    "industrial_reference_name",
    "industrial_category",
    "latitude",
    "longitude",
    "status",
    "location_accuracy",
    "eligible_for_land_matching",
]

missing = [
    c for c in required_ref
    if c not in ref.columns
]

if missing:
    raise RuntimeError(
        "Missing reference columns:\n"
        + "\n".join(missing)
    )


eligible = ref[
    ref["eligible_for_land_matching"]
    .fillna(False)
    .astype(bool)
].copy()

eligible = eligible[
    eligible["latitude"].notna()
    &
    eligible["longitude"].notna()
].copy()

print(
    f"      Eligible references: "
    f"{len(eligible):,}"
)

if len(eligible) == 0:
    raise RuntimeError(
        "No eligible industrial references."
    )


# ============================================================
# 3. BUILD BALLTREE
# ============================================================

print("\n[3/7] Building spatial index...")

reference_radians = np.radians(
    eligible[
        [
            "latitude",
            "longitude",
        ]
    ].to_numpy(
        dtype=float
    )
)

tree = BallTree(
    reference_radians,
    metric="haversine"
)

print("      BallTree ready.")


# ============================================================
# 4. QUERY ALL OBSERVATIONS
# ============================================================

print("\n[4/7] Querying nearest independent facility...")

observation_radians = np.radians(
    obs[
        [
            "latitude",
            "longitude",
        ]
    ].to_numpy(
        dtype=float
    )
)

distance_rad, index = tree.query(
    observation_radians,
    k=1
)

distance_km = (
    distance_rad[:, 0]
    * EARTH_RADIUS_KM
)

nearest_index = index[:, 0]

nearest = (
    eligible.iloc[
        nearest_index
    ]
    .reset_index(drop=True)
)


# ============================================================
# 5. BUILD MATCH TABLE
# ============================================================

print("\n[5/7] Building evidence table...")

result = pd.DataFrame(
    {
        "observation_id":
            obs["observation_id"].astype(str),

        "independent_industrial_distance_km":
            distance_km,

        "independent_industrial_reference_id":
            nearest[
                "industrial_reference_id"
            ].to_numpy(),

        "independent_industrial_reference_name":
            nearest[
                "industrial_reference_name"
            ].to_numpy(),

        "independent_industrial_category":
            nearest[
                "industrial_category"
            ].to_numpy(),

        "independent_industrial_status":
            nearest[
                "status"
            ].to_numpy(),

        "independent_industrial_location_accuracy":
            nearest[
                "location_accuracy"
            ].to_numpy(),

        "independent_industrial_reference_latitude":
            nearest[
                "latitude"
            ].to_numpy(),

        "independent_industrial_reference_longitude":
            nearest[
                "longitude"
            ].to_numpy(),
    }
)


# Evidence bands only.
#
# We intentionally retain several distances rather than
# immediately deciding one universal "industrial" threshold.

for threshold in [
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    25.0,
]:

    name = str(
        threshold
    ).replace(
        ".",
        "_"
    )

    result[
        f"independent_industrial_within_{name}km"
    ] = (
        result[
            "independent_industrial_distance_km"
        ]
        <= threshold
    )


result[
    "independent_industrial_reference_source"
] = "Global Energy Monitor"


result[
    "independent_industrial_reference_dataset"
] = (
    "Global Oil and Gas Extraction Tracker"
)


# ============================================================
# 6. AUDIT
# ============================================================

print("\n[6/7] Auditing distances...")


distance_series = result[
    "independent_industrial_distance_km"
]


summary = {
    "mean":
        float(
            distance_series.mean()
        ),

    "median":
        float(
            distance_series.median()
        ),

    "p25":
        float(
            distance_series.quantile(0.25)
        ),

    "p75":
        float(
            distance_series.quantile(0.75)
        ),

    "p90":
        float(
            distance_series.quantile(0.90)
        ),

    "p95":
        float(
            distance_series.quantile(0.95)
        ),

    "p99":
        float(
            distance_series.quantile(0.99)
        ),

    "min":
        float(
            distance_series.min()
        ),

    "max":
        float(
            distance_series.max()
        ),
}


print(
    f"      Minimum distance : "
    f"{summary['min']:.3f} km"
)

print(
    f"      Median distance  : "
    f"{summary['median']:.3f} km"
)

print(
    f"      Mean distance    : "
    f"{summary['mean']:.3f} km"
)


threshold_counts = {}

print("\n      Independent facility proximity:")

for threshold in [
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    25.0,
]:

    column = (
        "independent_industrial_within_"
        + str(threshold).replace(".", "_")
        + "km"
    )

    count = int(
        result[
            column
        ].sum()
    )

    percentage = (
        count
        / len(result)
        * 100
    )

    threshold_counts[
        str(threshold)
    ] = {
        "count":
            count,

        "percentage":
            percentage,
    }

    print(
        f"        <= {threshold:>4.1f} km : "
        f"{count:>6,} "
        f"({percentage:>6.3f}%)"
    )


# ============================================================
# CROSS-CHECK AGAINST EXISTING EVIDENCE
# ============================================================

audit = result.merge(
    obs,
    on="observation_id",
    how="left",
    validate="one_to_one",
)


print(
    "\n      Closest observations:"
)

display_columns = [
    "observation_id",
    "independent_industrial_distance_km",
    "independent_industrial_reference_name",
    "independent_industrial_location_accuracy",
]


# Add useful existing evidence when present.
for candidate in [
    "osm_evidence_relation_1_5km",
    "osm_evidence_relation_3km",
    "temporal_evidence_group",
    "satellite_evidence_available",
]:

    if candidate in audit.columns:
        display_columns.append(
            candidate
        )


closest = (
    audit[
        display_columns
    ]
    .sort_values(
        "independent_industrial_distance_km"
    )
    .head(30)
)


print(
    closest.to_string(
        index=False
    )
)


# ============================================================
# 7. SAVE
# ============================================================

print("\n[7/7] Saving evidence...")


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


result.to_parquet(
    OUTPUT_FILE,
    index=False
)


# Save <= 25 km observations for manual inspection.
close_matches = audit[
    audit[
        "independent_industrial_distance_km"
    ]
    <= 25.0
].copy()


close_matches = (
    close_matches
    .sort_values(
        "independent_industrial_distance_km"
    )
)


close_matches.to_csv(
    OUTPUT_CSV,
    index=False
)


reference_match_counts = (
    result[
        "independent_industrial_reference_name"
    ]
    .value_counts()
    .to_dict()
)


report = {
    "stage":
        "independent_industrial_spatial_matching",

    "development_observations":
        int(len(result)),

    "eligible_reference_facilities":
        int(len(eligible)),

    "reference_source":
        "Global Energy Monitor",

    "reference_dataset":
        "Global Oil and Gas Extraction Tracker",

    "distance_summary_km":
        summary,

    "threshold_counts":
        threshold_counts,

    "nearest_reference_assignment_counts": {
        str(k): int(v)
        for k, v
        in reference_match_counts.items()
    },

    "methodology": {
        "distance":
            "Haversine great-circle distance",

        "spatial_index":
            "BallTree",

        "distance_role":
            (
                "Independent positive industrial "
                "reference evidence only."
            ),

        "non_match_interpretation":
            (
                "No match does not imply natural "
                "or non-industrial."
            ),
    },

    "important_limitations": [
        (
            "Only six eligible onshore Indian oil/gas "
            "references are available in this source."
        ),
        (
            "This source does not cover all industrial "
            "thermal facilities in India."
        ),
        (
            "Approximate facility coordinates have "
            "greater positional uncertainty."
        ),
        (
            "Distance to a facility does not itself "
            "prove that a FIRMS observation was caused "
            "by that facility."
        ),
    ],

    "labels_created":
        False,

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
) as file:

    json.dump(
        report,
        file,
        indent=2
    )


print("\n" + "=" * 80)
print("INDEPENDENT INDUSTRIAL MATCHING COMPLETE")
print("=" * 80)

print(
    f"\nObservations       : "
    f"{len(result):,}"
)

print(
    f"Reference facilities: "
    f"{len(eligible):,}"
)

print(
    f"Within 25 km       : "
    f"{len(close_matches):,}"
)

print("\nEvidence:")
print(OUTPUT_FILE)

print("\nClose-match review:")
print(OUTPUT_CSV)

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
    "\nDecide whether this independent industrial"
    "\nevidence has enough positive coverage or whether"
    "\nan additional industrial reference source is needed."
)