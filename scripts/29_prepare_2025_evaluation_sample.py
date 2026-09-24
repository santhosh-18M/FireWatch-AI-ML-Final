from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "firms_with_osm_2025.parquet"
)

# Development report containing the ORIGINAL FRP
# sampling thresholds. We reuse them instead of
# calculating new thresholds from 2025.
DEVELOPMENT_SAMPLING_REPORT = (
    ROOT
    / "reports"
    / "experiments"
    / "satellite_sampling_summary.json"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "evaluation_sample_2025.parquet"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "evaluation_sample_2025.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "evaluation_sample_2025_summary.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FROZEN SAMPLING SETTINGS
# ============================================================

TARGET_SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
GRID_SIZE_DEG = 0.25


# ============================================================
# HELPERS
# ============================================================

def safe_sample(frame, n, seed):

    if len(frame) <= n:
        return frame.copy()

    return frame.sample(
        n=n,
        random_state=seed,
        replace=False,
    ).copy()


def count_dict(series):

    return {
        str(k): int(v)
        for k, v in (
            series
            .value_counts(dropna=False)
            .to_dict()
            .items()
        )
    }


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 29")
print("PREPARE 2025 FINAL EVALUATION SAMPLE")
print("=" * 80)

print(
    "\nPurpose:"
    "\n- Select 20,000 representative 2025 observations."
    "\n- Reuse frozen development sampling principles."
    "\n- Reuse development FRP thresholds."
    "\n- Preserve geographic/context/temporal diversity."
    "\n- Create NO source labels."
    "\n- Execute NO model."
)

start = time.time()


# ============================================================
# LOAD FROZEN DEVELOPMENT THRESHOLDS
# ============================================================

print(
    "\n[1/9] Loading frozen development sampling thresholds..."
)

if not DEVELOPMENT_SAMPLING_REPORT.exists():

    raise FileNotFoundError(
        "Development sampling report missing:\n"
        f"{DEVELOPMENT_SAMPLING_REPORT}"
    )


with open(
    DEVELOPMENT_SAMPLING_REPORT,
    "r",
    encoding="utf-8",
) as f:

    development_report = json.load(f)


thresholds = (
    development_report[
        "frp_sampling_thresholds"
    ]
)

frp_q50 = float(
    thresholds["q50"]
)

frp_q90 = float(
    thresholds["q90"]
)

frp_q95 = float(
    thresholds["q95"]
)

frp_q99 = float(
    thresholds["q99"]
)


print(
    f"      q50 = {frp_q50:.3f}"
)

print(
    f"      q90 = {frp_q90:.3f}"
)

print(
    f"      q95 = {frp_q95:.3f}"
)

print(
    f"      q99 = {frp_q99:.3f}"
)


# ============================================================
# LOAD 2025
# ============================================================

print(
    "\n[2/9] Loading 2025 FIRMS + temporal + OSM..."
)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input missing:\n{INPUT_FILE}"
    )


df = pd.read_parquet(
    INPUT_FILE
)


print(
    f"      Rows: {len(df):,}"
)


if len(df) != 646_376:

    raise RuntimeError(
        "Unexpected 2025 source row count."
    )


required_columns = [
    "observation_id",
    "acquired_at",
    "latitude",
    "longitude",

    "brightness",
    "bright_t31",
    "frp",
    "scan",
    "track",
    "confidence",
    "daynight",
    "type",

    "detections_3d",
    "detections_7d",
    "detections_10d",
    "detections_30d",
    "distinct_active_days_30d",

    "distance_to_strong_industrial_km",
    "nearest_strong_industrial_category",

    "distance_to_supporting_industrial_km",
    "nearest_supporting_industrial_category",

    "distance_to_infrastructure_km",
    "nearest_infrastructure_category",

    "distance_to_vegetation_km",
    "nearest_vegetation_category",

    "osm_context_relation",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


if df[
    "observation_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate observation IDs."
    )


# ============================================================
# TIME METADATA
# ============================================================

print(
    "\n[3/9] Building frozen sampling metadata..."
)

df["acquired_at"] = pd.to_datetime(
    df["acquired_at"],
    errors="raise",
)

df["sample_year"] = (
    df["acquired_at"]
    .dt.year
)

df["sample_month"] = (
    df["acquired_at"]
    .dt.month
)


if not (
    df["sample_year"] == 2025
).all():

    raise RuntimeError(
        "Non-2025 observation detected."
    )


# ============================================================
# FROZEN FRP STRATA
# ============================================================

print(
    "\n[4/9] Applying frozen FRP strata..."
)

df["sample_frp_group"] = pd.cut(
    df["frp"],
    bins=[
        -np.inf,
        frp_q50,
        frp_q90,
        frp_q95,
        frp_q99,
        np.inf,
    ],
    labels=[
        "low",
        "moderate",
        "high",
        "very_high",
        "extreme",
    ],
    include_lowest=True,
)


# ============================================================
# FROZEN TEMPORAL STRATA
# ============================================================

conditions = [

    (
        df["detections_30d"] == 0
    ),

    (
        (df["detections_30d"] >= 1)
        &
        (df["detections_30d"] <= 2)
    ),

    (
        (df["detections_30d"] >= 3)
        &
        (
            df[
                "distinct_active_days_30d"
            ] < 5
        )
    ),

    (
        (df["detections_30d"] >= 3)
        &
        (
            df[
                "distinct_active_days_30d"
            ] >= 5
        )
    ),
]


choices = [
    "no_prior",
    "limited_prior",
    "repeated_low_active_days",
    "high_recurrence",
]


df[
    "sample_temporal_group"
] = np.select(
    conditions,
    choices,
    default="other",
)


# ============================================================
# OSM CONTEXT
# ============================================================

valid_contexts = {
    "strong_industrial_context",
    "vegetation_context",
    "mixed_context",
    "no_close_context",
}


unexpected = (
    set(
        df[
            "osm_context_relation"
        ]
        .dropna()
        .unique()
    )
    -
    valid_contexts
)


if unexpected:

    raise RuntimeError(
        "Unexpected OSM context values: "
        f"{unexpected}"
    )


df[
    "sample_context_group"
] = df[
    "osm_context_relation"
]


# ============================================================
# GEOGRAPHIC CELLS
# ============================================================

print(
    "\n[5/9] Creating frozen 0.25-degree geographic cells..."
)

df["sample_grid_lat"] = np.floor(
    df["latitude"]
    / GRID_SIZE_DEG
).astype(
    np.int32
)


df["sample_grid_lon"] = np.floor(
    df["longitude"]
    / GRID_SIZE_DEG
).astype(
    np.int32
)


df["sample_grid_id"] = (
    df[
        "sample_grid_lat"
    ].astype(str)
    + "_"
    + df[
        "sample_grid_lon"
    ].astype(str)
)


grid_count = int(
    df[
        "sample_grid_id"
    ].nunique()
)


print(
    f"      2025 geographic cells: "
    f"{grid_count:,}"
)


# ============================================================
# STRATUM
# ============================================================

print(
    "\n[6/9] Creating holdout sampling strata..."
)

# Development included year here because there were
# three years. 2025 is constant, so adding "2025|"
# would not change grouping. We omit the redundant
# constant dimension.

df["sample_stratum"] = (
    df[
        "sample_context_group"
    ].astype(str)
    + "|"
    + df[
        "sample_temporal_group"
    ].astype(str)
    + "|"
    + df[
        "sample_frp_group"
    ].astype(str)
)


stratum_counts = (
    df[
        "sample_stratum"
    ]
    .value_counts()
)


n_strata = len(
    stratum_counts
)


print(
    f"      Non-empty strata: "
    f"{n_strata:,}"
)


# ============================================================
# STAGE A — SAME BASE ALLOCATION FORMULA
# ============================================================

print(
    "\n[7/9] Building stratified base sample..."
)


base_per_stratum = max(
    20,
    int(
        TARGET_SAMPLE_SIZE
        * 0.65
        / n_strata
    ),
)


print(
    f"      Base allocation per stratum: "
    f"{base_per_stratum:,}"
)


parts = []


for i, (
    stratum,
    group,
) in enumerate(
    df.groupby(
        "sample_stratum",
        observed=True,
        sort=True,
    )
):

    n = min(
        base_per_stratum,
        len(group),
    )

    sampled = safe_sample(
        group,
        n,
        RANDOM_STATE + i,
    )

    sampled[
        "sampling_reason"
    ] = "evidence_stratum"

    parts.append(
        sampled
    )


base_sample = pd.concat(
    parts,
    ignore_index=True,
)


base_sample = (
    base_sample
    .drop_duplicates(
        subset=[
            "observation_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    f"      Base sample: "
    f"{len(base_sample):,}"
)


# ============================================================
# STAGE B — GEOGRAPHIC COVERAGE
# ============================================================

print(
    "\n[8/9] Adding geographic coverage and diversity fill..."
)


remaining_capacity = (
    TARGET_SAMPLE_SIZE
    - len(base_sample)
)


if remaining_capacity < 0:

    raise RuntimeError(
        "Base sample exceeds 20,000 rows."
    )


selected_ids = set(
    base_sample[
        "observation_id"
    ].tolist()
)


remaining = df[
    ~df[
        "observation_id"
    ].isin(
        selected_ids
    )
].copy()


geo_parts = []


if remaining_capacity > 0:

    remaining = remaining.sample(
        frac=1,
        random_state=RANDOM_STATE,
    )


    one_per_cell = (
        remaining
        .groupby(
            "sample_grid_id",
            group_keys=False,
            sort=True,
        )
        .head(1)
    )


    if len(
        one_per_cell
    ) > remaining_capacity:

        one_per_cell = (
            one_per_cell
            .sample(
                n=remaining_capacity,
                random_state=RANDOM_STATE,
            )
        )


    one_per_cell = (
        one_per_cell.copy()
    )


    one_per_cell[
        "sampling_reason"
    ] = "geographic_coverage"


    geo_parts.append(
        one_per_cell
    )


    remaining_capacity -= len(
        one_per_cell
    )


# ------------------------------------------------------------
# Fill remaining capacity
# ------------------------------------------------------------

if remaining_capacity > 0:

    used_geo_ids = set()

    for part in geo_parts:

        used_geo_ids.update(
            part[
                "observation_id"
            ].tolist()
        )


    fill_pool = remaining[
        ~remaining[
            "observation_id"
        ].isin(
            used_geo_ids
        )
    ]


    fill_n = min(
        remaining_capacity,
        len(fill_pool),
    )


    if fill_n > 0:

        fill = fill_pool.sample(
            n=fill_n,
            random_state=(
                RANDOM_STATE
                + 999
            ),
        ).copy()


        fill[
            "sampling_reason"
        ] = "diversity_fill"


        geo_parts.append(
            fill
        )


# ============================================================
# COMBINE
# ============================================================

all_parts = [
    base_sample
]

all_parts.extend(
    geo_parts
)


sample = pd.concat(
    all_parts,
    ignore_index=True,
)


sample = (
    sample
    .drop_duplicates(
        subset=[
            "observation_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


if len(sample) > TARGET_SAMPLE_SIZE:

    base_ids = set(
        base_sample[
            "observation_id"
        ].tolist()
    )


    mandatory = sample[
        sample[
            "observation_id"
        ].isin(
            base_ids
        )
    ]


    optional = sample[
        ~sample[
            "observation_id"
        ].isin(
            base_ids
        )
    ]


    optional_n = (
        TARGET_SAMPLE_SIZE
        - len(mandatory)
    )


    optional = safe_sample(
        optional,
        optional_n,
        RANDOM_STATE + 2000,
    )


    sample = pd.concat(
        [
            mandatory,
            optional,
        ],
        ignore_index=True,
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\n[9/9] Validating and saving..."
)


if len(sample) != TARGET_SAMPLE_SIZE:

    raise RuntimeError(
        "Final evaluation sample is not exactly "
        f"{TARGET_SAMPLE_SIZE:,} rows. "
        f"Found {len(sample):,}."
    )


if sample[
    "observation_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate observations in final sample."
    )


sample_contexts = set(
    sample[
        "sample_context_group"
    ].unique()
)


missing_contexts = (
    valid_contexts
    -
    sample_contexts
)


if missing_contexts:

    raise RuntimeError(
        "Missing OSM context groups: "
        f"{missing_contexts}"
    )


sample_grid_count = int(
    sample[
        "sample_grid_id"
    ].nunique()
)


# ============================================================
# WORLDCOVER REQUEST METADATA
# ============================================================

sample[
    "worldcover_extraction_status"
] = "pending"


sample[
    "worldcover_source"
] = "ESA_WorldCover_2021_v200"


# ============================================================
# SAVE
# ============================================================

sample = (
    sample
    .sort_values(
        [
            "acquired_at",
            "observation_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


sample.to_parquet(
    OUTPUT_FILE,
    index=False,
)


sample.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

report = {

    "stage":
        "2025_final_evaluation_sampling",

    "source_rows":
        int(
            len(df)
        ),

    "target_sample_size":
        TARGET_SAMPLE_SIZE,

    "actual_sample_size":
        int(
            len(sample)
        ),

    "random_state":
        RANDOM_STATE,

    "grid_size_degrees":
        GRID_SIZE_DEG,

    "source_geographic_cells":
        grid_count,

    "sample_geographic_cells":
        sample_grid_count,

    "geographic_cell_coverage_percentage":
        float(
            sample_grid_count
            / grid_count
            * 100
        ),

    "frp_threshold_source":
        str(
            DEVELOPMENT_SAMPLING_REPORT
        ),

    "frp_sampling_thresholds":
        {
            "q50":
                frp_q50,

            "q90":
                frp_q90,

            "q95":
                frp_q95,

            "q99":
                frp_q99,
        },

    "sample_context_distribution":
        count_dict(
            sample[
                "sample_context_group"
            ]
        ),

    "sample_temporal_distribution":
        count_dict(
            sample[
                "sample_temporal_group"
            ]
        ),

    "sample_frp_distribution":
        count_dict(
            sample[
                "sample_frp_group"
            ]
        ),

    "sampling_reason_distribution":
        count_dict(
            sample[
                "sampling_reason"
            ]
        ),

    "model_used":
        False,

    "labels_generated":
        False,

    "sampling_policy_changed":
        False,

    "notes": [
        (
            "Development FRP thresholds were reused; "
            "2025 quantiles were not used to redefine "
            "sampling strata."
        ),
        (
            "The development sampler included year "
            "as a stratum dimension because it covered "
            "2022-2024. Year is constant for the 2025 "
            "holdout and is therefore redundant."
        ),
        (
            "OSM context is used only for sampling "
            "diversity here, not for label generation."
        ),
        (
            "The frozen source classifier has not "
            "been executed."
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
# FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 29 COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\nSource observations: "
    f"{len(df):,}"
)


print(
    f"Evaluation sample: "
    f"{len(sample):,}"
)


print(
    "\nCONTEXT DISTRIBUTION"
)

print(
    sample[
        "sample_context_group"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nTEMPORAL DISTRIBUTION"
)

print(
    sample[
        "sample_temporal_group"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nFRP DISTRIBUTION"
)

print(
    sample[
        "sample_frp_group"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nSAMPLING REASON"
)

print(
    sample[
        "sampling_reason"
    ]
    .value_counts()
    .to_string()
)


print(
    f"\nGeographic cells in source: "
    f"{grid_count:,}"
)


print(
    f"Geographic cells represented: "
    f"{sample_grid_count:,}"
)


print(
    f"Geographic coverage: "
    f"{sample_grid_count / grid_count * 100:.2f}%"
)


print(
    "\nFrozen development FRP thresholds:"
)

print(
    f"  q50 = {frp_q50:.3f}"
)

print(
    f"  q90 = {frp_q90:.3f}"
)

print(
    f"  q95 = {frp_q95:.3f}"
)

print(
    f"  q99 = {frp_q99:.3f}"
)


print(
    f"\nSaved:\n{OUTPUT_FILE}"
)


print(
    "\nIMPORTANT:"
    "\n- No source labels generated."
    "\n- No model predictions generated."
    "\n- Next step is independent WorldCover extraction."
)