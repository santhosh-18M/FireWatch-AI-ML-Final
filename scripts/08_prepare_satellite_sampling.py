from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "firms_with_osm_2022_2024.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "satellite"
    / "sentinel_sampling_2022_2024.parquet"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "satellite"
    / "sentinel_sampling_2022_2024.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "satellite_sampling_summary.json"
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
# SAMPLING SETTINGS
# ============================================================

TARGET_SAMPLE_SIZE = 20_000

RANDOM_STATE = 42

# Geographic grouping.
# 0.25 degrees is roughly tens of km and is used only
# for diversity sampling, NOT as a model feature.
GRID_SIZE_DEG = 0.25


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
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


# ============================================================
# HELPERS
# ============================================================

def safe_sample(frame, n, seed):
    """
    Reproducible sampling without replacement.
    """

    if len(frame) <= n:
        return frame.copy()

    return frame.sample(
        n=n,
        random_state=seed
    ).copy()


def count_dict(series):
    """
    JSON-friendly value counts.
    """

    return {
        str(k): int(v)
        for k, v in
        series.value_counts(
            dropna=False
        ).to_dict().items()
    }


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — SENTINEL SAMPLING PREPARATION")
print("=" * 80)

print(
    "\nPurpose:"
    "\nCreate a geographically and contextually diverse"
    "\n2022-2024 FIRMS subset for scalable satellite"
    "\nevidence extraction."
)

print(
    "\nThis stage:"
    "\n- does NOT download Sentinel imagery"
    "\n- does NOT create fire labels"
    "\n- does NOT train a model"
    "\n- does NOT access 2025"
)

start = time.time()


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n[1/9] Loading development evidence dataset..."
)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing input:\n{INPUT_FILE}"
    )

df = pd.read_parquet(
    INPUT_FILE
)

print(
    f"      Rows: {len(df):,}"
)

print(
    f"      Columns: {len(df.columns):,}"
)


# ============================================================
# VALIDATE SCHEMA
# ============================================================

print(
    "\n[2/9] Validating required columns..."
)

missing = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


if df["observation_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate observation_id detected."
    )


if not df["latitude"].between(
    -90, 90
).all():

    raise RuntimeError(
        "Invalid latitude detected."
    )


if not df["longitude"].between(
    -180, 180
).all():

    raise RuntimeError(
        "Invalid longitude detected."
    )


print(
    "      Schema validation passed."
)


# ============================================================
# TIME FEATURES
# ============================================================

print(
    "\n[3/9] Building sampling metadata..."
)

df["acquired_at"] = pd.to_datetime(
    df["acquired_at"]
)

df["sample_year"] = (
    df["acquired_at"].dt.year
)

df["sample_month"] = (
    df["acquired_at"].dt.month
)


years = sorted(
    df["sample_year"]
    .dropna()
    .unique()
    .tolist()
)

print(
    f"      Years: {years}"
)


# ============================================================
# THERMAL STRATA
# ============================================================

# Use quantiles only for sampling diversity.
# These are NOT labels and NOT production thresholds.

frp_q50 = float(
    df["frp"].quantile(0.50)
)

frp_q90 = float(
    df["frp"].quantile(0.90)
)

frp_q95 = float(
    df["frp"].quantile(0.95)
)

frp_q99 = float(
    df["frp"].quantile(0.99)
)


df["sample_frp_group"] = pd.cut(
    df["frp"],
    bins=[
        -np.inf,
        frp_q50,
        frp_q90,
        frp_q95,
        frp_q99,
        np.inf
    ],
    labels=[
        "low",
        "moderate",
        "high",
        "very_high",
        "extreme"
    ],
    include_lowest=True
)


print(
    "\n      FRP sampling thresholds:"
)

print(
    f"        q50 = {frp_q50:.3f}"
)

print(
    f"        q90 = {frp_q90:.3f}"
)

print(
    f"        q95 = {frp_q95:.3f}"
)

print(
    f"        q99 = {frp_q99:.3f}"
)


# ============================================================
# TEMPORAL STRATA
# ============================================================

# Again: these are sampling categories only.
# They are not final temporal labels.

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
    )
]


choices = [
    "no_prior",
    "limited_prior",
    "repeated_low_active_days",
    "high_recurrence"
]


df["sample_temporal_group"] = np.select(
    conditions,
    choices,
    default="other"
)


# ============================================================
# CONTEXT STRATA
# ============================================================

# Use the previously audited OSM relation.
# It is context evidence, not a source label.

valid_contexts = {
    "strong_industrial_context",
    "vegetation_context",
    "mixed_context",
    "no_close_context"
}


unexpected_contexts = (
    set(
        df[
            "osm_context_relation"
        ].dropna().unique()
    )
    -
    valid_contexts
)


if unexpected_contexts:

    raise RuntimeError(
        "Unexpected OSM context values: "
        f"{unexpected_contexts}"
    )


df["sample_context_group"] = (
    df["osm_context_relation"]
)


# ============================================================
# GEOGRAPHIC CELLS
# ============================================================

print(
    "\n[4/9] Creating geographic diversity cells..."
)

df["sample_grid_lat"] = np.floor(
    df["latitude"]
    / GRID_SIZE_DEG
).astype(np.int32)


df["sample_grid_lon"] = np.floor(
    df["longitude"]
    / GRID_SIZE_DEG
).astype(np.int32)


df["sample_grid_id"] = (
    df["sample_grid_lat"].astype(str)
    + "_"
    + df["sample_grid_lon"].astype(str)
)


grid_count = int(
    df["sample_grid_id"].nunique()
)

print(
    f"      Geographic cells: "
    f"{grid_count:,}"
)


# ============================================================
# CREATE STRATUM
# ============================================================

print(
    "\n[5/9] Creating multidimensional sampling strata..."
)

df["sample_stratum"] = (
    df["sample_year"].astype(str)
    + "|"
    + df["sample_context_group"].astype(str)
    + "|"
    + df["sample_temporal_group"].astype(str)
    + "|"
    + df["sample_frp_group"].astype(str)
)


stratum_counts = (
    df["sample_stratum"]
    .value_counts()
)

print(
    f"      Non-empty strata: "
    f"{len(stratum_counts):,}"
)


# ============================================================
# STAGE A — STRATIFIED BASE SAMPLE
# ============================================================

print(
    "\n[6/9] Building balanced evidence sample..."
)

# We first allocate a reasonable minimum across all
# existing evidence strata.
#
# Then remaining capacity is filled geographically.

n_strata = len(
    stratum_counts
)

base_per_stratum = max(
    20,
    int(
        TARGET_SAMPLE_SIZE
        * 0.65
        / n_strata
    )
)


print(
    f"      Base allocation per stratum: "
    f"{base_per_stratum:,}"
)


parts = []

for i, (
    stratum,
    group
) in enumerate(
    df.groupby(
        "sample_stratum",
        observed=True
    )
):

    n = min(
        base_per_stratum,
        len(group)
    )

    sampled = safe_sample(
        group,
        n,
        RANDOM_STATE + i
    )

    sampled[
        "sampling_reason"
    ] = "evidence_stratum"

    parts.append(
        sampled
    )


base_sample = pd.concat(
    parts,
    ignore_index=True
)


# Remove accidental duplicates just in case.

base_sample = (
    base_sample
    .drop_duplicates(
        subset=[
            "observation_id"
        ]
    )
    .reset_index(drop=True)
)


print(
    f"      Base sample: "
    f"{len(base_sample):,}"
)


# ============================================================
# STAGE B — GEOGRAPHIC COVERAGE
# ============================================================

print(
    "\n[7/9] Improving geographic coverage..."
)

remaining_capacity = (
    TARGET_SAMPLE_SIZE
    -
    len(base_sample)
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

    # Shuffle deterministically before selecting from cells.
    remaining = remaining.sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    # First pass:
    # try to select at least one additional point
    # from as many geographic cells as possible.

    one_per_cell = (
        remaining
        .groupby(
            "sample_grid_id",
            group_keys=False
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
                random_state=RANDOM_STATE
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


# ============================================================
# STAGE C — FILL REMAINING CAPACITY
# ============================================================

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


    if len(fill_pool):

        fill_n = min(
            remaining_capacity,
            len(fill_pool)
        )

        fill = fill_pool.sample(
            n=fill_n,
            random_state=RANDOM_STATE + 999
        ).copy()

        fill[
            "sampling_reason"
        ] = "diversity_fill"

        geo_parts.append(
            fill
        )


# ============================================================
# COMBINE SAMPLE
# ============================================================

all_parts = [
    base_sample
]

all_parts.extend(
    geo_parts
)


sample = pd.concat(
    all_parts,
    ignore_index=True
)


sample = (
    sample
    .drop_duplicates(
        subset=[
            "observation_id"
        ]
    )
    .reset_index(drop=True)
)


# If for any reason we exceeded target,
# deterministically reduce only the geographic/fill portion
# while keeping the evidence-stratum base.

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
        -
        len(mandatory)
    )

    optional = safe_sample(
        optional,
        optional_n,
        RANDOM_STATE + 2000
    )

    sample = pd.concat(
        [
            mandatory,
            optional
        ],
        ignore_index=True
    )


# ============================================================
# VALIDATE SAMPLE
# ============================================================

print(
    "\n[8/9] Validating satellite sample..."
)

if sample[
    "observation_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate observations in satellite sample."
    )


if len(sample) > TARGET_SAMPLE_SIZE:

    raise RuntimeError(
        "Satellite sample exceeds target size."
    )


if not set(
    sample[
        "sample_year"
    ].unique()
) == set(years):

    raise RuntimeError(
        "Not all development years are represented."
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
        "Missing context groups from sample: "
        f"{missing_contexts}"
    )


sample_grid_count = int(
    sample[
        "sample_grid_id"
    ].nunique()
)


print(
    f"      Final sample: "
    f"{len(sample):,}"
)

print(
    f"      Geographic cells represented: "
    f"{sample_grid_count:,}"
)

print(
    f"      Context groups represented: "
    f"{len(sample_contexts)}"
)


# ============================================================
# SATELLITE REQUEST METADATA
# ============================================================

# These columns will be consumed by the next stage.
#
# We use past-only imagery to avoid future leakage.

sample[
    "sentinel_window_start"
] = (
    sample[
        "acquired_at"
    ]
    -
    pd.Timedelta(
        days=14
    )
)


sample[
    "sentinel_window_end"
] = (
    sample[
        "acquired_at"
    ]
)


sample[
    "sentinel_search_days"
] = 14


sample[
    "satellite_extraction_status"
] = "pending"


sample[
    "satellite_available"
] = np.nan


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [

    # Identity
    "observation_id",

    # Event location/time
    "acquired_at",
    "latitude",
    "longitude",

    # FIRMS evidence
    "brightness",
    "bright_t31",
    "frp",
    "scan",
    "track",
    "confidence",
    "daynight",
    "type",

    # Temporal evidence
    "detections_3d",
    "detections_7d",
    "detections_10d",
    "detections_30d",
    "distinct_active_days_30d",

    # OSM evidence
    "distance_to_strong_industrial_km",
    "nearest_strong_industrial_category",

    "distance_to_supporting_industrial_km",
    "nearest_supporting_industrial_category",

    "distance_to_infrastructure_km",
    "nearest_infrastructure_category",

    "distance_to_vegetation_km",
    "nearest_vegetation_category",

    "osm_context_relation",

    # Sampling metadata
    "sample_year",
    "sample_month",
    "sample_context_group",
    "sample_temporal_group",
    "sample_frp_group",
    "sample_grid_id",
    "sample_stratum",
    "sampling_reason",

    # Sentinel request metadata
    "sentinel_window_start",
    "sentinel_window_end",
    "sentinel_search_days",
    "satellite_extraction_status",
    "satellite_available",
]


sample = sample[
    output_columns
].sort_values(
    [
        "sample_year",
        "acquired_at",
        "observation_id"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

print(
    "\n[9/9] Saving satellite sampling dataset..."
)

sample.to_parquet(
    OUTPUT_FILE,
    index=False
)

sample.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report = {

    "source_dataset_rows":
        int(len(df)),

    "target_sample_size":
        int(TARGET_SAMPLE_SIZE),

    "actual_sample_size":
        int(len(sample)),

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
        )
        if grid_count
        else 0.0,

    "frp_sampling_thresholds": {
        "q50":
            frp_q50,

        "q90":
            frp_q90,

        "q95":
            frp_q95,

        "q99":
            frp_q99
    },

    "sample_year_distribution":
        count_dict(
            sample[
                "sample_year"
            ]
        ),

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

    "sentinel_policy": {

        "lookback_days":
            14,

        "future_imagery_allowed":
            False,

        "reason":
            (
                "Prevent temporal leakage by using "
                "only Sentinel observations acquired "
                "at or before the FIRMS timestamp."
            )
    },

    "methodology_notes": [

        (
            "This is a satellite evidence sampling "
            "dataset, not a training dataset."
        ),

        (
            "Sampling intentionally covers multiple "
            "years, OSM contexts, temporal behaviors, "
            "thermal intensities and geographic cells."
        ),

        (
            "Sampling groups are not source labels."
        ),

        (
            "Latitude/longitude and grid IDs are "
            "sampling/GIS metadata and are not "
            "automatically model predictors."
        ),

        (
            "Sentinel imagery will use a past-only "
            "14-day search window."
        ),

        (
            "Missing satellite observations will be "
            "represented explicitly rather than "
            "replaced with fabricated values."
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
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "SATELLITE SAMPLING PREPARATION COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\nSource observations: "
    f"{len(df):,}"
)

print(
    f"Satellite sample: "
    f"{len(sample):,}"
)


print(
    "\nYEAR DISTRIBUTION"
)

print(
    sample[
        "sample_year"
    ]
    .value_counts()
    .sort_index()
    .to_string()
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
    "\nSentinel policy:"
    "\n  Search window = previous 14 days"
    "\n  Future imagery = NOT allowed"
    "\n  Missing imagery = explicit missing value"
)


print(
    f"\nParquet:\n{OUTPUT_FILE}"
)

print(
    f"\nCSV:\n{OUTPUT_CSV}"
)

print(
    f"\nReport:\n{REPORT_FILE}"
)

print(
    f"\nRuntime: "
    f"{time.time() - start:.2f} seconds"
)

print(
    "\nIMPORTANT:"
    "\nNo Sentinel API requests were made."
    "\nNo labels were created."
    "\n2025 holdout remains untouched."
)