from pathlib import Path
from collections import defaultdict, deque
import json
import math
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

HISTORY_FILE = (
    ROOT
    / "data"
    / "processed"
    / "firms_master_2022_2024.parquet"
)

HOLDOUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "firms_master_2025.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "firms_with_temporal_2025.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "temporal_features_2025_summary.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


RADIUS_KM = 1.0
MAX_HISTORY_DAYS = 30
CELL_SIZE_DEG = 0.01
EARTH_RADIUS_KM = 6371.0088

NS_PER_HOUR = 3_600_000_000_000
NS_PER_DAY = 86_400_000_000_000

WINDOW_3D = 3 * NS_PER_DAY
WINDOW_7D = 7 * NS_PER_DAY
WINDOW_10D = 10 * NS_PER_DAY
WINDOW_30D = 30 * NS_PER_DAY


# ============================================================
# HELPERS — SAME AS DEVELOPMENT PIPELINE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(
            np.sqrt(a)
        )
    )


def spatial_cell(
    lat,
    lon
):
    return (
        int(
            math.floor(
                lat / CELL_SIZE_DEG
            )
        ),
        int(
            math.floor(
                lon / CELL_SIZE_DEG
            )
        ),
    )


def neighboring_cells(
    cell
):
    x, y = cell

    # Exact same ±2-cell search used in development.
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            yield (
                x + dx,
                y + dy
            )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 27")
print("BUILD CAUSAL 2025 TEMPORAL FEATURES")
print("=" * 80)

print(
    "\nMethod:"
    "\n- Same 1 km radius as development."
    "\n- Same rolling 30-day history."
    "\n- Only STRICTLY EARLIER timestamps."
    "\n- Current observation excluded."
    "\n- Future observations excluded."
    "\n- December 2024 supplies boundary history."
    "\n- Final output contains ONLY 2025 rows."
)


# ============================================================
# LOAD 2025
# ============================================================

print(
    "\n[1/9] Loading 2025 holdout..."
)

holdout = pd.read_parquet(
    HOLDOUT_FILE
)

holdout["acquired_at"] = pd.to_datetime(
    holdout["acquired_at"],
    errors="raise"
)

holdout["acq_date"] = pd.to_datetime(
    holdout["acq_date"],
    errors="raise"
)

if not (
    holdout["year"] == 2025
).all():

    raise RuntimeError(
        "Holdout contains non-2025 rows."
    )

print(
    f"      2025 rows: "
    f"{len(holdout):,}"
)


# ============================================================
# LOAD ONLY NECESSARY 2024 HISTORY
# ============================================================

print(
    "\n[2/9] Loading development history..."
)

history = pd.read_parquet(
    HISTORY_FILE
)

history["acquired_at"] = pd.to_datetime(
    history["acquired_at"],
    errors="raise"
)

history["acq_date"] = pd.to_datetime(
    history["acq_date"],
    errors="raise"
)


first_holdout_time = (
    holdout["acquired_at"]
    .min()
)

history_cutoff = (
    first_holdout_time
    - pd.Timedelta(
        days=MAX_HISTORY_DAYS
    )
)


history = history[
    (
        history["acquired_at"]
        >= history_cutoff
    )
    &
    (
        history["acquired_at"]
        < first_holdout_time
    )
].copy()


print(
    f"      First 2025 observation: "
    f"{first_holdout_time}"
)

print(
    f"      History cutoff: "
    f"{history_cutoff}"
)

print(
    f"      2024 history rows loaded: "
    f"{len(history):,}"
)


if history.empty:

    raise RuntimeError(
        "No 2024 boundary history found."
    )


# ============================================================
# MARK SOURCE
# ============================================================

print(
    "\n[3/9] Combining boundary history "
    "with 2025 observations..."
)


history[
    "_temporal_source"
] = "history_2024"


holdout[
    "_temporal_source"
] = "holdout_2025"


combined = pd.concat(
    [
        history,
        holdout,
    ],
    ignore_index=True,
    sort=False,
)


combined = (
    combined
    .sort_values(
        [
            "acquired_at",
            "latitude",
            "longitude",
            "observation_id",
        ],
        kind="stable",
    )
    .reset_index(
        drop=True
    )
)


print(
    f"      Combined rows: "
    f"{len(combined):,}"
)


# ============================================================
# ARRAYS
# ============================================================

print(
    "\n[4/9] Preparing arrays..."
)

n = len(combined)

latitudes = (
    combined["latitude"]
    .to_numpy(
        dtype=np.float64
    )
)

longitudes = (
    combined["longitude"]
    .to_numpy(
        dtype=np.float64
    )
)

frps = (
    combined["frp"]
    .to_numpy(
        dtype=np.float64
    )
)

timestamps = (
    combined["acquired_at"]
    .to_numpy(
        dtype="datetime64[ns]"
    )
    .astype(np.int64)
)

dates = (
    combined["acq_date"]
    .values
    .astype("datetime64[D]")
)


# ============================================================
# OUTPUT ARRAYS
# ============================================================

detections_3d = np.zeros(
    n,
    dtype=np.int32
)

detections_7d = np.zeros(
    n,
    dtype=np.int32
)

detections_10d = np.zeros(
    n,
    dtype=np.int32
)

detections_30d = np.zeros(
    n,
    dtype=np.int32
)

distinct_active_days_30d = np.zeros(
    n,
    dtype=np.int16
)

hours_since_previous = np.full(
    n,
    np.nan,
    dtype=np.float32
)

mean_frp_previous_30d = np.full(
    n,
    np.nan,
    dtype=np.float32
)

median_frp_previous_30d = np.full(
    n,
    np.nan,
    dtype=np.float32
)

max_frp_previous_30d = np.full(
    n,
    np.nan,
    dtype=np.float32
)

frp_ratio_to_history = np.full(
    n,
    np.nan,
    dtype=np.float32
)

active_days_fraction_30d = np.zeros(
    n,
    dtype=np.float32
)


# ============================================================
# CAUSAL HISTORY
# ============================================================

print(
    "\n[5/9] Calculating causal temporal features..."
)

grid = defaultdict(
    deque
)

global_history = deque()

start_time = time.time()

progress_every = 100_000


for i in range(n):

    current_time = (
        timestamps[i]
    )

    current_lat = (
        latitudes[i]
    )

    current_lon = (
        longitudes[i]
    )

    cutoff_30d = (
        current_time
        - WINDOW_30D
    )


    # --------------------------------------------------------
    # Remove history older than 30 days
    # --------------------------------------------------------

    while (
        global_history
        and
        timestamps[
            global_history[0]
        ]
        < cutoff_30d
    ):

        old_idx = (
            global_history
            .popleft()
        )

        old_cell = spatial_cell(
            latitudes[old_idx],
            longitudes[old_idx]
        )

        cell_queue = (
            grid[
                old_cell
            ]
        )

        if (
            cell_queue
            and
            cell_queue[0]
            == old_idx
        ):

            cell_queue.popleft()

        else:

            try:
                cell_queue.remove(
                    old_idx
                )
            except ValueError:
                pass

        if not cell_queue:
            del grid[
                old_cell
            ]


    # --------------------------------------------------------
    # Candidate history
    # --------------------------------------------------------

    current_cell = spatial_cell(
        current_lat,
        current_lon
    )

    candidate_indexes = []


    for cell in neighboring_cells(
        current_cell
    ):

        if cell in grid:

            candidate_indexes.extend(
                grid[cell]
            )


    if candidate_indexes:

        candidate_indexes = np.asarray(
            candidate_indexes,
            dtype=np.int64
        )


        # EXACT causal rule:
        # historical timestamp must be strictly earlier.
        historical_mask = (
            timestamps[
                candidate_indexes
            ]
            < current_time
        )


        candidate_indexes = (
            candidate_indexes[
                historical_mask
            ]
        )


        if len(
            candidate_indexes
        ) > 0:

            distances = haversine_km(
                current_lat,
                current_lon,
                latitudes[
                    candidate_indexes
                ],
                longitudes[
                    candidate_indexes
                ],
            )


            nearby_indexes = (
                candidate_indexes[
                    distances
                    <= RADIUS_KM
                ]
            )


            if len(
                nearby_indexes
            ) > 0:

                time_deltas = (
                    current_time
                    - timestamps[
                        nearby_indexes
                    ]
                )


                # ============================================
                # COUNTS
                # ============================================

                detections_3d[i] = (
                    np.count_nonzero(
                        time_deltas
                        <= WINDOW_3D
                    )
                )

                detections_7d[i] = (
                    np.count_nonzero(
                        time_deltas
                        <= WINDOW_7D
                    )
                )

                detections_10d[i] = (
                    np.count_nonzero(
                        time_deltas
                        <= WINDOW_10D
                    )
                )

                detections_30d[i] = (
                    len(
                        nearby_indexes
                    )
                )


                # ============================================
                # ACTIVE DAYS
                # ============================================

                historical_dates = (
                    dates[
                        nearby_indexes
                    ]
                )

                distinct_days = len(
                    np.unique(
                        historical_dates
                    )
                )

                distinct_active_days_30d[
                    i
                ] = distinct_days


                active_days_fraction_30d[
                    i
                ] = (
                    distinct_days
                    / 31.0
                )


                # ============================================
                # PREVIOUS DETECTION
                # ============================================

                latest_previous_time = (
                    np.max(
                        timestamps[
                            nearby_indexes
                        ]
                    )
                )


                hours_since_previous[
                    i
                ] = (
                    (
                        current_time
                        - latest_previous_time
                    )
                    / NS_PER_HOUR
                )


                # ============================================
                # HISTORICAL FRP
                # ============================================

                history_frp = (
                    frps[
                        nearby_indexes
                    ]
                )

                mean_frp = (
                    np.mean(
                        history_frp
                    )
                )

                median_frp = (
                    np.median(
                        history_frp
                    )
                )

                max_frp = (
                    np.max(
                        history_frp
                    )
                )


                mean_frp_previous_30d[
                    i
                ] = mean_frp

                median_frp_previous_30d[
                    i
                ] = median_frp

                max_frp_previous_30d[
                    i
                ] = max_frp


                if median_frp > 0:

                    frp_ratio_to_history[
                        i
                    ] = (
                        frps[i]
                        / median_frp
                    )


    # --------------------------------------------------------
    # Insert CURRENT observation only AFTER calculation
    # --------------------------------------------------------

    grid[
        current_cell
    ].append(
        i
    )

    global_history.append(
        i
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        (i + 1)
        % progress_every
        == 0
        or
        i == n - 1
    ):

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            (i + 1)
            / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            (n - i - 1)
            / rate
            if rate > 0
            else 0
        )

        print(
            f"      {i + 1:,}/{n:,} "
            f"({(i + 1) / n * 100:.1f}%) | "
            f"{rate:,.0f} rows/sec | "
            f"ETA {remaining / 60:.1f} min"
        )


# ============================================================
# ATTACH FEATURES
# ============================================================

print(
    "\n[6/9] Attaching temporal features..."
)

combined[
    "detections_3d"
] = detections_3d

combined[
    "detections_7d"
] = detections_7d

combined[
    "detections_10d"
] = detections_10d

combined[
    "detections_30d"
] = detections_30d

combined[
    "distinct_active_days_30d"
] = distinct_active_days_30d

combined[
    "hours_since_previous_detection"
] = hours_since_previous

combined[
    "mean_frp_previous_30d"
] = mean_frp_previous_30d

combined[
    "median_frp_previous_30d"
] = median_frp_previous_30d

combined[
    "max_frp_previous_30d"
] = max_frp_previous_30d

combined[
    "frp_ratio_to_history"
] = frp_ratio_to_history

combined[
    "active_days_fraction_30d"
] = active_days_fraction_30d


# ============================================================
# KEEP ONLY 2025
# ============================================================

print(
    "\n[7/9] Removing boundary-history rows..."
)

result = combined[
    combined[
        "_temporal_source"
    ].eq(
        "holdout_2025"
    )
].copy()


result = result.drop(
    columns=[
        "_temporal_source"
    ]
)


result = (
    result
    .sort_values(
        [
            "acquired_at",
            "latitude",
            "longitude",
            "observation_id",
        ],
        kind="stable",
    )
    .reset_index(
        drop=True
    )
)


if len(result) != len(
    holdout
):

    raise RuntimeError(
        "2025 row count changed."
    )


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n[8/9] Validating temporal features..."
)


assert (
    result["year"]
    == 2025
).all()


assert (
    result["observation_id"]
    .str.startswith(
        "FW25_"
    )
    .all()
)


assert (
    result["observation_id"]
    .is_unique
)


assert (
    result["detections_3d"]
    <= result["detections_7d"]
).all()


assert (
    result["detections_7d"]
    <= result["detections_10d"]
).all()


assert (
    result["detections_10d"]
    <= result["detections_30d"]
).all()


assert (
    result[
        "distinct_active_days_30d"
    ]
    <=
    result[
        "detections_30d"
    ]
).all()


assert (
    result[
        "active_days_fraction_30d"
    ]
    .between(
        0,
        1
    )
    .all()
)


no_history = (
    result[
        "detections_30d"
    ]
    == 0
)


for column in [
    "hours_since_previous_detection",
    "mean_frp_previous_30d",
    "median_frp_previous_30d",
    "max_frp_previous_30d",
    "frp_ratio_to_history",
]:

    assert (
        result.loc[
            no_history,
            column
        ]
        .isna()
        .all()
    )


# ============================================================
# SAVE
# ============================================================

print(
    "\n[9/9] Saving 2025 temporal dataset..."
)

result.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# BOUNDARY DIAGNOSTIC
# ============================================================

first_30_days_end = (
    first_holdout_time
    + pd.Timedelta(
        days=30
    )
)


early_2025 = result[
    result["acquired_at"]
    < first_30_days_end
]


early_with_history = int(
    (
        early_2025[
            "detections_30d"
        ]
        > 0
    ).sum()
)


all_with_history = int(
    (
        result[
            "detections_30d"
        ]
        > 0
    ).sum()
)


# ============================================================
# REPORT
# ============================================================

report = {
    "stage":
        "2025_causal_temporal_features",

    "rows":
        int(
            len(result)
        ),

    "boundary_history_rows":
        int(
            len(history)
        ),

    "boundary_history_start":
        str(
            history[
                "acquired_at"
            ].min()
        ),

    "boundary_history_end":
        str(
            history[
                "acquired_at"
            ].max()
        ),

    "first_2025_timestamp":
        str(
            first_holdout_time
        ),

    "radius_km":
        RADIUS_KM,

    "maximum_history_days":
        MAX_HISTORY_DAYS,

    "strictly_earlier_only":
        True,

    "current_observation_excluded":
        True,

    "future_observations_excluded":
        True,

    "rows_with_prior_detection_30d":
        all_with_history,

    "percent_with_prior_detection_30d":
        float(
            all_with_history
            / len(result)
            * 100
        ),

    "early_2025_rows":
        int(
            len(early_2025)
        ),

    "early_2025_rows_with_history":
        early_with_history,

    "early_2025_percent_with_history":
        float(
            early_with_history
            / len(early_2025)
            * 100
        ),

    "detections_30d": {
        "mean":
            float(
                result[
                    "detections_30d"
                ].mean()
            ),

        "median":
            float(
                result[
                    "detections_30d"
                ].median()
            ),

        "p90":
            float(
                result[
                    "detections_30d"
                ].quantile(
                    0.90
                )
            ),

        "p95":
            float(
                result[
                    "detections_30d"
                ].quantile(
                    0.95
                )
            ),

        "p99":
            float(
                result[
                    "detections_30d"
                ].quantile(
                    0.99
                )
            ),

        "max":
            int(
                result[
                    "detections_30d"
                ].max()
            ),
    },

    "distinct_active_days_30d": {
        "mean":
            float(
                result[
                    "distinct_active_days_30d"
                ].mean()
            ),

        "median":
            float(
                result[
                    "distinct_active_days_30d"
                ].median()
            ),

        "p95":
            float(
                result[
                    "distinct_active_days_30d"
                ].quantile(
                    0.95
                )
            ),

        "p99":
            float(
                result[
                    "distinct_active_days_30d"
                ].quantile(
                    0.99
                )
            ),

        "max":
            int(
                result[
                    "distinct_active_days_30d"
                ].max()
            ),
    },

    "model_used":
        False,

    "labels_generated":
        False,

    "feature_policy_changed":
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


# ============================================================
# FINISH
# ============================================================

elapsed = (
    time.time()
    - start_time
)


print(
    "\n" + "=" * 80
)

print(
    "STEP 27 COMPLETE"
)

print(
    "=" * 80
)


print(
    f"\n2024 boundary history rows: "
    f"{len(history):,}"
)

print(
    f"2025 output rows: "
    f"{len(result):,}"
)


print(
    "\nRows with previous detection "
    "within 1 km / 30 days:"
)

print(
    f"  {all_with_history:,} / "
    f"{len(result):,} "
    f"({all_with_history / len(result) * 100:.2f}%)"
)


print(
    "\nFirst 30 days of 2025:"
)

print(
    f"  Rows: "
    f"{len(early_2025):,}"
)

print(
    f"  With prior history: "
    f"{early_with_history:,} "
    f"({early_with_history / len(early_2025) * 100:.2f}%)"
)


print(
    "\n30-day detection statistics:"
)

print(
    result[
        "detections_30d"
    ]
    .describe(
        percentiles=[
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
    "\nDistinct active days:"
)

print(
    result[
        "distinct_active_days_30d"
    ]
    .describe(
        percentiles=[
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
    f"{elapsed / 60:.2f} minutes"
)


print(
    f"\nSaved:\n{OUTPUT_FILE}"
)


print(
    "\nIMPORTANT:"
    "\n- December 2024 history was available to January 2025."
    "\n- Current observation excluded."
    "\n- Future observations excluded."
    "\n- Output contains only 2025 observations."
    "\n- Model has NOT been executed."
    "\n- Labels have NOT been generated."
)