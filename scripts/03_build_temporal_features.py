from pathlib import Path
from collections import defaultdict, deque
import json
import math
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
    / "firms_master_2022_2024.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "firms_with_temporal_2022_2024.parquet"
)

CHECKPOINT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "_temporal_checkpoint.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "temporal_features_2022_2024_summary.json"
)

RADIUS_KM = 1.0
MAX_HISTORY_DAYS = 30
CELL_SIZE_DEG = 0.01
EARTH_RADIUS_KM = 6371.0088

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance.

    lat1/lon1:
        scalar current observation

    lat2/lon2:
        arrays of historical candidate observations
    """

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

    a = np.clip(a, 0.0, 1.0)

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


def spatial_cell(lat, lon):
    """
    Convert coordinates to a coarse spatial grid cell.
    """

    return (
        int(math.floor(lat / CELL_SIZE_DEG)),
        int(math.floor(lon / CELL_SIZE_DEG))
    )


def neighboring_cells(cell):
    """
    Return the current grid cell and its eight neighbors.

    Exact 1 km filtering is performed later using Haversine
    distance.
    """

    x, y = cell

    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            yield (x + dx, y + dy)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("FIREWATCH — BUILD CAUSAL TEMPORAL FEATURES")
print("=" * 80)

print(f"\nInput:\n{INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_parquet(INPUT_FILE)

print(f"\nLoaded observations: {len(df):,}")

required = [
    "observation_id",
    "latitude",
    "longitude",
    "acquired_at",
    "acq_date",
    "frp"
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print("\n[1/7] Sorting chronologically...")

df["acquired_at"] = pd.to_datetime(
    df["acquired_at"],
    errors="raise"
)

df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="raise"
)

df = df.sort_values(
    [
        "acquired_at",
        "latitude",
        "longitude",
        "observation_id"
    ],
    kind="stable"
).reset_index(drop=True)


# ============================================================
# PREPARE ARRAYS
# ============================================================

print("[2/7] Preparing spatial/temporal arrays...")

n = len(df)

latitudes = df["latitude"].to_numpy(
    dtype=np.float64
)

longitudes = df["longitude"].to_numpy(
    dtype=np.float64
)

frps = df["frp"].to_numpy(
    dtype=np.float64
)

timestamps = (
    df["acquired_at"]
    .to_numpy(dtype="datetime64[ns]")
    .astype(np.int64)
)

dates = (
    df["acq_date"]
    .values
    .astype("datetime64[D]")
)

NS_PER_HOUR = 3_600_000_000_000
NS_PER_DAY = 86_400_000_000_000

window_3d = 3 * NS_PER_DAY
window_7d = 7 * NS_PER_DAY
window_10d = 10 * NS_PER_DAY
window_30d = 30 * NS_PER_DAY


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
# SPATIAL HISTORY INDEX
# ============================================================

print("[3/7] Building causal 30-day spatial history...")

# Each grid cell stores indexes of previously processed
# observations.
#
# The current observation is inserted only AFTER its features
# are calculated. Therefore current/future observations cannot
# leak into historical features.

grid = defaultdict(deque)

# Global chronological queue for removing observations older
# than the 30-day history window.
global_history = deque()

start_time = time.time()

progress_every = 100_000


# ============================================================
# CAUSAL FEATURE GENERATION
# ============================================================

for i in range(n):

    current_time = timestamps[i]
    current_lat = latitudes[i]
    current_lon = longitudes[i]

    cutoff_30d = current_time - window_30d

    # --------------------------------------------------------
    # REMOVE OBSERVATIONS OLDER THAN 30 DAYS
    # --------------------------------------------------------

    while (
        global_history
        and timestamps[global_history[0]] < cutoff_30d
    ):

        old_idx = global_history.popleft()

        old_cell = spatial_cell(
            latitudes[old_idx],
            longitudes[old_idx]
        )

        cell_queue = grid[old_cell]

        if (
            cell_queue
            and cell_queue[0] == old_idx
        ):
            cell_queue.popleft()

        else:
            try:
                cell_queue.remove(old_idx)
            except ValueError:
                pass

        if not cell_queue:
            del grid[old_cell]

    # --------------------------------------------------------
    # GET CANDIDATE OBSERVATIONS FROM NEIGHBORING CELLS
    # --------------------------------------------------------

    current_cell = spatial_cell(
        current_lat,
        current_lon
    )

    candidate_indexes = []

    for cell in neighboring_cells(current_cell):

        if cell in grid:
            candidate_indexes.extend(
                grid[cell]
            )

    if candidate_indexes:

        candidate_indexes = np.asarray(
            candidate_indexes,
            dtype=np.int64
        )

        # Strictly earlier timestamps only.
        historical_mask = (
            timestamps[candidate_indexes]
            < current_time
        )

        candidate_indexes = (
            candidate_indexes[
                historical_mask
            ]
        )

        if len(candidate_indexes) > 0:

            distances = haversine_km(
                current_lat,
                current_lon,
                latitudes[candidate_indexes],
                longitudes[candidate_indexes]
            )

            nearby_mask = (
                distances <= RADIUS_KM
            )

            nearby_indexes = (
                candidate_indexes[
                    nearby_mask
                ]
            )

            if len(nearby_indexes) > 0:

                time_deltas = (
                    current_time
                    - timestamps[
                        nearby_indexes
                    ]
                )

                # ============================================
                # DETECTION COUNTS
                # ============================================

                detections_3d[i] = (
                    np.count_nonzero(
                        time_deltas
                        <= window_3d
                    )
                )

                detections_7d[i] = (
                    np.count_nonzero(
                        time_deltas
                        <= window_7d
                    )
                )

                detections_10d[i] = (
                    np.count_nonzero(
                        time_deltas
                        <= window_10d
                    )
                )

                detections_30d[i] = (
                    len(nearby_indexes)
                )

                # ============================================
                # DISTINCT HISTORICAL ACTIVE DAYS
                # ============================================

                historical_dates = (
                    dates[nearby_indexes]
                )

                distinct_days = len(
                    np.unique(
                        historical_dates
                    )
                )

                distinct_active_days_30d[i] = (
                    distinct_days
                )

                # A rolling 30 x 24-hour interval may touch
                # portions of up to 31 calendar dates.
                active_days_fraction_30d[i] = (
                    distinct_days / 31.0
                )

                # ============================================
                # TIME SINCE MOST RECENT PREVIOUS DETECTION
                # ============================================

                latest_previous_time = (
                    np.max(
                        timestamps[
                            nearby_indexes
                        ]
                    )
                )

                hours_since_previous[i] = (
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
                    frps[nearby_indexes]
                )

                mean_frp = np.mean(
                    history_frp
                )

                median_frp = np.median(
                    history_frp
                )

                max_frp = np.max(
                    history_frp
                )

                mean_frp_previous_30d[i] = (
                    mean_frp
                )

                median_frp_previous_30d[i] = (
                    median_frp
                )

                max_frp_previous_30d[i] = (
                    max_frp
                )

                if median_frp > 0:

                    frp_ratio_to_history[i] = (
                        frps[i]
                        / median_frp
                    )

    # --------------------------------------------------------
    # INSERT CURRENT OBSERVATION INTO HISTORY
    # --------------------------------------------------------
    #
    # This happens AFTER all features for observation i have
    # been calculated.

    grid[current_cell].append(i)

    global_history.append(i)

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        (i + 1) % progress_every == 0
        or i == n - 1
    ):

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            (i + 1) / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            (n - i - 1) / rate
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

print("\n[4/7] Attaching temporal features...")

df["detections_3d"] = (
    detections_3d
)

df["detections_7d"] = (
    detections_7d
)

df["detections_10d"] = (
    detections_10d
)

df["detections_30d"] = (
    detections_30d
)

df["distinct_active_days_30d"] = (
    distinct_active_days_30d
)

df["hours_since_previous_detection"] = (
    hours_since_previous
)

df["mean_frp_previous_30d"] = (
    mean_frp_previous_30d
)

df["median_frp_previous_30d"] = (
    median_frp_previous_30d
)

df["max_frp_previous_30d"] = (
    max_frp_previous_30d
)

df["frp_ratio_to_history"] = (
    frp_ratio_to_history
)

df["active_days_fraction_30d"] = (
    active_days_fraction_30d
)


# ============================================================
# CHECKPOINT
# ============================================================

print(
    "      Writing temporary checkpoint..."
)

df.to_parquet(
    CHECKPOINT_FILE,
    index=False
)

print(
    f"      Checkpoint saved: "
    f"{CHECKPOINT_FILE.name}"
)


# ============================================================
# VALIDATION
# ============================================================

print("[5/7] Validating temporal consistency...")

# Nested historical windows must be monotonic.

assert (
    df["detections_3d"]
    <= df["detections_7d"]
).all(), (
    "detections_3d cannot exceed detections_7d"
)

assert (
    df["detections_7d"]
    <= df["detections_10d"]
).all(), (
    "detections_7d cannot exceed detections_10d"
)

assert (
    df["detections_10d"]
    <= df["detections_30d"]
).all(), (
    "detections_10d cannot exceed detections_30d"
)

assert (
    df["distinct_active_days_30d"]
    <= df["detections_30d"]
).all(), (
    "Distinct active days cannot exceed "
    "number of historical detections"
)

assert (
    df["active_days_fraction_30d"]
    >= 0
).all(), (
    "Active-day fraction cannot be negative"
)

assert (
    df["active_days_fraction_30d"]
    <= 1
).all(), (
    "Active-day fraction cannot exceed 1"
)

# If no previous detection exists, all features requiring
# historical measurements should remain NaN.

no_history = (
    df["detections_30d"] == 0
)

assert df.loc[
    no_history,
    "hours_since_previous_detection"
].isna().all(), (
    "Rows without history contain previous-detection gaps"
)

assert df.loc[
    no_history,
    "mean_frp_previous_30d"
].isna().all(), (
    "Rows without history contain historical mean FRP"
)

assert df.loc[
    no_history,
    "median_frp_previous_30d"
].isna().all(), (
    "Rows without history contain historical median FRP"
)

assert df.loc[
    no_history,
    "max_frp_previous_30d"
].isna().all(), (
    "Rows without history contain historical max FRP"
)

assert df.loc[
    no_history,
    "frp_ratio_to_history"
].isna().all(), (
    "Rows without history contain FRP history ratio"
)

print(
    "      Temporal consistency checks passed."
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

print("[6/7] Saving temporal dataset...")

df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# BUILD REPORT
# ============================================================

print("[7/7] Writing summary report...")

history_count = int(
    (
        df["detections_30d"] > 0
    ).sum()
)

no_history_count = int(
    (
        df["detections_30d"] == 0
    ).sum()
)

report = {

    "input_file": str(
        INPUT_FILE
    ),

    "output_file": str(
        OUTPUT_FILE
    ),

    "rows": int(
        len(df)
    ),

    "radius_km": (
        RADIUS_KM
    ),

    "maximum_history_days": (
        MAX_HISTORY_DAYS
    ),

    "causal_rule": (
        "Only observations with timestamps strictly "
        "earlier than the current observation are used."
    ),

    "current_observation_excluded": True,

    "future_observations_excluded": True,

    "rows_with_prior_detection_30d": (
        history_count
    ),

    "rows_without_prior_detection_30d": (
        no_history_count
    ),

    "percent_with_prior_detection_30d": (
        history_count
        / len(df)
        * 100
    ),

    "temporal_feature_summary": {

        "detections_3d": {

            "mean": float(
                df[
                    "detections_3d"
                ].mean()
            ),

            "median": float(
                df[
                    "detections_3d"
                ].median()
            ),

            "p95": float(
                df[
                    "detections_3d"
                ].quantile(0.95)
            ),

            "max": int(
                df[
                    "detections_3d"
                ].max()
            )
        },

        "detections_7d": {

            "mean": float(
                df[
                    "detections_7d"
                ].mean()
            ),

            "median": float(
                df[
                    "detections_7d"
                ].median()
            ),

            "p95": float(
                df[
                    "detections_7d"
                ].quantile(0.95)
            ),

            "max": int(
                df[
                    "detections_7d"
                ].max()
            )
        },

        "detections_10d": {

            "mean": float(
                df[
                    "detections_10d"
                ].mean()
            ),

            "median": float(
                df[
                    "detections_10d"
                ].median()
            ),

            "p95": float(
                df[
                    "detections_10d"
                ].quantile(0.95)
            ),

            "max": int(
                df[
                    "detections_10d"
                ].max()
            )
        },

        "detections_30d": {

            "mean": float(
                df[
                    "detections_30d"
                ].mean()
            ),

            "median": float(
                df[
                    "detections_30d"
                ].median()
            ),

            "p95": float(
                df[
                    "detections_30d"
                ].quantile(0.95)
            ),

            "p99": float(
                df[
                    "detections_30d"
                ].quantile(0.99)
            ),

            "max": int(
                df[
                    "detections_30d"
                ].max()
            )
        },

        "distinct_active_days_30d": {

            "mean": float(
                df[
                    "distinct_active_days_30d"
                ].mean()
            ),

            "median": float(
                df[
                    "distinct_active_days_30d"
                ].median()
            ),

            "p95": float(
                df[
                    "distinct_active_days_30d"
                ].quantile(0.95)
            ),

            "p99": float(
                df[
                    "distinct_active_days_30d"
                ].quantile(0.99)
            ),

            "max": int(
                df[
                    "distinct_active_days_30d"
                ].max()
            )
        },

        "active_days_fraction_30d": {

            "mean": float(
                df[
                    "active_days_fraction_30d"
                ].mean()
            ),

            "median": float(
                df[
                    "active_days_fraction_30d"
                ].median()
            ),

            "p95": float(
                df[
                    "active_days_fraction_30d"
                ].quantile(0.95)
            ),

            "max": float(
                df[
                    "active_days_fraction_30d"
                ].max()
            )
        }
    },

    "notes": [

        (
            "Temporal features are factual historical "
            "measurements, not class labels."
        ),

        (
            "No persistence class was assigned."
        ),

        (
            "No source-class labels were assigned."
        ),

        (
            "The current observation is inserted into "
            "history only after its own features are "
            "calculated."
        ),

        (
            "Only previous detections within 1 km and "
            "the previous 30 days are considered."
        ),

        (
            "active_days_fraction_30d divides distinct "
            "historical calendar dates by 31 because a "
            "rolling 30x24-hour interval can touch parts "
            "of 31 calendar dates."
        ),

        (
            "2025 holdout data was not accessed."
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
# REMOVE CHECKPOINT AFTER SUCCESS
# ============================================================

if CHECKPOINT_FILE.exists():

    CHECKPOINT_FILE.unlink()

    print(
        "      Temporary checkpoint removed."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print("\n" + "=" * 80)
print("TEMPORAL FEATURE BUILD COMPLETE")
print("=" * 80)

print(
    f"\nRows: "
    f"{len(df):,}"
)

print(
    f"\nRows with previous detection "
    f"within 1 km / 30 days: "
    f"{history_count:,} "
    f"({history_count / len(df) * 100:.2f}%)"
)

print(
    f"Rows with no previous detection: "
    f"{no_history_count:,}"
)

print(
    "\n30-day detection statistics:"
)

print(
    df[
        "detections_30d"
    ]
    .describe(
        percentiles=[
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
    "\nDistinct active days statistics:"
)

print(
    df[
        "distinct_active_days_30d"
    ]
    .describe(
        percentiles=[
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
    "\nActive-days fraction statistics:"
)

print(
    df[
        "active_days_fraction_30d"
    ]
    .describe(
        percentiles=[
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
    f"\nRuntime: "
    f"{elapsed / 60:.2f} minutes"
)

print(
    f"\nSaved:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nReport:\n"
    f"{REPORT_FILE}"
)

print(
    "\nIMPORTANT:"
    "\n- Current observation excluded from its own history."
    "\n- Future observations excluded."
    "\n- No persistence labels created."
    "\n- No source labels created."
    "\n- 2025 holdout untouched."
)