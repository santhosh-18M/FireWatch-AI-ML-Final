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
    / "firms_with_osm_2022_2024.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "combined_evidence_audit.json"
)

SAMPLE_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "combined_evidence_review_sample.csv"
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def count_pct(mask, total):

    count = int(mask.sum())

    return {
        "count": count,
        "percentage": float(
            count / total * 100
        )
    }


def numeric_summary(series):

    clean = series.dropna()

    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max())
    }


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — COMBINED EVIDENCE AUDIT")
print("=" * 80)

print(
    "\nPurpose:"
    "\nAudit thermal + temporal + OSM evidence"
    "\nBEFORE constructing source labels."
)

print(
    "\nNo labels are created."
    "\nNo model is trained."
    "\n2025 holdout is not accessed."
)

start = time.time()


# ============================================================
# LOAD
# ============================================================

print(
    "\n[1/7] Loading enriched development dataset..."
)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing input file:\n{INPUT_FILE}"
    )

df = pd.read_parquet(
    INPUT_FILE
)

total = len(df)

print(
    f"      Rows: {total:,}"
)

print(
    f"      Columns: {len(df.columns):,}"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

print(
    "\n[2/7] Checking required evidence columns..."
)

required = [
    "observation_id",
    "acquired_at",
    "latitude",
    "longitude",
    "brightness",
    "frp",
    "detections_3d",
    "detections_7d",
    "detections_10d",
    "detections_30d",
    "distinct_active_days_30d",
    "distance_to_strong_industrial_km",
    "distance_to_vegetation_km",
    "nearest_strong_industrial_category",
    "nearest_vegetation_category",
    "osm_context_relation",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print(
    "      Required columns present."
)


# ============================================================
# TEMPORAL AUDIT GROUPS
# ============================================================

print(
    "\n[3/7] Building audit-only temporal groups..."
)

# IMPORTANT:
# These are exploratory audit buckets.
# They are NOT final temporal labels.

no_prior = (
    df["detections_30d"] == 0
)

some_prior = (
    df["detections_30d"] >= 1
)

repeated_7d = (
    df["detections_7d"] >= 2
)

repeated_30d = (
    df["detections_30d"] >= 3
)

many_active_days = (
    df["distinct_active_days_30d"] >= 5
)

high_recurrence = (
    repeated_30d
    &
    many_active_days
)


print(
    f"      No prior detections (30d): "
    f"{no_prior.sum():,}"
)

print(
    f"      >=1 prior detection (30d): "
    f"{some_prior.sum():,}"
)

print(
    f"      Audit high-recurrence group: "
    f"{high_recurrence.sum():,}"
)


# ============================================================
# OSM CONTEXT GROUPS
# ============================================================

print(
    "\n[4/7] Auditing OSM evidence groups..."
)

strong_1_5 = (
    df[
        "distance_to_strong_industrial_km"
    ] <= 1.5
)

strong_3 = (
    df[
        "distance_to_strong_industrial_km"
    ] <= 3.0
)

vegetation_1_5 = (
    df[
        "distance_to_vegetation_km"
    ] <= 1.5
)

vegetation_3 = (
    df[
        "distance_to_vegetation_km"
    ] <= 3.0
)


industrial_only_3 = (
    strong_3
    &
    ~vegetation_3
)

vegetation_only_3 = (
    vegetation_3
    &
    ~strong_3
)

mixed_3 = (
    strong_3
    &
    vegetation_3
)

neither_3 = (
    ~strong_3
    &
    ~vegetation_3
)


print(
    f"      Strong industrial <=1.5 km: "
    f"{strong_1_5.sum():,}"
)

print(
    f"      Strong industrial <=3 km: "
    f"{strong_3.sum():,}"
)

print(
    f"      Vegetation <=1.5 km: "
    f"{vegetation_1_5.sum():,}"
)

print(
    f"      Vegetation <=3 km: "
    f"{vegetation_3.sum():,}"
)


# ============================================================
# THERMAL DISTRIBUTION
# ============================================================

print(
    "\n[5/7] Auditing thermal evidence..."
)

frp_p90 = float(
    df["frp"].quantile(0.90)
)

frp_p95 = float(
    df["frp"].quantile(0.95)
)

frp_p99 = float(
    df["frp"].quantile(0.99)
)

brightness_p90 = float(
    df["brightness"].quantile(0.90)
)

brightness_p95 = float(
    df["brightness"].quantile(0.95)
)


high_frp = (
    df["frp"] >= frp_p90
)

very_high_frp = (
    df["frp"] >= frp_p95
)


print(
    f"      FRP p90: {frp_p90:.3f}"
)

print(
    f"      FRP p95: {frp_p95:.3f}"
)

print(
    f"      FRP p99: {frp_p99:.3f}"
)

print(
    f"      Brightness p90: "
    f"{brightness_p90:.3f}"
)

print(
    f"      Brightness p95: "
    f"{brightness_p95:.3f}"
)


# ============================================================
# COMBINED EVIDENCE GROUPS
# ============================================================

print(
    "\n[6/7] Measuring evidence combinations..."
)

groups = {

    # --------------------------------------------------------
    # Clean contextual separation
    # --------------------------------------------------------

    "industrial_only_within_3km":
        industrial_only_3,

    "vegetation_only_within_3km":
        vegetation_only_3,

    "mixed_within_3km":
        mixed_3,

    "neither_within_3km":
        neither_3,


    # --------------------------------------------------------
    # Industrial-context behavior
    # --------------------------------------------------------

    "industrial_context_no_prior_30d":
        industrial_only_3
        &
        no_prior,

    "industrial_context_some_prior_30d":
        industrial_only_3
        &
        some_prior,

    "industrial_context_high_recurrence":
        industrial_only_3
        &
        high_recurrence,

    "industrial_context_high_frp":
        industrial_only_3
        &
        high_frp,

    "industrial_context_high_frp_no_prior":
        industrial_only_3
        &
        high_frp
        &
        no_prior,


    # --------------------------------------------------------
    # Vegetation-context behavior
    # --------------------------------------------------------

    "vegetation_context_no_prior_30d":
        vegetation_only_3
        &
        no_prior,

    "vegetation_context_some_prior_30d":
        vegetation_only_3
        &
        some_prior,

    "vegetation_context_high_recurrence":
        vegetation_only_3
        &
        high_recurrence,

    "vegetation_context_high_frp":
        vegetation_only_3
        &
        high_frp,

    "vegetation_context_high_frp_no_prior":
        vegetation_only_3
        &
        high_frp
        &
        no_prior,


    # --------------------------------------------------------
    # Ambiguous/context conflict
    # --------------------------------------------------------

    "mixed_context_high_frp":
        mixed_3
        &
        high_frp,

    "mixed_context_high_recurrence":
        mixed_3
        &
        high_recurrence,

    "no_close_context_high_frp":
        neither_3
        &
        high_frp,

    "no_close_context_high_recurrence":
        neither_3
        &
        high_recurrence,


    # --------------------------------------------------------
    # Stronger proximity evidence
    # --------------------------------------------------------

    "industrial_within_1_5km_only":
        strong_1_5
        &
        ~vegetation_3,

    "vegetation_within_1_5km_only":
        vegetation_1_5
        &
        ~strong_3,
}


group_report = {}

for name, mask in groups.items():

    subset = df.loc[mask]

    group_report[name] = {

        "count":
            int(len(subset)),

        "percentage":
            float(
                len(subset)
                / total
                * 100
            ),

        "frp":
            numeric_summary(
                subset["frp"]
            )
            if len(subset)
            else None,

        "brightness":
            numeric_summary(
                subset["brightness"]
            )
            if len(subset)
            else None,

        "detections_30d":
            numeric_summary(
                subset["detections_30d"]
            )
            if len(subset)
            else None,

        "distinct_active_days_30d":
            numeric_summary(
                subset[
                    "distinct_active_days_30d"
                ]
            )
            if len(subset)
            else None
    }


# ============================================================
# CROSS-TABLE
# ============================================================

cross_table = pd.crosstab(
    df["osm_context_relation"],
    pd.cut(
        df["detections_30d"],
        bins=[
            -1,
            0,
            2,
            10,
            50,
            np.inf
        ],
        labels=[
            "0",
            "1-2",
            "3-10",
            "11-50",
            "51+"
        ]
    )
)


# ============================================================
# CREATE REVIEW SAMPLE
# ============================================================

print(
    "\n[7/7] Creating evidence-review sample..."
)

# This is NOT a training set.
# It gives us examples from contrasting evidence groups
# for inspection before label construction.

review_parts = []

review_groups = {
    "industrial_no_prior":
        industrial_only_3
        &
        no_prior,

    "industrial_recurrent":
        industrial_only_3
        &
        high_recurrence,

    "vegetation_no_prior":
        vegetation_only_3
        &
        no_prior,

    "vegetation_recurrent":
        vegetation_only_3
        &
        high_recurrence,

    "mixed":
        mixed_3,

    "no_close_context":
        neither_3
}


for name, mask in review_groups.items():

    subset = df.loc[mask]

    n = min(
        100,
        len(subset)
    )

    if n == 0:
        continue

    sampled = subset.sample(
        n=n,
        random_state=RANDOM_STATE
    ).copy()

    sampled[
        "audit_group"
    ] = name

    review_parts.append(
        sampled
    )


review = pd.concat(
    review_parts,
    ignore_index=True
)


review_columns = [
    "observation_id",
    "audit_group",
    "acquired_at",
    "latitude",
    "longitude",
    "brightness",
    "frp",
    "detections_3d",
    "detections_7d",
    "detections_10d",
    "detections_30d",
    "distinct_active_days_30d",
    "distance_to_strong_industrial_km",
    "nearest_strong_industrial_category",
    "nearest_strong_industrial_strength",
    "distance_to_vegetation_km",
    "nearest_vegetation_category",
    "nearest_vegetation_strength",
    "osm_context_relation"
]


review[
    review_columns
].to_csv(
    SAMPLE_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report = {

    "dataset": {
        "rows": int(total),
        "columns": int(
            len(df.columns)
        )
    },

    "thermal_thresholds_for_audit_only": {
        "frp_p90":
            frp_p90,

        "frp_p95":
            frp_p95,

        "frp_p99":
            frp_p99,

        "brightness_p90":
            brightness_p90,

        "brightness_p95":
            brightness_p95
    },

    "temporal_groups_for_audit_only": {

        "no_prior_30d":
            count_pct(
                no_prior,
                total
            ),

        "some_prior_30d":
            count_pct(
                some_prior,
                total
            ),

        "repeated_7d":
            count_pct(
                repeated_7d,
                total
            ),

        "repeated_30d":
            count_pct(
                repeated_30d,
                total
            ),

        "high_recurrence":
            count_pct(
                high_recurrence,
                total
            )
    },

    "osm_context": {

        "strong_industrial_within_1_5km":
            count_pct(
                strong_1_5,
                total
            ),

        "strong_industrial_within_3km":
            count_pct(
                strong_3,
                total
            ),

        "vegetation_within_1_5km":
            count_pct(
                vegetation_1_5,
                total
            ),

        "vegetation_within_3km":
            count_pct(
                vegetation_3,
                total
            ),

        "industrial_only_3km":
            count_pct(
                industrial_only_3,
                total
            ),

        "vegetation_only_3km":
            count_pct(
                vegetation_only_3,
                total
            ),

        "mixed_3km":
            count_pct(
                mixed_3,
                total
            ),

        "neither_3km":
            count_pct(
                neither_3,
                total
            )
    },

    "combined_groups":
        group_report,

    "context_vs_recurrence":
        cross_table.to_dict(),

    "methodology_notes": [

        (
            "This stage performs exploratory "
            "evidence auditing only."
        ),

        (
            "No source labels or temporal labels "
            "are generated."
        ),

        (
            "Thresholds used here are audit "
            "thresholds, not finalized labeling "
            "rules."
        ),

        (
            "OSM context is contextual evidence "
            "and is not treated as confirmation "
            "of fire source."
        ),

        (
            "Historical recurrence describes "
            "prior FIRMS activity and does not "
            "by itself imply an industrial source."
        ),

        (
            "FRP and brightness describe thermal "
            "characteristics and do not independently "
            "identify source type."
        ),

        (
            "2025 FIRMS data remains locked and "
            "was not accessed."
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
# OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "COMBINED EVIDENCE AUDIT COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nOSM CONTEXT DISTRIBUTION"
)

for name in [
    "industrial_only_within_3km",
    "vegetation_only_within_3km",
    "mixed_within_3km",
    "neither_within_3km"
]:

    data = group_report[name]

    print(
        f"{name:35s} "
        f"{data['count']:>10,} "
        f"({data['percentage']:6.2f}%)"
    )


print(
    "\nINDUSTRIAL-CONTEXT EVIDENCE"
)

for name in [
    "industrial_context_no_prior_30d",
    "industrial_context_some_prior_30d",
    "industrial_context_high_recurrence",
    "industrial_context_high_frp",
    "industrial_context_high_frp_no_prior"
]:

    data = group_report[name]

    print(
        f"{name:42s} "
        f"{data['count']:>10,} "
        f"({data['percentage']:6.2f}%)"
    )


print(
    "\nVEGETATION-CONTEXT EVIDENCE"
)

for name in [
    "vegetation_context_no_prior_30d",
    "vegetation_context_some_prior_30d",
    "vegetation_context_high_recurrence",
    "vegetation_context_high_frp",
    "vegetation_context_high_frp_no_prior"
]:

    data = group_report[name]

    print(
        f"{name:42s} "
        f"{data['count']:>10,} "
        f"({data['percentage']:6.2f}%)"
    )


print(
    "\nAMBIGUOUS / CONFLICTING EVIDENCE"
)

for name in [
    "mixed_context_high_frp",
    "mixed_context_high_recurrence",
    "no_close_context_high_frp",
    "no_close_context_high_recurrence"
]:

    data = group_report[name]

    print(
        f"{name:42s} "
        f"{data['count']:>10,} "
        f"({data['percentage']:6.2f}%)"
    )


print(
    "\nCONTEXT VS PRIOR 30-DAY DETECTIONS"
)

print(
    cross_table.to_string()
)


print(
    f"\nReview sample rows: "
    f"{len(review):,}"
)

print(
    f"\nAudit report:\n"
    f"{REPORT_FILE}"
)

print(
    f"\nReview sample:\n"
    f"{SAMPLE_FILE}"
)

print(
    f"\nRuntime: "
    f"{time.time() - start:.2f} seconds"
)

print(
    "\nIMPORTANT:"
    "\nNo labels have been created."
    "\nDo NOT train a model from the audit groups."
    "\n2025 holdout remains untouched."
)