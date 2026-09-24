from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "training_dataset_v1_spatial_split.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "experiments"
    / "final_feature_group_experiment_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CLASS_NAMES = [
    "industrial_fire_candidate",
    "natural_vegetation_fire",
    "other_uncertain",
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


# ============================================================
# FEATURE GROUPS
# ============================================================

FIRMS = [
    "brightness",
    "bright_t31",
    "frp",
    "scan",
    "track",
    "confidence_code",
    "hour",
    "month",
    "brightness_t31_delta",
    "log_frp",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


TEMPORAL = [
    "detections_3d",
    "detections_7d",
    "detections_10d",
    "detections_30d",
    "distinct_active_days_30d",
    "frp_ratio_to_history",
    "active_days_fraction_30d",
]


OSM = [
    "distance_to_strong_industrial_km",
    "distance_to_supporting_industrial_km",
    "distance_to_infrastructure_km",
    "distance_to_vegetation_km",
    "strong_industrial_vs_vegetation_distance_km",
]


SENTINEL = [
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "NBR",
    "NDMI",
    "satellite_evidence_available",
]


EXPERIMENTS = {
    "firms_only":
        FIRMS,

    "firms_temporal":
        FIRMS + TEMPORAL,

    "firms_osm":
        FIRMS + OSM,

    "firms_sentinel":
        FIRMS + SENTINEL,

    "firms_temporal_osm":
        FIRMS + TEMPORAL + OSM,

    "firms_temporal_sentinel":
        FIRMS + TEMPORAL + SENTINEL,

    "firms_osm_sentinel":
        FIRMS + OSM + SENTINEL,

    "full_fusion":
        FIRMS + TEMPORAL + OSM + SENTINEL,
}


# ============================================================
# START
# ============================================================

print("=" * 90)
print("FIREWATCH — STEP 24")
print("FINAL FEATURE-GROUP ARCHITECTURE EXPERIMENT")
print("=" * 90)

print(
    "\nPURPOSE:"
    "\n- Compare all meaningful feature-group combinations."
    "\n- Same frozen CatBoost configuration."
    "\n- Same frozen spatial train/validation split."
    "\n- Calibration data is NOT used."
    "\n- 2025 holdout is NOT used."
    "\n- This is the final architecture experiment before freeze."
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    DATA_FILE
)


train_df = df[
    df["split"].eq("train")
].copy()


val_df = df[
    df["split"].eq("validation")
].copy()


cal_df = df[
    df["split"].eq("calibration")
].copy()


print(
    f"\nTrain       : {len(train_df):,}"
)

print(
    f"Validation  : {len(val_df):,}"
)

print(
    f"Calibration : {len(cal_df):,} "
    "(NOT USED)"
)


# ============================================================
# LABELS
# ============================================================

y_train = (
    train_df["source_class"]
    .map(CLASS_TO_ID)
    .astype(int)
    .to_numpy()
)


y_val = (
    val_df["source_class"]
    .map(CLASS_TO_ID)
    .astype(int)
    .to_numpy()
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.array(
    [0, 1, 2]
)


weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)


weight_map = {
    int(c): float(w)
    for c, w in zip(
        classes,
        weights
    )
}


sample_weights = np.array([
    weight_map[int(label)]
    for label in y_train
])


print(
    "\nClass weights:"
)

for class_id in classes:

    print(
        f"  {CLASS_NAMES[class_id]:<30}"
        f"{weight_map[class_id]:.4f}"
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []


for experiment_name, features in (
    EXPERIMENTS.items()
):

    print("\n" + "-" * 90)

    print(
        experiment_name.upper()
    )

    print("-" * 90)

    print(
        f"Feature count: "
        f"{len(features)}"
    )


    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train = train_df[
        features
    ].copy()


    X_val = val_df[
        features
    ].copy()


    for column in features:

        if X_train[column].dtype == bool:

            X_train[column] = (
                X_train[column]
                .astype(int)
            )

            X_val[column] = (
                X_val[column]
                .astype(int)
            )


    # Train-only medians
    medians = (
        X_train
        .median(
            numeric_only=True
        )
    )


    X_train = X_train.fillna(
        medians
    )


    X_val = X_val.fillna(
        medians
    )


    if X_train.isna().any().any():

        raise RuntimeError(
            f"Training NaNs remain "
            f"for {experiment_name}"
        )


    if X_val.isna().any().any():

        raise RuntimeError(
            f"Validation NaNs remain "
            f"for {experiment_name}"
        )


    # --------------------------------------------------------
    # Frozen CatBoost configuration
    # --------------------------------------------------------

    model = CatBoostClassifier(
        iterations=900,
        depth=7,
        learning_rate=0.025,
        l2_leaf_reg=5,
        random_strength=1,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


    start = time.perf_counter()


    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )


    training_seconds = (
        time.perf_counter()
        - start
    )


    y_pred = (
        model.predict(
            X_val
        )
        .reshape(-1)
        .astype(int)
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y_val,
            y_pred
        )
    )


    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_val,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )


    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=[0, 1, 2],
    )


    result = {
        "experiment":
            experiment_name,

        "feature_count":
            len(features),

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_f1":
            float(macro_f1),

        "industrial_precision":
            float(precision[0]),

        "industrial_recall":
            float(recall[0]),

        "industrial_f1":
            float(f1[0]),

        "natural_precision":
            float(precision[1]),

        "natural_recall":
            float(recall[1]),

        "natural_f1":
            float(f1[1]),

        "other_precision":
            float(precision[2]),

        "other_recall":
            float(recall[2]),

        "other_f1":
            float(f1[2]),

        "training_seconds":
            float(training_seconds),
    }


    results.append(
        result
    )


    # --------------------------------------------------------
    # Save experiment
    # --------------------------------------------------------

    experiment_dir = (
        OUTPUT_DIR
        / experiment_name
    )


    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        experiment_dir
        / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


    with open(
        experiment_dir
        / "features.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            features,
            f,
            indent=2
        )


    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )


    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"


    cm_df.to_csv(
        experiment_dir
        / "confusion_matrix.csv"
    )


    print(
        f"\nAccuracy             : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy    : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1             : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Industrial precision : "
        f"{precision[0]:.4f}"
    )

    print(
        f"Industrial recall    : "
        f"{recall[0]:.4f}"
    )

    print(
        f"Industrial F1        : "
        f"{f1[0]:.4f}"
    )

    print(
        f"Natural F1           : "
        f"{f1[1]:.4f}"
    )

    print(
        f"Other F1             : "
        f"{f1[2]:.4f}"
    )


# ============================================================
# COMPARISON
# ============================================================

comparison = pd.DataFrame(
    results
)


comparison.to_csv(
    OUTPUT_DIR
    / "final_feature_group_comparison.csv",
    index=False,
)


print("\n" + "=" * 90)
print("FINAL FEATURE-GROUP COMPARISON")
print("=" * 90)


display_columns = [
    "experiment",
    "feature_count",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "industrial_precision",
    "industrial_recall",
    "industrial_f1",
    "natural_f1",
    "other_f1",
]


print(
    comparison[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# INCREMENTAL DIFFERENCES
# ============================================================

lookup = (
    comparison
    .set_index(
        "experiment"
    )
)


print("\n" + "=" * 90)
print("KEY ARCHITECTURE DIFFERENCES")
print("=" * 90)


pairs = [
    (
        "firms_only",
        "firms_temporal",
        "Temporal added to FIRMS"
    ),
    (
        "firms_only",
        "firms_osm",
        "OSM added to FIRMS"
    ),
    (
        "firms_only",
        "firms_sentinel",
        "Sentinel added to FIRMS"
    ),
    (
        "firms_osm",
        "firms_temporal_osm",
        "Temporal added when OSM present"
    ),
    (
        "firms_temporal",
        "firms_temporal_osm",
        "OSM added when temporal present"
    ),
    (
        "firms_temporal_osm",
        "full_fusion",
        "Sentinel added to FIRMS+Temporal+OSM"
    ),
    (
        "firms_osm",
        "firms_osm_sentinel",
        "Sentinel added to FIRMS+OSM"
    ),
]


difference_rows = []


for before, after, description in pairs:

    macro_delta = (
        lookup.loc[
            after,
            "macro_f1"
        ]
        -
        lookup.loc[
            before,
            "macro_f1"
        ]
    )


    industrial_delta = (
        lookup.loc[
            after,
            "industrial_f1"
        ]
        -
        lookup.loc[
            before,
            "industrial_f1"
        ]
    )


    difference_rows.append({
        "comparison":
            description,

        "macro_f1_change":
            float(macro_delta),

        "industrial_f1_change":
            float(
                industrial_delta
            ),
    })


difference_df = pd.DataFrame(
    difference_rows
)


difference_df.to_csv(
    OUTPUT_DIR
    / "architecture_differences.csv",
    index=False,
)


print(
    difference_df.to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("STEP 24 COMPLETE")
print("=" * 90)

print(
    "\nSaved:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCALIBRATION SET USED: NO"
)

print(
    "2025 HOLDOUT USED: NO"
)

print(
    "\nDo not change the architecture "
    "after inspecting 2025."
)