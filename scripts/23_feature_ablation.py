from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "training_dataset_v1_spatial_split.parquet"
)

OUTPUT_DIR = (
    ROOT / "reports" / "experiments"
    / "feature_ablation_v1"
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

    "firms_temporal_osm":
        FIRMS + TEMPORAL + OSM,

    "full_fusion":
        FIRMS + TEMPORAL + OSM + SENTINEL,
}


print("=" * 80)
print("FIREWATCH — STEP 23")
print("FEATURE-GROUP ABLATION")
print("=" * 80)

print(
    "\nFrozen CatBoost configuration:"
    "\niterations=900"
    "\ndepth=7"
    "\nlearning_rate=0.025"
    "\nl2_leaf_reg=5"
    "\nrandom_strength=1"
)

print(
    "\n2025 HOLDOUT USED: NO"
)


df = pd.read_parquet(
    DATA_FILE
)


train_df = df[
    df["split"].eq("train")
].copy()

val_df = df[
    df["split"].eq("validation")
].copy()


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


classes = np.array([0, 1, 2])


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
    weight_map[int(y)]
    for y in y_train
])


results = []


for name, features in EXPERIMENTS.items():

    print("\n" + "-" * 80)
    print(name.upper())
    print("-" * 80)

    print(
        f"Features: {len(features)}"
    )


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


    medians = X_train.median(
        numeric_only=True
    )


    X_train = X_train.fillna(
        medians
    )

    X_val = X_val.fillna(
        medians
    )


    if X_train.isna().any().any():
        raise RuntimeError(
            f"Training NaNs remain: {name}"
        )

    if X_val.isna().any().any():
        raise RuntimeError(
            f"Validation NaNs remain: {name}"
        )


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


    experiment_dir = (
        OUTPUT_DIR / name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.index.name = "actual"

    cm_df.to_csv(
        experiment_dir
        / "confusion_matrix.csv"
    )


    metrics = {
        "experiment":
            name,

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

        "natural_f1":
            float(f1[1]),

        "other_f1":
            float(f1[2]),

        "training_seconds":
            float(
                training_seconds
            ),
    }


    results.append(
        metrics
    )


    with open(
        experiment_dir
        / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
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


comparison = pd.DataFrame(
    results
)


comparison.to_csv(
    OUTPUT_DIR
    / "feature_ablation_comparison.csv",
    index=False,
)


print("\n" + "=" * 80)
print("FEATURE ABLATION COMPARISON")
print("=" * 80)


print(
    comparison[
        [
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
    ].to_string(
        index=False
    )
)


# ============================================================
# MACRO F1 PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.bar(
    comparison["experiment"],
    comparison["macro_f1"],
)


ax.set_ylabel(
    "Macro F1"
)

ax.set_xlabel(
    "Feature configuration"
)

ax.set_title(
    "FireWatch Feature Ablation — Macro F1"
)

ax.tick_params(
    axis="x",
    rotation=20
)

fig.tight_layout()


fig.savefig(
    OUTPUT_DIR
    / "macro_f1_ablation.png",
    dpi=200,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# INDUSTRIAL F1 PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.bar(
    comparison["experiment"],
    comparison["industrial_f1"],
)


ax.set_ylabel(
    "Industrial Candidate F1"
)

ax.set_xlabel(
    "Feature configuration"
)

ax.set_title(
    "FireWatch Feature Ablation — Industrial F1"
)

ax.tick_params(
    axis="x",
    rotation=20
)

fig.tight_layout()


fig.savefig(
    OUTPUT_DIR
    / "industrial_f1_ablation.png",
    dpi=200,
    bbox_inches="tight",
)


plt.close(fig)


print("\n" + "=" * 80)
print("STEP 23 COMPLETE")
print("=" * 80)

print(
    "\nSaved:"
)

print(
    OUTPUT_DIR
)

print(
    "\n2025 HOLDOUT USED: NO"
)

print(
    "\nNo production model was changed."
)