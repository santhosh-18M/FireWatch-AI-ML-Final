from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
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

FEATURE_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "training_features_v1.json"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "experiments"
    / "catboost_tuning_v1"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "experiments"
    / "catboost_tuning_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RANDOM_STATE = 42


CLASS_NAMES = [
    "industrial_fire_candidate",
    "natural_vegetation_fire",
    "other_uncertain",
]


CLASS_TO_ID = {
    name: i
    for i, name
    in enumerate(CLASS_NAMES)
}


# ============================================================
# CONTROLLED EXPERIMENTS
# ============================================================

EXPERIMENTS = {

    # Step-17 baseline reproduced for comparison
    "baseline": {
        "iterations": 500,
        "depth": 7,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "random_strength": 1,
    },

    # Shallower tree - stronger regularization
    "shallow_d6": {
        "iterations": 700,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5,
        "random_strength": 1,
    },

    # Slightly deeper representation
    "deep_d8": {
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5,
        "random_strength": 1,
    },

    # Stronger L2 regularization
    "regularized": {
        "iterations": 700,
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 8,
        "random_strength": 1,
    },

    # Lower learning rate / more boosting
    "slow_boost": {
        "iterations": 900,
        "depth": 7,
        "learning_rate": 0.025,
        "l2_leaf_reg": 5,
        "random_strength": 1,
    },

    # Slightly stronger randomization
    "randomized": {
        "iterations": 700,
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5,
        "random_strength": 2,
    },
}


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 20")
print("CONTROLLED CATBOOST TUNING")
print("=" * 80)

print(
    "\nRules:"
    "\n- Frozen source labels."
    "\n- Frozen 34 predictors."
    "\n- Frozen spatial train/validation split."
    "\n- Calibration set untouched."
    "\n- 2025 holdout untouched."
    "\n- No threshold optimization in this step."
)


df = pd.read_parquet(
    DATA_FILE
)


with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_spec = json.load(f)


features = feature_spec["features"]


train_df = df[
    df["split"].eq("train")
].copy()


val_df = df[
    df["split"].eq("validation")
].copy()


calibration_df = df[
    df["split"].eq("calibration")
].copy()


print(
    f"\nTrain       : {len(train_df):,}"
)

print(
    f"Validation  : {len(val_df):,}"
)

print(
    f"Calibration : {len(calibration_df):,} "
    "(untouched)"
)

print(
    f"Features    : {len(features)}"
)


# ============================================================
# FEATURES
# ============================================================

X_train = train_df[
    features
].copy()


X_val = val_df[
    features
].copy()


# Convert booleans safely
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


# Train-only medians.
train_medians = (
    X_train
    .median(
        numeric_only=True
    )
)


X_train = X_train.fillna(
    train_medians
)


X_val = X_val.fillna(
    train_medians
)


if X_train.isna().any().any():

    raise RuntimeError(
        "NaN remains in training features."
    )


if X_val.isna().any().any():

    raise RuntimeError(
        "NaN remains in validation features."
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


class_weight_dict = {
    int(c): float(w)
    for c, w
    in zip(classes, weights)
}


sample_weights = np.array(
    [
        class_weight_dict[
            int(label)
        ]
        for label in y_train
    ]
)


print(
    "\nClass weights:"
)

for class_id in classes:

    print(
        f"  "
        f"{CLASS_NAMES[class_id]:<30}"
        f"{class_weight_dict[class_id]:.4f}"
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []


for experiment_name, params in EXPERIMENTS.items():

    print("\n" + "-" * 80)

    print(
        f"EXPERIMENT: "
        f"{experiment_name}"
    )

    print("-" * 80)

    print(params)


    model = CatBoostClassifier(

        **params,

        loss_function="MultiClass",
        eval_metric="MultiClass",

        random_seed=RANDOM_STATE,

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
        model.predict(X_val)
        .reshape(-1)
        .astype(int)
    )


    probabilities = (
        model.predict_proba(
            X_val
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

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


    weighted_f1 = f1_score(
        y_val,
        y_pred,
        average="weighted",
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

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

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


    results.append(result)


    # ========================================================
    # SAVE EXPERIMENT
    # ========================================================

    experiment_dir = (
        OUTPUT_DIR
        / experiment_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    model_output_dir = (
        MODEL_DIR
        / experiment_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Metrics
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


    # Parameters
    with open(
        experiment_dir
        / "parameters.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            params,
            f,
            indent=2
        )


    # Confusion matrix
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


    # Predictions
    prediction_df = pd.DataFrame({

        "observation_id":
            val_df[
                "observation_id"
            ].to_numpy(),

        "actual_class":
            val_df[
                "source_class"
            ].to_numpy(),

        "predicted_class":
            [
                CLASS_NAMES[i]
                for i in y_pred
            ],

        "prob_industrial":
            probabilities[:, 0],

        "prob_natural":
            probabilities[:, 1],

        "prob_other":
            probabilities[:, 2],
    })


    prediction_df.to_csv(
        experiment_dir
        / "predictions.csv",
        index=False,
    )


    # Native model
    model.save_model(
        model_output_dir
        / "model.cbm"
    )


    joblib.dump(
        model,
        model_output_dir
        / "model.joblib"
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
        f"Training seconds     : "
        f"{training_seconds:.2f}"
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    results
)


comparison = comparison.sort_values(
    by=[
        "macro_f1",
        "industrial_f1",
    ],
    ascending=False,
).reset_index(
    drop=True
)


comparison.to_csv(
    OUTPUT_DIR
    / "catboost_tuning_comparison.csv",
    index=False,
)


print("\n" + "=" * 80)
print("CATBOOST TUNING COMPARISON")
print("=" * 80)


display_columns = [

    "experiment",

    "accuracy",

    "balanced_accuracy",

    "macro_f1",

    "industrial_precision",

    "industrial_recall",

    "industrial_f1",

    "natural_f1",

    "other_f1",

    "training_seconds",
]


print(
    comparison[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("STEP 20 COMPLETE")
print("=" * 80)

print(
    "\nNo production model "
    "has been selected automatically."
)

print(
    "\nCALIBRATION SET USED: NO"
)

print(
    "2025 HOLDOUT USED: NO"
)

print(
    "\nSaved results:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSaved experiment models:"
)

print(
    MODEL_DIR
)