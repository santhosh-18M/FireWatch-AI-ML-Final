from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    log_loss,
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

FROZEN_MODEL_FILE = (
    ROOT
    / "models"
    / "experiments"
    / "catboost_tuning_v1"
    / "slow_boost"
    / "model.cbm"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "experiments"
    / "calibration_v1"
)

PRODUCTION_DIR = (
    ROOT
    / "models"
    / "production"
    / "source_classifier_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PRODUCTION_DIR.mkdir(
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

RANDOM_STATE = 42

EPSILON = 1e-12


# ============================================================
# HELPERS
# ============================================================

def save_json(path, obj):

    def convert(x):

        if isinstance(x, dict):
            return {
                str(k): convert(v)
                for k, v in x.items()
            }

        if isinstance(x, list):
            return [
                convert(v)
                for v in x
            ]

        if isinstance(x, tuple):
            return [
                convert(v)
                for v in x
            ]

        if isinstance(x, np.integer):
            return int(x)

        if isinstance(x, np.floating):
            return float(x)

        if isinstance(x, np.bool_):
            return bool(x)

        return x

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            convert(obj),
            f,
            indent=2
        )


def multiclass_brier(
    y_true,
    probabilities,
    n_classes=3
):

    one_hot = np.eye(
        n_classes
    )[y_true]

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                ) ** 2,
                axis=1,
            )
        )
    )


def calculate_metrics(
    y_true,
    probabilities
):

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    return {
        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_f1":
            float(macro_f1),

        "log_loss":
            float(
                log_loss(
                    y_true,
                    probabilities,
                    labels=[0, 1, 2],
                )
            ),

        "multiclass_brier":
            multiclass_brier(
                y_true,
                probabilities,
            ),

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

        "support": {
            CLASS_NAMES[i]:
                int(support[i])
            for i in range(3)
        }
    }


def reliability_table(
    y_true_binary,
    probabilities,
    bins=10
):

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1
    )

    rows = []

    for i in range(bins):

        low = edges[i]
        high = edges[i + 1]

        if i == bins - 1:

            mask = (
                (probabilities >= low)
                &
                (probabilities <= high)
            )

        else:

            mask = (
                (probabilities >= low)
                &
                (probabilities < high)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        mean_probability = float(
            probabilities[
                mask
            ].mean()
        )

        observed_fraction = float(
            y_true_binary[
                mask
            ].mean()
        )

        rows.append({
            "bin_lower":
                float(low),

            "bin_upper":
                float(high),

            "count":
                count,

            "mean_probability":
                mean_probability,

            "observed_fraction":
                observed_fraction,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 21")
print("FROZEN CATBOOST PROBABILITY CALIBRATION")
print("=" * 80)

print(
    "\nFrozen model:"
    "\n  CatBoost slow_boost"
    "\n  iterations=900"
    "\n  depth=7"
    "\n  learning_rate=0.025"
    "\n  l2_leaf_reg=5"
)

print(
    "\nIMPORTANT:"
    "\n- Model hyperparameters are frozen."
    "\n- Calibration partition is now unlocked for calibration."
    "\n- Validation is NOT used to fit calibration."
    "\n- 2025 remains completely untouched."
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n[1/8] Loading frozen dataset..."
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


features = feature_spec[
    "features"
]


train_df = df[
    df["split"].eq("train")
].copy()

validation_df = df[
    df["split"].eq("validation")
].copy()

calibration_df = df[
    df["split"].eq("calibration")
].copy()


print(
    f"      Train       : "
    f"{len(train_df):,}"
)

print(
    f"      Validation  : "
    f"{len(validation_df):,}"
)

print(
    f"      Calibration : "
    f"{len(calibration_df):,}"
)

print(
    f"      Features    : "
    f"{len(features)}"
)


# ============================================================
# PREPROCESSING
# ============================================================

print(
    "\n[2/8] Reconstructing frozen preprocessing..."
)


X_train = train_df[
    features
].copy()

X_cal = calibration_df[
    features
].copy()


for column in features:

    if X_train[column].dtype == bool:

        X_train[column] = (
            X_train[column]
            .astype(int)
        )

        X_cal[column] = (
            X_cal[column]
            .astype(int)
        )


# IMPORTANT:
# Same strategy used during training:
# medians derived from TRAIN only.

train_medians = (
    X_train
    .median(
        numeric_only=True
    )
)


X_cal = X_cal.fillna(
    train_medians
)


if X_cal.isna().any().any():

    bad = (
        X_cal.columns[
            X_cal.isna().any()
        ]
        .tolist()
    )

    raise RuntimeError(
        "Calibration NaNs remain:\n"
        + "\n".join(bad)
    )


y_cal = (
    calibration_df[
        "source_class"
    ]
    .map(CLASS_TO_ID)
    .astype(int)
    .to_numpy()
)


print(
    "      Train-derived median "
    "preprocessing restored."
)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

print(
    "\n[3/8] Loading frozen slow_boost model..."
)


model = CatBoostClassifier()

model.load_model(
    FROZEN_MODEL_FILE
)


print(
    "      Frozen model loaded."
)


# ============================================================
# RAW CALIBRATION PROBABILITIES
# ============================================================

print(
    "\n[4/8] Predicting calibration partition..."
)


raw_probabilities = (
    model.predict_proba(
        X_cal
    )
)


raw_probabilities = np.asarray(
    raw_probabilities,
    dtype=float,
)


raw_metrics = calculate_metrics(
    y_cal,
    raw_probabilities,
)


print(
    "\nRAW probability behavior "
    "on calibration partition:"
)

print(
    f"  Accuracy          : "
    f"{raw_metrics['accuracy']:.4f}"
)

print(
    f"  Macro F1          : "
    f"{raw_metrics['macro_f1']:.4f}"
)

print(
    f"  Log loss          : "
    f"{raw_metrics['log_loss']:.4f}"
)

print(
    f"  Multiclass Brier  : "
    f"{raw_metrics['multiclass_brier']:.4f}"
)

print(
    f"  Industrial P/R/F1 : "
    f"{raw_metrics['industrial_precision']:.4f} / "
    f"{raw_metrics['industrial_recall']:.4f} / "
    f"{raw_metrics['industrial_f1']:.4f}"
)


# ============================================================
# MULTINOMIAL LOGISTIC CALIBRATOR
# ============================================================

print(
    "\n[5/8] Fitting multinomial logistic calibrator..."
)


# Convert probabilities to log-probability features.
# Using all 3 log probabilities lets the logistic model
# learn a conservative multiclass recalibration.

clipped_raw = np.clip(
    raw_probabilities,
    EPSILON,
    1.0
)


log_probability_features = (
    np.log(
        clipped_raw
    )
)


calibrator = LogisticRegression(
    max_iter=5000,
    random_state=RANDOM_STATE,
)


calibrator.fit(
    log_probability_features,
    y_cal
)


calibrated_probabilities = (
    calibrator.predict_proba(
        log_probability_features
    )
)


calibrated_metrics = (
    calculate_metrics(
        y_cal,
        calibrated_probabilities,
    )
)


print(
    "\nCALIBRATED apparent behavior "
    "on calibration partition:"
)

print(
    f"  Accuracy          : "
    f"{calibrated_metrics['accuracy']:.4f}"
)

print(
    f"  Macro F1          : "
    f"{calibrated_metrics['macro_f1']:.4f}"
)

print(
    f"  Log loss          : "
    f"{calibrated_metrics['log_loss']:.4f}"
)

print(
    f"  Multiclass Brier  : "
    f"{calibrated_metrics['multiclass_brier']:.4f}"
)

print(
    f"  Industrial P/R/F1 : "
    f"{calibrated_metrics['industrial_precision']:.4f} / "
    f"{calibrated_metrics['industrial_recall']:.4f} / "
    f"{calibrated_metrics['industrial_f1']:.4f}"
)

print(
    "\nNOTE:"
    "\nThese calibrated metrics are apparent calibration-set "
    "metrics because the calibrator was fitted on this same "
    "partition. They are NOT independent final performance."
)


# ============================================================
# INDUSTRIAL RELIABILITY TABLES
# ============================================================

print(
    "\n[6/8] Building Industrial reliability diagnostics..."
)


industrial_true = (
    y_cal == 0
).astype(int)


raw_reliability = reliability_table(
    industrial_true,
    raw_probabilities[:, 0],
    bins=10,
)


calibrated_reliability = reliability_table(
    industrial_true,
    calibrated_probabilities[:, 0],
    bins=10,
)


raw_reliability.to_csv(
    OUTPUT_DIR
    / "industrial_reliability_raw.csv",
    index=False,
)


calibrated_reliability.to_csv(
    OUTPUT_DIR
    / "industrial_reliability_calibrated_apparent.csv",
    index=False,
)


# Plot raw vs calibrated apparent reliability.

fig, ax = plt.subplots(
    figsize=(8, 7)
)


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)


if len(raw_reliability):

    ax.plot(
        raw_reliability[
            "mean_probability"
        ],
        raw_reliability[
            "observed_fraction"
        ],
        marker="o",
        label="Raw CatBoost",
    )


if len(calibrated_reliability):

    ax.plot(
        calibrated_reliability[
            "mean_probability"
        ],
        calibrated_reliability[
            "observed_fraction"
        ],
        marker="o",
        label="Calibrated (apparent)",
    )


ax.set_xlabel(
    "Mean predicted Industrial probability"
)

ax.set_ylabel(
    "Observed Industrial fraction"
)

ax.set_title(
    "Industrial Candidate Reliability"
)

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    0,
    1
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()


fig.savefig(
    OUTPUT_DIR
    / "industrial_reliability.png",
    dpi=200,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# CALIBRATION-SET INDUSTRIAL OPERATING POINTS
# ============================================================

print(
    "\n[7/8] Inspecting calibrated Industrial operating points..."
)


thresholds = np.arange(
    0.02,
    0.81,
    0.01
)


threshold_rows = []


for threshold in thresholds:

    predicted = (
        calibrated_probabilities[
            :,
            0
        ]
        >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            industrial_true,
            predicted,
            labels=[0, 1],
        )
        .ravel()
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


    threshold_rows.append({

        "threshold":
            round(
                float(threshold),
                2
            ),

        "true_positive":
            int(tp),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_negative":
            int(tn),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "predicted_industrial":
            int(tp + fp),
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


threshold_df.to_csv(
    OUTPUT_DIR
    / "calibrated_industrial_operating_points.csv",
    index=False,
)


best_index = (
    threshold_df[
        "f1"
    ].idxmax()
)


best = threshold_df.loc[
    best_index
]


print(
    "\nHighest apparent Industrial F1 "
    "on calibration partition:"
)

print(
    f"  Threshold      : "
    f"{best['threshold']:.2f}"
)

print(
    f"  Precision      : "
    f"{best['precision']:.4f}"
)

print(
    f"  Recall         : "
    f"{best['recall']:.4f}"
)

print(
    f"  F1             : "
    f"{best['f1']:.4f}"
)

print(
    f"  TP / FP / FN   : "
    f"{int(best['true_positive'])} / "
    f"{int(best['false_positive'])} / "
    f"{int(best['false_negative'])}"
)


# Useful fixed operating points to inspect.

fixed_thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]


fixed = threshold_df[
    threshold_df[
        "threshold"
    ].isin(
        fixed_thresholds
    )
].copy()


fixed.to_csv(
    OUTPUT_DIR
    / "important_calibrated_thresholds.csv",
    index=False,
)


print(
    "\nIMPORTANT CALIBRATED THRESHOLDS"
)

print(
    fixed[
        [
            "threshold",
            "true_positive",
            "false_positive",
            "precision",
            "recall",
            "f1",
            "predicted_industrial",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

print(
    "\n[8/8] Saving calibration artifacts..."
)


# Save calibrator

joblib.dump(
    calibrator,
    PRODUCTION_DIR
    / "probability_calibrator.joblib"
)


# Copy/save preprocessing medians

joblib.dump(
    train_medians.to_dict(),
    PRODUCTION_DIR
    / "training_medians.joblib"
)


# Feature specification

with open(
    PRODUCTION_DIR
    / "feature_spec.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "features":
                features,

            "class_names":
                CLASS_NAMES,

            "source_model":
                "CatBoost slow_boost",

            "iterations":
                900,

            "depth":
                7,

            "learning_rate":
                0.025,

            "l2_leaf_reg":
                5,

            "random_strength":
                1,
        },
        f,
        indent=2
    )


# Save a production copy of the frozen CatBoost model.

production_model = (
    PRODUCTION_DIR
    / "source_classifier.cbm"
)

model.save_model(
    production_model
)


# Save calibration predictions

prediction_df = pd.DataFrame({

    "observation_id":
        calibration_df[
            "observation_id"
        ].to_numpy(),

    "actual_class":
        calibration_df[
            "source_class"
        ].to_numpy(),

    "raw_prob_industrial":
        raw_probabilities[:, 0],

    "raw_prob_natural":
        raw_probabilities[:, 1],

    "raw_prob_other":
        raw_probabilities[:, 2],

    "calibrated_prob_industrial":
        calibrated_probabilities[:, 0],

    "calibrated_prob_natural":
        calibrated_probabilities[:, 1],

    "calibrated_prob_other":
        calibrated_probabilities[:, 2],
})


prediction_df.to_csv(
    OUTPUT_DIR
    / "calibration_predictions.csv",
    index=False,
)


summary = {

    "model":
        "CatBoost slow_boost",

    "calibration_method":
        (
            "multinomial logistic regression "
            "on log class probabilities"
        ),

    "calibration_rows":
        len(calibration_df),

    "raw_metrics_on_calibration":
        raw_metrics,

    "calibrated_apparent_metrics_on_calibration":
        calibrated_metrics,

    "best_apparent_industrial_threshold":
        {
            key:
                (
                    float(value)
                    if isinstance(
                        value,
                        (float, np.floating)
                    )
                    else int(value)
                )

            for key, value
            in best.to_dict().items()
        },

    "warning":
        (
            "The calibrator was fitted on the calibration "
            "partition. Calibrated metrics on that same "
            "partition are apparent metrics, not independent "
            "final performance. Independent evaluation must "
            "use the locked 2025 holdout after the complete "
            "pipeline is frozen."
        ),

    "holdout_2025_used":
        False,
}


save_json(
    OUTPUT_DIR
    / "calibration_summary.json",
    summary,
)


print("\n" + "=" * 80)
print("STEP 21 COMPLETE")
print("=" * 80)

print(
    "\nProduction candidate artifacts:"
)

print(
    PRODUCTION_DIR
)

print(
    "\n2025 HOLDOUT USED: NO"
)

print(
    "\nIMPORTANT:"
    "\nDo not report the apparent calibrated "
    "calibration-set metrics as final accuracy."
)