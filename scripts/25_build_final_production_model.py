from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd

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
    ROOT / "data" / "processed"
    / "training_dataset_v1_spatial_split.parquet"
)

OUTPUT_DIR = (
    ROOT / "reports" / "final"
    / "source_classifier_v1"
)

MODEL_DIR = (
    ROOT / "models" / "production"
    / "source_classifier_v1_final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
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
# FINAL FROZEN 26 FEATURES
# ============================================================

FIRMS_FEATURES = [
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


TEMPORAL_FEATURES = [
    "detections_3d",
    "detections_7d",
    "detections_10d",
    "detections_30d",
    "distinct_active_days_30d",
    "frp_ratio_to_history",
    "active_days_fraction_30d",
]


OSM_FEATURES = [
    "distance_to_strong_industrial_km",
    "distance_to_supporting_industrial_km",
    "distance_to_infrastructure_km",
    "distance_to_vegetation_km",
    "strong_industrial_vs_vegetation_distance_km",
]


FEATURES = (
    FIRMS_FEATURES
    + TEMPORAL_FEATURES
    + OSM_FEATURES
)


assert len(FEATURES) == 26
assert len(set(FEATURES)) == 26


# ============================================================
# HELPERS
# ============================================================

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


def metrics_from_probabilities(
    y_true,
    probabilities
):
    y_pred = np.argmax(
        probabilities,
        axis=1
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
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

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

        "support":
            {
                CLASS_NAMES[i]:
                    int(support[i])
                for i in range(3)
            }
    }


# ============================================================
# START
# ============================================================

print("=" * 90)
print("FIREWATCH — STEP 25")
print("BUILD FINAL 26-FEATURE PRODUCTION MODEL")
print("=" * 90)

print(
    "\nFINAL ARCHITECTURE:"
    "\n  FIRMS thermal/time"
    "\n  + causal historical temporal"
    "\n  + OSM context"
    "\n  = 26 predictors"
)

print(
    "\nEXCLUDED FROM SOURCE CLASSIFIER:"
    "\n  Sentinel spectral features"
    "\n  WorldCover label evidence"
    "\n  WRI/GEM industrial reference distances"
    "\n  latitude / longitude"
    "\n  temporal_status"
)

print(
    "\n2025 HOLDOUT USED: NO"
)


# ============================================================
# LOAD FROZEN DEVELOPMENT DATA
# ============================================================

df = pd.read_parquet(
    DATA_FILE
)


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
    f"\nTrain       : {len(train_df):,}"
)

print(
    f"Validation  : {len(validation_df):,}"
)

print(
    f"Calibration : {len(calibration_df):,}"
)

print(
    f"Features    : {len(FEATURES)}"
)


# ============================================================
# PREPROCESSING
# ============================================================

X_train = train_df[
    FEATURES
].copy()


X_val = validation_df[
    FEATURES
].copy()


X_cal = calibration_df[
    FEATURES
].copy()


for column in FEATURES:

    if X_train[column].dtype == bool:

        X_train[column] = (
            X_train[column]
            .astype(int)
        )

        X_val[column] = (
            X_val[column]
            .astype(int)
        )

        X_cal[column] = (
            X_cal[column]
            .astype(int)
        )


# Fit preprocessing on TRAIN ONLY.
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

X_cal = X_cal.fillna(
    train_medians
)


for name, X in [
    ("train", X_train),
    ("validation", X_val),
    ("calibration", X_cal),
]:

    if X.isna().any().any():

        bad = X.columns[
            X.isna().any()
        ].tolist()

        raise RuntimeError(
            f"{name} still contains NaN: "
            f"{bad}"
        )


# ============================================================
# LABELS
# ============================================================

def encode_labels(frame):

    encoded = (
        frame["source_class"]
        .map(CLASS_TO_ID)
    )

    if encoded.isna().any():

        raise RuntimeError(
            "Unknown source_class found."
        )

    return (
        encoded
        .astype(int)
        .to_numpy()
    )


y_train = encode_labels(
    train_df
)

y_val = encode_labels(
    validation_df
)

y_cal = encode_labels(
    calibration_df
)


# ============================================================
# CLASS WEIGHTS — TRAIN ONLY
# ============================================================

classes = np.array(
    [0, 1, 2]
)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)


class_weight_map = {
    int(c): float(w)
    for c, w in zip(
        classes,
        class_weights
    )
}


sample_weights = np.array([
    class_weight_map[
        int(label)
    ]
    for label in y_train
])


print(
    "\nTraining class weights:"
)

for class_id in classes:

    print(
        f"  {CLASS_NAMES[class_id]:<30}"
        f"{class_weight_map[class_id]:.4f}"
    )


# ============================================================
# FINAL FROZEN MODEL
# ============================================================

print(
    "\nTraining final frozen "
    "26-feature CatBoost..."
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


model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
)


# ============================================================
# VALIDATION — RAW MODEL
# ============================================================

val_probabilities = (
    model.predict_proba(
        X_val
    )
)


validation_metrics = (
    metrics_from_probabilities(
        y_val,
        val_probabilities,
    )
)


print(
    "\nVALIDATION — FINAL 26-FEATURE MODEL"
)

print(
    f"  Accuracy             : "
    f"{validation_metrics['accuracy']:.4f}"
)

print(
    f"  Balanced accuracy    : "
    f"{validation_metrics['balanced_accuracy']:.4f}"
)

print(
    f"  Macro F1             : "
    f"{validation_metrics['macro_f1']:.4f}"
)

print(
    f"  Industrial precision : "
    f"{validation_metrics['industrial_precision']:.4f}"
)

print(
    f"  Industrial recall    : "
    f"{validation_metrics['industrial_recall']:.4f}"
)

print(
    f"  Industrial F1        : "
    f"{validation_metrics['industrial_f1']:.4f}"
)


# ============================================================
# CALIBRATION
# ============================================================

print(
    "\nFitting probability calibrator "
    "on calibration partition..."
)


raw_cal_probabilities = (
    model.predict_proba(
        X_cal
    )
)


raw_cal_metrics = (
    metrics_from_probabilities(
        y_cal,
        raw_cal_probabilities,
    )
)


EPSILON = 1e-12


log_cal_features = np.log(
    np.clip(
        raw_cal_probabilities,
        EPSILON,
        1.0,
    )
)


calibrator = LogisticRegression(
    max_iter=5000,
    random_state=42,
)


calibrator.fit(
    log_cal_features,
    y_cal,
)


calibrated_cal_probabilities = (
    calibrator.predict_proba(
        log_cal_features
    )
)


apparent_calibrated_metrics = (
    metrics_from_probabilities(
        y_cal,
        calibrated_cal_probabilities,
    )
)


print(
    "\nCALIBRATION PARTITION"
)

print(
    "Raw frozen-model probabilities:"
)

print(
    f"  Log loss           : "
    f"{raw_cal_metrics['log_loss']:.4f}"
)

print(
    f"  Brier              : "
    f"{raw_cal_metrics['multiclass_brier']:.4f}"
)

print(
    f"  Industrial P/R/F1  : "
    f"{raw_cal_metrics['industrial_precision']:.4f} / "
    f"{raw_cal_metrics['industrial_recall']:.4f} / "
    f"{raw_cal_metrics['industrial_f1']:.4f}"
)


print(
    "\nCalibrated apparent behavior:"
)

print(
    f"  Log loss           : "
    f"{apparent_calibrated_metrics['log_loss']:.4f}"
)

print(
    f"  Brier              : "
    f"{apparent_calibrated_metrics['multiclass_brier']:.4f}"
)

print(
    f"  Industrial P/R/F1  : "
    f"{apparent_calibrated_metrics['industrial_precision']:.4f} / "
    f"{apparent_calibrated_metrics['industrial_recall']:.4f} / "
    f"{apparent_calibrated_metrics['industrial_f1']:.4f}"
)


# ============================================================
# CALIBRATED INDUSTRIAL OPERATING POINTS
# ============================================================

industrial_true = (
    y_cal == 0
).astype(int)


threshold_rows = []


for threshold in np.arange(
    0.02,
    0.81,
    0.01
):

    prediction = (
        calibrated_cal_probabilities[
            :,
            0
        ]
        >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            industrial_true,
            prediction,
            labels=[0, 1],
        )
        .ravel()
    )


    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )


    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )


    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
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
    / "industrial_operating_points_calibration.csv",
    index=False,
)


best = threshold_df.loc[
    threshold_df[
        "f1"
    ].idxmax()
]


print(
    "\nHighest apparent Industrial "
    "candidate F1 on calibration:"
)

print(
    f"  Threshold : "
    f"{best['threshold']:.2f}"
)

print(
    f"  Precision : "
    f"{best['precision']:.4f}"
)

print(
    f"  Recall    : "
    f"{best['recall']:.4f}"
)

print(
    f"  F1        : "
    f"{best['f1']:.4f}"
)

print(
    f"  TP/FP/FN  : "
    f"{int(best['true_positive'])}/"
    f"{int(best['false_positive'])}/"
    f"{int(best['false_negative'])}"
)


# ============================================================
# SAVE FINAL ARTIFACTS
# ============================================================

print(
    "\nSaving FINAL production artifacts..."
)


model.save_model(
    MODEL_DIR
    / "source_classifier.cbm"
)


joblib.dump(
    model,
    MODEL_DIR
    / "source_classifier.joblib"
)


joblib.dump(
    calibrator,
    MODEL_DIR
    / "probability_calibrator.joblib"
)


joblib.dump(
    train_medians.to_dict(),
    MODEL_DIR
    / "training_medians.joblib"
)


feature_spec = {
    "model_version":
        "source_classifier_v1_final",

    "architecture":
        "FIRMS + causal temporal + OSM",

    "feature_count":
        26,

    "features":
        FEATURES,

    "feature_groups": {
        "firms":
            FIRMS_FEATURES,

        "temporal":
            TEMPORAL_FEATURES,

        "osm":
            OSM_FEATURES,
    },

    "excluded_from_classifier": [
        "Sentinel spectral features",
        "WorldCover label evidence",
        "WRI/GEM label-generating distances",
        "latitude",
        "longitude",
        "temporal_status",
    ],

    "classes":
        CLASS_NAMES,

    "catboost": {
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

        "random_seed":
            42,
    },

    "calibration":
        (
            "Multinomial logistic regression "
            "on log CatBoost probabilities"
        ),
}


with open(
    MODEL_DIR
    / "feature_spec.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_spec,
        f,
        indent=2
    )


# Validation predictions
val_prediction_df = pd.DataFrame({
    "observation_id":
        validation_df[
            "observation_id"
        ].to_numpy(),

    "actual_class":
        validation_df[
            "source_class"
        ].to_numpy(),

    "prob_industrial":
        val_probabilities[:, 0],

    "prob_natural":
        val_probabilities[:, 1],

    "prob_other":
        val_probabilities[:, 2],
})


val_prediction_df[
    "predicted_class"
] = [
    CLASS_NAMES[i]
    for i in np.argmax(
        val_probabilities,
        axis=1
    )
]


val_prediction_df.to_csv(
    OUTPUT_DIR
    / "validation_predictions.csv",
    index=False,
)


# Calibration predictions
cal_prediction_df = pd.DataFrame({
    "observation_id":
        calibration_df[
            "observation_id"
        ].to_numpy(),

    "actual_class":
        calibration_df[
            "source_class"
        ].to_numpy(),

    "raw_prob_industrial":
        raw_cal_probabilities[:, 0],

    "raw_prob_natural":
        raw_cal_probabilities[:, 1],

    "raw_prob_other":
        raw_cal_probabilities[:, 2],

    "calibrated_prob_industrial":
        calibrated_cal_probabilities[:, 0],

    "calibrated_prob_natural":
        calibrated_cal_probabilities[:, 1],

    "calibrated_prob_other":
        calibrated_cal_probabilities[:, 2],
})


cal_prediction_df.to_csv(
    OUTPUT_DIR
    / "calibration_predictions.csv",
    index=False,
)


summary = {
    "status":
        "FROZEN_BEFORE_2025",

    "model_version":
        "source_classifier_v1_final",

    "architecture":
        "FIRMS + causal temporal + OSM",

    "feature_count":
        26,

    "validation_metrics":
        validation_metrics,

    "raw_calibration_metrics":
        raw_cal_metrics,

    "apparent_calibrated_metrics":
        apparent_calibrated_metrics,

    "calibration_operating_point_diagnostic": {
        "threshold":
            float(
                best["threshold"]
            ),

        "precision":
            float(
                best["precision"]
            ),

        "recall":
            float(
                best["recall"]
            ),

        "f1":
            float(
                best["f1"]
            ),

        "true_positive":
            int(
                best["true_positive"]
            ),

        "false_positive":
            int(
                best["false_positive"]
            ),

        "false_negative":
            int(
                best["false_negative"]
            ),
    },

    "important_warning":
        (
            "Calibration-set calibrated metrics and threshold "
            "performance are apparent development metrics. "
            "Independent final evaluation is the locked 2025 "
            "temporal holdout."
        ),

    "holdout_2025_used":
        False,
}


with open(
    OUTPUT_DIR
    / "final_model_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# Freeze marker
freeze_manifest = {
    "freeze_status":
        "LOCKED",

    "model_version":
        "source_classifier_v1_final",

    "feature_architecture":
        "FIRMS + causal temporal + OSM",

    "feature_count":
        26,

    "model_family":
        "CatBoost",

    "model_configuration": {
        "iterations": 900,
        "depth": 7,
        "learning_rate": 0.025,
        "l2_leaf_reg": 5,
        "random_strength": 1,
        "random_seed": 42,
    },

    "development_complete":
        True,

    "allowed_next_step":
        (
            "Construct 2025 features using frozen rules "
            "and perform one-time temporal holdout evaluation."
        ),

    "prohibited_after_2025_inspection": [
        "changing source labels based on 2025 results",
        "changing model hyperparameters based on 2025 results",
        "changing predictor set based on 2025 results",
        "changing preprocessing based on 2025 results",
        "retraining to improve reported 2025 score",
    ],
}


with open(
    MODEL_DIR
    / "FREEZE_MANIFEST.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        freeze_manifest,
        f,
        indent=2
    )


print("\n" + "=" * 90)
print("STEP 25 COMPLETE — MODEL FROZEN")
print("=" * 90)

print(
    "\nFinal production artifacts:"
)

print(
    MODEL_DIR
)

print(
    "\nFinal report:"
)

print(
    OUTPUT_DIR
)

print(
    "\n2025 HOLDOUT USED: NO"
)

print(
    "\nDEVELOPMENT IS NOW FROZEN."
)

print(
    "\nNext:"
    "\nBuild the 2025 holdout using the exact frozen "
    "feature definitions without changing the model."
)