from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    log_loss,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "source_labels_v1_2025.parquet"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "production"
    / "source_classifier_v1_final"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "model_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_ROWS = 20_000

CLASS_NAMES = [
    "industrial_fire_candidate",
    "natural_vegetation_fire",
    "other_uncertain",
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}

# FROZEN DEVELOPMENT OPERATING POINT.
INDUSTRIAL_THRESHOLD = 0.27

EPSILON = 1e-12


# ============================================================
# HELPERS
# ============================================================

def multiclass_brier(
    y_true,
    probabilities,
    n_classes=3,
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


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10,
):
    """
    Multiclass top-label ECE.

    Confidence = maximum predicted probability.
    Correctness = whether argmax class is correct.
    """

    predicted = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    correct = (
        predicted == y_true
    ).astype(float)

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    rows = []

    for i in range(n_bins):

        lower = bins[i]
        upper = bins[i + 1]

        if i == n_bins - 1:
            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin_lower": lower,
                "bin_upper": upper,
                "count": 0,
                "mean_confidence": None,
                "accuracy": None,
                "absolute_gap": None,
            })
            continue

        mean_confidence = float(
            confidence[mask].mean()
        )

        bin_accuracy = float(
            correct[mask].mean()
        )

        gap = abs(
            mean_confidence
            - bin_accuracy
        )

        ece += (
            count
            / len(y_true)
            * gap
        )

        rows.append({
            "bin_lower":
                float(lower),

            "bin_upper":
                float(upper),

            "count":
                count,

            "mean_confidence":
                mean_confidence,

            "accuracy":
                bin_accuracy,

            "absolute_gap":
                float(gap),
        })

    return float(ece), pd.DataFrame(rows)


def evaluate_multiclass(
    y_true,
    probabilities,
):
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1, 2],
    )

    metrics = {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    predictions,
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

        "per_class": {},
    }

    for i, name in enumerate(
        CLASS_NAMES
    ):
        metrics[
            "per_class"
        ][name] = {
            "precision":
                float(precision[i]),

            "recall":
                float(recall[i]),

            "f1":
                float(f1[i]),

            "support":
                int(support[i]),
        }

    return (
        metrics,
        predictions,
        matrix,
    )


# ============================================================
# START
# ============================================================

print("=" * 90)
print("FIREWATCH — STEP 34")
print("ONE-TIME FINAL 2025 TEMPORAL HOLDOUT EVALUATION")
print("=" * 90)

print(
    "\nIMPORTANT:"
    "\n- Frozen production artifacts only."
    "\n- No training."
    "\n- No hyperparameter tuning."
    "\n- No feature changes."
    "\n- No preprocessing fitted on 2025."
    "\n- No threshold tuning."
    "\n- 2025 weak labels remain unchanged."
)


# ============================================================
# 1. LOAD HOLDOUT
# ============================================================

print("\n[1/10] Loading final 2025 holdout...")


if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Holdout file missing:\n{DATA_FILE}"
    )


df = pd.read_parquet(
    DATA_FILE
)


print(
    f"      Rows: "
    f"{len(df):,}"
)


if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,}, "
        f"found {len(df):,}."
    )


if df["observation_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate observation IDs."
    )


# ============================================================
# 2. LOAD FROZEN ARTIFACT CONTRACT
# ============================================================

print("\n[2/10] Loading frozen production contract...")


feature_spec_file = (
    MODEL_DIR
    / "feature_spec.json"
)

freeze_file = (
    MODEL_DIR
    / "FREEZE_MANIFEST.json"
)

summary_file = (
    ROOT
    / "reports"
    / "final"
    / "source_classifier_v1"
    / "final_model_summary.json"
)


for path in [
    feature_spec_file,
    freeze_file,
    summary_file,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen artifact missing:\n{path}"
        )


with open(
    feature_spec_file,
    "r",
    encoding="utf-8",
) as f:
    feature_spec = json.load(f)


with open(
    freeze_file,
    "r",
    encoding="utf-8",
) as f:
    freeze_manifest = json.load(f)


with open(
    summary_file,
    "r",
    encoding="utf-8",
) as f:
    development_summary = json.load(f)


FEATURES = feature_spec[
    "features"
]

ARTIFACT_CLASSES = feature_spec[
    "classes"
]


if len(FEATURES) != 26:
    raise RuntimeError(
        f"Frozen feature count is "
        f"{len(FEATURES)}, expected 26."
    )


if ARTIFACT_CLASSES != CLASS_NAMES:
    raise RuntimeError(
        "Frozen class ordering does not match "
        "the expected production class ordering."
    )


if (
    freeze_manifest.get(
        "freeze_status"
    )
    != "LOCKED"
):
    raise RuntimeError(
        "Production model is not marked LOCKED."
    )


if (
    development_summary.get(
        "holdout_2025_used"
    )
    is not False
):
    raise RuntimeError(
        "Development summary does not confirm "
        "that 2025 was unused."
    )


print(
    "      Freeze status : LOCKED"
)

print(
    f"      Feature count : "
    f"{len(FEATURES)}"
)

print(
    "      Architecture  : "
    f"{feature_spec['architecture']}"
)


# ============================================================
# 3. VALIDATE FEATURES
# ============================================================

print("\n[3/10] Validating frozen 26 predictors...")


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:
    raise RuntimeError(
        "2025 holdout is missing frozen predictors:\n"
        + "\n".join(
            missing_features
        )
    )


print(
    "      All frozen predictors present."
)


# ============================================================
# 4. LOAD MODEL / CALIBRATOR / MEDIANS
# ============================================================

print("\n[4/10] Loading frozen production artifacts...")


model_file = (
    MODEL_DIR
    / "source_classifier.joblib"
)

calibrator_file = (
    MODEL_DIR
    / "probability_calibrator.joblib"
)

median_file = (
    MODEL_DIR
    / "training_medians.joblib"
)


for path in [
    model_file,
    calibrator_file,
    median_file,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Production artifact missing:\n{path}"
        )


model = joblib.load(
    model_file
)

calibrator = joblib.load(
    calibrator_file
)

training_medians = joblib.load(
    median_file
)


print(
    "      Model      : loaded"
)

print(
    "      Calibrator : loaded"
)

print(
    "      Medians    : loaded"
)


# ============================================================
# 5. PREPROCESS USING TRAINING MEDIANS ONLY
# ============================================================

print("\n[5/10] Applying frozen preprocessing...")


X = df[
    FEATURES
].copy()


for column in FEATURES:

    if X[column].dtype == bool:
        X[column] = (
            X[column]
            .astype(int)
        )

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce",
    )


missing_before = int(
    X.isna().sum().sum()
)


print(
    f"      Missing feature values before fill: "
    f"{missing_before:,}"
)


missing_medians = [
    feature
    for feature in FEATURES
    if feature not in training_medians
]


if missing_medians:
    raise RuntimeError(
        "Frozen training medians missing for:\n"
        + "\n".join(
            missing_medians
        )
    )


X = X.fillna(
    training_medians
)


if X.isna().any().any():

    bad = X.columns[
        X.isna().any()
    ].tolist()

    raise RuntimeError(
        "NaN remains after frozen preprocessing:\n"
        + "\n".join(bad)
    )


print(
    "      Remaining missing values: 0"
)


# ============================================================
# 6. ENCODE FROZEN WEAK LABELS
# ============================================================

print("\n[6/10] Encoding frozen 2025 weak labels...")


encoded = (
    df[
        "source_class"
    ]
    .map(
        CLASS_TO_ID
    )
)


if encoded.isna().any():
    bad = (
        df.loc[
            encoded.isna(),
            "source_class",
        ]
        .unique()
        .tolist()
    )

    raise RuntimeError(
        f"Unknown labels: {bad}"
    )


y_true = (
    encoded
    .astype(int)
    .to_numpy()
)


print(
    "      Label counts:"
)

for i, name in enumerate(
    CLASS_NAMES
):
    print(
        f"        {name:<30}"
        f"{int((y_true == i).sum()):>6,}"
    )


# ============================================================
# 7. EXECUTE FROZEN MODEL
# ============================================================

print(
    "\n[7/10] EXECUTING FROZEN MODEL ON 2025..."
)


raw_probabilities = (
    model.predict_proba(
        X
    )
)


if raw_probabilities.shape != (
    EXPECTED_ROWS,
    3,
):
    raise RuntimeError(
        "Unexpected raw probability shape: "
        f"{raw_probabilities.shape}"
    )


log_probability_features = np.log(
    np.clip(
        raw_probabilities,
        EPSILON,
        1.0,
    )
)


calibrated_probabilities = (
    calibrator.predict_proba(
        log_probability_features
    )
)


if calibrated_probabilities.shape != (
    EXPECTED_ROWS,
    3,
):
    raise RuntimeError(
        "Unexpected calibrated probability shape: "
        f"{calibrated_probabilities.shape}"
    )


print(
    "      Frozen inference complete."
)


# ============================================================
# 8. FINAL MULTICLASS METRICS
# ============================================================

print(
    "\n[8/10] Calculating final holdout metrics..."
)


(
    raw_metrics,
    raw_predictions,
    raw_cm,
) = evaluate_multiclass(
    y_true,
    raw_probabilities,
)


(
    calibrated_metrics,
    calibrated_predictions,
    calibrated_cm,
) = evaluate_multiclass(
    y_true,
    calibrated_probabilities,
)


industrial_binary_true = (
    y_true == 0
).astype(int)


raw_industrial_pr_auc = float(
    average_precision_score(
        industrial_binary_true,
        raw_probabilities[:, 0],
    )
)


calibrated_industrial_pr_auc = float(
    average_precision_score(
        industrial_binary_true,
        calibrated_probabilities[:, 0],
    )
)


raw_ece, raw_reliability = (
    expected_calibration_error(
        y_true,
        raw_probabilities,
    )
)


calibrated_ece, calibrated_reliability = (
    expected_calibration_error(
        y_true,
        calibrated_probabilities,
    )
)


raw_metrics[
    "industrial_pr_auc"
] = raw_industrial_pr_auc

raw_metrics[
    "ece_top_label_10_bins"
] = raw_ece


calibrated_metrics[
    "industrial_pr_auc"
] = calibrated_industrial_pr_auc

calibrated_metrics[
    "ece_top_label_10_bins"
] = calibrated_ece


# ============================================================
# FROZEN INDUSTRIAL OPERATING POINT = 0.27
# ============================================================

threshold_prediction = (
    calibrated_probabilities[
        :,
        0
    ]
    >= INDUSTRIAL_THRESHOLD
).astype(int)


tn, fp, fn, tp = (
    confusion_matrix(
        industrial_binary_true,
        threshold_prediction,
        labels=[0, 1],
    )
    .ravel()
)


threshold_precision = (
    tp / (tp + fp)
    if tp + fp > 0
    else 0.0
)


threshold_recall = (
    tp / (tp + fn)
    if tp + fn > 0
    else 0.0
)


threshold_f1 = (
    2
    * threshold_precision
    * threshold_recall
    / (
        threshold_precision
        + threshold_recall
    )
    if (
        threshold_precision
        + threshold_recall
        > 0
    )
    else 0.0
)


threshold_metrics = {
    "threshold":
        INDUSTRIAL_THRESHOLD,

    "true_positive":
        int(tp),

    "false_positive":
        int(fp),

    "false_negative":
        int(fn),

    "true_negative":
        int(tn),

    "precision":
        float(threshold_precision),

    "recall":
        float(threshold_recall),

    "f1":
        float(threshold_f1),

    "predicted_industrial":
        int(tp + fp),
}


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_metrics(
    title,
    metrics,
    matrix,
):
    print(
        "\n" + "-" * 90
    )

    print(title)

    print(
        "-" * 90
    )

    print(
        f"Accuracy             : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy    : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1             : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Log loss             : "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Multiclass Brier     : "
        f"{metrics['multiclass_brier']:.4f}"
    )

    print(
        f"Industrial PR-AUC    : "
        f"{metrics['industrial_pr_auc']:.4f}"
    )

    print(
        f"Top-label ECE        : "
        f"{metrics['ece_top_label_10_bins']:.4f}"
    )

    print(
        "\nPer-class P / R / F1 / Support"
    )

    for name in CLASS_NAMES:

        m = metrics[
            "per_class"
        ][name]

        print(
            f"  {name:<30}"
            f"{m['precision']:.4f} / "
            f"{m['recall']:.4f} / "
            f"{m['f1']:.4f} / "
            f"{m['support']:,}"
        )

    print(
        "\nConfusion matrix"
    )

    print(
        "Rows = actual, columns = predicted"
    )

    print(
        pd.DataFrame(
            matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_string()
    )


print_metrics(
    "FINAL 2025 — RAW CATBOOST",
    raw_metrics,
    raw_cm,
)


print_metrics(
    "FINAL 2025 — CALIBRATED ARGMAX",
    calibrated_metrics,
    calibrated_cm,
)


print(
    "\n" + "-" * 90
)

print(
    "FINAL 2025 — FROZEN INDUSTRIAL THRESHOLD 0.27"
)

print(
    "-" * 90
)

print(
    f"Precision            : "
    f"{threshold_precision:.4f}"
)

print(
    f"Recall               : "
    f"{threshold_recall:.4f}"
)

print(
    f"F1                   : "
    f"{threshold_f1:.4f}"
)

print(
    f"TP / FP / FN / TN    : "
    f"{tp} / {fp} / {fn} / {tn}"
)

print(
    f"Predicted industrial : "
    f"{tp + fp:,}"
)


# ============================================================
# 9. SAVE PREDICTIONS / MATRICES / RELIABILITY
# ============================================================

print(
    "\n[9/10] Saving final evaluation artifacts..."
)


prediction_df = pd.DataFrame({
    "observation_id":
        df[
            "observation_id"
        ].astype(str).to_numpy(),

    "actual_class":
        df[
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

    "raw_predicted_class": [
        CLASS_NAMES[i]
        for i in raw_predictions
    ],

    "calibrated_predicted_class": [
        CLASS_NAMES[i]
        for i in calibrated_predictions
    ],

    "frozen_threshold_027_industrial_candidate":
        threshold_prediction.astype(bool),
})


prediction_df.to_csv(
    OUTPUT_DIR
    / "predictions_2025.csv",
    index=False,
)


raw_cm_df = pd.DataFrame(
    raw_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

raw_cm_df.index.name = (
    "actual_class"
)

raw_cm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix_raw.csv"
)


calibrated_cm_df = pd.DataFrame(
    calibrated_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

calibrated_cm_df.index.name = (
    "actual_class"
)

calibrated_cm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix_calibrated.csv"
)


raw_reliability.to_csv(
    OUTPUT_DIR
    / "reliability_raw.csv",
    index=False,
)


calibrated_reliability.to_csv(
    OUTPUT_DIR
    / "reliability_calibrated.csv",
    index=False,
)


# ============================================================
# 10. FINAL REPORT
# ============================================================

print(
    "\n[10/10] Writing final evaluation report..."
)


report = {
    "evaluation_status":
        "FINAL_ONE_TIME_2025_TEMPORAL_HOLDOUT",

    "model_version":
        feature_spec[
            "model_version"
        ],

    "freeze_status":
        freeze_manifest[
            "freeze_status"
        ],

    "evaluation_rows":
        int(len(df)),

    "feature_count":
        len(FEATURES),

    "architecture":
        feature_spec[
            "architecture"
        ],

    "label_counts": {
        name:
            int(
                (
                    df[
                        "source_class"
                    ]
                    == name
                ).sum()
            )
        for name in CLASS_NAMES
    },

    "raw_catboost":
        raw_metrics,

    "calibrated_argmax":
        calibrated_metrics,

    "frozen_industrial_threshold_0_27":
        threshold_metrics,

    "raw_confusion_matrix":
        raw_cm.tolist(),

    "calibrated_confusion_matrix":
        calibrated_cm.tolist(),

    "evaluation_notes": [
        (
            "The 2025 source labels are contextual "
            "weak labels, not confirmed real-world "
            "fire-cause ground truth."
        ),
        (
            "The evaluation sample was deliberately "
            "diversity/stratification sampled and "
            "does not estimate natural 2025 class prevalence."
        ),
        (
            "WorldCover 2021 is a static land-cover "
            "reference used for 2025 label evidence."
        ),
        (
            "WRI/GEM industrial references provide "
            "facility-proximity evidence but do not prove "
            "the cause of a FIRMS thermal anomaly."
        ),
        (
            "The 0.27 industrial operating threshold "
            "was frozen during development and was not "
            "optimized on the 2025 holdout."
        ),
        (
            "No model retraining, feature selection, "
            "hyperparameter tuning, preprocessing fitting, "
            "or label-policy changes were performed "
            "using these 2025 results."
        ),
    ],

    "post_evaluation_rule":
        (
            "Do not modify and re-evaluate this model "
            "on the same 2025 holdout as a new final result."
        ),
}


with open(
    OUTPUT_DIR
    / "final_2025_evaluation.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


print(
    "\n" + "=" * 90
)

print(
    "STEP 34 COMPLETE — FINAL 2025 EVALUATION RECORDED"
)

print(
    "=" * 90
)


print(
    f"\nResults directory:"
    f"\n{OUTPUT_DIR}"
)


print(
    "\nTHIS HOLDOUT IS NOW SPENT."
)

print(
    "\nDo NOT tune the model, features, labels, "
    "preprocessing or threshold using these results "
    "and then report another evaluation on this same "
    "2025 sample as independent final performance."
)