from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed"
    / "training_dataset_v1_spatial_split.parquet"
)

FEATURE_FILE = (
    ROOT / "reports" / "experiments"
    / "training_features_v1.json"
)

OUTPUT_ROOT = (
    ROOT / "reports" / "experiments"
    / "model_benchmark_v1"
)

MODEL_ROOT = (
    ROOT / "models" / "experiments"
    / "model_benchmark_v1"
)

RANDOM_STATE = 42


CLASS_NAMES = [
    "industrial_fire_candidate",
    "natural_vegetation_fire",
    "other_uncertain",
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}

ID_TO_CLASS = {
    i: name
    for name, i in CLASS_TO_ID.items()
}


# ============================================================
# HELPERS
# ============================================================

def make_dirs():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


def clean_for_json(obj):

    if isinstance(obj, dict):
        return {
            str(k): clean_for_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            clean_for_json(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return [
            clean_for_json(v)
            for v in obj
        ]

    if isinstance(
        obj,
        (
            np.integer,
        )
    ):
        return int(obj)

    if isinstance(
        obj,
        (
            np.floating,
        )
    ):
        return float(obj)

    if isinstance(
        obj,
        (
            np.bool_,
        )
    ):
        return bool(obj)

    return obj


def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            clean_for_json(data),
            f,
            indent=2
        )


def save_confusion_matrix(
    cm,
    model_name,
    output_dir
):

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"

    cm_df.to_csv(
        output_dir
        / "confusion_matrix.csv"
    )


    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(cm)

    ax.set_title(
        f"{model_name} - Validation Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "Actual class"
    )

    ax.set_xticks(
        np.arange(
            len(CLASS_NAMES)
        )
    )

    ax.set_yticks(
        np.arange(
            len(CLASS_NAMES)
        )
    )

    short_names = [
        "Industrial",
        "Natural",
        "Other",
    ]

    ax.set_xticklabels(
        short_names,
        rotation=20,
        ha="right",
    )

    ax.set_yticklabels(
        short_names
    )


    threshold = (
        cm.max() / 2.0
        if cm.size
        else 0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm[i, j]

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )


    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def evaluate_model(
    model_name,
    model,
    X_val,
    y_val,
    validation_meta,
    training_seconds,
    output_dir,
):

    print(
        f"\nEvaluating {model_name}..."
    )

    y_pred = model.predict(
        X_val
    )

    y_pred = np.asarray(
        y_pred
    ).reshape(-1)

    y_pred = y_pred.astype(int)


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                X_val
            )
        )

        probabilities = np.asarray(
            probabilities
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
            labels=[
                0,
                1,
                2,
            ],
            zero_division=0,
        )
    )


    per_class = {}

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        per_class[
            class_name
        ] = {
            "precision":
                float(
                    precision[i]
                ),

            "recall":
                float(
                    recall[i]
                ),

            "f1":
                float(
                    f1[i]
                ),

            "support":
                int(
                    support[i]
                ),
        }


    metrics = {
        "model":
            model_name,

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "weighted_f1":
            float(
                weighted_f1
            ),

        "industrial_precision":
            float(
                per_class[
                    "industrial_fire_candidate"
                ]["precision"]
            ),

        "industrial_recall":
            float(
                per_class[
                    "industrial_fire_candidate"
                ]["recall"]
            ),

        "industrial_f1":
            float(
                per_class[
                    "industrial_fire_candidate"
                ]["f1"]
            ),

        "natural_precision":
            float(
                per_class[
                    "natural_vegetation_fire"
                ]["precision"]
            ),

        "natural_recall":
            float(
                per_class[
                    "natural_vegetation_fire"
                ]["recall"]
            ),

        "natural_f1":
            float(
                per_class[
                    "natural_vegetation_fire"
                ]["f1"]
            ),

        "other_precision":
            float(
                per_class[
                    "other_uncertain"
                ]["precision"]
            ),

        "other_recall":
            float(
                per_class[
                    "other_uncertain"
                ]["recall"]
            ),

        "other_f1":
            float(
                per_class[
                    "other_uncertain"
                ]["f1"]
            ),

        "training_seconds":
            float(
                training_seconds
            ),

        "per_class":
            per_class,
    }


    save_json(
        output_dir
        / "metrics.json",
        metrics
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_val,
        y_pred,
        labels=[
            0,
            1,
            2,
        ],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(
            report
        ).transpose()
    )

    report_df.to_csv(
        output_dir
        / "classification_report.csv"
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=[
            0,
            1,
            2,
        ],
    )


    save_confusion_matrix(
        cm,
        model_name,
        output_dir,
    )


    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    predictions = (
        validation_meta.copy()
    )


    predictions[
        "actual_class_id"
    ] = y_val


    predictions[
        "actual_class"
    ] = [
        ID_TO_CLASS[int(v)]
        for v in y_val
    ]


    predictions[
        "predicted_class_id"
    ] = y_pred


    predictions[
        "predicted_class"
    ] = [
        ID_TO_CLASS[int(v)]
        for v in y_pred
    ]


    predictions[
        "correct"
    ] = (
        predictions[
            "actual_class_id"
        ]
        ==
        predictions[
            "predicted_class_id"
        ]
    )


    if probabilities is not None:

        if (
            probabilities.ndim == 2
            and
            probabilities.shape[1] == 3
        ):

            for i, class_name in enumerate(
                CLASS_NAMES
            ):

                predictions[
                    f"prob_{class_name}"
                ] = probabilities[
                    :,
                    i
                ]


            predictions[
                "max_probability"
            ] = probabilities.max(
                axis=1
            )


    predictions.to_csv(
        output_dir
        / "predictions.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Misclassifications only
    # --------------------------------------------------------

    predictions[
        ~predictions["correct"]
    ].to_csv(
        output_dir
        / "misclassifications.csv",
        index=False,
    )


    return metrics


# ============================================================
# LOAD DATA
# ============================================================

make_dirs()


print("=" * 80)
print("FIREWATCH — STEP 17")
print("THREE-MODEL SPATIAL VALIDATION BENCHMARK")
print("=" * 80)


print(
    "\nIMPORTANT:"
    "\n- Training split is used for fitting."
    "\n- Validation split is used for model comparison."
    "\n- Calibration split is NOT used."
    "\n- 2025 holdout is NOT used."
    "\n- All models use the same frozen features."
)


print("\n[1/7] Loading dataset...")

df = pd.read_parquet(
    DATA_FILE
)


with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    feature_spec = json.load(
        f
    )


features = feature_spec[
    "features"
]


print(
    f"      Dataset rows : "
    f"{len(df):,}"
)

print(
    f"      Features     : "
    f"{len(features)}"
)


missing_features = [
    f for f in features
    if f not in df.columns
]


if missing_features:

    raise RuntimeError(
        "Missing features:\n"
        + "\n".join(
            missing_features
        )
    )


# ============================================================
# SPLITS
# ============================================================

print("\n[2/7] Preparing frozen splits...")


train_df = df[
    df["split"].eq(
        "train"
    )
].copy()


val_df = df[
    df["split"].eq(
        "validation"
    )
].copy()


calibration_df = df[
    df["split"].eq(
        "calibration"
    )
].copy()


print(
    f"      Train       : "
    f"{len(train_df):,}"
)

print(
    f"      Validation  : "
    f"{len(val_df):,}"
)

print(
    f"      Calibration : "
    f"{len(calibration_df):,} "
    "(untouched)"
)


X_train = train_df[
    features
].copy()


X_val = val_df[
    features
].copy()


y_train = (
    train_df[
        "source_class"
    ]
    .map(
        CLASS_TO_ID
    )
    .astype(int)
    .to_numpy()
)


y_val = (
    val_df[
        "source_class"
    ]
    .map(
        CLASS_TO_ID
    )
    .astype(int)
    .to_numpy()
)


# Metadata saved with predictions.
validation_meta = val_df[
    [
        "observation_id",
        "latitude",
        "longitude",
        "spatial_group",
    ]
].reset_index(
    drop=True
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n[3/7] Preparing missing-value strategy...")


# Fit medians using TRAINING ONLY.
train_medians = (
    X_train
    .median(
        numeric_only=True
    )
)


X_train_imputed = (
    X_train
    .fillna(
        train_medians
    )
)


X_val_imputed = (
    X_val
    .fillna(
        train_medians
    )
)


if X_train_imputed.isna().any().any():

    bad = (
        X_train_imputed
        .columns[
            X_train_imputed
            .isna()
            .any()
        ]
        .tolist()
    )

    raise RuntimeError(
        "Training NaN remains in:\n"
        + "\n".join(bad)
    )


if X_val_imputed.isna().any().any():

    bad = (
        X_val_imputed
        .columns[
            X_val_imputed
            .isna()
            .any()
        ]
        .tolist()
    )

    raise RuntimeError(
        "Validation NaN remains in:\n"
        + "\n".join(bad)
    )


median_path = (
    OUTPUT_ROOT
    / "training_medians.json"
)


save_json(
    median_path,
    {
        str(k): float(v)
        for k, v
        in train_medians.items()
    }
)


print(
    "      Median imputation fitted "
    "on TRAIN only."
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n[4/7] Computing class weights...")


classes = np.array(
    [
        0,
        1,
        2,
    ]
)


weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)


class_weight_dict = {
    int(cls):
        float(weight)

    for cls, weight
    in zip(
        classes,
        weights
    )
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
    "      Class weights:"
)

for class_id in classes:

    print(
        f"        "
        f"{ID_TO_CLASS[int(class_id)]:<30} "
        f"{class_weight_dict[int(class_id)]:.4f}"
    )


# ============================================================
# MODELS
# ============================================================

print("\n[5/7] Creating baseline models...")


models = {}


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

models[
    "random_forest"
] = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

models[
    "xgboost"
] = XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=500,
    learning_rate=0.05,

    max_depth=6,
    min_child_weight=2,

    subsample=0.85,
    colsample_bytree=0.85,

    reg_alpha=0.1,
    reg_lambda=1.0,

    eval_metric="mlogloss",

    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ------------------------------------------------------------
# CATBOOST
# ------------------------------------------------------------

models[
    "catboost"
] = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,

    loss_function="MultiClass",
    eval_metric="MultiClass",

    random_seed=RANDOM_STATE,

    verbose=False,

    allow_writing_files=False,
)


print(
    "      Random Forest ready."
)

print(
    "      XGBoost ready."
)

print(
    "      CatBoost ready."
)


# ============================================================
# TRAIN + EVALUATE
# ============================================================

print("\n[6/7] Training models...")


all_metrics = []


for model_name, model in models.items():

    print("\n" + "-" * 80)
    print(
        f"TRAINING: "
        f"{model_name.upper()}"
    )
    print("-" * 80)


    model_output_dir = (
        OUTPUT_ROOT
        / model_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    model_dir = (
        MODEL_ROOT
        / model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    start = time.perf_counter()


    if model_name == "random_forest":

        model.fit(
            X_train_imputed,
            y_train,
        )


    elif model_name == "xgboost":

        model.fit(
            X_train_imputed,
            y_train,
            sample_weight=sample_weights,
        )


    elif model_name == "catboost":

        model.fit(
            X_train_imputed,
            y_train,
            sample_weight=sample_weights,
        )


    training_seconds = (
        time.perf_counter()
        - start
    )


    print(
        f"Training time: "
        f"{training_seconds:.2f} sec"
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        model_dir
        / f"{model_name}.joblib"
    )


    # Native formats as well where useful.
    if model_name == "xgboost":

        model.save_model(
            model_dir
            / "xgboost_model.json"
        )


    if model_name == "catboost":

        model.save_model(
            model_dir
            / "catboost_model.cbm"
        )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model_name=model_name,
        model=model,
        X_val=X_val_imputed,
        y_val=y_val,
        validation_meta=validation_meta,
        training_seconds=training_seconds,
        output_dir=model_output_dir,
    )


    all_metrics.append(
        metrics
    )


    # --------------------------------------------------------
    # Save training information
    # --------------------------------------------------------

    training_info = {
        "model":
            model_name,

        "train_rows":
            len(train_df),

        "validation_rows":
            len(val_df),

        "calibration_rows_untouched":
            len(calibration_df),

        "feature_count":
            len(features),

        "features":
            features,

        "training_seconds":
            training_seconds,

        "random_state":
            RANDOM_STATE,

        "class_weights":
            class_weight_dict,

        "spatial_split":
            True,

        "calibration_used":
            False,

        "holdout_2025_used":
            False,

        "parameters":
            model.get_params(),
    }


    save_json(
        model_output_dir
        / "training_info.json",
        training_info,
    )


    print(
        f"\n{model_name} validation:"
    )

    print(
        f"  Accuracy             : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Balanced accuracy    : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Macro F1             : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"  Industrial precision : "
        f"{metrics['industrial_precision']:.4f}"
    )

    print(
        f"  Industrial recall    : "
        f"{metrics['industrial_recall']:.4f}"
    )

    print(
        f"  Industrial F1        : "
        f"{metrics['industrial_f1']:.4f}"
    )


# ============================================================
# COMPARISON
# ============================================================

print("\n[7/7] Building model comparison...")


comparison_columns = [
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",

    "industrial_precision",
    "industrial_recall",
    "industrial_f1",

    "natural_precision",
    "natural_recall",
    "natural_f1",

    "other_precision",
    "other_recall",
    "other_f1",

    "training_seconds",
]


comparison = pd.DataFrame(
    [
        {
            column:
                metric[
                    column
                ]

            for column
            in comparison_columns
        }

        for metric
        in all_metrics
    ]
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
    OUTPUT_ROOT
    / "model_comparison.csv",
    index=False,
)


print("\nMODEL COMPARISON")
print("=" * 80)

display_columns = [
    "model",
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


# ------------------------------------------------------------
# Comparison chart
# ------------------------------------------------------------

chart = comparison.set_index(
    "model"
)[
    [
        "accuracy",
        "macro_f1",
        "industrial_f1",
        "natural_f1",
        "other_f1",
    ]
]


ax = chart.plot(
    kind="bar",
    figsize=(11, 7),
)

ax.set_title(
    "FireWatch Model Validation Comparison"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1
)

ax.set_xlabel(
    "Model"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_ROOT
    / "model_comparison.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Benchmark summary JSON
# ------------------------------------------------------------

save_json(
    OUTPUT_ROOT
    / "benchmark_summary.json",
    {
        "comparison":
            all_metrics,

        "comparison_sorted_by":
            [
                "macro_f1",
                "industrial_f1",
            ],

        "calibration_used":
            False,

        "holdout_2025_used":
            False,

        "note":
            (
                "Validation metrics measure agreement "
                "with contextual weak labels on "
                "spatially disjoint development data. "
                "They are not confirmed real-world "
                "fire-classification accuracy."
            ),
    },
)


print("\n" + "=" * 80)
print("STEP 17 BENCHMARK COMPLETE")
print("=" * 80)

print("\nSaved results:")
print(OUTPUT_ROOT)

print("\nSaved models:")
print(MODEL_ROOT)

print(
    "\nCALIBRATION SET USED: NO"
)

print(
    "2025 HOLDOUT USED: NO"
)

print(
    "\nDo NOT select the final model "
    "from accuracy alone."
)

print(
    "\nNEXT:"
    "\nReview all three validation results and "
    "select the model family for controlled tuning."
)