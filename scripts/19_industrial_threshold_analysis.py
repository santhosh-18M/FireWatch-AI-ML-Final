from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "model_benchmark_v1"
    / "catboost"
    / "predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "experiments"
    / "industrial_threshold_analysis_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INDUSTRIAL = "industrial_fire_candidate"

PROB_COLUMN = (
    "prob_industrial_fire_candidate"
)


print("=" * 80)
print("FIREWATCH — STEP 19")
print("CATBOOST INDUSTRIAL THRESHOLD ANALYSIS")
print("=" * 80)

print(
    "\nThis is diagnostic only."
    "\nNo model retraining."
    "\nNo calibration data."
    "\nNo 2025 holdout."
)


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


if PROB_COLUMN not in df.columns:

    raise RuntimeError(
        f"Missing probability column: "
        f"{PROB_COLUMN}"
    )


y_true = (
    df["actual_class"]
    .eq(INDUSTRIAL)
    .astype(int)
    .to_numpy()
)


industrial_probability = (
    df[PROB_COLUMN]
    .to_numpy()
)


print(
    f"\nValidation observations : "
    f"{len(df):,}"
)

print(
    f"Actual Industrial       : "
    f"{y_true.sum():,}"
)

print(
    f"Non-Industrial          : "
    f"{(y_true == 0).sum():,}"
)


# ============================================================
# THRESHOLD SWEEP
# ============================================================

thresholds = np.arange(
    0.02,
    0.81,
    0.01
)


rows = []


for threshold in thresholds:

    y_pred = (
        industrial_probability
        >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )


    rows.append({
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

        "false_positive_rate":
            float(
                false_positive_rate
            ),

        "predicted_industrial":
            int(
                tp + fp
            ),
    })


results = pd.DataFrame(
    rows
)


results.to_csv(
    OUTPUT_DIR
    / "catboost_industrial_thresholds.csv",
    index=False,
)


# ============================================================
# BEST F1 — DIAGNOSTIC ONLY
# ============================================================

best_index = (
    results["f1"]
    .idxmax()
)

best_row = results.loc[
    best_index
]


print(
    "\nHighest validation Industrial F1 "
    "(diagnostic only):"
)

print(
    f"  Threshold           : "
    f"{best_row['threshold']:.2f}"
)

print(
    f"  Precision           : "
    f"{best_row['precision']:.4f}"
)

print(
    f"  Recall              : "
    f"{best_row['recall']:.4f}"
)

print(
    f"  F1                  : "
    f"{best_row['f1']:.4f}"
)

print(
    f"  True positives      : "
    f"{int(best_row['true_positive'])}"
)

print(
    f"  False positives     : "
    f"{int(best_row['false_positive'])}"
)

print(
    f"  False negatives     : "
    f"{int(best_row['false_negative'])}"
)


# ============================================================
# IMPORTANT THRESHOLDS
# ============================================================

important_thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
]


selected = results[
    results["threshold"]
    .isin(
        important_thresholds
    )
].copy()


selected.to_csv(
    OUTPUT_DIR
    / "important_thresholds.csv",
    index=False,
)


print(
    "\nIMPORTANT THRESHOLDS"
)

print(
    selected[
        [
            "threshold",
            "true_positive",
            "false_positive",
            "false_negative",
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
# PRECISION / RECALL / F1 PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.plot(
    results["threshold"],
    results["precision"],
    label="Precision",
)

ax.plot(
    results["threshold"],
    results["recall"],
    label="Recall",
)

ax.plot(
    results["threshold"],
    results["f1"],
    label="F1",
)


ax.set_xlabel(
    "Industrial probability threshold"
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "CatBoost Industrial Candidate Threshold Analysis"
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
    / "precision_recall_f1_vs_threshold.png",
    dpi=200,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# FALSE POSITIVE / TRUE POSITIVE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.plot(
    results["threshold"],
    results["true_positive"],
    label="True positives",
)

ax.plot(
    results["threshold"],
    results["false_positive"],
    label="False positives",
)


ax.set_xlabel(
    "Industrial probability threshold"
)

ax.set_ylabel(
    "Number of validation observations"
)

ax.set_title(
    "Industrial Detection vs False Alerts"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()


fig.savefig(
    OUTPUT_DIR
    / "industrial_detection_vs_false_alerts.png",
    dpi=200,
    bbox_inches="tight",
)


plt.close(fig)


print("\n" + "=" * 80)
print("STEP 19 COMPLETE")
print("=" * 80)

print(
    "\nSaved:"
)

print(
    OUTPUT_DIR
)

print(
    "\nIMPORTANT:"
    "\nThe threshold with the highest validation F1 "
    "is NOT automatically the production threshold."
)

print(
    "\nCALIBRATION SET USED: NO"
)

print(
    "2025 HOLDOUT USED: NO"
)