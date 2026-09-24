from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]

LABELS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "source_labels_v1_2025.parquet"
)

PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "model_evaluation"
    / "predictions_2025.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "experiments"
    / "industrial_context_sanity_audit_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDUSTRIAL = "industrial_fire_candidate"
NATURAL = "natural_vegetation_fire"
OTHER = "other_uncertain"


def pct(x):
    return f"{100.0 * x:.2f}%"


def safe_median(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return np.nan
    return float(s.median())


def class_distribution(df, column):
    counts = df[column].value_counts(dropna=False)
    percentages = (
        df[column]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    result = pd.DataFrame(
        {
            "count": counts,
            "percent": percentages,
        }
    )

    return result


def evaluate_subset(name, df):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print("Rows:", len(df))

    if len(df) == 0:
        print("No observations in this subset.")
        return

    print("\nActual class distribution:")
    print(class_distribution(df, "source_class").to_string())

    print("\nCalibrated prediction distribution:")
    print(
        class_distribution(
            df,
            "calibrated_predicted_class",
        ).to_string()
    )

    actual_industrial = df["source_class"] == INDUSTRIAL
    predicted_industrial = (
        df["calibrated_predicted_class"] == INDUSTRIAL
    )

    actual_industrial_count = int(actual_industrial.sum())

    if actual_industrial_count > 0:
        correct = int(
            (
                actual_industrial
                & predicted_industrial
            ).sum()
        )

        missed = actual_industrial & ~predicted_industrial
        missed_count = int(missed.sum())

        print("\nIndustrial weak-label cases:", actual_industrial_count)
        print("Correctly classified Industrial:", correct)
        print("Missed Industrial:", missed_count)

        print(
            "Industrial recall in subset:",
            pct(correct / actual_industrial_count),
        )

        if missed_count > 0:
            print("\nWhere missed Industrial cases went:")
            print(
                df.loc[
                    missed,
                    "calibrated_predicted_class",
                ]
                .value_counts()
                .to_string()
            )

            print("\nMedian distances for MISSED Industrial cases:")
            for col in [
                "distance_to_strong_industrial_km",
                "distance_to_supporting_industrial_km",
                "distance_to_infrastructure_km",
                "distance_to_vegetation_km",
                "strong_industrial_vs_vegetation_distance_km",
            ]:
                print(
                    f"{col}: "
                    f"{safe_median(df.loc[missed, col]):.3f}"
                )

    print("\nMedian calibrated probabilities:")
    print(
        df[
            [
                "calibrated_prob_industrial",
                "calibrated_prob_natural",
                "calibrated_prob_other",
            ]
        ]
        .median()
        .round(4)
        .to_string()
    )


def main():
    print("Loading FireWatch V1 holdout evidence...")

    labels = pd.read_parquet(LABELS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    print("Labels:", labels.shape)
    print("Predictions:", predictions.shape)

    required_label_columns = [
        "observation_id",
        "source_class",
        "label_quality",
        "label_source",
        "label_conflict_industrial_vs_vegetation",
        "temporal_status",
        "detections_30d",
        "distinct_active_days_30d",
        "frp",
        "brightness",
        "distance_to_strong_industrial_km",
        "distance_to_supporting_industrial_km",
        "distance_to_infrastructure_km",
        "distance_to_vegetation_km",
        "strong_industrial_vs_vegetation_distance_km",
        "strong_industrial_within_1_0km",
        "strong_industrial_within_1_5km",
        "supporting_industrial_within_1_0km",
        "supporting_industrial_within_1_5km",
        "independent_industrial_distance_km",
        "industrial_reference_le_1km",
        "industrial_reference_le_2km",
    ]

    required_prediction_columns = [
        "observation_id",
        "actual_class",
        "raw_prob_industrial",
        "raw_prob_natural",
        "raw_prob_other",
        "calibrated_prob_industrial",
        "calibrated_prob_natural",
        "calibrated_prob_other",
        "raw_predicted_class",
        "calibrated_predicted_class",
    ]

    missing_labels = [
        c for c in required_label_columns
        if c not in labels.columns
    ]

    missing_predictions = [
        c for c in required_prediction_columns
        if c not in predictions.columns
    ]

    if missing_labels:
        raise RuntimeError(
            "Missing label columns: "
            + ", ".join(missing_labels)
        )

    if missing_predictions:
        raise RuntimeError(
            "Missing prediction columns: "
            + ", ".join(missing_predictions)
        )

    evidence = labels[required_label_columns].copy()

    df = evidence.merge(
        predictions[required_prediction_columns],
        on="observation_id",
        how="inner",
        validate="one_to_one",
    )

    print("Joined rows:", len(df))

    if len(df) != 20000:
        print(
            "WARNING: expected 20,000 joined holdout observations."
        )

    mismatch = (
        df["source_class"]
        != df["actual_class"]
    )

    print(
        "Label/prediction actual-class mismatches:",
        int(mismatch.sum()),
    )

    if mismatch.any():
        print(
            "WARNING: source_class and actual_class differ."
        )

    # ---------------------------------------------------------
    # OVERALL FROZEN V1
    # ---------------------------------------------------------

    print("\n")
    print("#" * 80)
    print("OVERALL FROZEN V1")
    print("#" * 80)

    print(
        classification_report(
            df["source_class"],
            df["calibrated_predicted_class"],
            digits=4,
        )
    )

    labels_order = [
        INDUSTRIAL,
        NATURAL,
        OTHER,
    ]

    cm = confusion_matrix(
        df["source_class"],
        df["calibrated_predicted_class"],
        labels=labels_order,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{x}" for x in labels_order],
        columns=[f"pred_{x}" for x in labels_order],
    )

    print("Confusion matrix:")
    print(cm_df.to_string())

    # ---------------------------------------------------------
    # DEFINE INDUSTRIAL CONTEXT
    # ---------------------------------------------------------

    strong_1km = (
        pd.to_numeric(
            df["distance_to_strong_industrial_km"],
            errors="coerce",
        )
        <= 1.0
    )

    strong_1_5km = (
        pd.to_numeric(
            df["distance_to_strong_industrial_km"],
            errors="coerce",
        )
        <= 1.5
    )

    supporting_1km = (
        pd.to_numeric(
            df["distance_to_supporting_industrial_km"],
            errors="coerce",
        )
        <= 1.0
    )

    supporting_1_5km = (
        pd.to_numeric(
            df["distance_to_supporting_industrial_km"],
            errors="coerce",
        )
        <= 1.5
    )

    independent_2km = (
        pd.to_numeric(
            df["independent_industrial_distance_km"],
            errors="coerce",
        )
        <= 2.0
    )

    near_industrial_1km = (
        strong_1km
        | supporting_1km
    )

    near_industrial_1_5km = (
        strong_1_5km
        | supporting_1_5km
    )

    # Puliyangulam-like:
    # supporting industrial <= 1 km
    # episodic/no prior history
    puliyangulam_like = (
        supporting_1km
        & (
            pd.to_numeric(
                df["detections_30d"],
                errors="coerce",
            ).fillna(0)
            == 0
        )
    )

    # ---------------------------------------------------------
    # SUBSET AUDITS
    # ---------------------------------------------------------

    evaluate_subset(
        "A. Strong industrial context <= 1.0 km",
        df.loc[strong_1km],
    )

    evaluate_subset(
        "B. Supporting industrial context <= 1.0 km",
        df.loc[supporting_1km],
    )

    evaluate_subset(
        "C. Any model industrial context <= 1.0 km",
        df.loc[near_industrial_1km],
    )

    evaluate_subset(
        "D. Any model industrial context <= 1.5 km",
        df.loc[near_industrial_1_5km],
    )

    evaluate_subset(
        "E. Independent industrial reference <= 2.0 km",
        df.loc[independent_2km],
    )

    evaluate_subset(
        "F. Puliyangulam-like: supporting industrial <=1 km + no prior detections",
        df.loc[puliyangulam_like],
    )

    # ---------------------------------------------------------
    # ACTUAL INDUSTRIAL: CORRECT VS MISSED
    # ---------------------------------------------------------

    actual_industrial = (
        df["source_class"] == INDUSTRIAL
    )

    industrial_correct = (
        actual_industrial
        & (
            df["calibrated_predicted_class"]
            == INDUSTRIAL
        )
    )

    industrial_missed = (
        actual_industrial
        & (
            df["calibrated_predicted_class"]
            != INDUSTRIAL
        )
    )

    print("\n")
    print("#" * 80)
    print("ACTUAL INDUSTRIAL: CORRECT VS MISSED")
    print("#" * 80)

    compare_columns = [
        "frp",
        "brightness",
        "detections_30d",
        "distinct_active_days_30d",
        "distance_to_strong_industrial_km",
        "distance_to_supporting_industrial_km",
        "distance_to_infrastructure_km",
        "distance_to_vegetation_km",
        "strong_industrial_vs_vegetation_distance_km",
        "calibrated_prob_industrial",
    ]

    rows = []

    for col in compare_columns:
        rows.append(
            {
                "feature": col,
                "correct_industrial_median": safe_median(
                    df.loc[
                        industrial_correct,
                        col,
                    ]
                ),
                "missed_industrial_median": safe_median(
                    df.loc[
                        industrial_missed,
                        col,
                    ]
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    comparison["difference_missed_minus_correct"] = (
        comparison["missed_industrial_median"]
        - comparison["correct_industrial_median"]
    )

    print(comparison.round(4).to_string(index=False))

    # ---------------------------------------------------------
    # INFRASTRUCTURE DISTANCE BINS
    # ---------------------------------------------------------

    print("\n")
    print("#" * 80)
    print("INDUSTRIAL RECALL BY INFRASTRUCTURE DISTANCE")
    print("#" * 80)

    industrial_df = df.loc[actual_industrial].copy()

    industrial_df["infrastructure_distance_bin"] = pd.cut(
        pd.to_numeric(
            industrial_df["distance_to_infrastructure_km"],
            errors="coerce",
        ),
        bins=[
            -np.inf,
            1,
            3,
            5,
            10,
            20,
            50,
            np.inf,
        ],
        labels=[
            "<=1 km",
            "1-3 km",
            "3-5 km",
            "5-10 km",
            "10-20 km",
            "20-50 km",
            ">50 km",
        ],
    )

    infra_rows = []

    for distance_bin, group in industrial_df.groupby(
        "infrastructure_distance_bin",
        observed=False,
    ):
        if len(group) == 0:
            continue

        correct = (
            group["calibrated_predicted_class"]
            == INDUSTRIAL
        ).sum()

        infra_rows.append(
            {
                "infrastructure_distance": str(distance_bin),
                "actual_industrial_count": len(group),
                "correct_industrial": int(correct),
                "missed_industrial": int(
                    len(group) - correct
                ),
                "industrial_recall": (
                    correct / len(group)
                ),
                "median_industrial_probability": (
                    group[
                        "calibrated_prob_industrial"
                    ].median()
                ),
            }
        )

    infra_table = pd.DataFrame(infra_rows)

    print(
        infra_table
        .round(4)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # PULIYANGULAM-LIKE BREAKDOWN
    # ---------------------------------------------------------

    print("\n")
    print("#" * 80)
    print("PULIYANGULAM-LIKE CASES")
    print("#" * 80)

    p = df.loc[puliyangulam_like].copy()

    print("Total:", len(p))

    if len(p):
        cross = pd.crosstab(
            p["source_class"],
            p["calibrated_predicted_class"],
            margins=True,
        )

        print("\nActual vs prediction:")
        print(cross.to_string())

        print("\nMedian feature values:")
        for col in [
            "frp",
            "brightness",
            "distance_to_strong_industrial_km",
            "distance_to_supporting_industrial_km",
            "distance_to_infrastructure_km",
            "distance_to_vegetation_km",
            "calibrated_prob_industrial",
            "calibrated_prob_natural",
            "calibrated_prob_other",
        ]:
            print(
                f"{col}: {safe_median(p[col]):.4f}"
            )

    # ---------------------------------------------------------
    # SAVE USEFUL FILES
    # ---------------------------------------------------------

    comparison.to_csv(
        OUTPUT_DIR
        / "industrial_correct_vs_missed.csv",
        index=False,
    )

    infra_table.to_csv(
        OUTPUT_DIR
        / "industrial_recall_by_infrastructure_distance.csv",
        index=False,
    )

    p.sort_values(
        "calibrated_prob_industrial",
        ascending=True,
    ).to_csv(
        OUTPUT_DIR
        / "puliyangulam_like_cases.csv",
        index=False,
    )

    missed_export_columns = [
        "observation_id",
        "source_class",
        "calibrated_predicted_class",
        "label_quality",
        "label_source",
        "temporal_status",
        "frp",
        "brightness",
        "detections_30d",
        "distinct_active_days_30d",
        "distance_to_strong_industrial_km",
        "distance_to_supporting_industrial_km",
        "distance_to_infrastructure_km",
        "distance_to_vegetation_km",
        "independent_industrial_distance_km",
        "calibrated_prob_industrial",
        "calibrated_prob_natural",
        "calibrated_prob_other",
    ]

    df.loc[
        industrial_missed,
        missed_export_columns,
    ].sort_values(
        "calibrated_prob_industrial",
        ascending=True,
    ).to_csv(
        OUTPUT_DIR
        / "missed_industrial_cases.csv",
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(
        "Reports saved to:",
        OUTPUT_DIR,
    )

    print(
        "\nIMPORTANT: Do not retrain yet. "
        "Review these audit results first."
    )


if __name__ == "__main__":
    main()