from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_DIR = (
    ROOT / "reports" / "experiments"
    / "model_benchmark_v1"
)

OUTPUT_DIR = (
    ROOT / "reports" / "experiments"
    / "industrial_error_analysis_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODELS = [
    "random_forest",
    "xgboost",
    "catboost",
]

INDUSTRIAL = "industrial_fire_candidate"
NATURAL = "natural_vegetation_fire"
OTHER = "other_uncertain"


print("=" * 80)
print("FIREWATCH — STEP 18")
print("INDUSTRIAL PROBABILITY ERROR ANALYSIS")
print("=" * 80)

print(
    "\nPurpose:"
    "\n- No retraining."
    "\n- Analyze the 57 Industrial validation examples."
    "\n- Determine whether missed Industrial cases are near the decision boundary."
    "\n- Calibration and 2025 remain untouched."
)


summary_rows = []


for model_name in MODELS:

    print("\n" + "=" * 80)
    print(model_name.upper())
    print("=" * 80)

    prediction_file = (
        BENCHMARK_DIR
        / model_name
        / "predictions.csv"
    )

    df = pd.read_csv(
        prediction_file
    )

    prob_ind = (
        f"prob_{INDUSTRIAL}"
    )

    prob_nat = (
        f"prob_{NATURAL}"
    )

    prob_other = (
        f"prob_{OTHER}"
    )

    required = [
        "actual_class",
        "predicted_class",
        prob_ind,
        prob_nat,
        prob_other,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{model_name} missing columns:\n"
            + "\n".join(missing)
        )


    # ========================================================
    # ACTUAL INDUSTRIAL ONLY
    # ========================================================

    industrial = df[
        df["actual_class"]
        .eq(INDUSTRIAL)
    ].copy()


    industrial[
        "industrial_probability"
    ] = industrial[
        prob_ind
    ]


    industrial[
        "winning_probability"
    ] = industrial[
        [
            prob_ind,
            prob_nat,
            prob_other,
        ]
    ].max(
        axis=1
    )


    industrial[
        "probability_gap_to_winner"
    ] = (
        industrial[
            "winning_probability"
        ]
        -
        industrial[
            "industrial_probability"
        ]
    )


    industrial[
        "industrial_rank"
    ] = (
        industrial[
            [
                prob_ind,
                prob_nat,
                prob_other,
            ]
        ]
        .rank(
            axis=1,
            ascending=False,
            method="min",
        )[
            prob_ind
        ]
        .astype(int)
    )


    correct = industrial[
        industrial[
            "predicted_class"
        ].eq(INDUSTRIAL)
    ].copy()


    missed = industrial[
        ~industrial[
            "predicted_class"
        ].eq(INDUSTRIAL)
    ].copy()


    missed_other = missed[
        missed[
            "predicted_class"
        ].eq(OTHER)
    ].copy()


    missed_natural = missed[
        missed[
            "predicted_class"
        ].eq(NATURAL)
    ].copy()


    print(
        f"\nActual Industrial examples : "
        f"{len(industrial)}"
    )

    print(
        f"Correct Industrial         : "
        f"{len(correct)}"
    )

    print(
        f"Missed                     : "
        f"{len(missed)}"
    )

    print(
        f"  -> Other                 : "
        f"{len(missed_other)}"
    )

    print(
        f"  -> Natural               : "
        f"{len(missed_natural)}"
    )


    # ========================================================
    # INDUSTRIAL PROBABILITY DISTRIBUTION
    # ========================================================

    print(
        "\nIndustrial probability "
        "for ACTUAL Industrial:"
    )

    description = (
        industrial[
            "industrial_probability"
        ]
        .describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
    )

    print(
        description.to_string()
    )


    print(
        "\nIndustrial probability "
        "for MISSED Industrial:"
    )

    if len(missed) > 0:

        print(
            missed[
                "industrial_probability"
            ]
            .describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                ]
            )
            .to_string()
        )


    # ========================================================
    # THRESHOLD COUNTS
    # ========================================================

    print(
        "\nActual Industrial observations "
        "above Industrial probability:"
    )

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    threshold_results = {}

    for threshold in thresholds:

        count = int(
            (
                industrial[
                    "industrial_probability"
                ]
                >= threshold
            ).sum()
        )

        pct = (
            count
            / len(industrial)
            * 100
        )

        threshold_results[
            str(threshold)
        ] = count

        print(
            f"  >= {threshold:.2f}: "
            f"{count:>2}/{len(industrial)} "
            f"({pct:5.1f}%)"
        )


    # ========================================================
    # INDUSTRIAL RANK
    # ========================================================

    print(
        "\nIndustrial probability rank "
        "among three classes:"
    )

    rank_counts = (
        industrial[
            "industrial_rank"
        ]
        .value_counts()
        .sort_index()
    )

    for rank in [
        1,
        2,
        3,
    ]:

        count = int(
            rank_counts.get(
                rank,
                0
            )
        )

        print(
            f"  Rank {rank}: "
            f"{count}"
        )


    # ========================================================
    # MISSED OTHER: HOW CLOSE?
    # ========================================================

    if len(
        missed_other
    ) > 0:

        missed_other[
            "other_minus_industrial"
        ] = (
            missed_other[
                prob_other
            ]
            -
            missed_other[
                prob_ind
            ]
        )

        print(
            "\nIndustrial -> Other probability gap:"
        )

        print(
            missed_other[
                "other_minus_industrial"
            ]
            .describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                ]
            )
            .to_string()
        )


    # ========================================================
    # SAVE INDUSTRIAL CASES
    # ========================================================

    industrial = industrial.sort_values(
        by="industrial_probability",
        ascending=False,
    )


    industrial.to_csv(
        OUTPUT_DIR
        / f"{model_name}_actual_industrial_cases.csv",
        index=False,
    )


    missed.sort_values(
        by="industrial_probability",
        ascending=False,
    ).to_csv(
        OUTPUT_DIR
        / f"{model_name}_missed_industrial.csv",
        index=False,
    )


    # ========================================================
    # FALSE POSITIVE ANALYSIS
    # ========================================================

    predicted_industrial = df[
        df[
            "predicted_class"
        ].eq(INDUSTRIAL)
    ].copy()


    false_positive = (
        predicted_industrial[
            ~predicted_industrial[
                "actual_class"
            ].eq(INDUSTRIAL)
        ]
        .copy()
    )


    false_positive.to_csv(
        OUTPUT_DIR
        / f"{model_name}_industrial_false_positives.csv",
        index=False,
    )


    print(
        "\nPredicted Industrial total : "
        f"{len(predicted_industrial)}"
    )

    print(
        "Industrial false positives: "
        f"{len(false_positive)}"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows.append({
        "model":
            model_name,

        "actual_industrial":
            len(industrial),

        "correct_industrial":
            len(correct),

        "missed_industrial":
            len(missed),

        "missed_as_other":
            len(missed_other),

        "missed_as_natural":
            len(missed_natural),

        "mean_industrial_probability":
            industrial[
                "industrial_probability"
            ].mean(),

        "median_industrial_probability":
            industrial[
                "industrial_probability"
            ].median(),

        "industrial_probability_ge_0_10":
            threshold_results[
                "0.1"
            ],

        "industrial_probability_ge_0_20":
            threshold_results[
                "0.2"
            ],

        "industrial_probability_ge_0_30":
            threshold_results[
                "0.3"
            ],

        "industrial_probability_ge_0_40":
            threshold_results[
                "0.4"
            ],

        "industrial_probability_ge_0_50":
            threshold_results[
                "0.5"
            ],

        "industrial_rank_1":
            int(
                rank_counts.get(
                    1,
                    0
                )
            ),

        "industrial_rank_2":
            int(
                rank_counts.get(
                    2,
                    0
                )
            ),

        "industrial_rank_3":
            int(
                rank_counts.get(
                    3,
                    0
                )
            ),

        "predicted_industrial":
            len(
                predicted_industrial
            ),

        "industrial_false_positives":
            len(
                false_positive
            ),
    })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUTPUT_DIR
    / "industrial_error_summary.csv",
    index=False,
)


print("\n" + "=" * 80)
print("STEP 18 COMPLETE")
print("=" * 80)

print(
    "\nSUMMARY:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    OUTPUT_DIR
)

print(
    "\nNO MODEL RETRAINING PERFORMED."
)

print(
    "CALIBRATION SET USED: NO"
)

print(
    "2025 HOLDOUT USED: NO"
)