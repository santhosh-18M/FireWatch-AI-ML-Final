from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "training_dataset_v1.parquet"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "training_dataset_v1_spatial_split.parquet"
)

REPORT_FILE = (
    ROOT / "reports" / "experiments"
    / "spatial_split_v1_report.json"
)

RANDOM_STATE = 42

TARGET_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "calibration": 0.15,
}

SPLITS = [
    "train",
    "validation",
    "calibration",
]

CLASSES = [
    "industrial_fire_candidate",
    "natural_vegetation_fire",
    "other_uncertain",
]

# Number of candidate spatial assignments to test.
N_TRIALS = 5000

# Require at least this many industrial examples
# in validation and calibration.
MIN_INDUSTRIAL_EVAL = 40


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 16 V2")
print("ROBUST LEAKAGE-SAFE SPATIAL SPLITS")
print("=" * 80)

print(
    "\nPolicy:"
    "\n- Entire 0.5-degree spatial groups remain together."
    "\n- No spatial group may occur in multiple splits."
    "\n- Target approximately 70 / 15 / 15."
    "\n- Search multiple group-level assignments."
    "\n- Preserve Industrial examples in both evaluation splits."
    "\n- No model fitting."
    "\n- 2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/8] Loading training dataset...")

df = pd.read_parquet(INPUT_FILE)

print(f"      Rows          : {len(df):,}")
print(
    f"      Spatial groups: "
    f"{df['spatial_group'].nunique():,}"
)

required = [
    "observation_id",
    "spatial_group",
    "source_class",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

if df["observation_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate observation IDs detected."
    )


# ============================================================
# 2. GROUP × CLASS MATRIX
# ============================================================

print("\n[2/8] Building group-level matrix...")

group_class = pd.crosstab(
    df["spatial_group"],
    df["source_class"],
)

group_class = group_class.reindex(
    columns=CLASSES,
    fill_value=0,
)

group_class["total"] = (
    group_class[CLASSES]
    .sum(axis=1)
)

groups = group_class.reset_index()

print(
    f"      Groups: "
    f"{len(groups):,}"
)


class_totals = {
    cls: int(
        (df["source_class"] == cls).sum()
    )
    for cls in CLASSES
}


print("\n      Overall class counts:")

for cls in CLASSES:

    print(
        f"        {cls:<30} "
        f"{class_totals[cls]:>6,}"
    )


# ============================================================
# 3. TARGETS
# ============================================================

print("\n[3/8] Building split targets...")


total_rows = len(df)

target_rows = {
    split:
        total_rows
        * TARGET_FRACTIONS[split]

    for split in SPLITS
}


target_classes = {
    split: {
        cls:
            class_totals[cls]
            * TARGET_FRACTIONS[split]

        for cls in CLASSES
    }

    for split in SPLITS
}


for split in SPLITS:

    print(
        f"\n      {split.upper()}"
    )

    print(
        f"        rows       : "
        f"{target_rows[split]:.1f}"
    )

    for cls in CLASSES:

        print(
            f"        {cls:<30} "
            f"{target_classes[split][cls]:.1f}"
        )


# ============================================================
# 4. RANDOMIZED GROUP-LEVEL SEARCH
# ============================================================

print(
    f"\n[4/8] Searching {N_TRIALS:,} "
    "spatial assignments..."
)


group_ids = groups[
    "spatial_group"
].to_numpy()

group_totals = groups[
    "total"
].to_numpy(
    dtype=int
)

group_class_arrays = {
    cls:
        groups[cls]
        .to_numpy(dtype=int)

    for cls in CLASSES
}


rng = np.random.default_rng(
    RANDOM_STATE
)


best_score = np.inf
best_assignment = None
best_summary = None
valid_trials = 0


def evaluate_assignment(
    assignment
):

    summary = {
        split: {
            "rows": 0,
            **{
                cls: 0
                for cls in CLASSES
            },
        }
        for split in SPLITS
    }


    for split_index, split in enumerate(
        SPLITS
    ):

        mask = (
            assignment
            == split_index
        )

        summary[
            split
        ]["rows"] = int(
            group_totals[
                mask
            ].sum()
        )

        for cls in CLASSES:

            summary[
                split
            ][cls] = int(
                group_class_arrays[
                    cls
                ][mask].sum()
            )


    # --------------------------------------------------------
    # HARD REQUIREMENTS
    # --------------------------------------------------------

    for split in SPLITS:

        if summary[
            split
        ]["rows"] == 0:

            return None, np.inf

        for cls in CLASSES:

            if summary[
                split
            ][cls] == 0:

                return None, np.inf


    if (
        summary[
            "validation"
        ][
            "industrial_fire_candidate"
        ]
        < MIN_INDUSTRIAL_EVAL
    ):

        return None, np.inf


    if (
        summary[
            "calibration"
        ][
            "industrial_fire_candidate"
        ]
        < MIN_INDUSTRIAL_EVAL
    ):

        return None, np.inf


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0.0


    # Row-count balance.
    for split in SPLITS:

        target = (
            target_rows[
                split
            ]
        )

        error = (
            summary[
                split
            ]["rows"]
            - target
        ) / target

        score += (
            error ** 2
            * 3.0
        )


    # Class balance.
    class_weights = {
        "industrial_fire_candidate":
            6.0,

        "natural_vegetation_fire":
            1.0,

        "other_uncertain":
            1.5,
    }


    for split in SPLITS:

        for cls in CLASSES:

            target = (
                target_classes[
                    split
                ][cls]
            )

            error = (
                summary[
                    split
                ][cls]
                - target
            ) / max(
                target,
                1.0
            )

            score += (
                class_weights[
                    cls
                ]
                * error ** 2
            )


    return summary, score


# ------------------------------------------------------------
# Generate candidate group assignments.
#
# Probabilities are exactly 70/15/15 at group level.
# Since group sizes differ, actual row proportions vary.
# We select the best row/class-balanced result.
# ------------------------------------------------------------

probabilities = np.array(
    [
        TARGET_FRACTIONS[
            "train"
        ],
        TARGET_FRACTIONS[
            "validation"
        ],
        TARGET_FRACTIONS[
            "calibration"
        ],
    ]
)


for trial in range(
    N_TRIALS
):

    assignment = rng.choice(
        3,
        size=len(groups),
        replace=True,
        p=probabilities,
    )


    summary, score = (
        evaluate_assignment(
            assignment
        )
    )


    if summary is None:
        continue


    valid_trials += 1


    if score < best_score:

        best_score = score

        best_assignment = (
            assignment.copy()
        )

        best_summary = summary


if best_assignment is None:

    raise RuntimeError(
        "Could not find a valid spatial "
        "assignment. Increase N_TRIALS."
    )


print(
    f"      Valid trials: "
    f"{valid_trials:,}"
)

print(
    f"      Best score  : "
    f"{best_score:.8f}"
)


# ============================================================
# 5. APPLY BEST ASSIGNMENT
# ============================================================

print("\n[5/8] Applying best assignment...")


group_to_split = {
    group_ids[i]:
        SPLITS[
            best_assignment[i]
        ]

    for i in range(
        len(group_ids)
    )
}


df["split"] = (
    df[
        "spatial_group"
    ]
    .map(
        group_to_split
    )
)


if df["split"].isna().any():

    raise RuntimeError(
        "Some observations were not assigned."
    )


# ============================================================
# 6. AUDIT DISTRIBUTION
# ============================================================

print("\n[6/8] Auditing selected split...")


print("\n      Row counts:")

for split in SPLITS:

    count = int(
        (
            df["split"]
            == split
        ).sum()
    )

    percentage = (
        count
        / len(df)
        * 100
    )

    print(
        f"        {split:<12} "
        f"{count:>6,} "
        f"({percentage:>6.2f}%)"
    )


cross = pd.crosstab(
    df["source_class"],
    df["split"],
)

cross = cross.reindex(
    index=CLASSES,
    columns=SPLITS,
    fill_value=0,
)


print("\n      Class × split:")

print(
    cross.to_string()
)


group_split_counts = (
    df[
        [
            "spatial_group",
            "split",
        ]
    ]
    .drop_duplicates()
    ["split"]
    .value_counts()
    .reindex(
        SPLITS,
        fill_value=0,
    )
)


print(
    "\n      Spatial groups per split:"
)

print(
    group_split_counts.to_string()
)


# ============================================================
# 7. LEAKAGE / QUALITY CHECKS
# ============================================================

print("\n[7/8] Running leakage checks...")


group_split_nunique = (
    df.groupby(
        "spatial_group"
    )["split"]
    .nunique()
)


leaking_groups = (
    group_split_nunique[
        group_split_nunique > 1
    ]
)


print(
    f"      Spatial groups in >1 split: "
    f"{len(leaking_groups)}"
)


if len(leaking_groups) > 0:

    raise RuntimeError(
        "Spatial leakage detected."
    )


sets = {
    split: set(
        df.loc[
            df["split"] == split,
            "observation_id",
        ]
    )
    for split in SPLITS
}


overlaps = {
    "train_validation":
        len(
            sets["train"]
            &
            sets["validation"]
        ),

    "train_calibration":
        len(
            sets["train"]
            &
            sets["calibration"]
        ),

    "validation_calibration":
        len(
            sets["validation"]
            &
            sets["calibration"]
        ),
}


for name, count in overlaps.items():

    print(
        f"      {name:<25}: "
        f"{count}"
    )


if any(
    overlaps.values()
):

    raise RuntimeError(
        "Observation leakage detected."
    )


print(
    "\n      Industrial evaluation coverage:"
)

print(
    "        validation : "
    f"{cross.loc['industrial_fire_candidate', 'validation']}"
)

print(
    "        calibration: "
    f"{cross.loc['industrial_fire_candidate', 'calibration']}"
)


# ============================================================
# 8. SAVE
# ============================================================

print("\n[8/8] Saving valid spatial split...")


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_parquet(
    OUTPUT_FILE,
    index=False
)


report = {
    "stage":
        "spatial_split_v1_v2",

    "strategy":
        (
            "randomized whole-spatial-group "
            "search with class-balance objective"
        ),

    "random_state":
        RANDOM_STATE,

    "trials":
        N_TRIALS,

    "valid_trials":
        valid_trials,

    "best_score":
        float(
            best_score
        ),

    "total_rows":
        int(
            len(df)
        ),

    "spatial_groups":
        int(
            df[
                "spatial_group"
            ].nunique()
        ),

    "target_fractions":
        TARGET_FRACTIONS,

    "actual_rows": {
        split:
            int(
                (
                    df["split"]
                    == split
                ).sum()
            )

        for split in SPLITS
    },

    "class_by_split": {
        cls: {
            split:
                int(
                    cross.loc[
                        cls,
                        split
                    ]
                )

            for split in SPLITS
        }

        for cls in CLASSES
    },

    "groups_by_split": {
        split:
            int(
                group_split_counts[
                    split
                ]
            )

        for split in SPLITS
    },

    "leakage": {
        "spatial_group_overlap":
            int(
                len(
                    leaking_groups
                )
            ),

        **overlaps,
    },

    "model_training_performed":
        False,

    "2025_holdout_accessed":
        False,
}


REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print("\n" + "=" * 80)
print("SPATIAL SPLIT V1 PASSED")
print("=" * 80)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nReport:")
print(REPORT_FILE)

print(
    "\nSPATIAL GROUP OVERLAP: 0"
)

print(
    "OBSERVATION OVERLAP: 0"
)

print(
    "2025 HOLDOUT ACCESSED: NO"
)

print(
    "MODEL TRAINING PERFORMED: NO"
)

print(
    "\nNEXT:"
    "\nMODEL BENCHMARKING."
)