from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool


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

MODEL_FILE = (
    ROOT
    / "models"
    / "production"
    / "source_classifier_v1"
    / "source_classifier.cbm"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "experiments"
    / "shap_analysis_v1"
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


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 22")
print("SHAP ANALYSIS — FROZEN CATBOOST")
print("=" * 80)

print(
    "\nIMPORTANT:"
    "\n- No model training."
    "\n- No hyperparameter tuning."
    "\n- No threshold tuning."
    "\n- 2025 remains untouched."
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


validation_df = df[
    df["split"].eq("validation")
].copy()


# ============================================================
# RECONSTRUCT PREPROCESSING
# ============================================================

X_train = train_df[
    features
].copy()


X_val = validation_df[
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


train_medians = (
    X_train
    .median(
        numeric_only=True
    )
)


X_val = X_val.fillna(
    train_medians
)


if X_val.isna().any().any():

    raise RuntimeError(
        "NaN remains in SHAP input."
    )


print(
    f"\nValidation rows : {len(X_val):,}"
)

print(
    f"Features        : {len(features)}"
)


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

model = CatBoostClassifier()

model.load_model(
    MODEL_FILE
)


print(
    "\nFrozen production-candidate "
    "CatBoost loaded."
)


# ============================================================
# CATBOOST SHAP VALUES
# ============================================================

print(
    "\nCalculating CatBoost SHAP values..."
)


pool = Pool(
    X_val,
    feature_names=features,
)


shap_values = model.get_feature_importance(
    pool,
    type="ShapValues",
)


shap_values = np.asarray(
    shap_values
)


print(
    f"Raw SHAP shape: "
    f"{shap_values.shape}"
)


# CatBoost multiclass usually returns:
# (rows, classes, features + 1)
#
# Last column is expected value.

if (
    shap_values.ndim != 3
):

    raise RuntimeError(
        "Unexpected multiclass SHAP shape: "
        f"{shap_values.shape}"
    )


if (
    shap_values.shape[1]
    != len(CLASS_NAMES)
):

    raise RuntimeError(
        "Unexpected number of SHAP classes."
    )


feature_shap = (
    shap_values[
        :,
        :,
        :-1
    ]
)


expected_values = (
    shap_values[
        :,
        :,
        -1
    ]
)


# ============================================================
# GLOBAL IMPORTANCE PER CLASS
# ============================================================

print(
    "\nBuilding class-specific "
    "global SHAP importance..."
)


class_tables = {}


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    values = feature_shap[
        :,
        class_index,
        :
    ]


    mean_abs = np.mean(
        np.abs(values),
        axis=0
    )


    mean_signed = np.mean(
        values,
        axis=0
    )


    table = pd.DataFrame({
        "feature":
            features,

        "mean_abs_shap":
            mean_abs,

        "mean_signed_shap":
            mean_signed,
    })


    table = table.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )


    table.to_csv(
        OUTPUT_DIR
        / f"{class_name}_shap_importance.csv",
        index=False,
    )


    class_tables[
        class_name
    ] = table


    print(
        "\n"
        + class_name.upper()
    )

    print(
        table[
            [
                "feature",
                "mean_abs_shap",
                "mean_signed_shap",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# INDUSTRIAL TOP-20 PLOT
# ============================================================

industrial_table = (
    class_tables[
        "industrial_fire_candidate"
    ]
    .head(20)
    .copy()
)


industrial_plot = (
    industrial_table
    .sort_values(
        "mean_abs_shap",
        ascending=True,
    )
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)


ax.barh(
    industrial_plot["feature"],
    industrial_plot["mean_abs_shap"],
)


ax.set_xlabel(
    "Mean absolute SHAP value"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Industrial Fire Candidate — "
    "Top SHAP Features"
)


fig.tight_layout()


fig.savefig(
    OUTPUT_DIR
    / "industrial_top20_shap.png",
    dpi=200,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# OVERALL GLOBAL IMPORTANCE
# ============================================================

overall_mean_abs = np.mean(
    np.abs(
        feature_shap
    ),
    axis=(0, 1),
)


overall_table = pd.DataFrame({
    "feature":
        features,

    "mean_abs_shap":
        overall_mean_abs,
})


overall_table = (
    overall_table
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


overall_table.to_csv(
    OUTPUT_DIR
    / "overall_shap_importance.csv",
    index=False,
)


print(
    "\nOVERALL TOP 20"
)

print(
    overall_table
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# FEATURE GROUP IMPORTANCE
# ============================================================

FIRMS_FEATURES = {
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
}


TEMPORAL_FEATURES = {
    "detections_3d",
    "detections_7d",
    "detections_10d",
    "detections_30d",
    "distinct_active_days_30d",
    "frp_ratio_to_history",
    "active_days_fraction_30d",
}


OSM_FEATURES = {
    "distance_to_strong_industrial_km",
    "distance_to_supporting_industrial_km",
    "distance_to_infrastructure_km",
    "distance_to_vegetation_km",
    "strong_industrial_vs_vegetation_distance_km",
}


SATELLITE_FEATURES = {
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "NBR",
    "NDMI",
    "satellite_evidence_available",
}


feature_groups = {
    "FIRMS thermal/time":
        FIRMS_FEATURES,

    "Historical temporal":
        TEMPORAL_FEATURES,

    "OSM context":
        OSM_FEATURES,

    "Sentinel satellite":
        SATELLITE_FEATURES,
}


industrial_importance_map = dict(
    zip(
        class_tables[
            "industrial_fire_candidate"
        ]["feature"],
        class_tables[
            "industrial_fire_candidate"
        ]["mean_abs_shap"],
    )
)


group_rows = []


for group_name, group_features in (
    feature_groups.items()
):

    group_features_present = [
        feature
        for feature in group_features
        if feature
        in industrial_importance_map
    ]


    total_importance = sum(
        industrial_importance_map[
            feature
        ]
        for feature
        in group_features_present
    )


    group_rows.append({
        "feature_group":
            group_name,

        "feature_count":
            len(
                group_features_present
            ),

        "industrial_total_mean_abs_shap":
            float(
                total_importance
            ),
    })


group_table = pd.DataFrame(
    group_rows
)


total_group_importance = (
    group_table[
        "industrial_total_mean_abs_shap"
    ].sum()
)


group_table[
    "importance_share"
] = (
    group_table[
        "industrial_total_mean_abs_shap"
    ]
    / total_group_importance
)


group_table = (
    group_table
    .sort_values(
        "industrial_total_mean_abs_shap",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


group_table.to_csv(
    OUTPUT_DIR
    / "industrial_feature_group_importance.csv",
    index=False,
)


print(
    "\nINDUSTRIAL FEATURE-GROUP IMPORTANCE"
)

print(
    group_table.to_string(
        index=False
    )
)


# ============================================================
# LOCAL INDUSTRIAL EXPLANATIONS
# ============================================================

print(
    "\nSaving local Industrial explanations..."
)


actual_classes = (
    validation_df[
        "source_class"
    ]
    .to_numpy()
)


industrial_indices = np.where(
    actual_classes
    == "industrial_fire_candidate"
)[0]


local_rows = []


industrial_shap = (
    feature_shap[
        :,
        0,
        :
    ]
)


for row_index in industrial_indices:

    row_values = (
        industrial_shap[
            row_index
        ]
    )


    order = np.argsort(
        np.abs(
            row_values
        )
    )[::-1]


    record = {
        "observation_id":
            validation_df.iloc[
                row_index
            ][
                "observation_id"
            ],
    }


    for rank, feature_index in enumerate(
        order[:10],
        start=1
    ):

        feature_name = (
            features[
                feature_index
            ]
        )

        record[
            f"feature_{rank}"
        ] = feature_name

        record[
            f"value_{rank}"
        ] = (
            X_val.iloc[
                row_index
            ][
                feature_name
            ]
        )

        record[
            f"shap_{rank}"
        ] = (
            row_values[
                feature_index
            ]
        )


    local_rows.append(
        record
    )


local_df = pd.DataFrame(
    local_rows
)


local_df.to_csv(
    OUTPUT_DIR
    / "actual_industrial_local_explanations.csv",
    index=False,
)


# ============================================================
# EXPECTED VALUES
# ============================================================

expected_summary = {
    CLASS_NAMES[i]:
        float(
            np.mean(
                expected_values[
                    :,
                    i
                ]
            )
        )

    for i in range(
        len(CLASS_NAMES)
    )
}


with open(
    OUTPUT_DIR
    / "shap_expected_values.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        expected_summary,
        f,
        indent=2
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 80)
print("STEP 22 COMPLETE")
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
    "\nINTERPRETATION RULE:"
    "\nSHAP explains the model's prediction behavior."
    "\nIt does NOT prove that a feature caused a real fire."
)