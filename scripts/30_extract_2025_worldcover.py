from pathlib import Path
import json
import time

import ee
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "holdout_2025"
    / "evaluation_sample_2025.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "final"
    / "holdout_2025"
    / "worldcover_export_2025_summary.json"
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EARTH ENGINE CONFIG
# ============================================================

EE_PROJECT = "firewatch-ai-sih"

WORLDCOVER_COLLECTION = (
    "ESA/WorldCover/v200"
)

WORLDCOVER_BAND = "Map"

BUFFER_METERS = 100
SCALE_METERS = 10

# 20,000 observations -> 20 tasks of 1,000
BATCH_SIZE = 1000

EXPORT_FOLDER = "FireWatch_WorldCover_2025"
EXPORT_PREFIX = "firewatch_worldcover_2025"


# ============================================================
# WORLDCOVER CLASSES
# ============================================================

WORLD_COVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 30")
print("2025 HOLDOUT — ESA WORLDCOVER EXTRACTION")
print("=" * 80)

print(
    "\nPurpose:"
    "\n- Extract independent land-cover evidence."
    "\n- ESA WorldCover 2021 v200."
    "\n- Point class at FIRMS coordinate."
    "\n- 100 m neighborhood class fractions."
    "\n- 10 m extraction scale."
    "\n- NO source labels generated."
    "\n- NO model predictions generated."
)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Evaluation sample not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD SAMPLE
# ============================================================

print(
    "\n[1/7] Loading frozen 2025 evaluation sample..."
)

df = pd.read_parquet(
    INPUT_FILE
)


print(
    f"      Rows: {len(df):,}"
)


if len(df) != 20_000:

    raise RuntimeError(
        "Expected exactly 20,000 evaluation rows. "
        f"Found {len(df):,}."
    )


required_columns = [
    "observation_id",
    "latitude",
    "longitude",
    "acquired_at",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


if df[
    "observation_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate observation IDs detected."
    )


if not df[
    "latitude"
].between(
    -90,
    90
).all():

    raise RuntimeError(
        "Invalid latitude detected."
    )


if not df[
    "longitude"
].between(
    -180,
    180
).all():

    raise RuntimeError(
        "Invalid longitude detected."
    )


# ============================================================
# EARTH ENGINE
# ============================================================

print(
    "\n[2/7] Initializing Google Earth Engine..."
)


try:

    ee.Initialize(
        project=EE_PROJECT
    )

except Exception as exc:

    print(
        "\nEarth Engine initialization failed."
    )

    print(
        "Trying interactive authentication..."
    )

    ee.Authenticate()

    ee.Initialize(
        project=EE_PROJECT
    )


print(
    f"      Project: {EE_PROJECT}"
)


# ============================================================
# LOAD WORLDCOVER
# ============================================================

print(
    "\n[3/7] Loading ESA WorldCover 2021 v200..."
)


worldcover = (
    ee.ImageCollection(
        WORLDCOVER_COLLECTION
    )
    .first()
    .select(
        WORLDCOVER_BAND
    )
)


if worldcover is None:

    raise RuntimeError(
        "Could not load ESA WorldCover."
    )


print(
    f"      Dataset: {WORLDCOVER_COLLECTION}"
)

print(
    f"      Buffer: {BUFFER_METERS} m"
)

print(
    f"      Scale: {SCALE_METERS} m"
)


# ============================================================
# CLASS FRACTION IMAGES
# ============================================================

print(
    "\n[4/7] Preparing WorldCover class masks..."
)


class_codes = list(
    WORLD_COVER_CLASSES.keys()
)


class_fraction_images = []


for code in class_codes:

    mask = (
        worldcover
        .eq(code)
        .rename(
            f"wc_fraction_{code}"
        )
    )

    class_fraction_images.append(
        mask
    )


fraction_image = ee.Image.cat(
    class_fraction_images
)


# ============================================================
# FEATURE BUILDER
# ============================================================

def build_ee_feature(row):

    observation_id = str(
        row.observation_id
    )

    latitude = float(
        row.latitude
    )

    longitude = float(
        row.longitude
    )


    point = ee.Geometry.Point(
        [
            longitude,
            latitude,
        ]
    )


    buffer_geometry = (
        point.buffer(
            BUFFER_METERS
        )
    )


    # --------------------------------------------------------
    # POINT CLASS
    # --------------------------------------------------------

    point_result = (
        worldcover
        .reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=SCALE_METERS,
            bestEffort=False,
            maxPixels=100_000,
        )
        .get(
            WORLDCOVER_BAND
        )
    )


    # --------------------------------------------------------
    # 100 m CLASS FRACTIONS
    # --------------------------------------------------------

    fraction_result = (
        fraction_image
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer_geometry,
            scale=SCALE_METERS,
            bestEffort=False,
            maxPixels=100_000,
        )
    )


    properties = {
        "observation_id":
            observation_id,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "worldcover_point_class":
            point_result,

        "worldcover_buffer_m":
            BUFFER_METERS,

        "worldcover_scale_m":
            SCALE_METERS,

        "worldcover_dataset":
            "ESA_WorldCover_2021_v200",
    }


    # Add each class fraction
    for code in class_codes:

        property_name = (
            f"wc_fraction_{code}"
        )

        properties[
            property_name
        ] = fraction_result.get(
            property_name
        )


    return ee.Feature(
        point,
        properties
    )


# ============================================================
# EXPORT BATCHES
# ============================================================

print(
    "\n[5/7] Building Earth Engine export tasks..."
)


task_records = []


total_rows = len(df)

number_of_batches = (
    total_rows
    + BATCH_SIZE
    - 1
) // BATCH_SIZE


print(
    f"      Batch size: {BATCH_SIZE:,}"
)

print(
    f"      Export tasks: {number_of_batches}"
)


for batch_number in range(
    number_of_batches
):

    start_index = (
        batch_number
        * BATCH_SIZE
    )

    end_index = min(
        start_index
        + BATCH_SIZE,
        total_rows,
    )


    batch = (
        df.iloc[
            start_index:end_index
        ]
        .copy()
    )


    print(
        "\n"
        + "-" * 70
    )

    print(
        f"Batch {batch_number + 1}/"
        f"{number_of_batches}"
    )

    print(
        f"Rows: "
        f"{start_index:,} "
        f"to "
        f"{end_index - 1:,}"
    )


    features = []


    for row in batch.itertuples(
        index=False
    ):

        feature = build_ee_feature(
            row
        )

        features.append(
            feature
        )


    feature_collection = (
        ee.FeatureCollection(
            features
        )
    )


    batch_id = (
        f"{batch_number + 1:02d}"
    )


    description = (
        f"{EXPORT_PREFIX}_"
        f"batch_{batch_id}"
    )


    file_prefix = (
        f"{EXPORT_PREFIX}_"
        f"batch_{batch_id}"
    )


    task = (
        ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=description,
            folder=EXPORT_FOLDER,
            fileNamePrefix=file_prefix,
            fileFormat="CSV",
        )
    )


    task.start()


    task_id = task.id


    print(
        f"Started task: {task_id}"
    )

    print(
        f"Description: {description}"
    )


    task_records.append(
        {
            "batch_number":
                batch_number + 1,

            "start_index":
                start_index,

            "end_index_exclusive":
                end_index,

            "row_count":
                int(
                    len(batch)
                ),

            "task_id":
                task_id,

            "description":
                description,

            "drive_folder":
                EXPORT_FOLDER,

            "file_prefix":
                file_prefix,
        }
    )


# ============================================================
# VALIDATE TASKS
# ============================================================

print(
    "\n[6/7] Validating submitted tasks..."
)


if len(
    task_records
) != number_of_batches:

    raise RuntimeError(
        "Not all Earth Engine tasks were created."
    )


submitted_rows = sum(
    task[
        "row_count"
    ]
    for task in task_records
)


if submitted_rows != total_rows:

    raise RuntimeError(
        "Submitted row count does not equal "
        "evaluation sample size."
    )


print(
    f"      Submitted rows: "
    f"{submitted_rows:,}"
)


print(
    f"      Tasks created: "
    f"{len(task_records)}"
)


# ============================================================
# SAVE REPORT
# ============================================================

print(
    "\n[7/7] Saving export manifest..."
)


report = {
    "stage":
        "2025_worldcover_export",

    "input_file":
        str(INPUT_FILE),

    "input_rows":
        int(
            total_rows
        ),

    "earth_engine_project":
        EE_PROJECT,

    "dataset":
        WORLDCOVER_COLLECTION,

    "dataset_version":
        "2021_v200",

    "band":
        WORLDCOVER_BAND,

    "buffer_meters":
        BUFFER_METERS,

    "scale_meters":
        SCALE_METERS,

    "batch_size":
        BATCH_SIZE,

    "number_of_batches":
        number_of_batches,

    "submitted_rows":
        int(
            submitted_rows
        ),

    "drive_folder":
        EXPORT_FOLDER,

    "worldcover_classes":
        {
            str(k): v
            for k, v in (
                WORLD_COVER_CLASSES.items()
            )
        },

    "tasks":
        task_records,

    "model_used":
        False,

    "source_labels_generated":
        False,

    "notes": [
        (
            "WorldCover is independent land-cover "
            "reference evidence."
        ),
        (
            "WorldCover 2021 is static reference "
            "evidence and is not proof of the "
            "cause of a 2025 FIRMS anomaly."
        ),
        (
            "WorldCover evidence is used by the "
            "frozen V1 weak-label policy."
        ),
        (
            "OSM evidence is not used to create "
            "the frozen source labels."
        ),
        (
            "The frozen classifier has not been "
            "executed."
        ),
    ],
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 30 EXPORTS SUBMITTED"
)

print(
    "=" * 80
)


print(
    f"\nEvaluation observations: "
    f"{total_rows:,}"
)


print(
    f"Earth Engine tasks: "
    f"{number_of_batches}"
)


print(
    f"Rows submitted: "
    f"{submitted_rows:,}"
)


print(
    f"\nGoogle Drive folder:"
    f"\n{EXPORT_FOLDER}"
)


print(
    "\nTASKS"
)


for task in task_records:

    print(
        f"\nBatch "
        f"{task['batch_number']:02d}"
    )

    print(
        f"  Rows : "
        f"{task['row_count']:,}"
    )

    print(
        f"  ID   : "
        f"{task['task_id']}"
    )

    print(
        f"  File : "
        f"{task['file_prefix']}.csv"
    )


print(
    f"\nManifest:\n{REPORT_FILE}"
)


print(
    "\nIMPORTANT:"
    "\n- Earth Engine processing is asynchronous."
    "\n- Do NOT generate source labels yet."
    "\n- Wait until all 4 CSV exports complete."
    "\n- Do NOT run the classifier yet."
)