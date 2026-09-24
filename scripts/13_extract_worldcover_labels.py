from pathlib import Path
import json
import ee
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "satellite"
    / "sentinel_sampling_2022_2024.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "worldcover_export_manifest.json"
)

EE_PROJECT = "firewatch-ai-sih"

# ESA WorldCover 2021 v200
WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"

DRIVE_FOLDER = "FireWatch_WorldCover"

BATCH_SIZE = 5000

EXPECTED_ROWS = 20_000


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 13A: ESA WORLDCOVER EXTRACTION")
print("=" * 80)

print(
    "\nPurpose:"
    "\nExtract external ESA WorldCover 2021 land-cover"
    "\nevidence for the 20,000 development observations."
)

print(
    "\nIMPORTANT:"
    "\n- This does NOT create final ML labels."
    "\n- WorldCover is label/reference evidence."
    "\n- OSM is not used to create these WorldCover values."
    "\n- 2025 holdout remains untouched."
)

print("\nInitializing Earth Engine...")

try:

    ee.Initialize(
        project=EE_PROJECT
    )

except Exception as exc:

    print(
        "\nEarth Engine initialization failed."
    )

    print(
        "\nRun:"
        "\n  earthengine authenticate"
    )

    raise exc


print("Earth Engine initialized.")


# ============================================================
# LOAD SAMPLE
# ============================================================

print("\n[1/5] Loading 20,000-point development sample...")


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file missing:\n{INPUT_FILE}"
    )


df = pd.read_parquet(
    INPUT_FILE
)


print(
    f"      Rows: {len(df):,}"
)


if len(df) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} rows, "
        f"found {len(df):,}."
    )


required_columns = [
    "observation_id",
    "latitude",
    "longitude",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


if df["observation_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate observation IDs detected."
    )


if df[
    [
        "latitude",
        "longitude"
    ]
].isna().any().any():

    raise RuntimeError(
        "Missing coordinates detected."
    )


# Deterministic order

df = (
    df
    .sort_values(
        "observation_id"
    )
    .reset_index(
        drop=True
    )
)


df[
    "worldcover_export_row"
] = range(
    len(df)
)


df[
    "worldcover_export_batch"
] = (
    df[
        "worldcover_export_row"
    ]
    // BATCH_SIZE
    + 1
)


print(
    f"      Unique IDs: "
    f"{df['observation_id'].nunique():,}"
)

print(
    f"      Batches: "
    f"{df['worldcover_export_batch'].nunique()}"
)


# ============================================================
# LOAD WORLDCOVER
# ============================================================

print("\n[2/5] Loading ESA WorldCover 2021...")


worldcover = (
    ee.ImageCollection(
        WORLDCOVER_COLLECTION
    )
    .first()
    .select(
        "Map"
    )
)


print(
    f"      Collection: {WORLDCOVER_COLLECTION}"
)

print(
    "      Band: Map"
)

print(
    "      Nominal resolution: 10 m"
)


# ============================================================
# WORLDCOVER CLASS REFERENCE
# ============================================================

# ESA WorldCover v200 classes:
#
# 10 = Tree cover
# 20 = Shrubland
# 30 = Grassland
# 40 = Cropland
# 50 = Built-up
# 60 = Bare / sparse vegetation
# 70 = Snow and ice
# 80 = Permanent water bodies
# 90 = Herbaceous wetland
# 95 = Mangroves
# 100 = Moss and lichen
#
# IMPORTANT:
#
# "Built-up" does NOT automatically mean industrial.
# Cropland/grass/tree classes do NOT automatically prove
# that a FIRMS detection is a natural fire.
#
# These are external land-cover observations only.


# ============================================================
# CREATE EE FEATURES
# ============================================================

print("\n[3/5] Preparing Earth Engine features...")


def create_feature(row):

    point = ee.Geometry.Point(
        [
            float(
                row["longitude"]
            ),
            float(
                row["latitude"]
            )
        ]
    )

    return ee.Feature(
        point,
        {
            "observation_id":
                str(
                    row[
                        "observation_id"
                    ]
                ),

            "latitude":
                float(
                    row[
                        "latitude"
                    ]
                ),

            "longitude":
                float(
                    row[
                        "longitude"
                    ]
                ),

            "worldcover_export_row":
                int(
                    row[
                        "worldcover_export_row"
                    ]
                ),

            "worldcover_export_batch":
                int(
                    row[
                        "worldcover_export_batch"
                    ]
                ),
        }
    )


# ============================================================
# EXTRACTION FUNCTION
# ============================================================

def extract_worldcover(feature):

    geometry = feature.geometry()

    # --------------------------------------------------------
    # Point class
    # --------------------------------------------------------

    point_result = worldcover.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=geometry,
        scale=10,
        bestEffort=False,
        maxPixels=10000,
    )

    point_class = point_result.get(
        "Map"
    )

    point_available = ee.Algorithms.If(
    ee.Algorithms.IsEqual(
        point_class,
        None
    ),
    False,
    True
)


    # --------------------------------------------------------
    # 100 m neighborhood mode
    #
    # A FIRMS coordinate is not an exact fire boundary.
    # Therefore we also record the dominant land-cover class
    # within a 100 m buffer.
    # --------------------------------------------------------

    buffer_geometry = geometry.buffer(
        100
    )

    mode_result = worldcover.reduceRegion(
        reducer=ee.Reducer.mode(),
        geometry=buffer_geometry,
        scale=10,
        bestEffort=False,
        maxPixels=100000,
    )

    mode_class = mode_result.get(
        "Map"
    )


    # --------------------------------------------------------
    # Histogram within 100 m
    #
    # This preserves mixed land-cover evidence instead of
    # reducing everything to one class.
    # --------------------------------------------------------

    histogram_result = worldcover.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=buffer_geometry,
        scale=10,
        bestEffort=False,
        maxPixels=100000,
    )

    histogram = ee.Dictionary(
        ee.Algorithms.If(
            histogram_result.get(
                "Map"
            ),
            histogram_result.get(
                "Map"
            ),
            ee.Dictionary({})
        )
    )


    total_pixels = ee.Number(
        histogram.values().reduce(
            ee.Reducer.sum()
        )
    )


    # Helper: percentage of one class inside 100 m.

    def class_fraction(class_code):

        key = ee.Number(
            class_code
        ).format()

        count = ee.Number(
            histogram.get(
                key,
                0
            )
        )

        return ee.Algorithms.If(
            total_pixels.gt(0),
            count.divide(
                total_pixels
            ),
            None
        )


    tree_fraction = class_fraction(10)
    shrub_fraction = class_fraction(20)
    grass_fraction = class_fraction(30)
    cropland_fraction = class_fraction(40)
    built_fraction = class_fraction(50)
    bare_fraction = class_fraction(60)
    water_fraction = class_fraction(80)
    wetland_fraction = class_fraction(90)
    mangrove_fraction = class_fraction(95)


    return feature.set(
        {
            "worldcover_point_class":
                point_class,

            "worldcover_point_available":
                point_available,

            "worldcover_mode_100m":
                mode_class,

            "worldcover_total_pixels_100m":
                total_pixels,

            "worldcover_tree_fraction_100m":
                tree_fraction,

            "worldcover_shrub_fraction_100m":
                shrub_fraction,

            "worldcover_grass_fraction_100m":
                grass_fraction,

            "worldcover_cropland_fraction_100m":
                cropland_fraction,

            "worldcover_built_fraction_100m":
                built_fraction,

            "worldcover_bare_fraction_100m":
                bare_fraction,

            "worldcover_water_fraction_100m":
                water_fraction,

            "worldcover_wetland_fraction_100m":
                wetland_fraction,

            "worldcover_mangrove_fraction_100m":
                mangrove_fraction,

            "worldcover_source":
                "ESA_WorldCover_2021_v200",

            "worldcover_buffer_m":
                100,

            "worldcover_scale_m":
                10,
        }
    )


# ============================================================
# EXPORT BATCHES
# ============================================================

print("\n[4/5] Creating Earth Engine export tasks...")


selectors = [
    "observation_id",
    "latitude",
    "longitude",
    "worldcover_export_row",
    "worldcover_export_batch",

    "worldcover_point_class",
    "worldcover_point_available",

    "worldcover_mode_100m",

    "worldcover_total_pixels_100m",

    "worldcover_tree_fraction_100m",
    "worldcover_shrub_fraction_100m",
    "worldcover_grass_fraction_100m",
    "worldcover_cropland_fraction_100m",
    "worldcover_built_fraction_100m",
    "worldcover_bare_fraction_100m",
    "worldcover_water_fraction_100m",
    "worldcover_wetland_fraction_100m",
    "worldcover_mangrove_fraction_100m",

    "worldcover_source",
    "worldcover_buffer_m",
    "worldcover_scale_m",
]


tasks = []


for batch_number in range(
    1,
    5
):

    batch = df[
        df[
            "worldcover_export_batch"
        ]
        == batch_number
    ].copy()


    print(
        f"\n      Preparing Batch {batch_number}: "
        f"{len(batch):,} observations"
    )


    feature_list = [
        create_feature(row)
        for _, row
        in batch.iterrows()
    ]


    feature_collection = (
        ee.FeatureCollection(
            feature_list
        )
        .map(
            extract_worldcover
        )
    )


    description = (
        f"firewatch_worldcover_batch_{batch_number:02d}"
    )


    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,

        description=description,

        folder=DRIVE_FOLDER,

        fileNamePrefix=description,

        fileFormat="CSV",

        selectors=selectors,
    )


    task.start()


    status = task.status()


    task_id = status.get(
        "id",
        "UNKNOWN"
    )


    print(
        f"      Task started: {task_id}"
    )


    tasks.append(
        {
            "batch":
                batch_number,

            "rows":
                int(
                    len(batch)
                ),

            "description":
                description,

            "task_id":
                task_id,
        }
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

print("\n[5/5] Saving export manifest...")


manifest = {

    "stage":
        "worldcover_external_label_evidence",

    "source":
        "ESA WorldCover 2021 v200",

    "earth_engine_collection":
        WORLDCOVER_COLLECTION,

    "resolution_m":
        10,

    "buffer_m":
        100,

    "sample_rows":
        EXPECTED_ROWS,

    "batch_size":
        BATCH_SIZE,

    "number_of_batches":
        4,

    "drive_folder":
        DRIVE_FOLDER,

    "tasks":
        tasks,

    "class_reference": {

        "10":
            "Tree cover",

        "20":
            "Shrubland",

        "30":
            "Grassland",

        "40":
            "Cropland",

        "50":
            "Built-up",

        "60":
            "Bare / sparse vegetation",

        "70":
            "Snow and ice",

        "80":
            "Permanent water bodies",

        "90":
            "Herbaceous wetland",

        "95":
            "Mangroves",

        "100":
            "Moss and lichen",
    },

    "methodology": {

        "point_class":
            (
                "WorldCover class at the FIRMS "
                "observation coordinate."
            ),

        "mode_100m":
            (
                "Dominant WorldCover class within "
                "100 metres."
            ),

        "fractions_100m":
            (
                "Fractional land-cover composition "
                "within 100 metres."
            ),

        "role":
            (
                "External land-cover label/reference "
                "evidence. Not automatically converted "
                "into final source class."
            ),
    },

    "important_limitations": [

        (
            "WorldCover Built-up class does not "
            "specifically identify industrial facilities."
        ),

        (
            "Vegetation land cover does not by itself "
            "prove that a FIRMS thermal anomaly is "
            "a natural vegetation fire."
        ),

        (
            "WorldCover 2021 is a static reference "
            "used for observations from 2022-2024."
        ),
    ],

    "final_labels_assigned":
        False,

    "holdout_2025_accessed":
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
) as file:

    json.dump(
        manifest,
        file,
        indent=2
    )


print(
    "\n" + "=" * 80
)

print(
    "WORLDCOVER EXPORT TASKS STARTED"
)

print(
    "=" * 80
)


for task in tasks:

    print(
        f"\nBatch {task['batch']}"
    )

    print(
        f"  Rows    : {task['rows']:,}"
    )

    print(
        f"  Task ID : {task['task_id']}"
    )


print(
    "\nGoogle Drive folder:"
)

print(
    DRIVE_FOLDER
)


print(
    "\nManifest:"
)

print(
    REPORT_FILE
)


print(
    "\nCheck status with:"
)

print(
    f"earthengine --project {EE_PROJECT} task list"
)


print(
    "\n2025 HOLDOUT REMAINS LOCKED."
)

print(
    "\nDo NOT create source labels yet."
)