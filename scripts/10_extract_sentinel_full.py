from pathlib import Path
import json
import time

import ee
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "satellite"
    / "sentinel_sampling_2022_2024.parquet"
)

MANIFEST_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "sentinel_full_export_manifest.json"
)

PROJECT_ID = "firewatch-ai-sih"

EXPECTED_ROWS = 20_000

BATCH_SIZE = 5_000

LOOKBACK_DAYS = 14

MAX_SCENE_CLOUD_PERCENTAGE = 80

BUFFER_METERS = 100

REDUCE_SCALE_METERS = 20

EXPORT_FOLDER = "FireWatch_Sentinel_Full"


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — FULL SENTINEL-2 BATCH EXTRACTION")
print("=" * 80)

print(
    "\nPurpose:"
    "\nExtract validated Sentinel-2 contextual evidence"
    "\nfor the complete 20,000-observation development sample."
)

print(
    "\nConfiguration:"
    f"\n  Total observations : {EXPECTED_ROWS:,}"
    f"\n  Batch size         : {BATCH_SIZE:,}"
    f"\n  Expected batches   : {EXPECTED_ROWS // BATCH_SIZE}"
    f"\n  Lookback           : {LOOKBACK_DAYS} days"
    f"\n  Buffer             : {BUFFER_METERS} m"
    f"\n  Reduction scale    : {REDUCE_SCALE_METERS} m"
)

print(
    "\nImportant:"
    "\n- Same methodology as validated 500-point V2"
    "\n- No future Sentinel imagery"
    "\n- No fabricated spectral values"
    "\n- No labels are created"
    "\n- 2025 holdout is untouched"
)


# ============================================================
# EARTH ENGINE
# ============================================================

print(
    "\nInitializing Earth Engine..."
)

ee.Initialize(
    project=PROJECT_ID
)

print(
    "Earth Engine initialized successfully."
)


# ============================================================
# LOAD SAMPLE
# ============================================================

print(
    "\n[1/6] Loading satellite sampling dataset..."
)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing input:\n{INPUT_FILE}"
    )


sample = pd.read_parquet(
    INPUT_FILE
)


print(
    f"      Rows: {len(sample):,}"
)


if len(sample) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} rows, "
        f"found {len(sample):,}."
    )


if sample[
    "observation_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate observation IDs detected."
    )


# ------------------------------------------------------------
# IMPORTANT
#
# Keep the sample in a deterministic order.
#
# We do NOT randomly resample here.
# Every one of the original 20,000 observations must appear
# exactly once across the four exports.
# ------------------------------------------------------------

sample = (
    sample
    .sort_values(
        [
            "observation_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


sample[
    "full_export_row"
] = range(
    len(sample)
)


sample[
    "full_export_batch"
] = (
    sample[
        "full_export_row"
    ]
    // BATCH_SIZE
) + 1


batch_counts = (
    sample[
        "full_export_batch"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\n      Batch distribution:"
)

for batch_number, count in batch_counts.items():

    print(
        f"        Batch {batch_number}: "
        f"{count:,}"
    )


if len(batch_counts) != 4:

    raise RuntimeError(
        "Expected exactly four batches."
    )


if not all(
    batch_counts == BATCH_SIZE
):

    raise RuntimeError(
        "Each batch must contain exactly 5,000 rows."
    )


# ============================================================
# SENTINEL FUNCTIONS
# ============================================================

def mask_s2_scl(image):

    """
    Apply Sentinel-2 SCL quality masking and convert
    B4/B8/B11/B12 to reflectance.

    Removed SCL classes:

        0  No data
        1  Saturated / defective
        3  Cloud shadow
        7  Unclassified
        8  Medium probability cloud
        9  High probability cloud
        10 Cirrus
        11 Snow / ice

    This is the same policy used by the validated V2 run.
    """

    scl = image.select(
        "SCL"
    )


    mask = (
        scl.neq(0)
        .And(
            scl.neq(1)
        )
        .And(
            scl.neq(3)
        )
        .And(
            scl.neq(7)
        )
        .And(
            scl.neq(8)
        )
        .And(
            scl.neq(9)
        )
        .And(
            scl.neq(10)
        )
        .And(
            scl.neq(11)
        )
    )


    reflectance = (
        image
        .select(
            [
                "B4",
                "B8",
                "B11",
                "B12"
            ]
        )
        .multiply(
            0.0001
        )
        .updateMask(
            mask
        )
    )


    return reflectance.copyProperties(
        image,
        [
            "system:time_start",
            "CLOUDY_PIXEL_PERCENTAGE",
            "MGRS_TILE"
        ]
    )


def create_empty_sentinel_image():

    """
    Completely masked fallback.

    The expected bands exist, but no pixel contains
    a valid measurement.

    Therefore missing imagery stays missing.
    """

    return (
        ee.Image.constant(
            [
                0,
                0,
                0,
                0
            ]
        )
        .rename(
            [
                "B4",
                "B8",
                "B11",
                "B12"
            ]
        )
        .updateMask(
            ee.Image.constant(0)
        )
    )


def extract_point(feature):

    """
    Extract past-only Sentinel evidence for one FIRMS
    observation.
    """

    point = feature.geometry()


    event_date = ee.Date(
        feature.get(
            "event_time"
        )
    )


    start_date = event_date.advance(
        -LOOKBACK_DAYS,
        "day"
    )


    # Earth Engine filterDate() uses an exclusive end.
    #
    # One second is added only to permit a scene exactly
    # at the event timestamp.

    end_date = event_date.advance(
        1,
        "second"
    )


    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(
            point
        )
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_SCENE_CLOUD_PERCENTAGE
            )
        )
        .sort(
            "system:time_start",
            False
        )
    )


    scene_count = collection.size()

    has_scene = scene_count.gt(
        0
    )


    # ========================================================
    # Latest candidate scene metadata
    # ========================================================

    latest_image = ee.Image(
        ee.Algorithms.If(
            has_scene,
            collection.first(),
            ee.Image.constant(0)
        )
    )


    latest_time = ee.Algorithms.If(
        has_scene,
        latest_image.get(
            "system:time_start"
        ),
        None
    )


    latest_cloud = ee.Algorithms.If(
        has_scene,
        latest_image.get(
            "CLOUDY_PIXEL_PERCENTAGE"
        ),
        None
    )


    latest_tile = ee.Algorithms.If(
        has_scene,
        latest_image.get(
            "MGRS_TILE"
        ),
        None
    )


    # ========================================================
    # Cloud-mask all candidate scenes
    # ========================================================

    processed = collection.map(
        mask_s2_scl
    )


    empty_fallback = (
        create_empty_sentinel_image()
    )


    # ========================================================
    # Past-14-day median composite
    # ========================================================

    composite = ee.Image(
        ee.Algorithms.If(
            has_scene,
            processed.median(),
            empty_fallback
        )
    )


    # ========================================================
    # Spectral indices
    # ========================================================

    ndvi = (
        composite
        .normalizedDifference(
            [
                "B8",
                "B4"
            ]
        )
        .rename(
            "NDVI"
        )
    )


    nbr = (
        composite
        .normalizedDifference(
            [
                "B8",
                "B12"
            ]
        )
        .rename(
            "NBR"
        )
    )


    ndmi = (
        composite
        .normalizedDifference(
            [
                "B8",
                "B11"
            ]
        )
        .rename(
            "NDMI"
        )
    )


    evidence_image = (
        composite
        .addBands(
            ndvi
        )
        .addBands(
            nbr
        )
        .addBands(
            ndmi
        )
    )


    # ========================================================
    # 100 m context around FIRMS observation
    # ========================================================

    region = point.buffer(
        BUFFER_METERS
    )


    statistics = (
        evidence_image
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=REDUCE_SCALE_METERS,
            bestEffort=True,
            maxPixels=100000
        )
    )


    # ========================================================
    # Valid pixel count
    # ========================================================

    valid_pixel_raw = (
        composite
        .select(
            "B8"
        )
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=REDUCE_SCALE_METERS,
            bestEffort=True,
            maxPixels=100000
        )
        .get(
            "B8"
        )
    )


    valid_pixels = ee.Number(
        ee.Algorithms.If(
            valid_pixel_raw,
            valid_pixel_raw,
            0
        )
    )


    satellite_available = (
        has_scene
        .And(
            valid_pixels.gt(
                0
            )
        )
    )


    # ========================================================
    # Latest candidate scene age
    # ========================================================

    image_age_days = ee.Algorithms.If(
        has_scene,

        event_date.difference(
            ee.Date(
                latest_time
            ),
            "day"
        ),

        None
    )


    # ========================================================
    # Extraction status
    # ========================================================

    extraction_status = (
        ee.Algorithms.If(
            satellite_available,

            "available",

            ee.Algorithms.If(
                has_scene,

                "scene_present_no_valid_pixels",

                "no_scene"
            )
        )
    )


    # ========================================================
    # Result
    # ========================================================

    result = feature.set(
        {
            "sentinel_scene_count":
                scene_count,

            "sentinel_latest_time":
                latest_time,

            "sentinel_latest_cloud_pct":
                latest_cloud,

            "sentinel_mgrs_tile":
                latest_tile,

            "sentinel_image_age_days":
                image_age_days,

            "sentinel_valid_pixel_count":
                valid_pixels,

            "satellite_available":
                satellite_available,

            "satellite_extraction_status":
                extraction_status,

            "sentinel_window_days":
                LOOKBACK_DAYS,

            "sentinel_buffer_m":
                BUFFER_METERS,

            "sentinel_scale_m":
                REDUCE_SCALE_METERS,
        }
    )


    result = result.set(
        statistics
    )


    return result


# ============================================================
# EXPORT SELECTORS
# ============================================================

SELECTORS = [

    # Identity
    "observation_id",
    "event_time",
    "latitude",
    "longitude",

    # Batch tracking
    "full_export_batch",
    "full_export_row",

    # Sampling metadata
    "sample_year",
    "sample_context_group",
    "sample_temporal_group",
    "sample_frp_group",
    "sample_grid_id",

    # Sentinel metadata
    "sentinel_scene_count",
    "sentinel_latest_time",
    "sentinel_latest_cloud_pct",
    "sentinel_mgrs_tile",
    "sentinel_image_age_days",
    "sentinel_valid_pixel_count",
    "satellite_available",
    "satellite_extraction_status",

    # Reflectance
    "B4",
    "B8",
    "B11",
    "B12",

    # Spectral indices
    "NDVI",
    "NBR",
    "NDMI",

    # Extraction methodology
    "sentinel_window_days",
    "sentinel_buffer_m",
    "sentinel_scale_m",
]


# ============================================================
# SUBMIT FOUR BATCHES
# ============================================================

print(
    "\n[2/6] Preparing four non-overlapping Earth Engine batches..."
)


submitted_tasks = []


for batch_number in range(
    1,
    5
):

    print(
        "\n" + "-" * 80
    )

    print(
        f"PREPARING BATCH {batch_number}/4"
    )

    print(
        "-" * 80
    )


    batch = sample[
        sample[
            "full_export_batch"
        ] == batch_number
    ].copy()


    if len(batch) != BATCH_SIZE:

        raise RuntimeError(
            f"Batch {batch_number} contains "
            f"{len(batch):,} rows instead of "
            f"{BATCH_SIZE:,}."
        )


    print(
        f"Rows: {len(batch):,}"
    )

    print(
        f"Period: "
        f"{batch['acquired_at'].min()} "
        f"to "
        f"{batch['acquired_at'].max()}"
    )

    print(
        f"Unique IDs: "
        f"{batch['observation_id'].nunique():,}"
    )


    # ========================================================
    # Convert local rows to EE features
    # ========================================================

    features = []


    for row in batch.itertuples(
        index=False
    ):

        acquired_at = pd.Timestamp(
            row.acquired_at
        )


        event_iso = acquired_at.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )


        geometry = ee.Geometry.Point(
            [
                float(
                    row.longitude
                ),

                float(
                    row.latitude
                )
            ]
        )


        feature = ee.Feature(
            geometry,
            {
                "observation_id":
                    str(
                        row.observation_id
                    ),

                "event_time":
                    event_iso,

                "latitude":
                    float(
                        row.latitude
                    ),

                "longitude":
                    float(
                        row.longitude
                    ),

                "full_export_batch":
                    int(
                        row.full_export_batch
                    ),

                "full_export_row":
                    int(
                        row.full_export_row
                    ),

                "sample_year":
                    int(
                        row.sample_year
                    ),

                "sample_context_group":
                    str(
                        row.sample_context_group
                    ),

                "sample_temporal_group":
                    str(
                        row.sample_temporal_group
                    ),

                "sample_frp_group":
                    str(
                        row.sample_frp_group
                    ),

                "sample_grid_id":
                    str(
                        row.sample_grid_id
                    ),
            }
        )


        features.append(
            feature
        )


    points = ee.FeatureCollection(
        features
    )


    print(
        "Earth Engine FeatureCollection prepared."
    )


    # ========================================================
    # Server-side processing
    # ========================================================

    results = points.map(
        extract_point
    )


    print(
        "Sentinel processing graph prepared."
    )


    # ========================================================
    # Unique export names
    # ========================================================

    export_description = (
        f"firewatch_sentinel_full_batch_{batch_number:02d}"
    )


    export_prefix = (
        f"firewatch_sentinel_full_batch_{batch_number:02d}"
    )


    # ========================================================
    # Export
    # ========================================================

    task = ee.batch.Export.table.toDrive(

        collection=results,

        description=export_description,

        folder=EXPORT_FOLDER,

        fileNamePrefix=export_prefix,

        fileFormat="CSV",

        selectors=SELECTORS
    )


    task.start()


    # Allow task registration.

    time.sleep(
        2
    )


    status = task.status()


    task_id = status.get(
        "id",
        "unknown"
    )


    task_state = status.get(
        "state",
        "unknown"
    )


    print(
        f"Task ID: "
        f"{task_id}"
    )

    print(
        f"Initial state: "
        f"{task_state}"
    )


    submitted_tasks.append(
        {
            "batch":
                batch_number,

            "rows":
                len(batch),

            "first_export_row":
                int(
                    batch[
                        "full_export_row"
                    ].min()
                ),

            "last_export_row":
                int(
                    batch[
                        "full_export_row"
                    ].max()
                ),

            "description":
                export_description,

            "filename":
                f"{export_prefix}.csv",

            "task_id":
                task_id,

            "initial_state":
                task_state,
        }
    )


# ============================================================
# VERIFY LOCAL BATCH PARTITION
# ============================================================

print(
    "\n[3/6] Verifying batch partition..."
)


all_batch_ids = []


for batch_number in range(
    1,
    5
):

    ids = sample.loc[
        sample[
            "full_export_batch"
        ] == batch_number,

        "observation_id"
    ].tolist()


    all_batch_ids.extend(
        ids
    )


if len(all_batch_ids) != EXPECTED_ROWS:

    raise RuntimeError(
        "Batch partition row-count mismatch."
    )


if len(
    set(
        all_batch_ids
    )
) != EXPECTED_ROWS:

    raise RuntimeError(
        "Observation overlap exists between batches."
    )


original_ids = set(
    sample[
        "observation_id"
    ].tolist()
)


batch_ids = set(
    all_batch_ids
)


if original_ids != batch_ids:

    raise RuntimeError(
        "Batch IDs do not exactly match original sample."
    )


print(
    "      20,000/20,000 observations assigned."
)

print(
    "      No cross-batch duplicates."
)

print(
    "      No observations missing."
)


# ============================================================
# MANIFEST
# ============================================================

print(
    "\n[4/6] Saving export manifest..."
)


manifest = {

    "pipeline":
        "FireWatch Sentinel full extraction",

    "version":
        "validated_v2_method",

    "project":
        PROJECT_ID,

    "source_file":
        str(
            INPUT_FILE
        ),

    "total_observations":
        EXPECTED_ROWS,

    "batch_size":
        BATCH_SIZE,

    "number_of_batches":
        4,

    "dataset":
        "COPERNICUS/S2_SR_HARMONIZED",

    "lookback_days":
        LOOKBACK_DAYS,

    "future_imagery_allowed":
        False,

    "scene_cloud_prefilter_percentage":
        MAX_SCENE_CLOUD_PERCENTAGE,

    "pixel_mask":
        "Sentinel-2 SCL",

    "buffer_meters":
        BUFFER_METERS,

    "reduce_scale_meters":
        REDUCE_SCALE_METERS,

    "composite":
        "median of valid past-window scenes",

    "bands": [
        "B4",
        "B8",
        "B11",
        "B12"
    ],

    "indices": [
        "NDVI",
        "NBR",
        "NDMI"
    ],

    "missing_data_policy": {

        "no_scene":
            "fully masked fallback",

        "scene_without_valid_pixels":
            "missing spectral evidence",

        "zero_fill":
            False
    },

    "export_folder":
        EXPORT_FOLDER,

    "tasks":
        submitted_tasks,

    "methodology_notes": [

        (
            "Methodology is unchanged from the "
            "validated 500-point V2 extraction."
        ),

        (
            "The 20,000-observation sample is partitioned "
            "deterministically into four non-overlapping "
            "5,000-row batches."
        ),

        (
            "Spectral features represent a past-14-day "
            "cloud-masked Sentinel-2 median composite."
        ),

        (
            "Latest scene metadata describes the latest "
            "candidate scene in the search window and "
            "should not be interpreted as the sole image "
            "used to generate the composite."
        ),

        (
            "Satellite evidence is contextual evidence "
            "and does not itself constitute a fire label."
        ),

        (
            "Unavailable satellite evidence remains "
            "explicitly missing."
        ),

        (
            "2025 FIRMS holdout was not accessed."
        )
    ]
}


MANIFEST_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    MANIFEST_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# TASK SUMMARY
# ============================================================

print(
    "\n[5/6] Submitted task summary..."
)


for task_info in submitted_tasks:

    print(
        f"\nBatch "
        f"{task_info['batch']}:"
    )

    print(
        f"  Rows: "
        f"{task_info['rows']:,}"
    )

    print(
        f"  Task ID: "
        f"{task_info['task_id']}"
    )

    print(
        f"  State: "
        f"{task_info['initial_state']}"
    )

    print(
        f"  File: "
        f"{task_info['filename']}"
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n[6/6] Submission complete."
)

print(
    "\n" + "=" * 80
)

print(
    "FULL SENTINEL EXTRACTION SUBMITTED"
)

print(
    "=" * 80
)


print(
    "\nGoogle Drive folder:"
)

print(
    f"  {EXPORT_FOLDER}"
)


print(
    "\nExpected files:"
)

for batch_number in range(
    1,
    5
):

    print(
        f"  firewatch_sentinel_full_batch_"
        f"{batch_number:02d}.csv"
    )


print(
    "\nCheck tasks using:"
)

print(
    "\nearthengine --project "
    "firewatch-ai-sih task list"
)


print(
    "\nDo NOT rerun this script while these "
    "tasks are active."
)


print(
    "\nAfter all four tasks reach COMPLETED,"
    "\ndownload all four CSV files."
)


print(
    "\nPlace them in:"
)

print(
    "\nD:\\FireWatch_ML_Final\\data\\satellite\\exports\\full\\"
)


print(
    f"\nManifest:"
    f"\n{MANIFEST_FILE}"
)


print(
    "\nNext stage:"
    "\nMerge + validate all 20,000 Sentinel observations."
)

print(
    "\n2025 holdout remains untouched."
)