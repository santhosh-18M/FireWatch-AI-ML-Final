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
    / "sentinel_batch_export_manifest_v2.json"
)

PROJECT_ID = "firewatch-ai-sih"


# ------------------------------------------------------------
# VALIDATION BATCH ONLY
#
# DO NOT change this to 20,000 yet.
# We first validate the complete pipeline using 500 points.
# ------------------------------------------------------------

BATCH_SIZE = 500

RANDOM_STATE = 42

LOOKBACK_DAYS = 14

MAX_SCENE_CLOUD_PERCENTAGE = 80

BUFFER_METERS = 100

REDUCE_SCALE_METERS = 20


# ------------------------------------------------------------
# Export configuration
# ------------------------------------------------------------

EXPORT_DESCRIPTION = (
    "firewatch_sentinel_validation_500_v2"
)

EXPORT_FOLDER = "FireWatch_Sentinel"

EXPORT_PREFIX = (
    "firewatch_sentinel_validation_500_v2"
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — SENTINEL-2 SERVER-SIDE BATCH EXTRACTION V2")
print("=" * 80)

print(
    "\nDataset:"
    "\nCOPERNICUS/S2_SR_HARMONIZED"
)

print(
    f"\nEarth Engine project:"
    f"\n{PROJECT_ID}"
)

print(
    f"\nValidation batch: "
    f"{BATCH_SIZE:,} observations"
)

print(
    f"Past-only Sentinel window: "
    f"{LOOKBACK_DAYS} days"
)

print(
    "\nImportant:"
    "\n- No future Sentinel imagery"
    "\n- No fire labels"
    "\n- No fabricated satellite values"
    "\n- Missing imagery remains missing"
    "\n- 2025 FIRMS holdout is not accessed"
)


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

print(
    "\nInitializing Earth Engine..."
)

try:

    ee.Initialize(
        project=PROJECT_ID
    )

except Exception as exc:

    print(
        "\nEarth Engine initialization failed."
    )

    print(
        "\nAuthenticate using:"
        "\n  earthengine authenticate"
    )

    print(
        "\nThen run this script again."
    )

    raise exc


print(
    "Earth Engine initialized successfully."
)


# ============================================================
# LOAD SATELLITE SAMPLE
# ============================================================

print(
    "\n[1/6] Loading satellite sampling dataset..."
)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Missing input file:\n{INPUT_FILE}"
    )


sample = pd.read_parquet(
    INPUT_FILE
)


if len(sample) < BATCH_SIZE:

    raise RuntimeError(
        "Sampling dataset is smaller "
        "than requested validation batch."
    )


# ------------------------------------------------------------
# Deterministic validation subset
#
# Same random seed as V1 so we test the same observations.
# ------------------------------------------------------------

batch = (
    sample
    .sample(
        n=BATCH_SIZE,
        random_state=RANDOM_STATE
    )
    .sort_values(
        "acquired_at"
    )
    .reset_index(
        drop=True
    )
)


print(
    f"      Source sample: "
    f"{len(sample):,}"
)

print(
    f"      Validation batch: "
    f"{len(batch):,}"
)

print(
    f"      Period: "
    f"{batch['acquired_at'].min()} "
    f"to "
    f"{batch['acquired_at'].max()}"
)


# ============================================================
# BUILD EARTH ENGINE FEATURE COLLECTION
# ============================================================

print(
    "\n[2/6] Creating Earth Engine FeatureCollection..."
)


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
            float(row.longitude),
            float(row.latitude)
        ]
    )


    feature = ee.Feature(
        geometry,
        {
            "observation_id":
                str(row.observation_id),

            "event_time":
                event_iso,

            "latitude":
                float(row.latitude),

            "longitude":
                float(row.longitude),

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

            "sample_year":
                int(
                    row.sample_year
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
    f"      Features prepared: "
    f"{len(features):,}"
)


# ============================================================
# SENTINEL CLOUD MASK
# ============================================================

def mask_s2_scl(image):

    """
    Prepare Sentinel-2 surface reflectance.

    SCL classes removed:

        0  No data
        1  Saturated / defective
        3  Cloud shadow
        7  Unclassified / low-probability cloud
        8  Medium probability cloud
        9  High probability cloud
        10 Cirrus
        11 Snow / ice

    Remaining surface classes include vegetation,
    bare soil, water, etc.

    B4/B8/B11/B12 are converted from scaled integer
    reflectance to approximately 0-1 reflectance.
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


# ============================================================
# SAFE EMPTY IMAGE
# ============================================================

def create_empty_sentinel_image():

    """
    Create a completely MASKED fallback image.

    This image has the expected Sentinel bands so
    calculations such as NDVI/NBR/NDMI can still
    be constructed by Earth Engine.

    IMPORTANT:

    The pixels are fully masked.

    Therefore:

        B4/B8/B11/B12 -> missing
        NDVI/NBR/NDMI -> missing
        valid pixels -> 0
        satellite_available -> False

    Zero is NOT exported as a measurement.
    """

    empty = (
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
    )


    # Completely mask every pixel.

    empty = empty.updateMask(
        ee.Image.constant(0)
    )


    return empty


# ============================================================
# EXTRACT SENTINEL EVIDENCE
# ============================================================

print(
    "\n[3/6] Building server-side Sentinel processing graph..."
)


def extract_point(feature):

    """
    Extract Sentinel contextual evidence for one FIRMS
    observation.

    Workflow:

    FIRMS event
        ↓
    previous 14 days only
        ↓
    Sentinel-2 SR Harmonized
        ↓
    location filter
        ↓
    scene cloud filter
        ↓
    SCL pixel mask
        ↓
    median composite
        ↓
    B4/B8/B11/B12
        ↓
    NDVI/NBR/NDMI
        ↓
    100 m spatial mean

    No imagery after the FIRMS timestamp is used.
    """


    point = feature.geometry()


    # --------------------------------------------------------
    # FIRMS event timestamp
    # --------------------------------------------------------

    event_date = ee.Date(
        feature.get(
            "event_time"
        )
    )


    start_date = event_date.advance(
        -LOOKBACK_DAYS,
        "day"
    )


    # filterDate() has an exclusive end.
    #
    # Advancing one second allows observations exactly
    # at the event timestamp while preventing meaningful
    # future imagery from entering the search.

    end_date = event_date.advance(
        1,
        "second"
    )


    # --------------------------------------------------------
    # Sentinel collection
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Latest scene metadata
    #
    # Use conditional values instead of assuming a scene
    # always exists.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Process all available Sentinel scenes
    # --------------------------------------------------------

    processed = collection.map(
        mask_s2_scl
    )


    # --------------------------------------------------------
    # SAFE FALLBACK
    #
    # This fixes the V1 error:
    #
    # Image.normalizedDifference:
    # No band named 'B8'
    #
    # If no Sentinel scene exists, Earth Engine receives a
    # masked image with correctly named bands.
    # --------------------------------------------------------

    empty_fallback = (
        create_empty_sentinel_image()
    )


    composite = ee.Image(
        ee.Algorithms.If(
            has_scene,
            processed.median(),
            empty_fallback
        )
    )


    # --------------------------------------------------------
    # SPECTRAL INDICES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Spatial context
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # VALID PIXEL COUNT
    #
    # This is important because a scene can exist while
    # clouds/shadows leave no usable pixels around the point.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SATELLITE AVAILABILITY
    #
    # Scene existing is NOT enough.
    #
    # We require:
    #   scene_count > 0
    # AND
    #   at least one usable pixel after masking
    # --------------------------------------------------------

    satellite_available = (
        has_scene
        .And(
            valid_pixels.gt(
                0
            )
        )
    )


    # --------------------------------------------------------
    # IMAGE AGE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Extraction status
    # --------------------------------------------------------

    extraction_status = ee.Algorithms.If(
        satellite_available,
        "available",
        ee.Algorithms.If(
            has_scene,
            "scene_present_no_valid_pixels",
            "no_scene"
        )
    )


    # --------------------------------------------------------
    # Attach evidence to FIRMS feature
    # --------------------------------------------------------

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


    # Add B4/B8/B11/B12 + indices.
    #
    # For unavailable satellite observations these properties
    # remain missing because the fallback image is masked.

    result = result.set(
        statistics
    )


    return result


results = points.map(
    extract_point
)


print(
    "      Processing graph created successfully."
)


# ============================================================
# EXPORT COLUMNS
# ============================================================

print(
    "\n[4/6] Preparing Google Drive export..."
)


selectors = [

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    "observation_id",

    "event_time",

    "latitude",

    "longitude",


    # --------------------------------------------------------
    # Sampling metadata
    # --------------------------------------------------------

    "sample_year",

    "sample_context_group",

    "sample_temporal_group",

    "sample_frp_group",


    # --------------------------------------------------------
    # Sentinel metadata
    # --------------------------------------------------------

    "sentinel_scene_count",

    "sentinel_latest_time",

    "sentinel_latest_cloud_pct",

    "sentinel_mgrs_tile",

    "sentinel_image_age_days",

    "sentinel_valid_pixel_count",

    "satellite_available",

    "satellite_extraction_status",


    # --------------------------------------------------------
    # Reflectance
    # --------------------------------------------------------

    "B4",

    "B8",

    "B11",

    "B12",


    # --------------------------------------------------------
    # Spectral indices
    # --------------------------------------------------------

    "NDVI",

    "NBR",

    "NDMI",


    # --------------------------------------------------------
    # Method metadata
    # --------------------------------------------------------

    "sentinel_window_days",

    "sentinel_buffer_m",

    "sentinel_scale_m",
]


# ============================================================
# CREATE EXPORT TASK
# ============================================================

task = ee.batch.Export.table.toDrive(

    collection=results,

    description=EXPORT_DESCRIPTION,

    folder=EXPORT_FOLDER,

    fileNamePrefix=EXPORT_PREFIX,

    fileFormat="CSV",

    selectors=selectors
)


# ============================================================
# START TASK
# ============================================================

print(
    "\n[5/6] Starting Earth Engine batch task..."
)


task.start()


# Give Earth Engine a moment to register the task.

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
    f"      Task ID: "
    f"{task_id}"
)

print(
    f"      State: "
    f"{task_state}"
)


# ============================================================
# SAVE MANIFEST
# ============================================================

print(
    "\n[6/6] Saving export manifest..."
)


manifest = {

    "version":
        "v2_empty_collection_safe",

    "project":
        PROJECT_ID,

    "input_file":
        str(
            INPUT_FILE
        ),

    "batch_size":
        BATCH_SIZE,

    "random_state":
        RANDOM_STATE,

    "dataset":
        "COPERNICUS/S2_SR_HARMONIZED",

    "lookback_days":
        LOOKBACK_DAYS,

    "future_imagery_allowed":
        False,

    "max_scene_cloud_percentage":
        MAX_SCENE_CLOUD_PERCENTAGE,

    "cloud_mask":
        "Sentinel-2 SCL",

    "buffer_meters":
        BUFFER_METERS,

    "reduce_scale_meters":
        REDUCE_SCALE_METERS,

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
            (
                "Use fully masked fallback image. "
                "No spectral values are fabricated."
            ),

        "scene_but_no_valid_pixels":
            (
                "Satellite availability is false "
                "and spectral values remain missing."
            ),

        "zero_fill":
            False
    },

    "export": {

        "description":
            EXPORT_DESCRIPTION,

        "drive_folder":
            EXPORT_FOLDER,

        "filename_prefix":
            EXPORT_PREFIX,

        "format":
            "CSV"
    },

    "task": {

        "id":
            task_id,

        "state":
            task_state
    },

    "notes": [

        (
            "This is a 500-observation validation "
            "batch before scaling to the complete "
            "20,000-observation satellite sample."
        ),

        (
            "The same random seed as the failed V1 "
            "batch is used so the same observations "
            "are tested."
        ),

        (
            "Sentinel imagery is restricted to the "
            "14 days at or before each FIRMS event."
        ),

        (
            "SCL masking removes cloud, shadow, "
            "cirrus, snow and invalid pixels."
        ),

        (
            "B4/B8/B11/B12 are converted to "
            "reflectance using the 0.0001 scale."
        ),

        (
            "No-scene cases use a fully masked "
            "fallback image containing the expected "
            "band names."
        ),

        (
            "The fallback does not fabricate "
            "reflectance or spectral-index values."
        ),

        (
            "Satellite evidence is contextual "
            "evidence, not a fire label."
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
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "SENTINEL VALIDATION EXPORT V2 SUBMITTED"
)

print(
    "=" * 80
)


print(
    f"\nTask ID:"
    f"\n{task_id}"
)


print(
    f"\nCurrent state: "
    f"{task_state}"
)


print(
    "\nGoogle Drive destination:"
)

print(
    f"  Folder: "
    f"{EXPORT_FOLDER}"
)

print(
    f"  File: "
    f"{EXPORT_PREFIX}.csv"
)


print(
    f"\nManifest:"
    f"\n{MANIFEST_FILE}"
)


print(
    "\nIMPORTANT:"
)

print(
    "Do NOT change BATCH_SIZE to 20,000 yet."
)


print(
    "\nCheck the task using:"
)

print(
    "\nearthengine --project "
    "firewatch-ai-sih task list"
)


print(
    "\nWhen completed, download:"
)

print(
    "\nfirewatch_sentinel_validation_500_v2.csv"
)


print(
    "\nand place it at:"
)

print(
    "\ndata\\satellite\\exports\\"
    "firewatch_sentinel_validation_500_v2.csv"
)


print(
    "\nThe next validation stage will check:"
)

print(
    "  - exactly 500 observations"
)

print(
    "  - observation ID uniqueness"
)

print(
    "  - Sentinel availability rate"
)

print(
    "  - no future-image leakage"
)

print(
    "  - scene count"
)

print(
    "  - image age"
)

print(
    "  - valid pixel counts"
)

print(
    "  - missing-data behavior"
)

print(
    "  - B4/B8/B11/B12 ranges"
)

print(
    "  - NDVI/NBR/NDMI ranges"
)

print(
    "  - context-group coverage"
)

print(
    "  - temporal-group coverage"
)

print(
    "\n2025 holdout remains untouched."
)