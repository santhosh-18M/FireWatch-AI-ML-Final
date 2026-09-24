from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT
    / "data"
    / "labels"
    / "industrial_reference"
    / "raw"
)

INPUT_FILE = (
    RAW_DIR
    / "Global-Oil-and-Gas-Extraction-Tracker-March-2026.xlsx"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "labels"
    / "industrial_reference"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "india_industrial_reference.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "india_industrial_reference.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "industrial_reference_build.json"
)

SHEET = "Field-level main data"


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 13B-2")
print("BUILD INDEPENDENT INDUSTRIAL REFERENCE")
print("=" * 80)

print(
    "\nPurpose:"
    "\nBuild a clean India oil/gas industrial-reference"
    "\ndataset from Global Energy Monitor."
)

print(
    "\nImportant:"
    "\n- This is positive industrial evidence only."
    "\n- Absence from this dataset does NOT mean natural."
    "\n- Offshore fields will NOT create land-based labels."
    "\n- OSM is NOT used here."
    "\n- No ML labels are created."
    "\n- 2025 holdout remains untouched."
)


# ============================================================
# 1. LOAD
# ============================================================

print("\n[1/7] Loading GEM oil/gas extraction tracker...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing input file:\n{INPUT_FILE}"
    )

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET
)

print(f"      Global field rows: {len(df):,}")


# ============================================================
# 2. CHECK COLUMNS
# ============================================================

print("\n[2/7] Checking schema...")

required = [
    "Unit ID",
    "Unit Name",
    "Fuel type",
    "Country/Area",
    "Production Type",
    "Status",
    "Latitude",
    "Longitude",
    "Location accuracy",
    "Onshore/Offshore",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print("      Required columns present.")


# ============================================================
# 3. FILTER INDIA
# ============================================================

print("\n[3/7] Filtering India...")

india = df[
    df["Country/Area"]
    .astype(str)
    .str.strip()
    .str.casefold()
    .eq("india")
].copy()

print(f"      India records: {len(india):,}")


# ============================================================
# 4. CLEAN COORDINATES
# ============================================================

print("\n[4/7] Cleaning coordinates...")

india["Latitude"] = pd.to_numeric(
    india["Latitude"],
    errors="coerce"
)

india["Longitude"] = pd.to_numeric(
    india["Longitude"],
    errors="coerce"
)

india["has_coordinates"] = (
    india["Latitude"].notna()
    &
    india["Longitude"].notna()
)

print(
    f"      With coordinates   : "
    f"{india['has_coordinates'].sum():,}"
)

print(
    f"      Without coordinates: "
    f"{(~india['has_coordinates']).sum():,}"
)


# India geographic sanity check.
valid_coordinate = (
    india["has_coordinates"]
    &
    india["Latitude"].between(6, 38)
    &
    india["Longitude"].between(68, 98)
)

india["coordinate_valid_india"] = (
    valid_coordinate
)

invalid_geo = int(
    (
        india["has_coordinates"]
        &
        ~india["coordinate_valid_india"]
    ).sum()
)

print(
    f"      Coordinates outside India bounds: "
    f"{invalid_geo:,}"
)


# ============================================================
# 5. NORMALIZE METADATA
# ============================================================

print("\n[5/7] Normalizing industrial metadata...")


def clean_text(series):
    return (
        series
        .astype("string")
        .str.strip()
    )


for column in [
    "Unit ID",
    "Unit Name",
    "Fuel type",
    "Production Type",
    "Status",
    "Location accuracy",
    "Onshore/Offshore",
]:

    india[column] = clean_text(
        india[column]
    )


india["status_normalized"] = (
    india["Status"]
    .str.lower()
)


india["location_accuracy_normalized"] = (
    india["Location accuracy"]
    .str.lower()
)


india["onshore_offshore_normalized"] = (
    india["Onshore/Offshore"]
    .str.lower()
)


# Active/relevant operational status.
#
# Both operating and in-development are preserved as
# industrial facilities. Status itself remains available.

india["industrial_status_relevant"] = (
    india["status_normalized"]
    .isin(
        [
            "operating",
            "in-development",
        ]
    )
)


india["is_onshore"] = (
    india[
        "onshore_offshore_normalized"
    ]
    .eq("onshore")
)


india["is_offshore"] = (
    india[
        "onshore_offshore_normalized"
    ]
    .eq("offshore")
)


india["location_exact"] = (
    india[
        "location_accuracy_normalized"
    ]
    .eq("exact")
)


india["location_approximate"] = (
    india[
        "location_accuracy_normalized"
    ]
    .eq("approximate")
)


# ============================================================
# 6. BUILD NORMALIZED REFERENCE
# ============================================================

print("\n[6/7] Building normalized reference...")


reference = pd.DataFrame(
    {
        "industrial_reference_id":
            india["Unit ID"],

        "industrial_reference_name":
            india["Unit Name"],

        "industrial_category":
            "oil_gas_extraction",

        "fuel_type":
            india["Fuel type"],

        "production_type":
            india["Production Type"],

        "status":
            india["Status"],

        "latitude":
            india["Latitude"],

        "longitude":
            india["Longitude"],

        "location_accuracy":
            india["Location accuracy"],

        "onshore_offshore":
            india["Onshore/Offshore"],

        "has_coordinates":
            india["has_coordinates"],

        "coordinate_valid_india":
            india["coordinate_valid_india"],

        "industrial_status_relevant":
            india["industrial_status_relevant"],

        "is_onshore":
            india["is_onshore"],

        "is_offshore":
            india["is_offshore"],

        "location_exact":
            india["location_exact"],

        "location_approximate":
            india["location_approximate"],

        "reference_source":
            "Global Energy Monitor",

        "reference_dataset":
            "Global Oil and Gas Extraction Tracker",

        "reference_release":
            "March 2026",

        "reference_role":
            "independent_positive_industrial_evidence",
    }
)


# Remove exact duplicate reference IDs.
duplicate_ids = int(
    reference[
        "industrial_reference_id"
    ].duplicated().sum()
)

print(
    f"      Duplicate reference IDs: "
    f"{duplicate_ids:,}"
)

if duplicate_ids:

    reference = (
        reference
        .drop_duplicates(
            subset=[
                "industrial_reference_id"
            ],
            keep="first"
        )
        .copy()
    )


# ============================================================
# MATCHING ELIGIBILITY
# ============================================================

# For our land-based FIRMS development data, only onshore
# facilities with valid coordinates and relevant status
# are allowed to become spatial positive evidence.

reference[
    "eligible_for_land_matching"
] = (
    reference[
        "coordinate_valid_india"
    ]
    &
    reference[
        "industrial_status_relevant"
    ]
    &
    reference[
        "is_onshore"
    ]
)


eligible = reference[
    reference[
        "eligible_for_land_matching"
    ]
].copy()


print(
    f"      Total India references       : "
    f"{len(reference):,}"
)

print(
    f"      Eligible land references     : "
    f"{len(eligible):,}"
)

print(
    f"      Exact eligible locations     : "
    f"{eligible['location_exact'].sum():,}"
)

print(
    f"      Approximate eligible locations: "
    f"{eligible['location_approximate'].sum():,}"
)


# ============================================================
# 7. SAVE + AUDIT
# ============================================================

print("\n[7/7] Saving reference dataset...")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


reference.to_parquet(
    OUTPUT_FILE,
    index=False
)


reference.to_csv(
    OUTPUT_CSV,
    index=False
)


status_counts = (
    reference["status"]
    .fillna("MISSING")
    .value_counts()
    .to_dict()
)


onshore_counts = (
    reference["onshore_offshore"]
    .fillna("MISSING")
    .value_counts()
    .to_dict()
)


accuracy_counts = (
    reference["location_accuracy"]
    .fillna("MISSING")
    .value_counts()
    .to_dict()
)


eligible_status_counts = (
    eligible["status"]
    .fillna("MISSING")
    .value_counts()
    .to_dict()
)


report = {
    "stage":
        "independent_industrial_reference_build",

    "source":
        "Global Energy Monitor",

    "dataset":
        "Global Oil and Gas Extraction Tracker",

    "release":
        "March 2026",

    "global_field_rows":
        int(len(df)),

    "india_rows":
        int(len(india)),

    "india_with_coordinates":
        int(
            india[
                "has_coordinates"
            ].sum()
        ),

    "india_without_coordinates":
        int(
            (~india[
                "has_coordinates"
            ]).sum()
        ),

    "coordinates_outside_india_bounds":
        invalid_geo,

    "reference_rows":
        int(len(reference)),

    "eligible_land_matching_rows":
        int(len(eligible)),

    "eligible_exact_locations":
        int(
            eligible[
                "location_exact"
            ].sum()
        ),

    "eligible_approximate_locations":
        int(
            eligible[
                "location_approximate"
            ].sum()
        ),

    "status_distribution":
        {
            str(k): int(v)
            for k, v
            in status_counts.items()
        },

    "onshore_offshore_distribution":
        {
            str(k): int(v)
            for k, v
            in onshore_counts.items()
        },

    "location_accuracy_distribution":
        {
            str(k): int(v)
            for k, v
            in accuracy_counts.items()
        },

    "eligible_status_distribution":
        {
            str(k): int(v)
            for k, v
            in eligible_status_counts.items()
        },

    "matching_policy": {
        "requires_valid_coordinate": True,
        "requires_onshore": True,
        "accepted_status": [
            "operating",
            "in-development",
        ],
        "exact_and_approximate_retained": True,
    },

    "important_limitations": [
        (
            "This reference does not represent every "
            "industrial facility in India."
        ),
        (
            "Absence from the tracker must not be "
            "interpreted as non-industrial."
        ),
        (
            "Approximate coordinates have greater "
            "spatial uncertainty than exact coordinates."
        ),
        (
            "Offshore fields are retained for provenance "
            "but excluded from land-based matching."
        ),
        (
            "This dataset represents oil/gas extraction "
            "and does not cover all power plants, steel "
            "plants, refineries, mines or other industries."
        ),
    ],

    "labels_created":
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
        report,
        file,
        indent=2
    )


print("\n" + "=" * 80)
print("INDUSTRIAL REFERENCE BUILD COMPLETE")
print("=" * 80)

print(
    f"\nIndia reference rows : "
    f"{len(reference):,}"
)

print(
    f"Eligible for matching: "
    f"{len(eligible):,}"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nCSV:")
print(OUTPUT_CSV)

print("\nReport:")
print(REPORT_FILE)

print(
    "\nNO SOURCE LABELS CREATED."
)

print(
    "2025 HOLDOUT ACCESSED: NO"
)

print(
    "\nNEXT:"
    "\nSpatially compare these independent industrial"
    "\nreferences with the 20,000 development observations."
)