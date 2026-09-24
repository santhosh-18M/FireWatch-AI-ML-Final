from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ZIP_FILE = (
    ROOT
    / "data"
    / "labels"
    / "industrial_reference"
    / "raw"
    / "global-power-plant-database.zip"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "labels"
    / "industrial_reference"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "india_powerplant_reference.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "india_powerplant_reference.csv"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "powerplant_reference_build.json"
)

INNER_ZIP = "globalpowerplantdatabasev130.zip"
CSV_NAME = "global_power_plant_database.csv"

THERMAL_FUELS = {
    "Coal",
    "Gas",
    "Oil",
    "Biomass",
    "Nuclear",
}


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIREWATCH — STEP 13C-2")
print("BUILD INDIA POWER-PLANT INDUSTRIAL REFERENCE")
print("=" * 80)

print(
    "\nPurpose:"
    "\n- Read WRI Global Power Plant Database v1.3.0."
    "\n- Extract Indian power plants."
    "\n- Identify combustion/thermal industrial facilities."
    "\n- Preserve non-thermal plants for audit."
    "\n- Do NOT create source labels."
    "\n- Do NOT access 2025 holdout."
)


# ============================================================
# 1. READ NESTED ZIP
# ============================================================

print("\n[1/7] Reading nested WRI archive...")

if not ZIP_FILE.exists():
    raise FileNotFoundError(
        f"Missing ZIP:\n{ZIP_FILE}"
    )

with zipfile.ZipFile(ZIP_FILE, "r") as outer:

    names = outer.namelist()

    if INNER_ZIP not in names:
        raise RuntimeError(
            f"{INNER_ZIP} not found inside outer ZIP."
        )

    inner_bytes = outer.read(INNER_ZIP)


with zipfile.ZipFile(
    io.BytesIO(inner_bytes),
    "r"
) as inner:

    if CSV_NAME not in inner.namelist():
        raise RuntimeError(
            f"{CSV_NAME} not found inside inner ZIP."
        )

    with inner.open(CSV_NAME) as f:
        df = pd.read_csv(
            f,
            low_memory=False
        )


print(f"      Global plants: {len(df):,}")


# ============================================================
# 2. CHECK SCHEMA
# ============================================================

print("\n[2/7] Checking schema...")

required = [
    "country",
    "country_long",
    "name",
    "gppd_idnr",
    "capacity_mw",
    "latitude",
    "longitude",
    "primary_fuel",
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

print("      Required columns present.")


# ============================================================
# 3. INDIA ONLY
# ============================================================

print("\n[3/7] Filtering India...")

india = df[
    df["country_long"]
    .astype(str)
    .str.strip()
    .str.casefold()
    .eq("india")
].copy()

print(f"      India plants: {len(india):,}")


# ============================================================
# 4. COORDINATE VALIDATION
# ============================================================

print("\n[4/7] Validating coordinates...")

india["latitude"] = pd.to_numeric(
    india["latitude"],
    errors="coerce"
)

india["longitude"] = pd.to_numeric(
    india["longitude"],
    errors="coerce"
)

india["capacity_mw"] = pd.to_numeric(
    india["capacity_mw"],
    errors="coerce"
)

india["has_coordinates"] = (
    india["latitude"].notna()
    &
    india["longitude"].notna()
)

india["coordinate_valid_india"] = (
    india["has_coordinates"]
    &
    india["latitude"].between(6, 38)
    &
    india["longitude"].between(68, 98)
)

print(
    f"      Coordinates present : "
    f"{india['has_coordinates'].sum():,}"
)

print(
    f"      Valid India bounds  : "
    f"{india['coordinate_valid_india'].sum():,}"
)

print(
    f"      Invalid/out-of-range: "
    f"{(~india['coordinate_valid_india']).sum():,}"
)


# ============================================================
# 5. CLASSIFY REFERENCE ROLE
# ============================================================

print("\n[5/7] Classifying reference role...")

india["primary_fuel"] = (
    india["primary_fuel"]
    .astype("string")
    .str.strip()
)

india["thermal_industrial_reference"] = (
    india["primary_fuel"]
    .isin(THERMAL_FUELS)
    &
    india["coordinate_valid_india"]
)


def reference_group(fuel):

    if pd.isna(fuel):
        return "unknown"

    if fuel in {
        "Coal",
        "Gas",
        "Oil",
        "Biomass",
    }:
        return "combustion_power"

    if fuel == "Nuclear":
        return "nuclear_power"

    if fuel == "Hydro":
        return "hydro_power"

    if fuel == "Solar":
        return "solar_power"

    if fuel == "Wind":
        return "wind_power"

    return "other_power"


india["power_reference_group"] = (
    india["primary_fuel"]
    .apply(reference_group)
)


# ============================================================
# 6. BUILD NORMALIZED REFERENCE
# ============================================================

print("\n[6/7] Building normalized reference...")


reference = pd.DataFrame(
    {
        "industrial_reference_id":
            india["gppd_idnr"].astype("string"),

        "industrial_reference_name":
            india["name"].astype("string"),

        "industrial_category":
            "power_plant",

        "power_reference_group":
            india["power_reference_group"],

        "primary_fuel":
            india["primary_fuel"],

        "capacity_mw":
            india["capacity_mw"],

        "latitude":
            india["latitude"],

        "longitude":
            india["longitude"],

        "has_coordinates":
            india["has_coordinates"],

        "coordinate_valid_india":
            india["coordinate_valid_india"],

        "thermal_industrial_reference":
            india[
                "thermal_industrial_reference"
            ],

        "reference_source":
            "World Resources Institute",

        "reference_dataset":
            "Global Power Plant Database",

        "reference_version":
            "1.3.0",

        "reference_role":
            np.where(
                india[
                    "thermal_industrial_reference"
                ],
                "independent_positive_industrial_thermal_context",
                "reference_only_not_thermal_anchor",
            ),
    }
)


duplicate_ids = int(
    reference[
        "industrial_reference_id"
    ].duplicated().sum()
)

print(
    f"      Duplicate IDs: "
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


thermal = reference[
    reference[
        "thermal_industrial_reference"
    ]
].copy()


print(
    f"      Total India plants     : "
    f"{len(reference):,}"
)

print(
    f"      Thermal references     : "
    f"{len(thermal):,}"
)


print("\n      Thermal fuel distribution:")

print(
    thermal[
        "primary_fuel"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n      Thermal capacity summary (MW):"
)

print(
    thermal[
        "capacity_mw"
    ]
    .describe()
    .to_string()
)


# ============================================================
# 7. SAVE
# ============================================================

print("\n[7/7] Saving...")


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


fuel_counts = (
    reference[
        "primary_fuel"
    ]
    .fillna("MISSING")
    .value_counts()
    .to_dict()
)


thermal_fuel_counts = (
    thermal[
        "primary_fuel"
    ]
    .fillna("MISSING")
    .value_counts()
    .to_dict()
)


report = {
    "stage":
        "india_powerplant_reference_build",

    "source":
        "World Resources Institute",

    "dataset":
        "Global Power Plant Database",

    "version":
        "1.3.0",

    "global_rows":
        int(len(df)),

    "india_rows":
        int(len(reference)),

    "valid_coordinate_rows":
        int(
            reference[
                "coordinate_valid_india"
            ].sum()
        ),

    "thermal_industrial_references":
        int(len(thermal)),

    "fuel_distribution": {
        str(k): int(v)
        for k, v
        in fuel_counts.items()
    },

    "thermal_fuel_distribution": {
        str(k): int(v)
        for k, v
        in thermal_fuel_counts.items()
    },

    "thermal_reference_fuels": sorted(
        list(THERMAL_FUELS)
    ),

    "excluded_as_thermal_anchor": [
        "Solar",
        "Wind",
        "Hydro",
    ],

    "interpretation": (
        "Thermal-reference membership indicates "
        "independent industrial power-generation "
        "context. It does not prove that a nearby "
        "FIRMS observation was caused by the plant "
        "or represents an industrial fire."
    ),

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
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print("\n" + "=" * 80)
print("POWER-PLANT REFERENCE BUILD COMPLETE")
print("=" * 80)

print(
    f"\nIndia plants       : "
    f"{len(reference):,}"
)

print(
    f"Thermal references : "
    f"{len(thermal):,}"
)

print("\nOutput:")
print(OUTPUT_FILE)

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
    "\nCombine these power-plant references with the"
    "\noil/gas references and measure independent"
    "\nindustrial coverage of the 20,000 observations."
)