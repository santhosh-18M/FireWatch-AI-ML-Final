from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = ROOT / "data" / "satellite" / "exports" / "full"

files = [
    EXPORT_DIR / f"firewatch_sentinel_full_batch_{i:02d}.csv"
    for i in range(1, 5)
]

dfs = [
    pd.read_csv(f, low_memory=False)
    for f in files
]

df = pd.concat(dfs, ignore_index=True)

sat_cols = [
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "NBR",
    "NDMI"
]

available = (
    df["satellite_available"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })
)

mismatch = df[
    (~available)
    & df[sat_cols].notna().any(axis=1)
].copy()

print("=" * 90)
print("SENTINEL AVAILABILITY MISMATCH INSPECTION")
print("=" * 90)

print(f"\nTotal rows       : {len(df):,}")
print(f"Mismatch rows    : {len(mismatch):,}")

print("\nExtraction status:")
print(
    mismatch["satellite_extraction_status"]
    .value_counts(dropna=False)
)

print("\nScene count:")
print(
    mismatch["sentinel_scene_count"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nValid B8 pixel count:")
print(
    mismatch["sentinel_valid_pixel_count"]
    .describe()
)

print("\nNon-null values among mismatch rows:")

for col in sat_cols:
    print(
        f"{col:5s}: "
        f"{mismatch[col].notna().sum():3d} / {len(mismatch)}"
    )

print("\nMissing-value patterns:")

patterns = (
    mismatch[sat_cols]
    .notna()
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
    .value_counts()
)

for pattern, count in patterns.items():

    present = [
        col
        for col, flag in zip(sat_cols, pattern)
        if flag == "1"
    ]

    print(
        f"{count:3d} rows -> "
        f"{', '.join(present) if present else 'NONE'}"
    )

display_cols = [
    "observation_id",
    "full_export_batch",
    "event_time",
    "sentinel_scene_count",
    "sentinel_valid_pixel_count",
    "satellite_available",
    "satellite_extraction_status",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "NBR",
    "NDMI"
]

print("\nDetailed mismatch rows:")
print(
    mismatch[display_cols]
    .to_string(index=False)
)

OUTPUT = (
    ROOT
    / "reports"
    / "experiments"
    / "sentinel_availability_mismatch_16.csv"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

mismatch[display_cols].to_csv(
    OUTPUT,
    index=False
)

print(f"\nSaved inspection file:\n{OUTPUT}")