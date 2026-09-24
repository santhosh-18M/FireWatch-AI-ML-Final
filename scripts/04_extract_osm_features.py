from pathlib import Path
import json
import time

import osmium
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PBF_FILE = (
    ROOT
    / "data"
    / "osm"
    / "filtered"
    / "firewatch_context.osm.pbf"
)

INDUSTRIAL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "osm_industrial_features.parquet"
)

VEGETATION_FILE = (
    ROOT
    / "data"
    / "processed"
    / "osm_vegetation_features.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "experiments"
    / "osm_extraction_summary.json"
)

INDUSTRIAL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONTEXT RULES
# ============================================================

INDUSTRIAL_RULES = {

    ("landuse", "industrial"):
        ("industrial_landuse", 3),

    ("landuse", "quarry"):
        ("mining", 3),

    ("power", "plant"):
        ("power_plant", 3),

    ("power", "generator"):
        ("power_generation", 2),

    ("power", "substation"):
        ("power_substation", 1),

    ("man_made", "works"):
        ("manufacturing", 3),

    ("man_made", "storage_tank"):
        ("storage", 2),

    ("man_made", "petroleum_well"):
        ("oil_gas", 3),

    ("man_made", "kiln"):
        ("kiln", 3),

    ("man_made", "chimney"):
        ("industrial_chimney", 2),

    ("man_made", "silo"):
        ("storage", 1),

    ("man_made", "mineshaft"):
        ("mining", 3),

    ("man_made", "adit"):
        ("mining", 3),

    ("amenity", "fuel"):
        ("fuel_facility", 1),
}


VEGETATION_RULES = {

    ("landuse", "forest"):
        ("forest", 3),

    ("natural", "wood"):
        ("forest_woodland", 3),

    ("landuse", "farmland"):
        ("farmland", 3),

    ("landuse", "farmyard"):
        ("agricultural", 2),

    ("landuse", "orchard"):
        ("orchard", 3),

    ("landuse", "vineyard"):
        ("vineyard", 3),

    ("landuse", "meadow"):
        ("grass_meadow", 2),

    ("landuse", "grass"):
        ("grassland", 2),

    ("natural", "grassland"):
        ("grassland", 3),

    ("natural", "grass"):
        ("grassland", 2),

    ("natural", "scrub"):
        ("scrub", 3),

    ("natural", "heath"):
        ("heath", 2),
}


# ============================================================
# TAG MATCHING
# ============================================================

def inspect_tags(tags):

    industrial = []
    vegetation = []

    name = None

    for tag in tags:

        key = tag.k
        value = tag.v

        if (
            name is None
            and key in (
                "name",
                "operator",
                "brand"
            )
        ):
            name = value

        rule = INDUSTRIAL_RULES.get(
            (key, value)
        )

        if rule:

            category, strength = rule

            industrial.append(
                {
                    "key": key,
                    "value": value,
                    "category": category,
                    "strength": strength
                }
            )

        # industrial=* wildcard
        if key == "industrial" and value:

            industrial.append(
                {
                    "key": key,
                    "value": value,
                    "category": "industrial_tag",
                    "strength": 3
                }
            )

        rule = VEGETATION_RULES.get(
            (key, value)
        )

        if rule:

            category, strength = rule

            vegetation.append(
                {
                    "key": key,
                    "value": value,
                    "category": category,
                    "strength": strength
                }
            )

    return industrial, vegetation, name


def strongest(matches):

    return max(
        matches,
        key=lambda x: x["strength"]
    )


def serialize(matches):

    return ";".join(
        f'{x["key"]}={x["value"]}'
        for x in matches
    )


# ============================================================
# HANDLER
# ============================================================

class ContextHandler(osmium.SimpleHandler):

    def __init__(self):

        super().__init__()

        self.industrial = []
        self.vegetation = []

        self.nodes = 0
        self.ways = 0
        self.relations = 0

        self.matched_nodes = 0
        self.matched_ways = 0

        self.invalid_geometry = 0

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(
        self,
        osm_type,
        osm_id,
        lat,
        lon,
        industrial,
        vegetation,
        name
    ):

        if not (
            -90 <= lat <= 90
            and -180 <= lon <= 180
        ):

            self.invalid_geometry += 1
            return

        if industrial:

            best = strongest(
                industrial
            )

            self.industrial.append(
                {
                    "osm_type": osm_type,
                    "osm_id": int(osm_id),

                    "latitude": float(lat),
                    "longitude": float(lon),

                    "context_category":
                        best["category"],

                    "evidence_strength":
                        int(best["strength"]),

                    "feature_key":
                        best["key"],

                    "feature_value":
                        best["value"],

                    "matched_tags":
                        serialize(industrial),

                    "name": name
                }
            )

        if vegetation:

            best = strongest(
                vegetation
            )

            self.vegetation.append(
                {
                    "osm_type": osm_type,
                    "osm_id": int(osm_id),

                    "latitude": float(lat),
                    "longitude": float(lon),

                    "context_category":
                        best["category"],

                    "evidence_strength":
                        int(best["strength"]),

                    "feature_key":
                        best["key"],

                    "feature_value":
                        best["value"],

                    "matched_tags":
                        serialize(vegetation),

                    "name": name
                }
            )

    # --------------------------------------------------------
    # NODE
    # --------------------------------------------------------

    def node(self, n):

        self.nodes += 1

        industrial, vegetation, name = (
            inspect_tags(n.tags)
        )

        # Most nodes in filtered PBF are geometry-support
        # nodes for matching ways and have no matching tags.

        if not industrial and not vegetation:
            return

        self.matched_nodes += 1

        if not n.location.valid():

            self.invalid_geometry += 1
            return

        self.save(
            "node",
            n.id,
            n.location.lat,
            n.location.lon,
            industrial,
            vegetation,
            name
        )

    # --------------------------------------------------------
    # WAY
    # --------------------------------------------------------

    def way(self, w):

        self.ways += 1

        industrial, vegetation, name = (
            inspect_tags(w.tags)
        )

        if not industrial and not vegetation:
            return

        self.matched_ways += 1

        locations = []

        for node in w.nodes:

            if node.location.valid():

                locations.append(
                    (
                        node.location.lat,
                        node.location.lon
                    )
                )

        if not locations:

            self.invalid_geometry += 1
            return

        # Representative point only.
        #
        # We explicitly do NOT describe this as exact
        # polygon-boundary distance.

        lat = sum(
            x[0] for x in locations
        ) / len(locations)

        lon = sum(
            x[1] for x in locations
        ) / len(locations)

        self.save(
            "way",
            w.id,
            lat,
            lon,
            industrial,
            vegetation,
            name
        )

    # --------------------------------------------------------
    # RELATION
    # --------------------------------------------------------

    def relation(self, r):

        self.relations += 1


# ============================================================
# RUN
# ============================================================

print("=" * 80)
print("FIREWATCH — OSM CONTEXT EXTRACTION")
print("=" * 80)

if not PBF_FILE.exists():

    raise FileNotFoundError(
        f"Filtered OSM file missing:\n"
        f"{PBF_FILE}"
    )

size_mb = (
    PBF_FILE.stat().st_size
    / 1024**2
)

print(
    f"\nFiltered PBF:\n{PBF_FILE}"
)

print(
    f"\nSize: {size_mb:.2f} MB"
)

print(
    "\nIMPORTANT:"
    "\nOSM data is contextual evidence only."
    "\nNo fire labels are created."
)


# ============================================================
# PARSE
# ============================================================

print(
    "\n[1/4] Reading filtered OSM..."
)

start = time.time()

handler = ContextHandler()

handler.apply_file(
    str(PBF_FILE),
    locations=True
)

runtime = time.time() - start

print(
    f"      Complete in "
    f"{runtime:.2f} seconds."
)


# ============================================================
# TABLES
# ============================================================

print(
    "\n[2/4] Building evidence tables..."
)

industrial_df = pd.DataFrame(
    handler.industrial
)

vegetation_df = pd.DataFrame(
    handler.vegetation
)

if industrial_df.empty:

    raise RuntimeError(
        "No industrial context extracted."
    )

if vegetation_df.empty:

    raise RuntimeError(
        "No vegetation context extracted."
    )


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n[3/4] Validating..."
)

for label, df in [
    ("Industrial", industrial_df),
    ("Vegetation", vegetation_df)
]:

    invalid = (
        ~df["latitude"].between(-90, 90)
        |
        ~df["longitude"].between(
            -180,
            180
        )
    ).sum()

    duplicates = df.duplicated(
        [
            "osm_type",
            "osm_id"
        ]
    ).sum()

    print(
        f"      {label}: "
        f"{len(df):,} features"
    )

    print(
        f"      {label} invalid coordinates: "
        f"{invalid:,}"
    )

    print(
        f"      {label} duplicate object IDs: "
        f"{duplicates:,}"
    )

    if invalid:

        raise RuntimeError(
            f"{label} has invalid coordinates."
        )


# ============================================================
# SAVE
# ============================================================

print(
    "\n[4/4] Saving..."
)

industrial_df.to_parquet(
    INDUSTRIAL_FILE,
    index=False
)

vegetation_df.to_parquet(
    VEGETATION_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report = {

    "filtered_pbf":
        str(PBF_FILE),

    "filtered_pbf_size_mb":
        float(size_mb),

    "runtime_seconds":
        float(runtime),

    "objects_read": {

        "nodes":
            int(handler.nodes),

        "ways":
            int(handler.ways),

        "relations":
            int(handler.relations)
    },

    "matching_objects": {

        "nodes":
            int(handler.matched_nodes),

        "ways":
            int(handler.matched_ways)
    },

    "industrial_features":
        int(len(industrial_df)),

    "vegetation_features":
        int(len(vegetation_df)),

    "industrial_categories": {
        str(k): int(v)
        for k, v
        in industrial_df[
            "context_category"
        ].value_counts().items()
    },

    "vegetation_categories": {
        str(k): int(v)
        for k, v
        in vegetation_df[
            "context_category"
        ].value_counts().items()
    },

    "methodology": [
        (
            "OSM provides contextual evidence, "
            "not ground-truth fire labels."
        ),

        (
            "Industrial and vegetation/agricultural "
            "contexts are stored independently."
        ),

        (
            "Ways use representative mean-node "
            "coordinates for scalable contextual "
            "proximity analysis."
        ),

        (
            "Representative-point distance must not "
            "be described as exact polygon-boundary "
            "distance."
        ),

        (
            "The 2025 FIRMS holdout was not used."
        )
    ]
}


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


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "OSM EXTRACTION SUCCESSFUL"
)

print(
    "=" * 80
)

print(
    f"\nObjects read:"
    f"\n  Nodes: {handler.nodes:,}"
    f"\n  Ways: {handler.ways:,}"
    f"\n  Relations: {handler.relations:,}"
)

print(
    f"\nActual matching objects:"
    f"\n  Nodes: {handler.matched_nodes:,}"
    f"\n  Ways: {handler.matched_ways:,}"
)

print(
    f"\nIndustrial context features: "
    f"{len(industrial_df):,}"
)

print(
    industrial_df[
        "context_category"
    ]
    .value_counts()
    .to_string()
)

print(
    f"\nVegetation/agricultural context features: "
    f"{len(vegetation_df):,}"
)

print(
    vegetation_df[
        "context_category"
    ]
    .value_counts()
    .to_string()
)

print(
    f"\nSaved:\n{INDUSTRIAL_FILE}"
)

print(
    f"\nSaved:\n{VEGETATION_FILE}"
)

print(
    f"\nReport:\n{REPORT_FILE}"
)

print(
    "\nNo fire labels created."
    "\n2025 FIRMS holdout untouched."
)