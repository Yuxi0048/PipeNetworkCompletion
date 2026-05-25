"""Derive point anchors and type features from OSM building footprints.

This is an additive preprocessing utility for the anchor-augmented experiments.
It does not modify the original ISARC 2024-style manhole graph artifacts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd


BUILDING_GROUPS = {
    "residential": {
        "apartments",
        "bungalow",
        "cabin",
        "detached",
        "dormitory",
        "farm",
        "ger",
        "house",
        "houseboat",
        "residential",
        "semidetached_house",
        "static_caravan",
        "terrace",
    },
    "ancillary": {
        "barn",
        "carport",
        "conservatory",
        "garage",
        "garages",
        "greenhouse",
        "hut",
        "roof",
        "shed",
    },
    "commercial": {
        "commercial",
        "hotel",
        "kiosk",
        "office",
        "retail",
        "supermarket",
    },
    "industrial": {
        "construction",
        "hangar",
        "industrial",
        "service",
        "warehouse",
    },
    "institutional": {
        "civic",
        "college",
        "government",
        "hospital",
        "kindergarten",
        "public",
        "religious",
        "school",
        "university",
    },
    "utility": {
        "service",
        "transformer_tower",
        "transportation",
        "water_tower",
    },
}

AMENITY_TO_GROUP = {
    "school": "institutional",
    "university": "institutional",
    "college": "institutional",
    "kindergarten": "institutional",
    "hospital": "institutional",
    "clinic": "institutional",
    "place_of_worship": "institutional",
    "community_centre": "institutional",
    "fire_station": "institutional",
    "police": "institutional",
    "restaurant": "commercial",
    "cafe": "commercial",
    "fast_food": "commercial",
    "pub": "commercial",
    "bar": "commercial",
    "fuel": "commercial",
}


def safe_column(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return value or "unknown"


def normalize_building_type(building: object, amenity: object = None) -> str:
    amenity_value = "" if pd.isna(amenity) else str(amenity).strip().lower()
    if amenity_value in AMENITY_TO_GROUP:
        return AMENITY_TO_GROUP[amenity_value]

    value = "" if pd.isna(building) else str(building).strip().lower()
    if value in {"", "yes", "building"}:
        return "unknown"
    for group, values in BUILDING_GROUPS.items():
        if value in values:
            return group
    return "other"


def point_on_or_near_surface(footprints: gpd.GeoDataFrame) -> tuple[gpd.GeoSeries, pd.Series]:
    centroids = footprints.geometry.centroid
    inside = centroids.within(footprints.geometry)
    representative = footprints.geometry.representative_point()
    points = centroids.copy()
    points.loc[~inside] = representative.loc[~inside]
    method = pd.Series("centroid", index=footprints.index)
    method.loc[~inside] = "representative_point_fallback"
    return points, method


def derive_anchor_points(
    footprints: gpd.GeoDataFrame,
    *,
    target_crs: str,
    top_raw_types: int,
) -> gpd.GeoDataFrame:
    if footprints.crs is None:
        raise ValueError("OSM building footprint layer has no CRS")
    footprints = footprints.to_crs(target_crs).copy()
    footprints = footprints[~footprints.geometry.is_empty & footprints.geometry.notna()].copy()
    footprints = footprints.reset_index(drop=True)

    raw_type = (
        footprints["building"].fillna("unknown").astype(str)
        if "building" in footprints.columns
        else pd.Series("unknown", index=footprints.index)
    )
    amenity = (
        footprints["amenity"].fillna("")
        if "amenity" in footprints.columns
        else pd.Series("", index=footprints.index)
    )
    group = pd.Series(
        [normalize_building_type(b, a) for b, a in zip(raw_type, amenity)],
        index=footprints.index,
    )

    points, method = point_on_or_near_surface(footprints)
    centroids = footprints.geometry.centroid

    out = gpd.GeoDataFrame(
        {
            "anchor_id": [f"osm_building_{i:07d}" for i in range(len(footprints))],
            "anchor_family": "building",
            "anchor_source": "osm_building_footprint",
            "visibility": "surface_inferred_from_osm",
            "is_utility_asset": False,
            "is_surface_visible": True,
            "building_type_raw": raw_type.values,
            "building_type_group": group.values,
            "point_method": method.values,
            "footprint_area_m2": footprints.geometry.area.values,
            "footprint_perimeter_m": footprints.geometry.length.values,
            "centroid_x": centroids.x.values,
            "centroid_y": centroids.y.values,
        },
        geometry=points,
        crs=footprints.crs,
    )
    for optional_col in ["osm_id", "osm_type", "aoi_id", "name", "amenity"]:
        if optional_col in footprints.columns:
            out[optional_col] = footprints[optional_col].values

    for group_name in sorted(set(BUILDING_GROUPS) | {"unknown", "other"}):
        out[f"bt_group_{safe_column(group_name)}"] = (
            out["building_type_group"] == group_name
        ).astype(int)

    raw_counts = raw_type.value_counts().head(int(top_raw_types))
    for raw_value in raw_counts.index:
        col = f"bt_raw_{safe_column(raw_value)}"
        out[col] = (out["building_type_raw"] == raw_value).astype(int)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--footprints",
        type=Path,
        default=Path("data/raw/context/buildings/osm_buildings_selected_aois.geojson"),
    )
    parser.add_argument(
        "--output-points",
        type=Path,
        default=Path("data/processed/context/study_area/osm_building_anchor_points.geojson"),
    )
    parser.add_argument(
        "--output-features",
        type=Path,
        default=Path("data/processed/context/study_area/osm_building_anchor_features.csv"),
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--top-raw-types", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    footprints = gpd.read_file(args.footprints)
    anchors = derive_anchor_points(
        footprints,
        target_crs=args.target_crs,
        top_raw_types=args.top_raw_types,
    )
    args.output_points.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_file(args.output_points, driver="GeoJSON")

    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    feature_table = anchors.drop(columns="geometry").copy()
    feature_table.to_csv(args.output_features, index=False)

    print(f"Wrote {len(anchors)} OSM building anchor points: {args.output_points}")
    print(f"Wrote feature table: {args.output_features}")
    print("Building type group counts:")
    print(anchors["building_type_group"].value_counts().to_string())
    print("Point method counts:")
    print(anchors["point_method"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
