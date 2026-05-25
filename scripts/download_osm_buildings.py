"""Download OpenStreetMap building geometries for the selected Brisbane AOIs.

The script queries Overpass by AOI bounding box, converts OSM ways/relations to
GeoJSON polygons, deduplicates by OSM element id, and optionally clips the
result to the selected AOI polygons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import LineString, MultiPolygon, Polygon, shape
from shapely.ops import polygonize, unary_union


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def query_overpass(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    timeout: int,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["building"]({south:.8f},{west:.8f},{north:.8f},{east:.8f});
      relation["building"]({south:.8f},{west:.8f},{north:.8f},{east:.8f});
    );
    out body geom;
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=timeout + 30,
                headers={"User-Agent": "PipeNetworkCompletion OSM building downloader"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Overpass request failed after {retries} attempts") from last_error


def way_geometry(element: dict[str, Any]) -> Polygon | None:
    coords = [(pt["lon"], pt["lat"]) for pt in element.get("geometry", [])]
    if len(coords) < 4:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    polygon = Polygon(coords)
    if polygon.is_empty or not polygon.is_valid or polygon.area == 0:
        polygon = polygon.buffer(0)
    return polygon if not polygon.is_empty else None


def relation_geometry(element: dict[str, Any]) -> Polygon | MultiPolygon | None:
    lines = []
    for member in element.get("members", []):
        if member.get("role") != "outer":
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in member.get("geometry", [])]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    if not lines:
        return None
    polygons = list(polygonize(lines))
    if not polygons:
        return None
    merged = unary_union(polygons)
    if merged.is_empty:
        return None
    if not merged.is_valid:
        merged = merged.buffer(0)
    return merged if not merged.is_empty else None


def element_to_feature(element: dict[str, Any], aoi_id: str) -> dict[str, Any] | None:
    if element.get("type") == "way":
        geom = way_geometry(element)
    elif element.get("type") == "relation":
        geom = relation_geometry(element)
    else:
        geom = None
    if geom is None:
        return None

    tags = dict(element.get("tags", {}))
    properties: dict[str, Any] = {
        "osm_id": str(element.get("id")),
        "osm_type": element.get("type"),
        "aoi_id": aoi_id,
    }
    for key in [
        "building",
        "building:levels",
        "height",
        "name",
        "amenity",
        "addr:housenumber",
        "addr:street",
        "addr:suburb",
        "source",
    ]:
        if key in tags:
            properties[key.replace(":", "_")] = tags[key]
    properties["all_tags_json"] = json.dumps(tags, sort_keys=True)
    return {"geometry": geom, "properties": properties}


def build_aoi_windows(aoi_path: Path, max_aois: int | None) -> gpd.GeoDataFrame:
    aois = gpd.read_file(aoi_path)
    if aois.empty:
        raise ValueError(f"No AOIs found in {aoi_path}")
    if aois.crs is None:
        raise ValueError(f"AOI file has no CRS: {aoi_path}")
    aois = aois.to_crs("EPSG:4326")
    if "aoi_id" not in aois.columns:
        aois["aoi_id"] = [f"aoi_{i:04d}" for i in range(len(aois))]
    if max_aois is not None:
        aois = aois.head(max_aois).copy()
    return aois


def download_buildings(args: argparse.Namespace) -> gpd.GeoDataFrame:
    aois = build_aoi_windows(args.aoi_path, args.max_aois)
    records: dict[tuple[str, str], dict[str, Any]] = {}

    for idx, row in aois.iterrows():
        west, south, east, north = row.geometry.bounds
        pad = args.bbox_padding_degrees
        west -= pad
        south -= pad
        east += pad
        north += pad
        aoi_id = str(row["aoi_id"])
        print(
            f"[{idx + 1}/{len(aois)}] querying {aoi_id}: "
            f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}",
            flush=True,
        )
        payload = query_overpass(
            south,
            west,
            north,
            east,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
        )
        for element in payload.get("elements", []):
            feature = element_to_feature(element, aoi_id)
            if feature is None:
                continue
            key = (feature["properties"]["osm_type"], feature["properties"]["osm_id"])
            records.setdefault(key, feature)
        time.sleep(args.sleep_seconds)

    gdf = gpd.GeoDataFrame(
        [feature["properties"] for feature in records.values()],
        geometry=[feature["geometry"] for feature in records.values()],
        crs="EPSG:4326",
    )
    if args.clip_to_aois and not gdf.empty:
        clip_geom = unary_union(aois.geometry)
        gdf = gdf[gdf.intersects(clip_geom)].copy()
        gdf["geometry"] = gdf.geometry.intersection(clip_geom)
        gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aoi-path",
        type=Path,
        default=Path("data/processed/aois/anchor_free_2km_gap500_115/selected_aois.geojson"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/context/buildings/osm_buildings_selected_aois.geojson"),
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--max-aois", type=int)
    parser.add_argument("--bbox-padding-degrees", type=float, default=0.0001)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--clip-to-aois", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    buildings = download_buildings(args)
    if args.target_crs:
        buildings = buildings.to_crs(args.target_crs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(args.output, driver="GeoJSON")
    print(f"Wrote {len(buildings)} OSM building features to {args.output}")
    if len(buildings):
        print("Building tag counts:")
        print(buildings["building"].fillna("").value_counts().head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
