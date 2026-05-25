"""Compare building terminal definitions for representing sewer service laterals.

The original anchor graph assumes a line can be represented only when both line
endpoints match selected anchor supports. For service laterals, the second
endpoint is often a building/property terminal rather than a sewer manhole. This
script compares candidate terminal definitions such as OSM building centroids,
OSM footprint polygons, government building points, and built-up areas.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import fiona
import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.prepared import prep


VISIBLE_UTILITY_ANCHORS = [
    "SewerManholes_ExportFeatures.shp",
    "SewersqlSewerP_ExportFeature.shp",
    "SewerPumpStati_ExportFeature.shp",
    "SewerVent_ExportFeatures.shp",
    "SewerControlVa_ExportFeature.shp",
    "SewerSystemVal_ExportFeature.shp",
    "SewerDevice_ExportFeatures.shp",
    "UUSewertreatme_ExportFeature.shp",
]

OSM_ANCILLARY_BUILDINGS = {
    "barn",
    "carport",
    "conservatory",
    "garage",
    "garages",
    "greenhouse",
    "hut",
    "roof",
    "shed",
}


def transformer_for(src: fiona.Collection, target_crs: CRS) -> Transformer:
    source = CRS.from_user_input(src.crs_wkt or src.crs)
    return Transformer.from_crs(source, target_crs, always_xy=True)


def load_utility_anchor_coords(sewer_root: Path, target_crs: CRS) -> np.ndarray:
    coords = []
    for filename in VISIBLE_UTILITY_ANCHORS:
        path = sewer_root / filename
        with fiona.open(path) as src:
            transformer = transformer_for(src, target_crs)
            for feature in src:
                geom = feature.get("geometry")
                if not geom or geom.get("type") != "Point":
                    continue
                x, y = geom["coordinates"][:2]
                coords.append(transformer.transform(x, y))
    return np.asarray(coords, dtype=float)


def line_endpoints(geometry: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if geometry is None:
        return None
    coords = geometry.get("coordinates")
    if not coords:
        return None
    geom_type = geometry.get("type")
    if geom_type == "LineString" and len(coords) >= 2:
        return tuple(coords[0][:2]), tuple(coords[-1][:2])
    if geom_type == "MultiLineString":
        parts = [part for part in coords if len(part) >= 2]
        if not parts:
            return None
        return tuple(parts[0][0][:2]), tuple(parts[-1][-1][:2])
    return None


def load_service_lateral_endpoints(
    sewer_root: Path,
    target_crs: CRS,
    aoi_union,
) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    prepared_aoi = prep(aoi_union)
    path = sewer_root / "SewerService_ExportFeatures.shp"
    with fiona.open(path) as src:
        transformer = transformer_for(src, target_crs)
        for feature in src:
            endpoints = line_endpoints(feature.get("geometry"))
            if endpoints is None:
                continue
            start, end = endpoints
            start_xy = transformer.transform(*start)
            end_xy = transformer.transform(*end)
            # Use the endpoint chord for a fast AOI intersection test. Service
            # laterals are short, so this is sufficient for support auditing.
            if prepared_aoi.intersects(LineString([start_xy, end_xy])):
                starts.append(start_xy)
                ends.append(end_xy)
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def read_clipped_layer(path: Path, target_crs: str, aoi_union) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Layer has no CRS: {path}")
    gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.intersects(aoi_union)].copy()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    return gdf


def point_distances(coords: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(coords)
    start_dist, _ = tree.query(starts, k=1)
    end_dist, _ = tree.query(ends, k=1)
    return start_dist, end_dist


def polygon_distances(
    polygons: gpd.GeoDataFrame,
    starts: np.ndarray,
    ends: np.ndarray,
    target_crs: str,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    all_points = gpd.GeoDataFrame(
        {"endpoint_id": np.arange(len(starts) + len(ends), dtype=int)},
        geometry=[Point(xy) for xy in np.vstack([starts, ends])],
        crs=target_crs,
    )
    nearest = gpd.sjoin_nearest(
        all_points,
        polygons[["geometry"]],
        how="left",
        max_distance=max_distance,
        distance_col="distance_m",
    )
    min_dist = nearest.groupby("endpoint_id")["distance_m"].min()
    distances = np.full(len(all_points), np.inf, dtype=float)
    valid = min_dist.dropna()
    distances[valid.index.to_numpy(dtype=int)] = valid.to_numpy(dtype=float)
    return distances[: len(starts)], distances[len(starts) :]


def evaluate_candidate(
    name: str,
    candidate_kind: str,
    candidate_count: int,
    utility_start_dist: np.ndarray,
    utility_end_dist: np.ndarray,
    candidate_start_dist: np.ndarray | None,
    candidate_end_dist: np.ndarray | None,
    tolerances: Iterable[float],
) -> list[dict[str, object]]:
    rows = []
    total = len(utility_start_dist)
    for tolerance in tolerances:
        start_hit = utility_start_dist <= tolerance
        end_hit = utility_end_dist <= tolerance
        if candidate_start_dist is not None and candidate_end_dist is not None:
            start_hit = start_hit | (candidate_start_dist <= tolerance)
            end_hit = end_hit | (candidate_end_dist <= tolerance)
        both = start_hit & end_hit
        one_or_more = start_hit | end_hit
        rows.append(
            {
                "candidate": name,
                "candidate_kind": candidate_kind,
                "candidate_count": int(candidate_count),
                "tolerance_m": float(tolerance),
                "total_service_laterals": int(total),
                "both_supported": int(both.sum()),
                "orphan_lines": int((~both).sum()),
                "orphan_fraction": float((~both).mean()),
                "one_or_more_endpoint": int(one_or_more.sum()),
                "zero_endpoint": int((~one_or_more).sum()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sewer-root", type=Path, default=Path("data/raw/gis/sewer"))
    parser.add_argument(
        "--aoi-path",
        type=Path,
        default=Path("data/processed/aois/anchor_free_2km_gap500_115/selected_aois.geojson"),
    )
    parser.add_argument(
        "--osm-footprints",
        type=Path,
        default=Path("data/raw/context/buildings/osm_buildings_selected_aois.geojson"),
    )
    parser.add_argument(
        "--osm-centroids",
        type=Path,
        default=Path("data/processed/context/study_area/osm_building_anchor_points.geojson"),
    )
    parser.add_argument(
        "--government-building-points",
        type=Path,
        default=Path("data/processed/context/study_area/building_points_study_area.geojson"),
    )
    parser.add_argument(
        "--government-building-areas",
        type=Path,
        default=Path("data/processed/context/study_area/building_areas_study_area.geojson"),
    )
    parser.add_argument(
        "--built-up-areas",
        type=Path,
        default=Path("data/processed/context/study_area/build_up_areas_study_area.geojson"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/service_lateral_anchor_alternatives/service_lateral_terminal_support.csv"),
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--tolerances-m", type=float, nargs="+", default=[5, 10, 20, 30, 50])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_crs = CRS.from_user_input(args.target_crs)
    aoi = gpd.read_file(args.aoi_path).to_crs(args.target_crs)
    aoi_union = aoi.geometry.unary_union

    utility_coords = load_utility_anchor_coords(args.sewer_root, target_crs)
    starts, ends = load_service_lateral_endpoints(args.sewer_root, target_crs, aoi_union)
    print(f"Visible utility anchors: {len(utility_coords)}")
    print(f"Service laterals intersecting AOIs: {len(starts)}")

    utility_start_dist, utility_end_dist = point_distances(utility_coords, starts, ends)
    rows = evaluate_candidate(
        "visible_utility_only",
        "utility_points",
        len(utility_coords),
        utility_start_dist,
        utility_end_dist,
        None,
        None,
        args.tolerances_m,
    )

    max_tolerance = max(args.tolerances_m)

    osm_centroids = read_clipped_layer(args.osm_centroids, args.target_crs, aoi_union)
    osm_centroid_coords = np.asarray([(point.x, point.y) for point in osm_centroids.geometry], dtype=float)
    start_dist, end_dist = point_distances(osm_centroid_coords, starts, ends)
    rows += evaluate_candidate(
        "visible_utility_plus_osm_centroids",
        "point",
        len(osm_centroids),
        utility_start_dist,
        utility_end_dist,
        start_dist,
        end_dist,
        args.tolerances_m,
    )

    gov_points = read_clipped_layer(args.government_building_points, args.target_crs, aoi_union)
    gov_point_coords = np.asarray([(point.x, point.y) for point in gov_points.geometry], dtype=float)
    start_dist, end_dist = point_distances(gov_point_coords, starts, ends)
    rows += evaluate_candidate(
        "visible_utility_plus_government_building_points",
        "point",
        len(gov_points),
        utility_start_dist,
        utility_end_dist,
        start_dist,
        end_dist,
        args.tolerances_m,
    )

    polygon_candidates = []
    osm_footprints = read_clipped_layer(args.osm_footprints, args.target_crs, aoi_union)
    polygon_candidates.append(("visible_utility_plus_osm_footprints_all", "polygon", osm_footprints))
    if "building" in osm_footprints.columns:
        raw = osm_footprints["building"].fillna("").astype(str).str.lower()
        osm_demand = osm_footprints[~raw.isin(OSM_ANCILLARY_BUILDINGS)].copy()
        polygon_candidates.append(
            ("visible_utility_plus_osm_footprints_no_ancillary", "polygon", osm_demand)
        )

    gov_areas = read_clipped_layer(args.government_building_areas, args.target_crs, aoi_union)
    polygon_candidates.append(("visible_utility_plus_government_building_areas", "polygon", gov_areas))

    built_up = read_clipped_layer(args.built_up_areas, args.target_crs, aoi_union)
    polygon_candidates.append(("visible_utility_plus_built_up_areas", "polygon_zone", built_up))

    all_building_polygons = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(
            list(osm_footprints.geometry) + list(gov_areas.geometry),
            crs=args.target_crs,
        ),
        crs=args.target_crs,
    )
    polygon_candidates.append(
        ("visible_utility_plus_osm_and_government_footprints", "polygon", all_building_polygons)
    )

    for name, kind, polygons in polygon_candidates:
        print(f"Evaluating {name}: {len(polygons)} geometries")
        start_dist, end_dist = polygon_distances(
            polygons,
            starts,
            ends,
            args.target_crs,
            max_distance=max_tolerance,
        )
        rows += evaluate_candidate(
            name,
            kind,
            len(polygons),
            utility_start_dist,
            utility_end_dist,
            start_dist,
            end_dist,
            args.tolerances_m,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote comparison: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
